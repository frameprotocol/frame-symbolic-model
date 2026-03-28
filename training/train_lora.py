#!/usr/bin/env python3
"""Causal LM LoRA: <START> INPUT: NL OUTPUT: symbolic program <END> (per-family training)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
)
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import config as cfg
from training.dataset import load_family_dataset
from training.tokenizer_load import (
    load_causal_lm_with_fallback,
    load_tokenizer_for_training,
)


def lora_target_modules(model_name: str) -> tuple[list[str], bool]:
    m = model_name.lower()
    if "gpt2" in m or "distilgpt" in m:
        return (["c_attn", "c_proj"], True)
    if "pruned_base" in m:
        # Pruned model needs aggressive adaptation — include MLP layers
        return (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], False)
    if "llama" in m or "tinyllama" in m or "qwen" in m or "phi" in m:
        return (["q_proj", "k_proj", "v_proj", "o_proj"], False)
    raise ValueError(f"Set LoRA target_modules for base model {model_name!r} in training/train_lora.py")


def encode_pair(tokenizer: Any, inp: str, output: str, max_length: int) -> dict[str, torch.Tensor]:
    prefix = f"### Instruction:\nYou MUST output ONLY valid JSON. No extra text. No explanation.\nINPUT: {inp}\n\n### Response:\n"
    suffix = output
    ids_pre = tokenizer.encode(prefix, add_special_tokens=False)
    ids_suf = tokenizer.encode(suffix, add_special_tokens=False)
    # Append EOS so model learns to stop (lm_head knows EOS from pretraining)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        ids_suf = ids_suf + [eos_id]
    ids = (ids_pre + ids_suf)[:max_length]
    labels = ([-100] * len(ids_pre) + ids_suf)[:max_length]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    attn = [1] * len(ids)
    while len(ids) < max_length:
        ids.append(pad_id)
        labels.append(-100)
        attn.append(0)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SymbolicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict[str, str]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = self.records[i]
        return encode_pair(self.tokenizer, r["input"], r["output"], self.max_length)


class SymbolicSampleCallback(TrainerCallback):
    """Print greedy continuation for a fixed prompt every 100 steps."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, sample_nl: str = "get current time") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prefix = f"### Instruction:\nYou MUST output ONLY valid JSON. No extra text. No explanation.\nINPUT: {sample_nl}\n\n### Response:\n"

    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):  # type: ignore[no-untyped-def]
        step = int(state.global_step)
        if step == 0 or step % 100 != 0:
            return control
        was_training = self.model.training
        self.model.eval()
        try:
            dev = next(self.model.parameters()).device
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            eos_id = self.tokenizer.eos_token_id
            # Also stop at <END> token
            end_id = self.tokenizer.encode("<END>", add_special_tokens=False)
            stop_ids = [eos_id]
            if end_id:
                stop_ids.append(end_id[0])
            with torch.no_grad():
                ids = self.tokenizer.encode(self.prefix, return_tensors="pt", add_special_tokens=False).to(dev)
                out = self.model.generate(
                    ids,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=stop_ids,
                )
            full = self.tokenizer.decode(out[0], skip_special_tokens=False)
            tail = full.split("### Response:", 1)[-1].strip() if "### Response:" in full else full[:200]
            print(f"\n[SAMPLE step={step}] {tail[:280]!r}\n")
        except Exception as e:
            print(f"\n[SAMPLE step={step}] generate failed: {type(e).__name__}: {e}\n")
        finally:
            if was_training:
                self.model.train()
        return control


class AntiCollapseTrainer(Trainer):
    """Custom trainer that penalizes repeated tokens and missing <END>."""

    def __init__(self, end_token_id: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.end_token_id = end_token_id

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Standard cross-entropy loss (with label smoothing from TrainingArguments)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.args.label_smoothing_factor,
        )
        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Repetition penalty: penalize when consecutive token predictions are identical
        pred_tokens = shift_logits.argmax(dim=-1)  # (batch, seq)
        # Compare each token to the next
        if pred_tokens.size(-1) > 1:
            same_as_prev = (pred_tokens[:, 1:] == pred_tokens[:, :-1]).float()
            # Weight by how many valid (non-padding) positions
            valid_mask = (shift_labels[:, 1:] != -100).float()
            repeat_penalty = (same_as_prev * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            # Scale: 0.1 weight on repetition penalty
            loss = ce_loss + 0.1 * repeat_penalty
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss


def save_run_config(
    adapter_dir: Path,
    fam_root: Path,
    *,
    resolved_model: str,
    family: str,
    dataset: str | None,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()

    # Training config (detailed, for adapter directory)
    training_payload = {
        "FAMILY": family,
        "DATASET_DIR": dataset,
        "BASE_MODEL_NAME": cfg.BASE_MODEL_NAME,
        "BASE_MODEL_FALLBACK": cfg.BASE_MODEL_FALLBACK,
        "RESOLVED_BASE_MODEL_NAME": resolved_model,
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "LEARNING_RATE": cfg.LEARNING_RATE,
        "EPOCHS": cfg.EPOCHS,
        "MAX_LENGTH": cfg.MAX_LENGTH,
        "LORA_R": cfg.LORA_R,
        "LORA_ALPHA": cfg.LORA_ALPHA,
        "LORA_DROPOUT": cfg.LORA_DROPOUT,
        "DEDUPE_BY_PROGRAM_HASH": cfg.DEDUPE_BY_PROGRAM_HASH,
        "REBALANCE_DATASET": cfg.REBALANCE_DATASET,
        "timestamp": timestamp,
    }
    (adapter_dir / "training_config.json").write_text(
        json.dumps(training_payload, indent=2) + "\n", encoding="utf-8"
    )

    # Family-level config (concise, for family root directory)
    family_config = {
        "family": family,
        "base_model": resolved_model,
        "dataset": dataset,
        "timestamp": timestamp,
        "prompt_format": "### Instruction:\nYou MUST output ONLY valid JSON. No extra text. No explanation.\nINPUT: {input}\n\n### Response:\n",
    }
    (fam_root / "config.json").write_text(
        json.dumps(family_config, indent=2) + "\n", encoding="utf-8"
    )


def _family_paths(family: str) -> tuple[Path, Path]:
    """
    Return (adapter_dir, family_root) for models/{family}/adapter.
    """
    fam_root = ROOT / "models" / family
    adapter_dir = fam_root / "adapter"
    return adapter_dir, fam_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, help="Model family id (e.g. english, cjk, arabic, indic)")
    ap.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset directory override (expects canonical.jsonl and generated.jsonl)",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap (default: full epoch)",
    )
    args = ap.parse_args()

    adapter_dir, fam_root = _family_paths(args.family)
    dataset_dir = Path(args.dataset).expanduser().resolve() if args.dataset else None

    records = load_family_dataset(ROOT, args.family)
    print(f"Loaded {len(records)} training samples for family: {args.family}")

    model, resolved = load_causal_lm_with_fallback(
        cfg.BASE_MODEL_NAME, cfg.BASE_MODEL_FALLBACK
    )
    tokenizer, _used_symbolic = load_tokenizer_for_training(ROOT, resolved)

    model_vocab = None
    try:
        model_vocab = int(model.get_input_embeddings().num_embeddings)
    except Exception:
        model_vocab = None
    tok_vocab = len(tokenizer)
    if model_vocab is not None and tok_vocab != model_vocab:
        print(f"Resizing embeddings: {model_vocab} -> {tok_vocab}")
        model.resize_token_embeddings(tok_vocab)
    elif model_vocab is not None:
        model.resize_token_embeddings(tok_vocab)

    targets, fan_in = lora_target_modules(resolved)
    peft_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
        fan_in_fan_out=fan_in,
    )
    model = get_peft_model(model, peft_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = SymbolicDataset(records, tokenizer, cfg.MAX_LENGTH)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (fam_root / "merged").mkdir(parents=True, exist_ok=True)
    save_run_config(
        adapter_dir,
        fam_root,
        resolved_model=resolved,
        family=args.family,
        dataset=str(dataset_dir) if dataset_dir else None,
    )

    n = len(records)
    bs = cfg.BATCH_SIZE
    steps_per_epoch = max(1, (n + bs - 1) // bs)
    if args.max_steps is not None:
        effective_steps = args.max_steps
    else:
        effective_steps = steps_per_epoch * cfg.EPOCHS
    print(f"Dataset size: {n}")
    print(f"Effective steps: {effective_steps} (per_epoch={steps_per_epoch}, epochs={cfg.EPOCHS})")
    print(f"Device: {device}")
    print(f"Base model (resolved): {resolved}")
    print("Tokenizer: AutoTokenizer (base model)")
    print(f"Family: {args.family}")
    if dataset_dir:
        print(f"Dataset dir: {dataset_dir}")

    ta_kwargs: dict = dict(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        label_smoothing_factor=cfg.LABEL_SMOOTHING,
        logging_steps=10,
        save_strategy="epoch" if args.max_steps is None else "no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
    )
    if args.max_steps is not None:
        ta_kwargs["max_steps"] = args.max_steps
    else:
        ta_kwargs["num_train_epochs"] = cfg.EPOCHS
    targs = TrainingArguments(**ta_kwargs)

    # Resolve <END> token id for anti-collapse loss
    end_ids = tokenizer.encode("<END>", add_special_tokens=False)
    end_token_id = end_ids[0] if end_ids else None

    sample_cb = SymbolicSampleCallback(model, tokenizer)
    trainer = AntiCollapseTrainer(
        end_token_id=end_token_id,
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=default_data_collator,
        callbacks=[sample_cb],
    )
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    ex = ds[0]
    print("Example tokenized (first 32 ids):", ex["input_ids"][:32].tolist())
    print(f"Artifacts: {adapter_dir}")


if __name__ == "__main__":
    main()
