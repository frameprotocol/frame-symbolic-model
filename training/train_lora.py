#!/usr/bin/env python3
"""Causal LM LoRA: <INPUT> NL <OUTPUT> symbolic program (per-family training)."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import config as cfg
from training.dataset import load_training_records
from training.tokenizer_load import (
    load_causal_lm_with_fallback,
    load_tokenizer_for_training,
)


def lora_target_modules(model_name: str) -> tuple[list[str], bool]:
    m = model_name.lower()
    if "gpt2" in m or "distilgpt" in m:
        return (["c_attn", "c_proj"], True)
    if "llama" in m or "tinyllama" in m or "qwen" in m or "phi" in m:
        return (["q_proj", "k_proj", "v_proj", "o_proj"], False)
    raise ValueError(f"Set LoRA target_modules for base model {model_name!r} in training/train_lora.py")


def encode_pair(tokenizer: Any, inp: str, output: str, max_length: int) -> dict[str, torch.Tensor]:
    prefix = "<INPUT>\n" + inp + "\n<OUTPUT>\n"
    suffix = output + "\n<|endoftext|>\n"
    ids_pre = tokenizer.encode(prefix, add_special_tokens=False)
    ids_suf = tokenizer.encode(suffix, add_special_tokens=False)
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
        self.prefix = "<INPUT>\n" + sample_nl + "\n<OUTPUT>\n"

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
            with torch.no_grad():
                ids = self.tokenizer.encode(self.prefix, return_tensors="pt", add_special_tokens=False).to(dev)
                out = self.model.generate(
                    ids,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
            full = self.tokenizer.decode(out[0], skip_special_tokens=False)
            tail = full.split("<OUTPUT>", 1)[-1].strip() if "<OUTPUT>" in full else full[:200]
            print(f"\n[SAMPLE step={step}] {tail[:280]!r}\n")
        except Exception as e:
            print(f"\n[SAMPLE step={step}] generate failed: {type(e).__name__}: {e}\n")
        finally:
            if was_training:
                self.model.train()
        return control


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
        "prompt_format": "<INPUT>\n{input}\n<OUTPUT>\n",
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

    records = load_training_records(
        ROOT,
        dataset_dir=dataset_dir,
        dedupe_by_program_hash=cfg.DEDUPE_BY_PROGRAM_HASH,
        rebalance_ops=cfg.REBALANCE_DATASET,
    )
    if not records:
        raise SystemExit("No training records; populate data/canonical.jsonl and data/generated.jsonl")

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
    if model_vocab is not None and tok_vocab == model_vocab:
        # Safe no-op resize path when tokenizer/model vocab already match.
        model.resize_token_embeddings(tok_vocab)
    else:
        print(
            f"Skipping resize_token_embeddings (tokenizer_vocab={tok_vocab}, model_vocab={model_vocab})."
        )

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
        learning_rate=cfg.LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch" if args.max_steps is None else "no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    if args.max_steps is not None:
        ta_kwargs["max_steps"] = args.max_steps
    else:
        ta_kwargs["num_train_epochs"] = cfg.EPOCHS
    targs = TrainingArguments(**ta_kwargs)

    sample_cb = SymbolicSampleCallback(model, tokenizer)
    trainer = Trainer(
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
