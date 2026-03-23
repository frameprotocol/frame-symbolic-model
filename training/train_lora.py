#!/usr/bin/env python3
"""Causal LM LoRA: <INPUT> NL <OUTPUT> symbolic program (Qwen + symbolic tokenizer)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
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
    EXTRA_SPECIAL,
    load_causal_lm_with_fallback,
    load_tokenizer_for_training,
)
from tokenizer.symbolic_pre import prepare_for_tokenizer


def lora_target_modules(model_name: str) -> tuple[list[str], bool]:
    m = model_name.lower()
    if "gpt2" in m or "distilgpt" in m:
        return (["c_attn", "c_proj"], True)
    if "llama" in m or "tinyllama" in m or "qwen" in m or "phi" in m:
        return (["q_proj", "k_proj", "v_proj", "o_proj"], False)
    raise ValueError(f"Set LoRA target_modules for base model {model_name!r} in training/train_lora.py")


def encode_pair(tokenizer: Any, inp: str, output: str, max_length: int) -> dict[str, torch.Tensor]:
    pre_out = prepare_for_tokenizer(output)
    prefix = "<INPUT>\n" + inp + "\n<OUTPUT>\n"
    suffix = pre_out + "\n<|endoftext|>\n"
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
    out_dir: Path,
    *,
    resolved_model: str,
    used_symbolic_tokenizer: bool,
) -> None:
    payload = {
        "BASE_MODEL_NAME": cfg.BASE_MODEL_NAME,
        "BASE_MODEL_FALLBACK": cfg.BASE_MODEL_FALLBACK,
        "RESOLVED_BASE_MODEL_NAME": resolved_model,
        "USED_SYMBOLIC_TOKENIZER": used_symbolic_tokenizer,
        "OUTPUT_DIR": str(cfg.OUTPUT_DIR),
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "LEARNING_RATE": cfg.LEARNING_RATE,
        "EPOCHS": cfg.EPOCHS,
        "MAX_LENGTH": cfg.MAX_LENGTH,
        "LORA_R": cfg.LORA_R,
        "LORA_ALPHA": cfg.LORA_ALPHA,
        "LORA_DROPOUT": cfg.LORA_DROPOUT,
        "DEDUPE_BY_PROGRAM_HASH": cfg.DEDUPE_BY_PROGRAM_HASH,
        "REBALANCE_DATASET": cfg.REBALANCE_DATASET,
    }
    (out_dir / "training_config.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap (default: full epoch)",
    )
    args = ap.parse_args()

    records = load_training_records(
        ROOT,
        dedupe_by_program_hash=cfg.DEDUPE_BY_PROGRAM_HASH,
        rebalance_ops=cfg.REBALANCE_DATASET,
    )
    if not records:
        raise SystemExit("No training records; populate data/canonical.jsonl and data/generated.jsonl")

    model, resolved = load_causal_lm_with_fallback(
        cfg.BASE_MODEL_NAME, cfg.BASE_MODEL_FALLBACK
    )
    tokenizer, used_symbolic = load_tokenizer_for_training(ROOT, resolved)

    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        if not used_symbolic:
            raise
        print(
            f"WARNING: resize_token_embeddings failed for symbolic vocab "
            f"({type(e).__name__}: {e}). Switching to {resolved!r} tokenizer."
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        for s in EXTRA_SPECIAL:
            try:
                if s not in tokenizer.additional_special_tokens:
                    tokenizer.add_special_tokens({"additional_special_tokens": [s]})
            except Exception:
                pass
        used_symbolic = False
        model.resize_token_embeddings(len(tokenizer))

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
    out_dir = cfg.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(out_dir, resolved_model=resolved, used_symbolic_tokenizer=used_symbolic)

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
    print(f"Tokenizer: {'symbolic tokenizer.json' if used_symbolic else 'AutoTokenizer (base model)'}")

    ta_kwargs: dict = dict(
        output_dir=str(out_dir),
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
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    ex = ds[0]
    print("Example tokenized (first 32 ids):", ex["input_ids"][:32].tolist())
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
