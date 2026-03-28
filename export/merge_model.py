#!/usr/bin/env python3
"""Merge trained LoRA adapter into a full base model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]


def _family_dirs(family: str) -> tuple[Path, Path, Path]:
    fam_root = ROOT / "models" / family
    adapter_dir = fam_root / "adapter"
    training_config = adapter_dir / "training_config.json"
    merged_dir = fam_root / "merged"
    return adapter_dir, training_config, merged_dir


def _read_base_model_name(training_config: Path) -> str:
    if not training_config.is_file():
        raise FileNotFoundError(f"Missing training config: {training_config}")
    payload = json.loads(training_config.read_text(encoding="utf-8"))
    base_name = payload.get("RESOLVED_BASE_MODEL_NAME") or payload.get("BASE_MODEL_NAME")
    if not isinstance(base_name, str) or not base_name.strip():
        raise ValueError("training_config.json does not contain a valid base model name")
    return base_name.strip()


def _load_tokenizer(base_name: str, adapter_dir: Path) -> AutoTokenizer:
    # Prefer adapter tokenizer if present; fallback to base tokenizer.
    try:
        tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into a full HF checkpoint (per family)")
    ap.add_argument("--family", required=True, help="Model family id (e.g. english, cjk, arabic, indic)")
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    _, training_config, merged_dir = _family_dirs(args.family)
    adapter_dir = Path(args.input) if args.input else ROOT / "models" / args.family / "adapter"
    base_name = _read_base_model_name(training_config)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")

    print(f"Loading base model: {base_name}")
    tokenizer = _load_tokenizer(base_name, adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    # Resize embeddings to match tokenizer (may differ after special token additions)
    model_vocab = base.get_input_embeddings().num_embeddings
    tok_vocab = len(tokenizer)
    if tok_vocab != model_vocab:
        print(f"Resizing embeddings: {model_vocab} -> {tok_vocab}")
    base.resize_token_embeddings(tok_vocab)

    print(f"Loading LoRA adapter: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("Merging adapter weights into base model...")
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    # Also write a family-level config.json for convenience.
    (merged_dir.parent / "config.json").write_text(
        json.dumps({"family": args.family, "base_model": base_name}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved merged model and tokenizer to: {merged_dir}")


if __name__ == "__main__":
    main()
