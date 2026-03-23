#!/usr/bin/env python3
"""Greedy decode: NL → symbolic program (LoRA + saved or symbolic tokenizer)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.canonicalize import canonicalize
from pipeline.validate import validate
from training import config as cfg
from training.tokenizer_load import load_tokenizer_for_inference, read_resolved_base

ADAPTER_DIR = ROOT / "models" / "symbolic-lora"


def build_prompt(user_text: str) -> str:
    return "<INPUT>\n" + user_text + "\n<OUTPUT>\n"


def extract_after_output(decoded_full: str) -> str:
    if "<OUTPUT>" in decoded_full:
        return decoded_full.split("<OUTPUT>", 1)[-1].strip()
    return decoded_full.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run symbolic generation (greedy by default)")
    ap.add_argument("input_text", help="Natural language intent")
    ap.add_argument("--max-tokens", type=int, default=64, dest="max_tokens")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    adapter_weights = ADAPTER_DIR / "adapter_model.safetensors"
    adapter_bin = ADAPTER_DIR / "adapter_model.bin"
    if not adapter_weights.is_file() and not adapter_bin.is_file():
        raise FileNotFoundError(
            f"Missing LoRA adapter under {ADAPTER_DIR} (expected adapter_model.safetensors)."
        )

    resolved = read_resolved_base(ADAPTER_DIR, cfg.BASE_MODEL_NAME)
    tokenizer = load_tokenizer_for_inference(ROOT, ADAPTER_DIR, resolved)

    base = AutoModelForCausalLM.from_pretrained(resolved, trust_remote_code=True)
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = build_prompt(args.input_text)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    do_sample = args.temperature is not None and args.temperature > 0
    gen_kwargs: dict = {
        "max_new_tokens": args.max_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_id,
    }
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id
    if do_sample:
        gen_kwargs["temperature"] = max(args.temperature, 1e-5)

    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

    seq = out[0].tolist()
    full_decoded = tokenizer.decode(seq, skip_special_tokens=False)
    new_tokens = seq[input_ids.shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    parsed = extract_after_output(full_decoded)

    print("INPUT:")
    print(args.input_text)
    print()
    print("RAW:")
    print(raw)
    print()
    print("PARSED:")
    print(parsed)
    print()
    status = "INVALID"
    try:
        c = canonicalize(parsed)
        if validate(c):
            status = "VALID"
    except ValueError:
        pass
    print("STATUS:")
    print(status)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
