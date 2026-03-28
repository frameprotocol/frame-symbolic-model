#!/usr/bin/env python3
"""Prune Qwen2.5-0.5B for deterministic interlang compilation.

Two pruning strategies applied together:
1. VOCAB PRUNING: 151k → ~8k tokens (only keep tokens seen in training + byte fallbacks)
2. LAYER PRUNING: 24 → N layers (remove redundant middle layers)

This produces a smaller base model that can be LoRA-finetuned and exported to ~80-100MB GGUF.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = [
    "english", "cjk", "arabic", "indic", "cyrillic",
    "greek", "hebrew", "southeast_asian", "ethiopic",
]


def scan_used_tokens(tokenizer) -> set[int]:
    """Scan all training data and return the set of token IDs used."""
    all_ids: set[int] = set()

    for fam in FAMILIES:
        path = ROOT / "data" / "families" / f"{fam}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line.strip())
                inp = row["input"]
                out = row["intent"]

                # Tokenize with the prompt format (same as training)
                prefix = "<START>\nINPUT: " + inp + "\nOUTPUT: "
                suffix = out + "\n<END>"

                inp_ids = tokenizer.encode(prefix, add_special_tokens=False)
                out_ids = tokenizer.encode(suffix, add_special_tokens=False)
                all_ids.update(inp_ids)
                all_ids.update(out_ids)

    return all_ids


def get_essential_token_ids(tokenizer) -> set[int]:
    """Get IDs that must be kept regardless of training data usage."""
    essential: set[int] = set()

    # Special tokens
    for attr in ["eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            essential.add(tid)

    # Prompt-format special tokens
    for tok_str in ["<START>", "<END>", "<INPUT>", "<OUTPUT>"]:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        essential.update(ids)

    # All single-byte tokens (byte-level BPE fallbacks for any unseen chars)
    vocab = tokenizer.get_vocab()
    for token_str, tid in vocab.items():
        # Qwen uses byte tokens like <0x00> through <0xFF>
        if token_str.startswith("<0x") and token_str.endswith(">") and len(token_str) == 6:
            essential.add(tid)
        # Also keep single-character tokens
        if len(token_str) == 1:
            essential.add(tid)

    return essential


def get_buffer_tokens(tokenizer, used_ids: set[int], essential_ids: set[int], target_vocab: int) -> set[int]:
    """Add commonly-used tokens from the original vocab as a buffer for unseen inputs."""
    # We want to keep tokens that are likely to appear in unseen inputs
    # across all 9 language families. Sample some text and find common tokens.
    buffer: set[int] = set()
    already = used_ids | essential_ids

    # Keep all tokens that are ASCII printable (common in interlang output)
    vocab = tokenizer.get_vocab()
    for token_str, tid in vocab.items():
        if tid in already:
            continue
        # Keep short tokens (1-3 chars) - they're common subwords
        if len(token_str) <= 3 and token_str.isprintable():
            buffer.add(tid)
        if len(already) + len(buffer) >= target_vocab:
            break

    # If still under target, add by frequency rank (lower ID = more common in BPE)
    if len(already) + len(buffer) < target_vocab:
        remaining = sorted(set(range(len(vocab))) - already - buffer)
        needed = target_vocab - len(already) - len(buffer)
        buffer.update(remaining[:needed])

    return buffer


def build_pruned_tokenizer(
    tokenizer,
    keep_ids: list[int],
) -> tuple[PreTrainedTokenizerFast, dict[int, int]]:
    """Build a new tokenizer with only the kept token IDs.

    Returns (new_tokenizer, old_to_new_id_mapping).
    """
    # Build old->new mapping
    old_to_new: dict[int, int] = {}
    for new_id, old_id in enumerate(sorted(keep_ids)):
        old_to_new[old_id] = new_id

    # Get the tokenizer's internal data
    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())

    # Remap the vocabulary
    old_vocab = tok_json["model"]["vocab"]
    new_vocab: dict[str, int] = {}
    new_to_old_token: dict[int, str] = {}  # for reverse lookup

    keep_set = set(keep_ids)
    for token_str, old_id in old_vocab.items():
        if old_id in keep_set:
            new_id = old_to_new[old_id]
            new_vocab[token_str] = new_id
            new_to_old_token[new_id] = token_str

    tok_json["model"]["vocab"] = new_vocab

    # Filter merges: only keep merges where both resulting tokens are in our vocab
    if "merges" in tok_json["model"] and tok_json["model"]["merges"]:
        old_merges = tok_json["model"]["merges"]
        new_merges = []
        for merge in old_merges:
            # Merges can be "a b" strings or ["a", "b"] lists
            if isinstance(merge, str):
                parts = merge.split(" ")
            elif isinstance(merge, list):
                parts = merge
            else:
                continue
            if len(parts) == 2:
                a, b = parts[0], parts[1]
                merged = a + b
                # Keep merge if all involved tokens are in our vocab
                if a in new_vocab and b in new_vocab and merged in new_vocab:
                    new_merges.append(merge)
        tok_json["model"]["merges"] = new_merges
        print(f"  Merges: {len(old_merges)} -> {len(new_merges)}")

    # Update added_tokens with new IDs
    if "added_tokens" in tok_json:
        new_added = []
        for at in tok_json["added_tokens"]:
            old_id = at["id"]
            if old_id in old_to_new:
                at_copy = dict(at)
                at_copy["id"] = old_to_new[old_id]
                new_added.append(at_copy)
        tok_json["added_tokens"] = new_added

    # Rebuild tokenizer from modified JSON
    from tokenizers import Tokenizer as HFTokenizer

    backend_tok = HFTokenizer.from_str(json.dumps(tok_json))
    new_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tok,
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token,
        unk_token=tokenizer.unk_token,
        pad_token=tokenizer.pad_token,
    )

    return new_tokenizer, old_to_new


def prune_layers(model, config, keep_layer_indices: list[int]):
    """Remove transformer layers not in keep_layer_indices."""
    # Access the transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find model layers for pruning")

    original_count = len(layers)
    print(f"  Pruning layers: {original_count} -> {len(keep_layer_indices)}")
    print(f"  Keeping layers: {keep_layer_indices}")

    # Build new layer list
    new_layers = torch.nn.ModuleList([layers[i] for i in keep_layer_indices])
    model.model.layers = new_layers

    # Update config
    config.num_hidden_layers = len(keep_layer_indices)

    # Trim layer_types if present (Qwen2.5 uses this field)
    if hasattr(config, "layer_types") and config.layer_types is not None:
        config.layer_types = [config.layer_types[i] for i in keep_layer_indices]

    return model, config


def prune_embeddings(model, config, keep_ids: list[int], old_to_new: dict[int, int]):
    """Slice embedding matrix and lm_head to only keep specified token IDs."""
    keep_ids_sorted = sorted(keep_ids)
    keep_tensor = torch.tensor(keep_ids_sorted, dtype=torch.long)

    # Input embeddings
    old_embed = model.model.embed_tokens.weight.data
    new_embed_weight = old_embed[keep_tensor].clone()

    new_embed = torch.nn.Embedding(
        len(keep_ids_sorted),
        config.hidden_size,
        padding_idx=old_to_new.get(config.pad_token_id),
    )
    new_embed.weight.data = new_embed_weight
    model.model.embed_tokens = new_embed

    # Output head (lm_head)
    old_lm_head = model.lm_head.weight.data
    new_lm_head_weight = old_lm_head[keep_tensor].clone()

    new_lm_head = torch.nn.Linear(config.hidden_size, len(keep_ids_sorted), bias=False)
    new_lm_head.weight.data = new_lm_head_weight
    model.lm_head = new_lm_head

    # Update config
    config.vocab_size = len(keep_ids_sorted)
    if config.pad_token_id is not None and config.pad_token_id in old_to_new:
        config.pad_token_id = old_to_new[config.pad_token_id]
    if config.eos_token_id is not None:
        if isinstance(config.eos_token_id, int):
            config.eos_token_id = old_to_new.get(config.eos_token_id, 0)
        elif isinstance(config.eos_token_id, list):
            config.eos_token_id = [old_to_new.get(x, 0) for x in config.eos_token_id if x in old_to_new]
    if config.bos_token_id is not None and config.bos_token_id in old_to_new:
        config.bos_token_id = old_to_new[config.bos_token_id]

    return model, config


def main():
    ap = argparse.ArgumentParser(description="Prune Qwen2.5-0.5B for interlang compilation")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B", help="Base model to prune")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "models" / "pruned_base",
                    help="Where to save the pruned base model")
    ap.add_argument("--target-vocab", type=int, default=8000,
                    help="Target vocabulary size (default: 8000)")
    ap.add_argument("--keep-layers", type=int, default=12,
                    help="Number of transformer layers to keep (default: 12)")
    ap.add_argument("--layer-strategy", choices=["first", "uniform", "ends"],
                    default="ends",
                    help="How to select which layers to keep: "
                         "first=keep first N, "
                         "uniform=evenly spaced, "
                         "ends=keep first/last few + some middle (default: ends)")
    args = ap.parse_args()

    print(f"=== Pruning {args.base_model} ===")
    print(f"  Target vocab: {args.target_vocab}")
    print(f"  Keep layers: {args.keep_layers}")
    print(f"  Layer strategy: {args.layer_strategy}")
    print(f"  Output: {args.output_dir}")

    # Load model and tokenizer
    print("\n[1/6] Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, trust_remote_code=True
    )
    config = model.config

    orig_vocab = len(tokenizer)
    orig_layers = config.num_hidden_layers
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"  Original: {orig_params/1e6:.0f}M params, {orig_vocab} vocab, {orig_layers} layers")

    # Scan training data
    print("\n[2/6] Scanning training data for used tokens...")
    used_ids = scan_used_tokens(tokenizer)
    essential_ids = get_essential_token_ids(tokenizer)
    print(f"  Tokens used in training: {len(used_ids)}")
    print(f"  Essential tokens (specials + bytes): {len(essential_ids)}")

    # Build keep set
    all_keep = used_ids | essential_ids
    if len(all_keep) < args.target_vocab:
        buffer = get_buffer_tokens(tokenizer, used_ids, essential_ids, args.target_vocab)
        all_keep |= buffer
        print(f"  Buffer tokens added: {len(buffer)}")

    keep_ids = sorted(all_keep)
    print(f"  Final vocab size: {len(keep_ids)}")

    # Build pruned tokenizer
    print("\n[3/6] Building pruned tokenizer...")
    new_tokenizer, old_to_new = build_pruned_tokenizer(tokenizer, keep_ids)
    print(f"  New vocab size: {len(new_tokenizer)}")

    # Verify tokenizer works
    test_inputs = [
        "get current time",
        "获取当前时间",
        "الحصول على الوقت الحالي",
        "वर्तमान समय प्राप्त करें",
        ". time.now",
        '<START>\nINPUT: hello\nOUTPUT: . memory.store :text="hello"\n<END>',
    ]
    print("  Tokenizer verification:")
    for t in test_inputs:
        ids = new_tokenizer.encode(t, add_special_tokens=False)
        decoded = new_tokenizer.decode(ids)
        ok = "OK" if decoded.strip() == t.strip() else "LOSSY"
        print(f"    [{ok}] {t[:50]!r} -> {len(ids)} tokens")

    # Prune embeddings
    print("\n[4/6] Pruning embedding matrix...")
    model, config = prune_embeddings(model, config, keep_ids, old_to_new)

    # Select layers to keep
    print("\n[5/6] Pruning transformer layers...")
    n = args.keep_layers
    total = orig_layers

    if args.layer_strategy == "first":
        keep_layer_indices = list(range(n))
    elif args.layer_strategy == "uniform":
        keep_layer_indices = [int(i * total / n) for i in range(n)]
    elif args.layer_strategy == "ends":
        # Keep first 3, last 3, and evenly space the rest from the middle
        first_n = min(3, n // 3)
        last_n = min(3, n // 3)
        mid_n = n - first_n - last_n
        first = list(range(first_n))
        last = list(range(total - last_n, total))
        mid_range = list(range(first_n, total - last_n))
        if mid_n > 0 and mid_range:
            step = len(mid_range) / mid_n
            mid = [mid_range[int(i * step)] for i in range(mid_n)]
        else:
            mid = []
        keep_layer_indices = sorted(set(first + mid + last))[:n]
    else:
        keep_layer_indices = list(range(n))

    model, config = prune_layers(model, config, keep_layer_indices)

    # Stats
    new_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Pruned: {new_params/1e6:.0f}M params (was {orig_params/1e6:.0f}M)")
    print(f"  Reduction: {(1 - new_params/orig_params)*100:.1f}%")

    # Estimated GGUF sizes
    for quant, bpp in [("f16", 16), ("q4_k_m", 4.5), ("q3_k_m", 3.5), ("q2_K", 2.5)]:
        est_mb = new_params * bpp / 8 / 1024 / 1024
        print(f"  Estimated GGUF ({quant}): ~{est_mb:.0f}MB")

    # Save
    print(f"\n[6/6] Saving pruned model to {args.output_dir}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Update and save config
    model.config = config
    model.save_pretrained(str(args.output_dir), safe_serialization=True)
    new_tokenizer.save_pretrained(str(args.output_dir))

    # Save pruning metadata
    meta = {
        "source_model": args.base_model,
        "original_vocab": orig_vocab,
        "pruned_vocab": len(keep_ids),
        "original_layers": orig_layers,
        "pruned_layers": args.keep_layers,
        "layer_strategy": args.layer_strategy,
        "kept_layer_indices": keep_layer_indices,
        "original_params_M": round(orig_params / 1e6, 1),
        "pruned_params_M": round(new_params / 1e6, 1),
    }
    (args.output_dir / "pruning_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== DONE ===")
    print(f"Pruned base model saved to: {args.output_dir}")
    print(f"Next steps:")
    print(f"  1. Update training/config.py BASE_MODEL_NAME to point to {args.output_dir}")
    print(f"  2. Retrain all 9 LoRA adapters")
    print(f"  3. Merge and export to GGUF")


if __name__ == "__main__":
    main()
