#!/usr/bin/env python3
"""Train a BPE tokenizer on canonical + generated programs (HuggingFace tokenizers)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer

from tokenizer.corpus import iter_canonical_programs
from tokenizer.symbolic_pre import prepare_for_tokenizer

SPECIAL = ["[UNK]", "", ".", ";", ":", "=", "->", "*", "$"]
DEFAULT_VOCAB = 1024


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB,
        help="BPE vocab size (default 1024; use 2048 for more capacity)",
    )
    args = p.parse_args()
    vocab_size = max(256, args.vocab_size)

    data_dir = ROOT / "data"
    paths = (data_dir / "canonical.jsonl", data_dir / "generated.jsonl")
    programs = list(iter_canonical_programs(*paths))
    if not programs:
        raise SystemExit("No valid programs after canonicalize/validate in data/*.jsonl")

    corpus = [prepare_for_tokenizer(prog) for prog in programs]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    try:
        from tokenizers.pre_tokenizers import SplitDelimiterBehavior

        delim = SplitDelimiterBehavior.Removed
    except ImportError:
        delim = "removed"  # type: ignore[assignment]
    tokenizer.pre_tokenizer = Split(Regex(r"\x01"), behavior=delim)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        min_frequency=1,
    )
    tokenizer.train_from_iterator(corpus, trainer)

    out_dir = ROOT / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Wrote {out_path} ({len(programs)} programs, vocab_size={vocab_size})")


if __name__ == "__main__":
    main()
