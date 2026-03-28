#!/usr/bin/env python3
"""Train a BPE tokenizer on canonical + generated programs (HuggingFace tokenizers)."""

from __future__ import annotations

import argparse
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<START>", "<END>", "<INPUT>", "<OUTPUT>"]
    )

    tokenizer.train([args.input], trainer)

    os.makedirs(args.output, exist_ok=True)
    tokenizer.save(f"{args.output}/tokenizer.json")


if __name__ == "__main__":
    main()
