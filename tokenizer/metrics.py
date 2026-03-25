#!/usr/bin/env python3
"""Token length stats + program complexity distribution."""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interlang.parser import parse
from tokenizers import Tokenizer

from tokenizer.corpus import iter_canonical_programs

_WS = re.compile(r"\s+")


def whitespace_token_count(text: str) -> int:
    return len(_WS.split(text.strip())) if text.strip() else 0


def main() -> None:
    tok_path = ROOT / "tokenizer" / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(f"Missing {tok_path}; run tokenizer/train_tokenizer.py first")

    tokenizer = Tokenizer.from_file(str(tok_path))
    data_dir = ROOT / "data"
    paths = (data_dir / "canonical.jsonl", data_dir / "generated.jsonl")
    programs = list(iter_canonical_programs(*paths))
    if not programs:
        raise SystemExit("No programs to score")

    # Use programs directly (canonical interlang) without symbolic pre-tokenization.
    bpe_lens = [len(tokenizer.encode(p).ids) for p in programs]
    base_lens = [whitespace_token_count(p) for p in programs]

    op_len_counts: Counter[int] = Counter()
    for p in programs:
        try:
            n = len(parse(p))
        except ValueError:
            continue
        op_len_counts[n] += 1

    program_length_distribution: dict[str, int] = {
        f"{k}_ops": op_len_counts.get(k, 0) for k in range(1, 11)
    }
    program_length_distribution["11+_ops"] = sum(
        c for n, c in op_len_counts.items() if n >= 11
    )

    out = {
        "programs_count": len(programs),
        "bpe_avg_tokens_per_program": round(statistics.mean(bpe_lens), 4),
        "bpe_stdev_tokens": round(statistics.pstdev(bpe_lens), 4) if len(bpe_lens) > 1 else 0.0,
        "tokenizer_token_count_total": sum(bpe_lens),
        "whitespace_split_avg_tokens_per_program": round(statistics.mean(base_lens), 4),
        "whitespace_split_stdev_tokens": round(statistics.pstdev(base_lens), 4)
        if len(base_lens) > 1
        else 0.0,
        "avg_tokens_ratio_bpe_over_whitespace": round(
            statistics.mean(bpe_lens) / statistics.mean(base_lens), 4
        )
        if statistics.mean(base_lens) > 0
        else None,
        "program_length_distribution": program_length_distribution,
    }

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tik_lens = [len(enc.encode(p)) for p in programs]
        out["gpt_cl100k_avg_tokens_per_program"] = round(statistics.mean(tik_lens), 4)
        out["avg_tokens_ratio_bpe_over_gpt_cl100k"] = round(
            statistics.mean(bpe_lens) / statistics.mean(tik_lens), 4
        )
    except Exception:
        out["gpt_cl100k_avg_tokens_per_program"] = None

    metrics_path = ROOT / "data" / "tokenizer_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
