#!/usr/bin/env python3
"""Build canonical.jsonl from hardcoded seed intents (no LLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.canonicalize import canonicalize
from pipeline.validate import is_valid_program, sanitize_text, validate

SEEDS: list[tuple[str, str]] = [
    ("get current time", ". time.now"),
    ("store note hello", '. memory.store :text="hello"'),
    ("fetch example.com", '. http.fetch :url="example.com"'),
    ("read memory", ". memory.read"),
    ("write key x value y", '. memory.write :value="y" :key="x"'),
]


def main() -> None:
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "raw.jsonl"
    out_path = data_dir / "canonical.jsonl"

    raw_rows: list[dict[str, str]] = []
    canon_rows: list[dict[str, str]] = []

    for intent, program in SEEDS:
        intent = sanitize_text(intent)
        program = sanitize_text(program)
        raw_rows.append({"input": intent, "program": program})
        try:
            c = canonicalize(program)
        except ValueError as e:
            raise SystemExit(f"canonicalize failed for {intent!r}: {e}") from e
        c = sanitize_text(c)
        if not is_valid_program(c):
            raise SystemExit(f"charset validation failed for {intent!r}")
        if not validate(c):
            raise SystemExit(f"validate failed after canonicalize for {intent!r}")
        canon_rows.append({"input": intent, "program": c})

    raw_path.write_text(
        sanitize_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_rows)),
        encoding="utf-8",
    )
    out_path.write_text(
        sanitize_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in canon_rows)),
        encoding="utf-8",
    )
    print(f"Wrote {raw_path} and {out_path} ({len(canon_rows)} rows)")


if __name__ == "__main__":
    main()
