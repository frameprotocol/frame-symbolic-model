#!/usr/bin/env python3
"""Validate model outputs for training collapse indicators.

Rejects output if:
- Same token repeats > 5 times consecutively
- No <END> marker (in raw output)
- Invalid interlang syntax
- Output is longer than 20 tokens
- Output contains repeated digits (e.g. "22222")
- Output is empty or whitespace

Usage:
  python scripts/validate_output.py --text ". time.now"
  python scripts/validate_output.py --file outputs.jsonl
  python scripts/validate_output.py --interactive
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.validate import validate

# ---- Collapse detection patterns ----
# Same character repeated 4+ times
REPEATED_CHARS = re.compile(r"(.)\1{3,}")
# Same word/token repeated 3+ times consecutively
REPEATED_WORDS = re.compile(r"\b(\S+)(?:\s+\1){2,}\b")
# Long digit sequences (5+)
DIGIT_SPAM = re.compile(r"\d{5,}")
# Max output tokens
MAX_OUTPUT_TOKENS = 20


class OutputValidationResult:
    def __init__(self, text: str, is_valid: bool, reasons: list[str]):
        self.text = text
        self.is_valid = is_valid
        self.reasons = reasons

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        reason_str = ", ".join(self.reasons) if self.reasons else "OK"
        return f"[{status}] {self.text[:60]!r} — {reason_str}"


def validate_output(text: str, *, require_end: bool = False, raw_text: str = "") -> OutputValidationResult:
    """Validate a single model output.

    Args:
        text: The cleaned output text (interlang program).
        require_end: If True, checks that raw_text contains <END>.
        raw_text: The raw model output (before cleaning), used for <END> check.
    """
    reasons: list[str] = []

    # Empty
    if not text or not text.strip():
        return OutputValidationResult(text, False, ["empty_output"])

    stripped = text.strip()

    # Repeated characters (e.g., "22222", "aaaa")
    match = REPEATED_CHARS.search(stripped)
    if match:
        reasons.append(f"repeated_char:'{match.group()[:10]}'")

    # Repeated words/tokens
    match = REPEATED_WORDS.search(stripped)
    if match:
        reasons.append(f"repeated_word:'{match.group()[:20]}'")

    # Digit spam
    if DIGIT_SPAM.search(stripped):
        reasons.append("digit_spam")

    # Pure digit output
    clean = re.sub(r'[.:;="\s]', '', stripped)
    if clean and clean.isdigit():
        reasons.append("pure_digits")

    # Too long
    word_count = len(stripped.split())
    if word_count > MAX_OUTPUT_TOKENS:
        reasons.append(f"too_long:{word_count}_tokens")

    # Missing <END> in raw output (if checking)
    if require_end and raw_text and "<END>" not in raw_text:
        reasons.append("missing_END")

    # Invalid interlang syntax
    if not reasons:  # only check if no structural issues
        if not validate(stripped):
            reasons.append("invalid_syntax")

    return OutputValidationResult(stripped, len(reasons) == 0, reasons)


def validate_file(path: Path, require_end: bool = False) -> None:
    """Validate outputs in a JSONL file."""
    valid_count = 0
    invalid_count = 0
    reason_counts: dict[str, int] = {}

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # Try common output field names
            output = row.get("output") or row.get("intent") or ""
            raw = row.get("raw_output", "")

            result = validate_output(output, require_end=require_end, raw_text=raw)

            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if invalid_count <= 20:
                    inp = row.get("input", "")[:40]
                    print(f"  Line {i}: {result}")
                    if inp:
                        print(f"    input: {inp!r}")
                for r in result.reasons:
                    tag = r.split(":")[0]
                    reason_counts[tag] = reason_counts.get(tag, 0) + 1

    total = valid_count + invalid_count
    print(f"\n{'='*50}")
    print(f"File: {path.name}")
    print(f"Total: {total}")
    print(f"Valid: {valid_count} ({valid_count/max(total,1)*100:.1f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/max(total,1)*100:.1f}%)")

    if reason_counts:
        print(f"\nRejection reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate model outputs for collapse")
    ap.add_argument("--text", help="Validate a single output string")
    ap.add_argument("--file", type=Path, help="Validate outputs in JSONL file")
    ap.add_argument("--require-end", action="store_true", help="Require <END> in raw output")
    ap.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = ap.parse_args()

    if args.text:
        result = validate_output(args.text)
        print(result)
        sys.exit(0 if result.is_valid else 1)

    if args.file:
        validate_file(args.file, require_end=args.require_end)
        return

    if args.interactive:
        print("Enter outputs to validate (Ctrl+D to quit):")
        try:
            while True:
                text = input("> ").strip()
                if text:
                    result = validate_output(text)
                    print(f"  {result}")
        except EOFError:
            pass
        return

    # Default: validate all family datasets
    print("OUTPUT VALIDATION")
    print("=" * 60)

    families_dir = ROOT / "data" / "families"
    for path in sorted(families_dir.glob("*.jsonl")):
        print(f"\n--- {path.stem} ---")
        validate_file(path)


if __name__ == "__main__":
    main()
