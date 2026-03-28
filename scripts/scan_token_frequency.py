#!/usr/bin/env python3
"""Scan dataset for token frequency bias.

Flags any token exceeding 5% of total dataset tokens.
Identifies and reports:
- Repeated digits
- Placeholder spam
- Malformed outputs
- Over-represented tokens

Usage:
  python scripts/scan_token_frequency.py
  python scripts/scan_token_frequency.py --family english
  python scripts/scan_token_frequency.py --fix  # remove bad samples in-place
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES_DIR = ROOT / "data" / "families"

REPEAT_CHARS = re.compile(r"(.)\1{3,}")
REPEAT_TOKENS = re.compile(r"\b(\w+)(?:\s+\1){3,}\b")
DIGIT_SPAM = re.compile(r"\d{5,}")  # 5+ consecutive digits

FAMILIES = [
    "english", "cjk", "arabic", "indic", "cyrillic",
    "greek", "hebrew", "southeast_asian", "ethiopic",
]

BIAS_THRESHOLD = 0.05  # 5%


def scan_family(family: str) -> dict:
    """Scan a family dataset and return analysis."""
    path = FAMILIES_DIR / f"{family}.jsonl"
    if not path.exists():
        return {"family": family, "error": "file not found"}

    token_counts: Counter = Counter()
    total_tokens = 0
    bad_samples: list[dict] = []
    rows: list[dict] = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            intent = row.get("intent", "")

            # Tokenize by whitespace
            tokens = intent.split()
            for t in tokens:
                token_counts[t] += 1
                total_tokens += 1

            # Check for bad patterns
            reasons = []
            if REPEAT_CHARS.search(intent):
                reasons.append("repeated_chars")
            if REPEAT_TOKENS.search(intent):
                reasons.append("repeated_tokens")
            if DIGIT_SPAM.search(intent):
                reasons.append("digit_spam")

            inp = row.get("input", "")
            if REPEAT_CHARS.search(inp):
                reasons.append("input_repeated_chars")
            if DIGIT_SPAM.search(inp):
                reasons.append("input_digit_spam")

            if reasons:
                bad_samples.append({
                    "line": i,
                    "input": inp[:60],
                    "intent": intent[:60],
                    "reasons": reasons,
                })

    # Find biased tokens
    biased = []
    if total_tokens > 0:
        for token, count in token_counts.most_common(50):
            freq = count / total_tokens
            if freq > BIAS_THRESHOLD:
                biased.append({"token": token, "count": count, "freq": f"{freq:.1%}"})

    return {
        "family": family,
        "total_rows": len(rows),
        "total_tokens": total_tokens,
        "unique_tokens": len(token_counts),
        "biased_tokens": biased,
        "bad_samples": bad_samples,
        "top_20_tokens": [
            {"token": t, "count": c, "freq": f"{c/max(total_tokens,1):.1%}"}
            for t, c in token_counts.most_common(20)
        ],
    }


def fix_family(family: str) -> int:
    """Remove bad samples from family dataset. Returns count removed."""
    path = FAMILIES_DIR / f"{family}.jsonl"
    if not path.exists():
        return 0

    clean_rows = []
    removed = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            intent = row.get("intent", "")
            inp = row.get("input", "")

            bad = False
            if REPEAT_CHARS.search(intent) or REPEAT_CHARS.search(inp):
                bad = True
            if REPEAT_TOKENS.search(intent):
                bad = True
            if DIGIT_SPAM.search(intent) or DIGIT_SPAM.search(inp):
                bad = True

            # Check if output is just garbage
            stripped = re.sub(r'[.:;="\s]', '', intent)
            if stripped and stripped.isdigit():
                bad = True

            if bad:
                removed += 1
            else:
                clean_rows.append(row)

    if removed > 0:
        with open(path, "w", encoding="utf-8") as f:
            for row in clean_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", help="Scan specific family")
    ap.add_argument("--fix", action="store_true", help="Remove bad samples in-place")
    args = ap.parse_args()

    families = [args.family] if args.family else FAMILIES

    if args.fix:
        print("FIXING: Removing bad samples from datasets\n")
        for fam in families:
            removed = fix_family(fam)
            print(f"  {fam}: removed {removed} bad samples")
        print("\nDone.")
        return

    print("TOKEN FREQUENCY ANALYSIS")
    print("=" * 60)

    for fam in families:
        result = scan_family(fam)
        if "error" in result:
            print(f"\n{fam}: {result['error']}")
            continue

        print(f"\n{'='*60}")
        print(f"FAMILY: {fam}")
        print(f"  Rows: {result['total_rows']}")
        print(f"  Tokens: {result['total_tokens']} ({result['unique_tokens']} unique)")

        if result["biased_tokens"]:
            print(f"\n  BIASED TOKENS (>{BIAS_THRESHOLD:.0%} of dataset):")
            for bt in result["biased_tokens"]:
                print(f"    {bt['token']!r}: {bt['count']} ({bt['freq']})")

        if result["bad_samples"]:
            print(f"\n  BAD SAMPLES ({len(result['bad_samples'])} found):")
            for bs in result["bad_samples"][:10]:
                print(f"    Line {bs['line']}: {bs['reasons']}")
                print(f"      input: {bs['input']}")
                print(f"      intent: {bs['intent']}")
            if len(result["bad_samples"]) > 10:
                print(f"    ... and {len(result['bad_samples']) - 10} more")

        print(f"\n  TOP 20 TOKENS:")
        for tt in result["top_20_tokens"]:
            marker = " *** BIASED" if float(tt["freq"].rstrip("%")) / 100 > BIAS_THRESHOLD else ""
            print(f"    {tt['token']!r}: {tt['count']} ({tt['freq']}){marker}")


if __name__ == "__main__":
    main()
