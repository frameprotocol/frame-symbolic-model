#!/usr/bin/env python3
"""Convert data/families/*.jsonl (interlang DSL) to data/distill_{family}.jsonl (JSON format).

Source format:
    {"intent": ". time.now", "input": "...", "language": "...", "family": "..."}
    {"intent": ". memory.store :text=\"hello\"", "input": "...", ...}

Target format (matches data/distill_english.jsonl):
    {"input": "...", "output": {"intent": "time.now", "params": {}}}
    {"input": "...", "output": {"intent": "memory.store", "params": {"text": "مرحبا"}}}

SPAN RULE: every param value MUST be an exact substring of the input.
Records that fail span validation are dropped. Case mismatches are corrected using
the actual casing found in the input string.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Map from multilingual_intents.jsonl language keys to family names
LANG_TO_FAMILY: dict[str, str] = {
    "arabic": "arabic",
    "chinese": "cjk",
    "russian": "cyrillic",
    "amharic": "ethiopic",
    "greek": "greek",
    "hebrew": "hebrew",
    "hindi": "indic",
    "thai": "southeast_asian",
}

FAMILIES = [
    "arabic",
    "cjk",
    "cyrillic",
    "ethiopic",
    "greek",
    "hebrew",
    "indic",
    "southeast_asian",
]


def parse_dsl(dsl: str) -> tuple[str, dict[str, str]]:
    """Parse '. time.now :key="val"' into (intent, params)."""
    dsl = dsl.strip()
    if dsl.startswith(". "):
        dsl = dsl[2:]
    elif dsl.startswith("."):
        dsl = dsl[1:].lstrip()

    parts = dsl.split(" ", 1)
    intent = parts[0].strip()
    params: dict[str, str] = {}

    if len(parts) > 1:
        for m in re.finditer(r':(\w+)=(?:"([^"]*)"|([\S]*))', parts[1]):
            key = m.group(1)
            val = m.group(2) if m.group(2) is not None else m.group(3)
            params[key] = val

    return intent, params


def load_family_records(family: str) -> list[dict]:
    path = ROOT / "data" / "families" / f"{family}.jsonl"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        raw_intent = r.get("intent", "")
        inp = r.get("input", "").strip()
        if not raw_intent or not inp:
            continue

        try:
            intent, params = parse_dsl(raw_intent)
        except Exception:
            continue

        if not intent:
            continue

        records.append({"input": inp, "output": {"intent": intent, "params": params}})

    return records


def load_multilingual_records(family: str) -> list[dict]:
    """Extract records for this family from multilingual_intents.jsonl."""
    path = ROOT / "data" / "multilingual" / "multilingual_intents.jsonl"
    if not path.exists():
        return []

    # Which language key maps to this family?
    lang_keys = [lang for lang, fam in LANG_TO_FAMILY.items() if fam == family]
    if not lang_keys:
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        raw_intent = r.get("intent", "")
        inputs = r.get("inputs", {})

        try:
            intent, params = parse_dsl(raw_intent)
        except Exception:
            continue

        if not intent:
            continue

        for lang_key in lang_keys:
            inp = inputs.get(lang_key, "").strip()
            if inp:
                records.append({"input": inp, "output": {"intent": intent, "params": params}})

    return records


def enforce_spans(record: dict) -> dict | None:
    """Ensure every param value is an exact substring of input.

    - If a param value is already an exact substring: keep as-is.
    - If it matches case-insensitively: replace with the actual casing from input.
    - Otherwise: return None (record must be dropped).
    """
    text = record["input"]
    params = record["output"]["params"]
    if not params:
        return record  # no params to validate

    fixed: dict[str, str] = {}
    for k, v in params.items():
        v = str(v)
        if v in text:
            fixed[k] = v
        else:
            # Try case-insensitive find
            idx = text.lower().find(v.lower())
            if idx != -1:
                fixed[k] = text[idx : idx + len(v)]
            else:
                return None  # violation — cannot fix automatically

    return {"input": record["input"], "output": {"intent": record["output"]["intent"], "params": fixed}}


def dedup(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in records:
        key = r["input"] + "|" + r["output"]["intent"]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def validate_spans(record: dict) -> bool:
    text = record["input"]
    for v in record["output"]["params"].values():
        if str(v) not in text:
            return False
    return True


def convert_family(family: str) -> int:
    print(f"\n[{family}]")
    records = load_family_records(family)
    print(f"  families/{family}.jsonl: {len(records)} records")

    multi = load_multilingual_records(family)
    print(f"  multilingual ({family}): {len(multi)} records")

    all_records = dedup(records + multi)

    # Enforce span rule: fix case mismatches, drop true violations
    span_ok = []
    dropped = 0
    for r in all_records:
        fixed = enforce_spans(r)
        if fixed is None:
            dropped += 1
        else:
            span_ok.append(fixed)

    print(f"  after dedup: {len(all_records)}  span-dropped: {dropped}  kept: {len(span_ok)}")

    # Final assertion: every kept record passes span validation
    for r in span_ok:
        assert validate_spans(r), f"BUG: span validation failed after fix: {r}"

    out_path = ROOT / "data" / f"distill_{family}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in span_ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  written: {out_path}")
    return len(span_ok)


def main() -> None:
    families = sys.argv[1:] if len(sys.argv) > 1 else FAMILIES
    total = 0
    for family in families:
        total += convert_family(family)
    print(f"\nDone. Total records across {len(families)} families: {total}")


if __name__ == "__main__":
    main()
