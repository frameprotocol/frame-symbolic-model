"""Load NL → JSON intent pairs for supervised training.

Training target format (model output):
    {"intent": "message.send", "params": {"to": "alice", "text": "hello"}}

No DSL. No "missing" field. Strict JSON only.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_family_dataset(root: Path, family: str) -> list[dict[str, str]]:
    """Load training pairs from data/distill_{family}.jsonl.

    Each line must have {"input": ..., "output": {"intent": ..., "params": {...}}}.
    Returns list of {"input": str, "output": str} where output is a JSON string.
    """
    path = root / "data" / f"distill_{family}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run: python scripts/convert_families_to_distill.py {family}"
        )

    rows: list[dict[str, str]] = []
    skipped = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            inp = record.get("input", "")
            out = record.get("output", {})

            if not inp or not isinstance(out, dict):
                skipped += 1
                continue
            if "intent" not in out or "params" not in out:
                skipped += 1
                continue

            # Strip any runtime fields (e.g. "missing") before storing as training target
            model_output = {"intent": out["intent"], "params": out["params"]}
            rows.append({"input": inp, "output": json.dumps(model_output, ensure_ascii=False)})

    print(f"[{family}] Dataset loaded: {len(rows)} samples (skipped {skipped})")
    return rows


def load_english_dataset(root: Path) -> list[dict[str, str]]:
    """Alias for load_family_dataset(root, 'english')."""
    return load_family_dataset(root, "english")
