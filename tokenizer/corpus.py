"""Load rows from JSONL with defensive canonicalize + validate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pipeline.canonicalize import canonicalize
from pipeline.validate import validate


def iter_valid_rows(*paths: Path) -> Iterator[tuple[str, str]]:
    """Yield (input, canonical_program) for valid rows only."""
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = obj.get("input", "")
            raw = obj.get("program") or obj.get("output") or ""
            if not raw:
                continue
            try:
                c = canonicalize(raw)
            except ValueError:
                continue
            if not validate(c):
                continue
            yield (str(inp), c)


def iter_canonical_programs(*paths: Path) -> Iterator[str]:
    for _inp, c in iter_valid_rows(*paths):
        yield c
