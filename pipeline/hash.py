"""Deterministic program fingerprint and hash (for dedup)."""

from __future__ import annotations

import hashlib
from typing import List

from interlang.ast import Op


def ast_fingerprint(ops: List[Op]) -> str:
    """Stable string, e.g. time.now|memory.store:text=hello."""
    parts: list[str] = []
    for o in ops:
        if not o["args"]:
            parts.append(o["op"])
        else:
            arg_s = ",".join(f"{k}={v}" for k, v in sorted(o["args"].items()))
            parts.append(f"{o['op']}:{arg_s}")
    return "|".join(parts)


def hash_program(ast: List[Op]) -> str:
    """SHA-256 hex digest of the fingerprint."""
    fp = ast_fingerprint(ast)
    return hashlib.sha256(fp.encode("utf-8")).hexdigest()
