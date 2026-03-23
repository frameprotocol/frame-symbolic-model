"""Canonicalize interlang program strings."""

from __future__ import annotations

from typing import List, Tuple

from interlang.ast import Op
from interlang.parser import parse, serialize
from pipeline.validate import sanitize_text


def _op_key(o: Op) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    items = tuple(sorted(o["args"].items()))
    return (o["op"], items)


def canonicalize(program: str) -> str:
    program = sanitize_text(program)
    ops: List[Op] = parse(program)
    seen: set[Tuple[str, Tuple[Tuple[str, str], ...]]] = set()
    deduped: List[Op] = []
    for o in ops:
        k = _op_key(o)
        if k in seen:
            continue
        seen.add(k)
        deduped.append({"op": o["op"], "args": dict(sorted(o["args"].items()))})
    return sanitize_text(serialize(deduped))
