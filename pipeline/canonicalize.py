"""Canonicalize interlang program strings.

Includes an op-name normalization pass: any short/alias op name is
automatically rewritten to its fully-qualified namespace.op form using
the central CANONICAL_OPS registry.  Stats are accumulated per call and
accessible via module-level counters that callers may inspect.
"""

from __future__ import annotations

from typing import List, Tuple

from interlang.ast import Op
from interlang.parser import parse, serialize
from pipeline.validate import sanitize_text
from pipeline.op_registry import canonicalize_op, is_namespaced

# ---------------------------------------------------------------------------
# Module-level counters so callers can report totals after a batch run.
# ---------------------------------------------------------------------------
fixed_ops_count: int = 0
rejected_ops_count: int = 0


def _op_key(o: Op) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    items = tuple(sorted(o["args"].items()))
    return (o["op"], items)


def canonicalize(program: str) -> str:
    """Parse, normalize op names, deduplicate, and re-serialize *program*.

    Non-canonical (non-namespaced) op names are rewritten via the registry.
    If an op cannot be resolved a ValueError is raised (caller should skip
    the sample and increment rejected_ops_count).
    """
    global fixed_ops_count, rejected_ops_count

    program = sanitize_text(program)
    ops: List[Op] = parse(program)

    normalized: List[Op] = []
    for o in ops:
        op_name = o["op"]
        if not is_namespaced(op_name):
            try:
                canonical_name, was_fixed = canonicalize_op(op_name)
            except ValueError as exc:
                rejected_ops_count += 1
                print(f"WARNING: rejected non-canonical op {op_name!r}: {exc}")
                raise
            if was_fixed:
                fixed_ops_count += 1
                print(f"INFO: auto-fixed op {op_name!r} → {canonical_name!r}")
            op_name = canonical_name
        normalized.append({"op": op_name, "args": o["args"]})

    seen: set[Tuple[str, Tuple[Tuple[str, str], ...]]] = set()
    deduped: List[Op] = []
    for o in normalized:
        k = _op_key(o)
        if k in seen:
            continue
        seen.add(k)
        deduped.append({"op": o["op"], "args": dict(sorted(o["args"].items()))})
    return sanitize_text(serialize(deduped))
