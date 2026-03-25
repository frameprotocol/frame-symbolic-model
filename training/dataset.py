"""Load NL → canonical program pairs for supervised training."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from interlang.parser import parse
from pipeline.hash import hash_program
from tokenizer.corpus import iter_valid_rows

# No single primary op may exceed this share (post-trim).
_MAX_PRIMARY_OP_SHARE = 0.40
_MIN_ROWS_REBALANCE = 12


def ensure_dot_prefix(program: str) -> str:
    """Training targets must start like '. op' (anchor generation)."""
    p = program.strip()
    if not p:
        return p
    if p.startswith("."):
        return p
    return ". " + p


def primary_op(program: str) -> str:
    ops = parse(program)
    return ops[0]["op"] if ops else ""


def rebalance_by_primary_op(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Round-robin merge by primary op (deterministic), then greedy keep while
    no primary exceeds 40% of kept size.
    """
    if len(rows) < _MIN_ROWS_REBALANCE:
        return rows
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in sorted(rows, key=lambda x: x["input"]):
        buckets[primary_op(r["output"])].append(r)
    op_keys = sorted(buckets.keys())
    merged: list[dict[str, str]] = []
    while any(buckets[k] for k in op_keys):
        for k in op_keys:
            if buckets[k]:
                merged.append(buckets[k].pop(0))

    kept: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for r in merged:
        op = primary_op(r["output"])
        n2 = len(kept) + 1
        if n2 <= _MIN_ROWS_REBALANCE or (counts[op] + 1) / n2 <= _MAX_PRIMARY_OP_SHARE + 1e-12:
            kept.append(r)
            counts[op] += 1
    return kept


def load_training_records(
    root: Path,
    *,
    dataset_dir: Path | None = None,
    dedupe_by_program_hash: bool = False,
    rebalance_ops: bool = False,
) -> list[dict[str, str]]:
    """
    Load canonical.jsonl + generated.jsonl.
    Each row: canonicalize program, validate, skip invalid.
    Output: {"input": ..., "output": canonical_program} with '.' anchor.

    dedupe_by_program_hash: if True, keep at most one row per canonical program
    (reduces size). Default False so many NL phrases can map to the same op pattern.
    """
    data_dir = dataset_dir or (root / "data")
    paths = (data_dir / "canonical.jsonl", data_dir / "generated.jsonl")
    seen_hash: set[str] = set()
    seen_pair: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    for inp, canon in iter_valid_rows(*paths):
        out = ensure_dot_prefix(canon)
        if dedupe_by_program_hash:
            try:
                h = hash_program(parse(out))
            except ValueError:
                continue
            if h in seen_hash:
                continue
            seen_hash.add(h)
        else:
            key = (inp, out)
            if key in seen_pair:
                continue
            seen_pair.add(key)
        rows.append({"input": inp, "output": out})
    if rebalance_ops:
        rows = rebalance_by_primary_op(rows)
    return rows
