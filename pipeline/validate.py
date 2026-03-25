"""Validate interlang programs."""

from __future__ import annotations

import re

from interlang.parser import ParseError, parse

_OP_RE = re.compile(r"^[a-zA-Z0-9_.]+$")
_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_UNQUOTED_SAFE_RE = re.compile(r"^[A-Za-z0-9_./:@+-]+$")


def is_valid_program(s: str) -> bool:
    """
    Backward-compatible check: returns True if the program can be validated.

    NOTE: This function no longer enforces a strict ASCII-only charset.
    Unicode is now allowed in quoted string values. The actual structure
    validation happens in validate().
    """
    # Just check that it's non-empty and has printable characters.
    s = sanitize_text(s)
    if not s.strip():
        return False
    # Must start with '.' to be a valid program.
    if not s.strip().startswith("."):
        return False
    return True


def sanitize_text(s: str) -> str:
    # remove all non-printable control characters
    return "".join(c for c in s if c.isprintable())

def _split_chain_raw(s: str) -> list[str]:
    """Split op chain on ';' or '->' outside quoted strings (raw string version)."""
    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    escape = False
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if escape:
            buf.append(c)
            escape = False
            i += 1
            continue
        if c == "\\" and in_quote:
            escape = True
            buf.append(c)
            i += 1
            continue
        if c == '"':
            in_quote = not in_quote
            buf.append(c)
            i += 1
            continue
        if c == ";" and not in_quote:
            seg = "".join(buf).strip()
            if seg:
                parts.append(seg)
            buf = []
            i += 1
            continue
        if not in_quote and c == "-" and i + 1 < n and s[i + 1] == ">":
            seg = "".join(buf).strip()
            if seg:
                parts.append(seg)
            buf = []
            i += 2
            continue
        buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _scan_arg_quoting(segment_rest: str) -> dict[str, bool]:
    """
    Return {key: was_quoted}. This is a validator-only scan so we can enforce:
    - quoted values: allow full unicode (any printable chars; parser enforces quoting)
    - unquoted values: enforce a safe token regex
    """
    quoted: dict[str, bool] = {}
    i = 0
    n = len(segment_rest)
    while i < n:
        while i < n and segment_rest[i].isspace():
            i += 1
        if i >= n:
            break
        if segment_rest[i] != ":":
            raise ValueError(f"expected :key=value, got {segment_rest[i:]!r}")
        i += 1
        j = i
        while j < n and segment_rest[j] != "=" and not segment_rest[j].isspace():
            j += 1
        key = segment_rest[i:j]
        if j >= n or segment_rest[j] != "=":
            raise ValueError(f"missing = after :{key}")
        i = j + 1
        while i < n and segment_rest[i].isspace():
            i += 1
        if i >= n:
            raise ValueError(f"missing value for :{key}")
        if segment_rest[i] == '"':
            quoted[key] = True
            i += 1
            esc = False
            while i < n:
                c = segment_rest[i]
                if esc:
                    esc = False
                    i += 1
                    continue
                if c == "\\":
                    esc = True
                    i += 1
                    continue
                if c == '"':
                    i += 1
                    break
                i += 1
            else:
                raise ValueError("unterminated string")
        else:
            quoted[key] = False
            k = i
            while k < n and not segment_rest[k].isspace():
                k += 1
            i = k
    return quoted


def _arg_quoted_map(program: str) -> dict[tuple[int, str], bool]:
    """
    Build a {(segment_index, key): was_quoted} map by scanning raw text.
    We intentionally do NOT restrict unicode for quoted values here.
    """
    s = program.strip()
    if not s.startswith("."):
        raise ValueError("program must start with '.'")
    body = s[1:].strip()
    segs = _split_chain_raw(body)
    out: dict[tuple[int, str], bool] = {}
    for idx, seg in enumerate(segs):
        seg = seg.strip()
        if not seg:
            continue
        sp = seg.find(" ")
        rest = "" if sp == -1 else seg[sp + 1 :]
        if rest.strip():
            qm = _scan_arg_quoting(rest)
            for k, v in qm.items():
                out[(idx, k)] = v
    return out


def validate(program: str) -> bool:
    program = sanitize_text(program)
    try:
        ops = parse(program)
    except ParseError:
        return False
    try:
        quoted_map = _arg_quoted_map(program)
    except Exception:
        # If we cannot scan quoting deterministically, reject.
        return False
    for o in ops:
        op = o["op"]
        if not op or not _OP_RE.match(op):
            return False
        for k, v in o["args"].items():
            if not k or not _KEY_RE.match(k):
                return False
            if not isinstance(v, str):
                return False
    # Enforce unquoted token safety; allow full unicode in quoted values.
    for seg_i, o in enumerate(ops):
        for k, v in o["args"].items():
            was_quoted = quoted_map.get((seg_i, k), False)
            if not was_quoted:
                # Keep unquoted values conservative (no whitespace; safe chars only).
                if not v or not _UNQUOTED_SAFE_RE.match(v):
                    return False
    return True
