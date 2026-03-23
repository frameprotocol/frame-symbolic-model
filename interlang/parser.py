"""Deterministic interlang parser: . op ; op :k=v ..."""

from __future__ import annotations

import re
from typing import List

from interlang.ast import Op

_OP_RE = re.compile(r"^[a-zA-Z0-9_.]+$")
_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class ParseError(ValueError):
    pass


def split_chain(s: str) -> List[str]:
    """Split op chain on ';' or '->' outside quoted strings."""
    parts: List[str] = []
    buf: List[str] = []
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


def _parse_args(rest: str) -> dict[str, str]:
    args: dict[str, str] = {}
    i = 0
    n = len(rest)
    while i < n:
        while i < n and rest[i].isspace():
            i += 1
        if i >= n:
            break
        if rest[i] != ":":
            raise ParseError(f"expected :key=value, got {rest[i:]!r}")
        i += 1
        j = i
        while j < n and rest[j] != "=" and not rest[j].isspace():
            j += 1
        key = rest[i:j]
        if not key or not _KEY_RE.match(key):
            raise ParseError(f"invalid arg key {key!r}")
        if j >= n or rest[j] != "=":
            raise ParseError(f"missing = after :{key}")
        i = j + 1
        while i < n and rest[i].isspace():
            i += 1
        if i >= n:
            raise ParseError(f"missing value for :{key}")
        if rest[i] == '"':
            i += 1
            val_chars: List[str] = []
            while i < n:
                if rest[i] == "\\":
                    i += 1
                    if i >= n:
                        raise ParseError("unterminated escape in string")
                    val_chars.append(rest[i])
                    i += 1
                elif rest[i] == '"':
                    i += 1
                    break
                else:
                    val_chars.append(rest[i])
                    i += 1
            else:
                raise ParseError("unterminated string")
            args[key] = "".join(val_chars)
        else:
            k = i
            while k < n and not rest[k].isspace():
                k += 1
            args[key] = rest[i:k]
            i = k
    return args


def parse_segment(segment: str) -> Op:
    segment = segment.strip()
    if not segment:
        raise ParseError("empty segment")
    sp = segment.find(" ")
    if sp == -1:
        op = segment
        rest = ""
    else:
        op = segment[:sp]
        rest = segment[sp + 1 :]
    if not op or not _OP_RE.match(op):
        raise ParseError(f"invalid op {op!r}")
    args = _parse_args(rest) if rest.strip() else {}
    return {"op": op, "args": args}


def parse(program: str) -> List[Op]:
    s = program.strip()
    if not s.startswith("."):
        raise ParseError("program must start with '.'")
    body = s[1:].strip()
    if not body:
        raise ParseError("no operations after '.'")
    segments = split_chain(body)
    if not segments:
        raise ParseError("no operations")
    return [parse_segment(seg) for seg in segments]


def format_value(v: str) -> str:
    out = v.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{out}"'


def serialize(ops: List[Op]) -> str:
    parts: List[str] = []
    for o in ops:
        bits = [o["op"]]
        for k in sorted(o["args"]):
            bits.append(f":{k}={format_value(o['args'][k])}")
        parts.append(" ".join(bits))
    return ". " + " ; ".join(parts)
