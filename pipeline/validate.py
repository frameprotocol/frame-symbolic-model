"""Validate interlang programs."""

from __future__ import annotations

import re

from interlang.parser import ParseError, parse

_OP_RE = re.compile(r"^[a-zA-Z0-9_.]+$")
_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def sanitize_text(s: str) -> str:
    # remove all non-printable control characters
    return "".join(c for c in s if c.isprintable())


def is_valid_program(s: str) -> bool:
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:;="/ ')
    return all(c in allowed for c in s)


def validate(program: str) -> bool:
    program = sanitize_text(program)
    if not is_valid_program(program):
        return False
    try:
        ops = parse(program)
    except ParseError:
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
    return True
