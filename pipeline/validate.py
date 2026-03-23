"""Validate interlang programs."""

from __future__ import annotations

import re

from interlang.parser import ParseError, parse

_OP_RE = re.compile(r"^[a-zA-Z0-9_.]+$")
_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate(program: str) -> bool:
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
