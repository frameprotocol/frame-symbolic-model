"""Minimal AST types for interlang programs."""

from __future__ import annotations

from typing import TypedDict


class Op(TypedDict):
    op: str
    args: dict[str, str]
