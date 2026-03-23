"""Symbolic pre-tokenization: atomic chunks joined for BPE (see TOK_JOIN)."""

from __future__ import annotations

# Must not appear in canonical interlang programs.
TOK_JOIN = "\x01"


def symbolic_scan(s: str) -> list[str]:
    """
    Split a program into meaningful atoms: predicates (memory.store), ; : = -> * $N,
    quoted strings, leading program dot, etc.
    """
    t: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i].isspace():
            i += 1
            continue
        if s.startswith("->", i):
            t.append("->")
            i += 2
            continue
        if s[i] == ";":
            t.append(";")
            i += 1
            continue
        if s[i] == "*":
            t.append("*")
            i += 1
            continue
        if s[i] == "$" and i + 1 < n and s[i + 1].isdigit():
            j = i + 1
            while j < n and s[j].isdigit():
                j += 1
            t.append(s[i:j])
            i = j
            continue
        if s[i] == '"':
            j = i + 1
            esc = False
            while j < n:
                if esc:
                    esc = False
                    j += 1
                    continue
                if s[j] == "\\":
                    esc = True
                    j += 1
                    continue
                if s[j] == '"':
                    j += 1
                    break
                j += 1
            t.append(s[i:j])
            i = j
            continue
        if s[i] == ":":
            t.append(":")
            i += 1
            continue
        if s[i] == "=":
            t.append("=")
            i += 1
            continue
        if s[i] == "." and (i == 0 or s[i - 1].isspace()):
            t.append(".")
            i += 1
            continue
        if s[i].isalpha() or s[i] == "_":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] in "._"):
                j += 1
            t.append(s[i:j])
            i = j
            continue
        t.append(s[i])
        i += 1
    return t


def prepare_for_tokenizer(program: str) -> str:
    """Canonical interlang string -> BPE input (pretoken boundaries via TOK_JOIN)."""
    return TOK_JOIN.join(symbolic_scan(program))
