#!/usr/bin/env python3
"""Generate scaled (input, program) data via pluggable LLM, then canonicalize + validate."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Literal, TypedDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interlang.ast import Op
from interlang.parser import parse
from pipeline.canonicalize import canonicalize
from pipeline.hash import hash_program
from pipeline.validate import validate

Backend = Literal["mock", "ollama"]

MAX_OPS = 6

NOTE_WORDS = (
    "hello alpha beta gamma delta echo foxtrot hotel india juliet kilo lemon mike november "
    "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zebra "
    "apple bread coral drift ember frost glade haven ivory jade knoll lotus mist "
    "noble oasis prism quill ridge stone tide ulver vale wave xenon yield zephyr"
).split()

NAMES = (
    "kyle alex sam riley jordan casey morgan taylor quinn reese avery blake drew jamie "
    "skylar rowan sage river storm wren ash rowan finley emery logan reese"
).split()

# Base seeds (natural language); symbolic targets come from the mock classifier / future LLM.
BASE_SEEDS = [
    "get current time",
    "store note hello",
    "fetch example.com",
    "save my name as kyle",
    "read memory",
    "write key x value y",
]

RejectReason = Literal[
    "format", "multi_line", "no_dot", "parse_error", "invalid", "too_long", "duplicate"
]


class RejectRow(TypedDict):
    input: str
    raw: str
    reason: str


def strict_llm_output_reason(raw: str) -> tuple[str | None, RejectReason | None]:
    """
    Single program line only. Reasons: format (empty), multi_line, no_dot.
    """
    text = (raw or "").strip()
    if not text:
        return None, "format"
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 1:
        return None, "multi_line"
    if len(lines) == 0:
        return None, "format"
    line = lines[0]
    if not line.startswith("."):
        return None, "no_dot"
    return line, None


def strict_llm_output(raw: str) -> str | None:
    """Backward-compatible: program line or None."""
    line, _ = strict_llm_output_reason(raw)
    return line


def mock_generate_program(input_text: str) -> str:
    """Deterministic-ish NL → program for inputs produced by our variation templates."""
    s = input_text.lower()
    s = re.sub(r"\s+", " ", s).strip()

    if "example.com" in s or re.search(r"\b(fetch|retrieve|load|pull)\b.*\b(url|http|page|site)\b", s):
        return '. http.fetch :url="example.com"'
    if re.search(r"\b(save|store|set|record)\b", s) and "name" in s:
        for cand in sorted(NAMES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(cand)}\b", s):
                return f'. memory.write :key="name" :value="{cand}"'
        if "kyle" in s:
            return '. memory.write :key="name" :value="kyle"'
    if re.search(r"\b(write|set|put|assign)\b", s) and re.search(r"\bkey\b", s) and re.search(r"\bvalue\b", s):
        m = re.search(r"\bkey\s+([a-zA-Z0-9_]+)\s+value\s+([a-zA-Z0-9_]+)\b", s)
        if m:
            return f'. memory.write :key="{m.group(1)}" :value="{m.group(2)}"'
        m2 = re.search(r"\bvalue\s+([a-zA-Z0-9_]+)\s+for\s+key\s+([a-zA-Z0-9_]+)\b", s)
        if m2:
            return f'. memory.write :key="{m2.group(2)}" :value="{m2.group(1)}"'
        if "x" in s and "y" in s:
            return '. memory.write :key="x" :value="y"'
    if re.search(r"\b(read|recall|load|get|show)\b", s) and re.search(r"\b(memory|stored)\b", s):
        if not re.search(r"\b(time|timestamp|clock)\b", s) and "example.com" not in s:
            return ". memory.read"
    if re.search(r"\b(store|save|remember|stash|keep)\b", s) and (
        "note" in s or "memo" in s or "message" in s or "text" in s or "hello" in s
    ):
        m = re.search(r"\b(note|memo|message|text)\s+([a-zA-Z0-9_]+)\b", s)
        if not m:
            m = re.search(r"\b(message|text)\s*:\s*([a-zA-Z0-9_]+)\b", s)
        if m and re.match(r"^[a-zA-Z0-9_]+$", m.group(2)):
            w = m.group(2)
            return f'. memory.store :text="{w}"'
        if "hello" in s:
            return '. memory.store :text="hello"'
    if re.search(r"\b(get|obtain|retrieve|what|show|give|need|tell)\b", s) and re.search(
        r"\b(time|timestamp|clock)\b", s
    ):
        return ". time.now"

    return "unmapped_intent"


def _ollama_generate(input_text: str) -> str:
    """Call local Ollama; response must be program-only (one line)."""
    import os

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.environ.get("OLLAMA_MODEL", "")
    if not model:
        raise RuntimeError("Set OLLAMA_MODEL (and run ollama serve) to use backend=ollama")

    system = (
        "You output exactly one line: an interlang program. "
        "Format only: . op ; op :key=\"value\" — no prose, no markdown, no quotes around the line."
    )
    user = f"Natural language intent:\n{input_text}\n\nOutput the program on one line."
    payload = json.dumps(
        {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise RuntimeError(f"ollama request failed: {e}") from e

    msg = data.get("message") or {}
    content = (msg.get("content") or "").strip()
    return content


def generate_program(input_text: str, *, backend: Backend = "mock") -> str:
    """
    Produce raw program text from the model (must be trimmed to one program line by caller).
    """
    if backend == "mock":
        return mock_generate_program(input_text)
    if backend == "ollama":
        return _ollama_generate(input_text)
    raise ValueError(f"unknown backend {backend!r}")


def vary_input(seed: str, rng: random.Random) -> str:
    """Synonym / reorder variations; deterministic given rng state."""

    def noise() -> str:
        return rng.choice(["", " please", " now", " thanks", " for me", " if you can"])

    a = rng.choice(["get", "obtain", "retrieve", "what is", "show me", "i need"])
    b = rng.choice(["the ", ""])
    c = rng.choice(["current ", "present ", ""])
    d = rng.choice(["time", "timestamp", "clock", "time now"])
    time_phrases = [
        f"{a} {b}{c}{d}{noise()}".replace("  ", " ").strip(),
        f"{rng.choice(['tell me', 'give me', 'please return'])} {b}{c}{d}{noise()}".replace("  ", " ").strip(),
        f"{b}{c}{d} — {a}{noise()}".replace("  ", " ").strip(),
    ]

    s1 = rng.choice(["store", "save", "remember", "stash", "keep", "write down"])
    s2 = rng.choice(["note", "memo", "message", "text", "reminder"])
    nw = rng.choice(NOTE_WORDS)
    note_phrases = [
        f"{s1} {s2} {nw}{noise()}".replace("  ", " ").strip(),
        f"{s1} a {s2}: {nw}{noise()}".replace("  ", " ").strip(),
        f"{nw} — {s1} as {s2}{noise()}".replace("  ", " ").strip(),
        f"{s1} this {s2}: {nw}{noise()}".replace("  ", " ").strip(),
    ]

    f1 = rng.choice(["fetch", "retrieve", "load", "pull", "get"])
    f2 = rng.choice(["url", "page", "site", "resource", "address"])
    fetch_phrases = [
        f"{f1} example.com{noise()}".strip(),
        f"{f1} the {f2} example.com{noise()}".replace("  ", " ").strip(),
        f"{f1} http://example.com{noise()}".strip(),
        f"{f1} from example.com{noise()}".strip(),
    ]

    n1 = rng.choice(["save", "store", "set", "record"])
    nm = rng.choice(NAMES)
    n2_choices = [
        f"my name as {nm}",
        f"name {nm}",
        f"{nm} as my name",
        f"the name {nm}",
        f"my name {nm}",
    ]
    n2 = rng.choice(n2_choices)
    name_phrases = [
        f"{n1} {n2}{noise()}".strip(),
        f"{n2} — {n1} it{noise()}".strip(),
        f"{n1} that {n2}{noise()}".strip(),
    ]

    r1 = rng.choice(["read", "recall", "load", "get", "show"])
    r2 = rng.choice(["memory", "stored data", "what is saved", "saved memory"])
    read_phrases = [
        f"{r1} {r2}{noise()}".strip(),
        f"{r2}, {r1} it{noise()}".strip(),
        f"{r1} from {r2}{noise()}".strip(),
    ]

    w1 = rng.choice(["write", "set", "put", "assign"])
    wk, wv = rng.choice(NOTE_WORDS), rng.choice(NOTE_WORDS)
    write_phrases = [
        f"{w1} key x value y{noise()}".strip(),
        f"{w1} value y for key x{noise()}".strip(),
        f"key x {w1} value y{noise()}".strip(),
        f"for key x {w1} value y{noise()}".strip(),
        f"{w1} key {wk} value {wv}{noise()}".strip(),
        f"{w1} value {wv} for key {wk}{noise()}".strip(),
    ]

    table: dict[str, list[str]] = {
        "get current time": time_phrases,
        "store note hello": note_phrases,
        "fetch example.com": fetch_phrases,
        "save my name as kyle": name_phrases,
        "read memory": read_phrases,
        "write key x value y": write_phrases,
    }
    options = table.get(seed, [seed])
    return rng.choice(options)


def _accumulate_stats(ast: List[Op], op_counts: dict[str, int], arg_patterns: dict[str, int]) -> None:
    for o in ast:
        op_counts[o["op"]] = op_counts.get(o["op"], 0) + 1
        for k in o["args"]:
            arg_patterns[k] = arg_patterns.get(k, 0) + 1


def accept_or_reject(
    input_text: str,
    raw: str,
    *,
    seen_hashes: set[str] | None,
) -> tuple[str, str, List[Op]] | RejectRow:
    line, sr = strict_llm_output_reason(raw)
    if line is None:
        return {"input": input_text, "raw": raw, "reason": sr or "format"}
    try:
        canon = canonicalize(line)
        ast = parse(canon)
    except ValueError:
        return {"input": input_text, "raw": raw, "reason": "parse_error"}
    if not validate(canon):
        return {"input": input_text, "raw": raw, "reason": "invalid"}
    if len(ast) > MAX_OPS:
        return {"input": input_text, "raw": raw, "reason": "too_long"}
    h = hash_program(ast)
    if seen_hashes is not None:
        if h in seen_hashes:
            return {"input": input_text, "raw": raw, "reason": "duplicate"}
        seen_hashes.add(h)
    return (input_text, canon, ast)


def process_one(
    input_text: str,
    *,
    backend: Backend,
    generate_fn: Callable[[str], str] | None = None,
    seen_hashes: set[str] | None = None,
) -> tuple[str, str] | None:
    """
    generate → strict single-line → canonicalize → validate → optional hash dedup.
    If seen_hashes is None, hash dedup is skipped (same program may appear twice).
    """
    gen = generate_fn or (lambda t: generate_program(t, backend=backend))
    raw = gen(input_text)
    out = accept_or_reject(input_text, raw, seen_hashes=seen_hashes)
    if isinstance(out, dict):
        return None
    inp_c, canon, _ast = out
    return (inp_c, canon)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate data/generated.jsonl via LLM + canonicalize + validate")
    p.add_argument(
        "--n",
        type=int,
        default=500,
        help="target unique valid rows (clamped to 100–1000)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (deterministic variations)")
    p.add_argument("--backend", choices=("mock", "ollama"), default="mock")
    args = p.parse_args()

    n = max(100, min(args.n, 1000))
    rng = random.Random(args.seed)

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "generated.jsonl"
    reject_path = data_dir / "rejected.jsonl"
    stats_path = data_dir / "stats.json"

    seen_hashes: set[str] = set()
    rows: list[dict[str, str]] = []
    rejected: list[RejectRow] = []
    op_counts: dict[str, int] = {}
    arg_patterns: dict[str, int] = {}
    attempts = 0
    max_attempts = max(n * 200, n + 5000)

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        base = rng.choice(BASE_SEEDS)
        inp = vary_input(base, rng)
        raw = generate_program(inp, backend=args.backend)
        out = accept_or_reject(inp, raw, seen_hashes=seen_hashes)
        if isinstance(out, dict):
            rejected.append(out)
            continue
        inp_c, canon, ast = out
        rows.append({"input": inp_c, "program": canon})
        _accumulate_stats(ast, op_counts, arg_patterns)

    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    reject_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected),
        encoding="utf-8",
    )

    unique_ops = len(op_counts)
    stats = {
        "total": len(rows),
        "rejected_count": len(rejected),
        "unique_ops": unique_ops,
        "op_distribution": dict(sorted(op_counts.items())),
        "arg_distribution": dict(sorted(arg_patterns.items())),
        "max_ops_cap": MAX_OPS,
        "seed": args.seed,
        "target_n": n,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    if len(rows) < n:
        print(
            f"Note: hash saturation — got {len(rows)} unique programs (< target {n}). "
            "Expand mock diversity or lower --n."
        )
    print(
        f"Wrote {out_path} ({len(rows)} accepted), {reject_path} ({len(rejected)} rejected), {stats_path}"
    )


if __name__ == "__main__":
    main()

