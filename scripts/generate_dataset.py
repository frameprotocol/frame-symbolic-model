#!/usr/bin/env python3
"""Synthetic NL → program rows (no LLM). Appends validated canonical lines to data/generated.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.canonicalize import canonicalize
from pipeline.validate import is_valid_program, sanitize_text, validate

WORDS = (
    "hello alpha beta gamma note task result data payload echo foxtrot "
    "ledger cache buffer stream chunk frame packet session token vault "
    "signal bridge portal window thread kernel module record field entry"
).split()

URLS = [
    "https://example.com",
    "https://example.com/path",
    "https://api.example.com/v1",
    "https://cdn.example.net/asset",
    "https://docs.example.org/read",
    "https://status.example.io/ping",
]

NAMES = ("kyle", "alex", "sam", "river", "jordan", "casey", "morgan")

HIGH_SIGNAL_EXAMPLES = [
    ("get current time", ". time.now"),
    ("what time is it", ". time.now"),
    ('fetch example.com', '. web.request :url="https://example.com"'),
    ('store hello in memory', '. memory.store :text="hello"'),
    ('get current time and store it', '. time.now ; memory.store :text="current_time"'),
]

VARIANTS = [
    ("tell me the time", ". time.now"),
    ("current time please", ". time.now"),
    ("save current time", '. time.now ; memory.store :text="current_time"'),
]


def try_append(
    inp: str,
    raw: str,
    lines_out: list[str],
) -> bool:
    inp = sanitize_text(inp)
    raw = sanitize_text(raw)
    try:
        c = canonicalize(raw)
    except ValueError:
        return False
    c = sanitize_text(c)
    if not is_valid_program(c):
        return False
    if not validate(c):
        return False
    row = {
        "input": sanitize_text(inp),
        "program": sanitize_text(c),
        "output": sanitize_text(c),
    }
    lines_out.append(json.dumps(row, ensure_ascii=True))
    return True


def candidate_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    pairs.append(("get current time", ". time.now"))
    pairs.append(("what time is it", ". time.now"))
    pairs.append(("current timestamp please", ". time.now"))

    for w in WORDS:
        pairs.append((f"store {w} in memory", f'. memory.store :text="{w}"'))
        pairs.append((f"save {w} to memory", f'. memory.store :text="{w}"'))
        pairs.append((f"remember {w}", f'. memory.store :text="{w}"'))

    for url in URLS:
        host = url.split("//", 1)[-1].split("/")[0]
        pairs.append((f"fetch {host}", f'. web.request :url="{url}"'))
        pairs.append((f"get url {host}", f'. web.request :url="{url}"'))
        pairs.append((f"pull {url}", f'. web.request :url="{url}"'))

    for url in URLS:
        pairs.append(
            ("store result of web request", f'. web.request :url="{url}" -> memory.store :text=$0')
        )
        tail = url.split("//", 1)[-1]
        pairs.append(
            (
                f"fetch and save body from {tail}",
                f'. web.request :url="{url}" -> memory.store :text=$0',
            )
        )

    for w in WORDS:
        for url in URLS[:4]:
            pairs.append(
                (
                    f"get {w} from {url}",
                    f'. web.request :url="{url}" -> memory.store :text="{w}"',
                )
            )

    for w1 in WORDS[:24]:
        for w2 in WORDS[:24]:
            if w1 == w2:
                continue
            pairs.append(
                (
                    f"write key {w1} value {w2}",
                    f'. memory.write :key="{w1}" :value="{w2}"',
                )
            )

    for nm in NAMES:
        pairs.append((f"save my name as {nm}", f'. memory.write :key="name" :value="{nm}"'))

    for inp, raw in (
        ("read memory", ". memory.read"),
        ("recall stored data", ". memory.read"),
        ("show saved memory", ". memory.read"),
    ):
        pairs.append((inp, raw))

    i = 0
    while len(pairs) < 8000:
        w = WORDS[i % len(WORDS)]
        u = URLS[i % len(URLS)]
        pairs.append(
            (
                f"sync {w} from web index {i}",
                f'. web.request :url="{u}" -> memory.store :text="{w}"',
            )
        )
        i += 1

    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-n",
        "--num",
        type=int,
        default=2500,
        help="rows to append (1000–5000 typical)",
    )
    args = ap.parse_args()
    n_target = max(1, args.num)

    lines_out: list[str] = []

    # Balanced injection block (high-signal intent->operation mapping).
    # Cap each single mapping to avoid collapsing diversity.
    dataset_pairs: list[tuple[str, str]] = []

    # Keep time-only, fetch, and store balanced; reduce chain overlap so
    # "get current time" doesn't learn to emit only memory.store.
    for inp, out in HIGH_SIGNAL_EXAMPLES:
        chain = 'memory.store :text="current_time"' in out
        count = 50 if chain else 200
        for _ in range(count):
            dataset_pairs.append((inp, out))

    for inp, out in VARIANTS:
        chain = 'memory.store :text="current_time"' in out
        count = 0 if chain else 100
        for _ in range(count):
            dataset_pairs.append((inp, out))

    for inp, raw in dataset_pairs:
        if len(lines_out) >= n_target:
            break
        try_append(inp, raw, lines_out)

    for inp, raw in candidate_pairs():
        if len(lines_out) >= n_target:
            break
        try_append(inp, raw, lines_out)

    fill = 0
    while len(lines_out) < n_target and fill < n_target * 20:
        w = WORDS[fill % len(WORDS)]
        u = URLS[fill % len(URLS)]
        try_append(
            f"indexed fetch {fill} {w}",
            f'. web.request :url="{u}" -> memory.store :text="{w}_{fill}"',
            lines_out,
        )
        fill += 1

    out_path = ROOT / "data" / "generated.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for ln in lines_out:
            f.write(sanitize_text(ln) + "\n")
    print(f"Appended {len(lines_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
