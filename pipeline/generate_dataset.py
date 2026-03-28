#!/usr/bin/env python3
"""Generate English NL → partial intent JSON training data.

Output format per sample:
    {"input": "...", "output": {"intent": "...", "params": {...}}}

No "missing" field — the model ONLY extracts intent and params.
Missing-field detection is handled by the runtime validator.

Run:
    python pipeline/generate_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Seed dataset: (input, intent, params)
# params only contains values that ARE in the input string.
# ---------------------------------------------------------------------------

SEEDS: list[tuple[str, str, dict]] = [
    # ── message.send ─────────────────────────────────────────────────────────
    ("send alice hello",                   "message.send", {"to": "alice",  "text": "hello"}),
    ("send bob a message saying hi",       "message.send", {"to": "bob",    "text": "hi"}),
    ("message carol good morning",         "message.send", {"to": "carol",  "text": "good morning"}),
    ("tell dave I'll be late",             "message.send", {"to": "dave",   "text": "I'll be late"}),
    ("send a message to eve",              "message.send", {"to": "eve"}),        # missing text
    ("send alice",                         "message.send", {"to": "alice"}),      # missing text
    ("send a message",                     "message.send", {}),                   # missing to, text
    ("message frank that I'm on my way",   "message.send", {"to": "frank",  "text": "I'm on my way"}),
    ("text grace happy birthday",          "message.send", {"to": "grace",  "text": "happy birthday"}),
    ("send henry can you call me",         "message.send", {"to": "henry",  "text": "can you call me"}),
    ("drop ivan a note saying thanks",     "message.send", {"to": "ivan",   "text": "thanks"}),
    ("ping judy",                          "message.send", {"to": "judy"}),       # missing text
    ("write to karen saying see you soon", "message.send", {"to": "karen",  "text": "see you soon"}),
    ("shoot lee a message",                "message.send", {"to": "lee"}),        # missing text

    # ── payment.send ─────────────────────────────────────────────────────────
    ("send 10 dollars to alice",           "payment.send", {"to": "alice",  "amount": "10"}),
    ("pay bob 50",                         "payment.send", {"to": "bob",    "amount": "50"}),
    ("transfer 100 to carol",              "payment.send", {"to": "carol",  "amount": "100"}),
    ("send money to dave",                 "payment.send", {"to": "dave"}),       # missing amount
    ("pay 20 dollars",                     "payment.send", {"amount": "20"}),     # missing to
    ("send payment to eve",                "payment.send", {"to": "eve"}),        # missing amount
    ("transfer funds to frank",            "payment.send", {"to": "frank"}),      # missing amount
    ("pay",                                "payment.send", {}),                   # missing both
    ("send 5 dollars to grace",            "payment.send", {"to": "grace",  "amount": "5"}),
    ("pay henry 200 dollars",              "payment.send", {"to": "henry",  "amount": "200"}),
    ("send 75 bucks to ivan",              "payment.send", {"to": "ivan",   "amount": "75"}),
    ("wire 300 to judy",                   "payment.send", {"to": "judy",   "amount": "300"}),
    ("venmo karen 15",                     "payment.send", {"to": "karen",  "amount": "15"}),
    ("send lee 40 dollars",                "payment.send", {"to": "lee",    "amount": "40"}),

    # ── payment.balance ───────────────────────────────────────────────────────
    ("check my balance",                   "payment.balance", {}),
    ("what's my balance",                  "payment.balance", {}),
    ("show account balance",               "payment.balance", {}),
    ("how much money do I have",           "payment.balance", {}),
    ("get my balance",                     "payment.balance", {}),

    # ── payment.history ───────────────────────────────────────────────────────
    ("show payment history",               "payment.history", {}),
    ("list my transactions",               "payment.history", {}),
    ("recent payments",                    "payment.history", {}),
    ("show my transaction history",        "payment.history", {}),

    # ── time.now ─────────────────────────────────────────────────────────────
    ("what time is it",                    "time.now", {}),
    ("get current time",                   "time.now", {}),
    ("tell me the time",                   "time.now", {}),
    ("current time",                       "time.now", {}),
    ("what's the time",                    "time.now", {}),
    ("time now",                           "time.now", {}),

    # ── time.date ─────────────────────────────────────────────────────────────
    ("what's today's date",                "time.date", {}),
    ("get current date",                   "time.date", {}),
    ("today's date",                       "time.date", {}),
    ("what day is it",                     "time.date", {}),
    ("current date",                       "time.date", {}),

    # ── web.search ────────────────────────────────────────────────────────────
    ("search for python tutorials",        "web.search",  {"query": "python tutorials"}),
    ("look up the weather in tokyo",       "web.search",  {"query": "weather in tokyo"}),
    ("search news about AI",               "web.search",  {"query": "news about AI"}),
    ("find recipes for pasta",             "web.search",  {"query": "recipes for pasta"}),
    ("search",                             "web.search",  {}),                    # missing query
    ("look up bitcoin price",              "web.search",  {"query": "bitcoin price"}),
    ("search for how to cook rice",        "web.search",  {"query": "how to cook rice"}),
    ("google best restaurants nearby",     "web.search",  {"query": "best restaurants nearby"}),
    ("search the web",                     "web.search",  {}),                    # missing query

    # ── memory.store ─────────────────────────────────────────────────────────
    ("remember that my PIN is 1234",       "memory.store", {"text": "my PIN is 1234"}),
    ("save a note: buy milk",              "memory.store", {"text": "buy milk"}),
    ("store this: meeting at 3pm",         "memory.store", {"text": "meeting at 3pm"}),
    ("remember buy groceries",             "memory.store", {"text": "buy groceries"}),
    ("save note",                          "memory.store", {}),                   # missing text
    ("remember",                           "memory.store", {}),                   # missing text
    ("keep in memory: call dentist",       "memory.store", {"text": "call dentist"}),
    ("note that I need to renew passport", "memory.store", {"text": "I need to renew passport"}),

    # ── memory.read ──────────────────────────────────────────────────────────
    ("what did I save",                    "memory.read",  {}),
    ("show my notes",                      "memory.read",  {}),
    ("read memory",                        "memory.read",  {}),
    ("what do I have saved",               "memory.read",  {}),
    ("recall my notes",                    "memory.read",  {}),

    # ── timer.set ────────────────────────────────────────────────────────────
    ("set a timer for 10 minutes",         "timer.set",    {"duration": "10 minutes"}),
    ("set timer 5 minutes",                "timer.set",    {"duration": "5 minutes"}),
    ("start a 30 second timer",            "timer.set",    {"duration": "30 seconds"}),
    ("set a 1 hour timer",                 "timer.set",    {"duration": "1 hour"}),
    ("set timer",                          "timer.set",    {}),                   # missing duration
    ("start timer for 2 minutes",          "timer.set",    {"duration": "2 minutes"}),
    ("timer 15 minutes",                   "timer.set",    {"duration": "15 minutes"}),

    # ── alarm.set ─────────────────────────────────────────────────────────────
    ("set alarm for 7am",                  "alarm.set",    {"time": "7am"}),
    ("wake me up at 6:30",                 "alarm.set",    {"time": "6:30"}),
    ("set an alarm for 8 in the morning",  "alarm.set",    {"time": "8am"}),
    ("alarm at 9pm",                       "alarm.set",    {"time": "9pm"}),
    ("set alarm",                          "alarm.set",    {}),                   # missing time
    ("wake me up tomorrow morning",        "alarm.set",    {}),                   # no specific time

    # ── call.start ────────────────────────────────────────────────────────────
    ("call alice",                         "call.start",   {"to": "alice"}),
    ("ring bob",                           "call.start",   {"to": "bob"}),
    ("place a call to carol",              "call.start",   {"to": "carol"}),
    ("call",                               "call.start",   {}),                   # missing to
    ("dial dave",                          "call.start",   {"to": "dave"}),
    ("phone eve",                          "call.start",   {"to": "eve"}),

    # ── email.send ────────────────────────────────────────────────────────────
    ("email alice subject hello body hi there",  "email.send", {"to": "alice", "subject": "hello", "body": "hi there"}),
    ("send email to bob",                         "email.send", {"to": "bob"}),         # missing subject, body
    ("email carol",                               "email.send", {"to": "carol"}),       # missing subject, body
    ("send email",                                "email.send", {}),                    # missing all

    # ── weather.current ───────────────────────────────────────────────────────
    ("weather in london",                  "weather.current", {"location": "london"}),
    ("what's the weather in new york",     "weather.current", {"location": "new york"}),
    ("weather forecast for paris",         "weather.current", {"location": "paris"}),
    ("is it raining in tokyo",             "weather.current", {"location": "tokyo"}),
    ("weather",                            "weather.current", {}),                # missing location
    ("how's the weather in berlin",        "weather.current", {"location": "berlin"}),
    ("current weather sydney",             "weather.current", {"location": "sydney"}),

    # ── navigate.to ──────────────────────────────────────────────────────────
    ("navigate to downtown",               "navigate.to",  {"destination": "downtown"}),
    ("take me to the airport",             "navigate.to",  {"destination": "airport"}),
    ("directions to central park",         "navigate.to",  {"destination": "central park"}),
    ("go to 123 main street",              "navigate.to",  {"destination": "123 main street"}),
    ("navigate",                           "navigate.to",  {}),                   # missing destination
    ("get directions",                     "navigate.to",  {}),                   # missing destination
    ("route to the nearest hospital",      "navigate.to",  {"destination": "nearest hospital"}),

    # ── music.play ────────────────────────────────────────────────────────────
    ("play jazz music",                    "music.play",   {"query": "jazz music"}),
    ("play bohemian rhapsody",             "music.play",   {"query": "bohemian rhapsody"}),
    ("play some relaxing music",           "music.play",   {"query": "relaxing music"}),
    ("play",                               "music.play",   {}),                   # missing query
    ("play songs by the beatles",          "music.play",   {"query": "songs by the beatles"}),
    ("start playing hip hop",              "music.play",   {"query": "hip hop"}),

    # ── music.pause ──────────────────────────────────────────────────────────
    ("pause music",                        "music.pause",  {}),
    ("stop playing",                       "music.pause",  {}),
    ("pause",                              "music.pause",  {}),
    ("pause the music",                    "music.pause",  {}),

    # ── settings.volume ──────────────────────────────────────────────────────
    ("set volume to 75",                   "settings.volume", {"level": 75}),
    ("volume 50",                          "settings.volume", {"level": 50}),
    ("turn volume up to 80",               "settings.volume", {"level": 80}),
    ("set volume",                         "settings.volume", {}),                # missing level
    ("change volume to 30",                "settings.volume", {"level": 30}),
    ("volume at 100",                      "settings.volume", {"level": 100}),
    ("set volume to 0",                    "settings.volume", {"level": 0}),
    ("make it louder to 90",               "settings.volume", {"level": 90}),

    # ── settings.brightness ──────────────────────────────────────────────────
    ("set brightness to 60",              "settings.brightness", {"level": 60}),
    ("brightness 80",                     "settings.brightness", {"level": 80}),
    ("set brightness",                    "settings.brightness", {}),             # missing level
    ("dim screen to 20",                  "settings.brightness", {"level": 20}),

    # ── system.status ─────────────────────────────────────────────────────────
    ("system status",                      "system.status", {}),
    ("how is the system doing",            "system.status", {}),
    ("check system health",                "system.status", {}),

    # ── system.help ───────────────────────────────────────────────────────────
    ("help",                               "system.help",  {}),
    ("I need help",                        "system.help",  {}),
    ("show help",                          "system.help",  {}),

    # ── system.version ────────────────────────────────────────────────────────
    ("what version is this",               "system.version", {}),
    ("show version",                       "system.version", {}),
    ("version info",                       "system.version", {}),
]

# ---------------------------------------------------------------------------
# Phrase variations (prefix/suffix/synonym transforms)
# ---------------------------------------------------------------------------

_REPHRASINGS: dict[str, list[str]] = {
    "message.send": [
        "send {to} {text}",
        "message {to} {text}",
        "text {to} {text}",
        "tell {to} {text}",
        "send {to} saying {text}",
        "send a message to {to} saying {text}",
        "send a text to {to}: {text}",
        "drop {to} a message: {text}",
        "notify {to} that {text}",
        "write to {to}: {text}",
    ],
    "payment.send": [
        "send {amount} to {to}",
        "pay {to} {amount}",
        "transfer {amount} to {to}",
        "send {to} {amount} dollars",
        "pay {amount} dollars to {to}",
        "wire {amount} to {to}",
        "venmo {to} {amount}",
        "send payment of {amount} to {to}",
    ],
    "web.search": [
        "search for {query}",
        "look up {query}",
        "find {query}",
        "google {query}",
        "search {query} online",
        "what is {query}",
        "search the web for {query}",
    ],
}


def _make_sample(inp: str, intent: str, params: dict) -> dict:
    """Build a training sample dict. No 'missing' — model only extracts intent+params."""
    return {"input": inp, "output": {"intent": intent, "params": dict(params)}}


def generate_all_samples() -> list[dict]:
    samples: list[dict] = []
    seen: set[tuple] = set()

    for inp, intent, params in SEEDS:
        key = (inp.lower().strip(), intent, json.dumps(params, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        samples.append(_make_sample(inp, intent, params))

    return samples


def main() -> None:
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "distill_english.jsonl"

    samples = generate_all_samples()

    lines = [json.dumps(s, ensure_ascii=False) for s in samples]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {out_path}")

    # Show intent distribution
    from collections import Counter
    counts = Counter(s["output"]["intent"] for s in samples)
    for intent, n in sorted(counts.items()):
        print(f"  {intent}: {n}")


if __name__ == "__main__":
    main()
