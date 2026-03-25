#!/usr/bin/env python3
"""Generate phrase variations for canonical intents.

Takes canonical intents and expands each into 5-10 phrasing variations
while preserving the exact same meaning and intent.

Input: data/canonical/canonical_intents.jsonl
Output: data/canonical/variations.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANONICAL_DIR = ROOT / "data" / "canonical"

# =============================================================================
# VARIATION TEMPLATES
# These map base patterns to alternative phrasings
# =============================================================================

# Pattern-based variations
# Each key is a regex pattern, value is a list of template variations
# {0}, {1}, etc. are capture groups from the pattern

PATTERN_VARIATIONS: dict[str, list[str]] = {
    # TIME
    r"^get current time$": [
        "what time is it",
        "tell me the time",
        "show the time",
        "current time please",
        "what's the time",
        "time now",
    ],
    r"^what time is it$": [
        "get current time",
        "show time",
        "tell me the time",
        "time please",
    ],
    r"^get current date$": [
        "what is today's date",
        "show date",
        "tell me the date",
        "what date is it",
        "today's date",
    ],
    
    # MEMORY STORE
    r"^store note (.+)$": [
        "save note {0}",
        "remember {0}",
        "save {0}",
        "note {0}",
        "write down {0}",
        "keep note {0}",
        "memorize {0}",
    ],
    r"^save note (.+)$": [
        "store note {0}",
        "remember {0}",
        "note {0}",
        "write down {0}",
    ],
    r"^remember to (.+)$": [
        "remind me to {0}",
        "save note {0}",
        "store note {0}",
        "note {0}",
    ],
    
    # MEMORY READ
    r"^read memory$": [
        "show memory",
        "what did I save",
        "show stored notes",
        "list notes",
        "show notes",
    ],
    r"^show stored notes$": [
        "read memory",
        "list notes",
        "show notes",
        "what's saved",
    ],
    
    # MEMORY WRITE
    r"^save my (\w+) as (.+)$": [
        "set my {0} to {1}",
        "store my {0} as {1}",
        "remember my {0} is {1}",
        "my {0} is {1}",
        "update my {0} to {1}",
    ],
    
    # PAYMENT SEND
    r"^send (\d+) dollars? to (\w+)$": [
        "pay {1} {0} dollars",
        "transfer {0} to {1}",
        "give {1} {0} dollars",
        "send ${0} to {1}",
        "{0} dollars to {1}",
        "wire {0} to {1}",
        "pay {0} to {1}",
    ],
    r"^pay (\w+) (\d+) dollars?$": [
        "send {1} dollars to {0}",
        "transfer {1} to {0}",
        "give {0} {1} dollars",
        "send ${1} to {0}",
    ],
    r"^transfer (\d+) to (\w+)$": [
        "send {0} dollars to {1}",
        "pay {1} {0} dollars",
        "give {1} {0}",
        "wire {0} to {1}",
    ],
    
    # PAYMENT REQUEST
    r"^request (\d+) dollars? from (\w+)$": [
        "ask {1} for {0} dollars",
        "get {0} from {1}",
        "collect {0} from {1}",
        "ask for {0} from {1}",
    ],
    
    # PAYMENT BALANCE
    r"^check my balance$": [
        "show my balance",
        "what's my balance",
        "how much money do I have",
        "balance please",
        "my balance",
        "show balance",
    ],
    
    # MESSAGE
    r"^send message to (\w+) saying (.+)$": [
        "text {0} {1}",
        "message {0} {1}",
        "tell {0} {1}",
        "send {0} a message saying {1}",
    ],
    r"^text (\w+) (.+)$": [
        "send message to {0} saying {1}",
        "message {0} {1}",
        "tell {0} {1}",
    ],
    r"^message (\w+) (.+)$": [
        "text {0} {1}",
        "send message to {0} saying {1}",
        "tell {0} {1}",
    ],
    
    # CALL
    r"^call (\w+)$": [
        "phone {0}",
        "dial {0}",
        "ring {0}",
        "call up {0}",
        "make a call to {0}",
    ],
    
    # WEB
    r"^fetch (.+)$": [
        "load {0}",
        "open {0}",
        "go to {0}",
        "get {0}",
    ],
    r"^search for (.+)$": [
        "look up {0}",
        "find {0}",
        "search {0}",
        "google {0}",
    ],
    
    # WEATHER
    r"^what is the weather$": [
        "how's the weather",
        "current weather",
        "weather please",
        "show weather",
        "weather now",
    ],
    r"^weather in (.+)$": [
        "what's the weather in {0}",
        "how's the weather in {0}",
        "{0} weather",
        "show weather for {0}",
    ],
    
    # ALARM
    r"^set alarm for (.+)$": [
        "wake me up at {0}",
        "alarm at {0}",
        "set alarm at {0}",
        "alarm for {0}",
    ],
    r"^wake me up at (.+)$": [
        "set alarm for {0}",
        "alarm at {0}",
        "alarm for {0}",
    ],
    
    # TIMER
    r"^set timer for (\d+) minutes?$": [
        "timer {0} minutes",
        "{0} minute timer",
        "count down {0} minutes",
        "start {0} minute timer",
    ],
    
    # CALENDAR
    r"^add (.+) at (.+)$": [
        "schedule {0} at {1}",
        "put {0} on calendar at {1}",
        "create event {0} at {1}",
    ],
    
    # MUSIC
    r"^play music$": [
        "start music",
        "music please",
        "play some music",
        "start playing",
    ],
    r"^pause music$": [
        "stop music",
        "pause",
        "stop playing",
    ],
    r"^next song$": [
        "skip",
        "next track",
        "skip song",
        "play next",
    ],
    r"^previous song$": [
        "go back",
        "previous track",
        "last song",
        "play previous",
    ],
    r"^play (\w+)$": [
        "play some {0}",
        "start {0}",
        "put on {0}",
    ],
    
    # NAVIGATION
    r"^navigate to (.+)$": [
        "directions to {0}",
        "take me to {0}",
        "go to {0}",
        "route to {0}",
    ],
    r"^directions to (.+)$": [
        "navigate to {0}",
        "how do I get to {0}",
        "take me to {0}",
    ],
    
    # SHOPPING
    r"^add (.+) to shopping list$": [
        "put {0} on shopping list",
        "add {0} to my list",
        "shopping list add {0}",
    ],
    
    # LIGHTS
    r"^turn on lights$": [
        "lights on",
        "switch on lights",
        "enable lights",
    ],
    r"^turn off lights$": [
        "lights off",
        "switch off lights",
        "disable lights",
    ],
    r"^turn on (.+) lights$": [
        "{0} lights on",
        "switch on {0} lights",
    ],
    r"^turn off (.+) lights$": [
        "{0} lights off",
        "switch off {0} lights",
    ],
    
    # SETTINGS
    r"^set volume to (\d+)$": [
        "volume {0}",
        "volume to {0}",
        "change volume to {0}",
    ],
    r"^set brightness to (\d+)$": [
        "brightness {0}",
        "brightness to {0}",
        "change brightness to {0}",
    ],
    r"^set temperature to (\d+)$": [
        "temperature {0}",
        "thermostat to {0}",
        "change temperature to {0}",
    ],
    r"^set language to (\w+)$": [
        "language {0}",
        "change language to {0}",
        "switch to {0}",
    ],
    
    # SYSTEM
    r"^help$": [
        "show help",
        "what can you do",
        "commands",
        "help me",
    ],
    r"^show version$": [
        "version",
        "what version",
        "version number",
    ],
    r"^system status$": [
        "check system status",
        "status",
        "how is the system",
    ],
}


def generate_variations(base_input: str) -> list[str]:
    """Generate phrase variations for a given input."""
    variations: list[str] = [base_input]  # Always include original
    normalized = base_input.lower().strip()
    
    for pattern, templates in PATTERN_VARIATIONS.items():
        match = re.match(pattern, normalized, re.IGNORECASE)
        if match:
            groups = match.groups()
            for template in templates:
                try:
                    # Replace {0}, {1}, etc. with captured groups
                    variation = template
                    for i, group in enumerate(groups):
                        variation = variation.replace(f"{{{i}}}", group)
                    if variation.lower() != normalized:
                        variations.append(variation)
                except Exception:
                    continue
    
    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in variations:
        lower_v = v.lower().strip()
        if lower_v not in seen:
            seen.add(lower_v)
            unique.append(v)
    
    return unique


def main() -> None:
    input_path = CANONICAL_DIR / "canonical_intents.jsonl"
    output_path = CANONICAL_DIR / "variations.jsonl"
    
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run generate_canonical.py first.")
        sys.exit(1)
    
    # Load canonical intents
    rows: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    # Generate variations
    expanded: list[dict] = []
    for row in rows:
        intent = row["intent"]
        base_input = row["input"]
        variations = generate_variations(base_input)
        
        for var in variations:
            expanded.append({
                "intent": intent,
                "input": var,
                "base_input": base_input,
            })
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for row in expanded:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Stats
    print(f"Expanded {len(rows)} canonical examples to {len(expanded)} variations")
    print(f"Average variations per intent: {len(expanded) / len(rows):.1f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
