#!/usr/bin/env python3
"""Generate canonical intent dataset with deterministic interlang outputs.

Categories covered:
- payments: send, request, balance
- time: now, date, timestamp, timezone, alarm, timer, schedule
- memory: store, read, write, delete, list
- communication: message, call, email, notify
- system: settings, status, help, version

Output: data/canonical/canonical_intents.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.canonicalize import canonicalize
from pipeline.validate import validate

OUTPUT_DIR = ROOT / "data" / "canonical"

# =============================================================================
# CANONICAL INTENTS
# Each tuple: (interlang_program, english_input)
# The interlang is the GROUND TRUTH - never changes across languages
# =============================================================================

CANONICAL_INTENTS: list[tuple[str, str]] = [
    # -------------------------------------------------------------------------
    # TIME OPERATIONS
    # -------------------------------------------------------------------------
    ('. time.now', 'get current time'),
    ('. time.now', 'what time is it'),
    ('. time.now', 'tell me the time'),
    ('. time.now', 'show current time'),
    ('. time.date', 'get current date'),
    ('. time.date', 'what is today date'),
    ('. time.date', 'show today date'),
    ('. time.timestamp', 'get current timestamp'),
    ('. time.timestamp', 'unix timestamp now'),
    ('. time.timezone :value="utc"', 'set timezone to utc'),
    ('. time.timezone :value="pst"', 'set timezone to pst'),
    ('. time.timezone :value="est"', 'set timezone to est'),
    ('. alarm.set :time="07:00"', 'set alarm for 7 am'),
    ('. alarm.set :time="08:30"', 'set alarm for 8:30'),
    ('. alarm.set :time="06:00"', 'wake me up at 6 am'),
    ('. alarm.list', 'show my alarms'),
    ('. alarm.delete :id="1"', 'delete alarm 1'),
    ('. timer.set :minutes="5"', 'set timer for 5 minutes'),
    ('. timer.set :minutes="10"', 'set timer for 10 minutes'),
    ('. timer.set :minutes="30"', 'set timer for 30 minutes'),
    ('. timer.cancel', 'cancel timer'),
    ('. schedule.today', 'what is my schedule today'),
    ('. schedule.tomorrow', 'what is my schedule tomorrow'),
    
    # -------------------------------------------------------------------------
    # MEMORY OPERATIONS
    # -------------------------------------------------------------------------
    ('. memory.store :text="hello"', 'store note hello'),
    ('. memory.store :text="remember this"', 'save note remember this'),
    ('. memory.store :text="important"', 'store note important'),
    ('. memory.store :text="meeting at 3pm"', 'save note meeting at 3pm'),
    ('. memory.store :text="call mom"', 'remember to call mom'),
    ('. memory.read', 'read memory'),
    ('. memory.read', 'show stored notes'),
    ('. memory.read', 'what did I save'),
    ('. memory.read :key="name"', 'read my name'),
    ('. memory.read :key="address"', 'read my address'),
    ('. memory.write :key="name" :value="alice"', 'save my name as alice'),
    ('. memory.write :key="name" :value="bob"', 'save my name as bob'),
    ('. memory.write :key="name" :value="kyle"', 'save my name as kyle'),
    ('. memory.write :key="email" :value="test@example.com"', 'save my email as test@example.com'),
    ('. memory.write :key="phone" :value="555-1234"', 'save my phone as 555-1234'),
    ('. memory.write :key="city" :value="tokyo"', 'save my city as tokyo'),
    ('. memory.write :key="city" :value="london"', 'save my city as london'),
    ('. memory.write :key="city" :value="paris"', 'save my city as paris'),
    ('. memory.delete :key="name"', 'delete my name'),
    ('. memory.delete :key="email"', 'delete my email'),
    ('. memory.list', 'list all saved items'),
    ('. memory.list', 'show all memory'),
    ('. memory.clear', 'clear all memory'),
    
    # -------------------------------------------------------------------------
    # PAYMENT OPERATIONS
    # -------------------------------------------------------------------------
    ('. payment.send :amount="10" :to="alice"', 'send 10 dollars to alice'),
    ('. payment.send :amount="20" :to="bob"', 'send 20 dollars to bob'),
    ('. payment.send :amount="50" :to="charlie"', 'send 50 dollars to charlie'),
    ('. payment.send :amount="100" :to="david"', 'send 100 dollars to david'),
    ('. payment.send :amount="5" :to="eve"', 'pay eve 5 dollars'),
    ('. payment.send :amount="25" :to="frank"', 'transfer 25 to frank'),
    ('. payment.request :amount="10" :from="alice"', 'request 10 dollars from alice'),
    ('. payment.request :amount="50" :from="bob"', 'request 50 dollars from bob'),
    ('. payment.balance', 'check my balance'),
    ('. payment.balance', 'show my balance'),
    ('. payment.balance', 'how much money do I have'),
    ('. payment.history', 'show payment history'),
    ('. payment.history', 'recent transactions'),
    
    # -------------------------------------------------------------------------
    # COMMUNICATION OPERATIONS
    # -------------------------------------------------------------------------
    ('. message.send :to="alice" :text="hello"', 'send message to alice saying hello'),
    ('. message.send :to="bob" :text="hi there"', 'text bob hi there'),
    ('. message.send :to="mom" :text="I love you"', 'message mom I love you'),
    ('. message.send :to="dad" :text="call me"', 'text dad call me'),
    ('. message.read', 'read my messages'),
    ('. message.read', 'show messages'),
    ('. message.read :from="alice"', 'read messages from alice'),
    ('. call.start :to="alice"', 'call alice'),
    ('. call.start :to="bob"', 'call bob'),
    ('. call.start :to="mom"', 'call mom'),
    ('. call.start :to="dad"', 'call dad'),
    ('. call.start :to="911"', 'call 911'),
    ('. call.end', 'end call'),
    ('. call.end', 'hang up'),
    ('. email.send :to="alice@example.com" :subject="hello" :body="hi there"', 
     'email alice@example.com subject hello body hi there'),
    ('. email.read', 'read my emails'),
    ('. email.read', 'check inbox'),
    ('. notify.send :text="reminder"', 'send notification reminder'),
    ('. notify.send :text="meeting soon"', 'notify me meeting soon'),
    
    # -------------------------------------------------------------------------
    # WEB/FETCH OPERATIONS
    # -------------------------------------------------------------------------
    ('. web.request :url="https://example.com"', 'fetch example.com'),
    ('. web.request :url="https://example.com"', 'load example.com'),
    ('. web.request :url="https://api.example.com"', 'fetch api.example.com'),
    ('. web.request :url="https://google.com"', 'open google.com'),
    ('. web.search :query="weather"', 'search for weather'),
    ('. web.search :query="news"', 'search for news'),
    ('. web.search :query="restaurants nearby"', 'search for restaurants nearby'),
    
    # -------------------------------------------------------------------------
    # SYSTEM OPERATIONS
    # -------------------------------------------------------------------------
    ('. system.status', 'system status'),
    ('. system.status', 'check system status'),
    ('. system.version', 'show version'),
    ('. system.version', 'what version is this'),
    ('. system.help', 'help'),
    ('. system.help', 'show help'),
    ('. system.help', 'what can you do'),
    ('. settings.show', 'show settings'),
    ('. settings.show', 'open settings'),
    ('. settings.volume :level="50"', 'set volume to 50'),
    ('. settings.volume :level="100"', 'set volume to max'),
    ('. settings.volume :level="0"', 'mute'),
    ('. settings.brightness :level="50"', 'set brightness to 50'),
    ('. settings.brightness :level="100"', 'max brightness'),
    ('. settings.language :value="english"', 'set language to english'),
    ('. settings.language :value="chinese"', 'set language to chinese'),
    ('. settings.language :value="arabic"', 'set language to arabic'),
    
    # -------------------------------------------------------------------------
    # WEATHER OPERATIONS
    # -------------------------------------------------------------------------
    ('. weather.current', 'what is the weather'),
    ('. weather.current', 'current weather'),
    ('. weather.current', 'how is the weather'),
    ('. weather.current :location="tokyo"', 'weather in tokyo'),
    ('. weather.current :location="london"', 'weather in london'),
    ('. weather.current :location="paris"', 'weather in paris'),
    ('. weather.current :location="new york"', 'weather in new york'),
    ('. weather.forecast', 'weather forecast'),
    ('. weather.forecast', 'forecast for this week'),
    ('. weather.forecast :location="tokyo"', 'forecast for tokyo'),
    
    # -------------------------------------------------------------------------
    # CALENDAR OPERATIONS
    # -------------------------------------------------------------------------
    ('. calendar.today', 'show calendar'),
    ('. calendar.today', 'what is on my calendar'),
    ('. calendar.add :title="meeting" :time="15:00"', 'add meeting at 3pm'),
    ('. calendar.add :title="lunch" :time="12:00"', 'add lunch at noon'),
    ('. calendar.add :title="call" :time="10:00"', 'add call at 10am'),
    ('. calendar.delete :id="1"', 'delete event 1'),
    ('. calendar.next', 'next event'),
    ('. calendar.next', 'what is my next appointment'),
    
    # -------------------------------------------------------------------------
    # MUSIC OPERATIONS
    # -------------------------------------------------------------------------
    ('. music.play', 'play music'),
    ('. music.play', 'start music'),
    ('. music.pause', 'pause music'),
    ('. music.pause', 'stop music'),
    ('. music.next', 'next song'),
    ('. music.next', 'skip'),
    ('. music.previous', 'previous song'),
    ('. music.previous', 'go back'),
    ('. music.play :artist="beatles"', 'play beatles'),
    ('. music.play :song="imagine"', 'play imagine'),
    ('. music.play :genre="jazz"', 'play jazz'),
    ('. music.play :playlist="favorites"', 'play my favorites'),
    
    # -------------------------------------------------------------------------
    # NAVIGATION OPERATIONS
    # -------------------------------------------------------------------------
    ('. navigate.to :destination="home"', 'navigate home'),
    ('. navigate.to :destination="work"', 'navigate to work'),
    ('. navigate.to :destination="airport"', 'directions to airport'),
    ('. navigate.to :destination="hospital"', 'directions to hospital'),
    ('. navigate.eta', 'how long to destination'),
    ('. navigate.eta', 'eta'),
    
    # -------------------------------------------------------------------------
    # SHOPPING OPERATIONS
    # -------------------------------------------------------------------------
    ('. shopping.add :item="milk"', 'add milk to shopping list'),
    ('. shopping.add :item="bread"', 'add bread to shopping list'),
    ('. shopping.add :item="eggs"', 'add eggs to shopping list'),
    ('. shopping.list', 'show shopping list'),
    ('. shopping.clear', 'clear shopping list'),
    ('. shopping.remove :item="milk"', 'remove milk from shopping list'),
    
    # -------------------------------------------------------------------------
    # LIGHT/HOME OPERATIONS
    # -------------------------------------------------------------------------
    ('. lights.on', 'turn on lights'),
    ('. lights.on', 'lights on'),
    ('. lights.off', 'turn off lights'),
    ('. lights.off', 'lights off'),
    ('. lights.dim :level="50"', 'dim lights to 50'),
    ('. lights.on :room="bedroom"', 'turn on bedroom lights'),
    ('. lights.off :room="kitchen"', 'turn off kitchen lights'),
    ('. thermostat.set :temp="72"', 'set temperature to 72'),
    ('. thermostat.set :temp="68"', 'set temperature to 68'),
    ('. thermostat.status', 'what is the temperature'),
    
    # -------------------------------------------------------------------------
    # CHAINED OPERATIONS
    # -------------------------------------------------------------------------
    ('. time.now ; memory.store :text="current_time"', 'get time and save it'),
    ('. web.request :url="https://example.com" ; memory.store :text="result"', 'fetch example.com and save result'),
    ('. payment.balance ; message.send :to="alice" :text="balance checked"', 'check balance and notify alice'),
]


def generate_canonical_dataset() -> list[dict[str, str]]:
    """Generate canonical dataset with validation."""
    rows: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    
    for intent, english_input in CANONICAL_INTENTS:
        # Canonicalize the intent
        try:
            canonical_intent = canonicalize(intent)
        except ValueError as e:
            print(f"WARNING: Failed to canonicalize {intent!r}: {e}")
            continue
        
        # Validate
        if not validate(canonical_intent):
            print(f"WARNING: Invalid intent: {canonical_intent!r}")
            continue
        
        # Check for duplicates
        pair = (canonical_intent, english_input.lower().strip())
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        
        rows.append({
            "intent": canonical_intent,
            "input": english_input,
        })
    
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "canonical_intents.jsonl"
    
    rows = generate_canonical_dataset()
    
    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Stats
    intents = set(r["intent"] for r in rows)
    print(f"Generated {len(rows)} canonical examples")
    print(f"Unique intents: {len(intents)}")
    print(f"Output: {output_path}")
    
    # Category breakdown
    categories: dict[str, int] = {}
    for row in rows:
        op = row["intent"].split()[1].split(".")[0] if " " in row["intent"] else "unknown"
        categories[op] = categories.get(op, 0) + 1
    
    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
