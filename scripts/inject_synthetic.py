#!/usr/bin/env python3
"""Inject high-frequency synthetic training samples into english.jsonl.

Generates ~2500 deterministic input/output pairs with variations to
reinforce strict symbolic output patterns during training.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_PATH = ROOT / "data" / "families" / "english.jsonl"

# =============================================================================
# CORE PATTERNS: (intent, [input_variations])
# =============================================================================
CORE_PATTERNS: list[tuple[str, list[str]]] = [
    # TIME
    (". time.now", [
        "current time", "what time is it", "tell me the time", "time now",
        "show time", "time please", "get current time", "what's the time",
        "show the time", "current time please", "the time", "check time",
        "display time", "give me the time", "what time is it now",
    ]),
    (". time.date", [
        "what is the date", "today's date", "current date", "show date",
        "tell me the date", "what date is it", "date today", "the date",
        "get current date", "show today date", "date please", "display date",
        "what day is it", "give me the date", "check date",
    ]),
    (". time.timestamp", [
        "unix timestamp", "timestamp now", "current timestamp",
        "get timestamp", "show timestamp", "epoch time", "unix time",
        "get current timestamp", "unix timestamp now", "time stamp",
    ]),

    # MEMORY
    ('. memory.store :text="hello"', [
        "store note hello", "save hello", "remember hello", "note hello",
        "write down hello", "keep hello", "memorize hello", "store hello",
        "save note hello", "record hello",
    ]),
    ('. memory.store :text="meeting at 3pm"', [
        "save note meeting at 3pm", "remember meeting at 3pm",
        "store meeting at 3pm", "note meeting at 3pm",
        "write down meeting at 3pm", "save meeting at 3pm",
    ]),
    (". memory.read", [
        "read memory", "show notes", "what did I save", "show memory",
        "list notes", "read notes", "show stored notes", "get notes",
        "display memory", "check memory", "read my notes", "my notes",
    ]),
    (". memory.list", [
        "list all saved items", "show all memory", "list everything",
        "show all notes", "list all", "everything saved", "all notes",
    ]),
    (". memory.clear", [
        "clear all memory", "delete all notes", "clear memory",
        "erase memory", "wipe memory", "clear all", "reset memory",
    ]),

    # PAYMENT
    ('. payment.send :amount="10" :to="alice"', [
        "send 10 dollars to alice", "pay alice 10", "transfer 10 to alice",
        "give alice 10 dollars", "send alice 10", "wire 10 to alice",
        "10 dollars to alice", "pay 10 to alice",
    ]),
    ('. payment.send :amount="50" :to="bob"', [
        "send 50 dollars to bob", "pay bob 50", "transfer 50 to bob",
        "give bob 50 dollars", "send bob 50", "wire 50 to bob",
        "50 dollars to bob", "pay 50 to bob",
    ]),
    (". payment.balance", [
        "check my balance", "show my balance", "what's my balance",
        "how much money do I have", "balance please", "my balance",
        "show balance", "check balance", "account balance",
    ]),
    (". payment.history", [
        "show payment history", "recent transactions", "transaction history",
        "payment log", "show transactions", "my transactions",
    ]),

    # COMMUNICATION
    ('. message.send :to="alice" :text="hello"', [
        "send message to alice saying hello", "text alice hello",
        "message alice hello", "tell alice hello", "send alice hello",
    ]),
    ('. message.send :to="bob" :text="hi"', [
        "send message to bob saying hi", "text bob hi",
        "message bob hi", "tell bob hi", "send bob hi",
    ]),
    (". message.read", [
        "read my messages", "show messages", "check messages",
        "read messages", "my messages", "inbox", "show my messages",
    ]),
    ('. call.start :to="alice"', [
        "call alice", "phone alice", "dial alice", "ring alice",
        "call up alice", "make a call to alice",
    ]),
    ('. call.start :to="mom"', [
        "call mom", "phone mom", "dial mom", "ring mom",
        "call up mom", "make a call to mom",
    ]),
    (". call.end", [
        "end call", "hang up", "end the call", "disconnect",
        "stop call", "hang up the phone",
    ]),
    (". email.read", [
        "read my emails", "check inbox", "show emails", "read emails",
        "my emails", "check email", "show my email",
    ]),

    # WEB
    ('. web.search :query="weather"', [
        "search for weather", "look up weather", "find weather",
        "search weather", "google weather",
    ]),
    ('. web.search :query="news"', [
        "search for news", "look up news", "find news",
        "search news", "google news",
    ]),
    ('. web.search :query="restaurants nearby"', [
        "search for restaurants nearby", "find restaurants nearby",
        "look up restaurants nearby", "search restaurants near me",
    ]),

    # WEATHER
    (". weather.current", [
        "what is the weather", "current weather", "how is the weather",
        "weather now", "show weather", "weather please", "check weather",
        "the weather", "weather today", "display weather",
    ]),
    ('. weather.current :location="tokyo"', [
        "weather in tokyo", "tokyo weather", "what's the weather in tokyo",
        "how's the weather in tokyo", "show weather for tokyo",
    ]),
    ('. weather.current :location="london"', [
        "weather in london", "london weather", "what's the weather in london",
        "how's the weather in london", "show weather for london",
    ]),
    ('. weather.current :location="new york"', [
        "weather in new york", "new york weather",
        "what's the weather in new york", "show weather for new york",
    ]),
    (". weather.forecast", [
        "weather forecast", "forecast", "this week forecast",
        "weekly forecast", "forecast for this week", "show forecast",
    ]),

    # ALARM/TIMER
    ('. alarm.set :time="07:00"', [
        "set alarm for 7 am", "alarm at 7", "wake me up at 7",
        "alarm 7am", "set alarm 7:00", "7 am alarm",
    ]),
    ('. alarm.set :time="08:30"', [
        "set alarm for 8:30", "alarm at 8:30", "wake me up at 8:30",
        "alarm 8:30", "set alarm 8:30",
    ]),
    (". alarm.list", [
        "show my alarms", "list alarms", "my alarms", "show alarms",
        "check alarms", "display alarms",
    ]),
    ('. timer.set :minutes="5"', [
        "set timer for 5 minutes", "5 minute timer", "timer 5 minutes",
        "count down 5 minutes", "start 5 minute timer",
    ]),
    ('. timer.set :minutes="10"', [
        "set timer for 10 minutes", "10 minute timer", "timer 10 minutes",
        "count down 10 minutes", "start 10 minute timer",
    ]),
    (". timer.cancel", [
        "cancel timer", "stop timer", "delete timer", "end timer",
        "cancel the timer",
    ]),

    # CALENDAR
    (". calendar.today", [
        "show calendar", "what is on my calendar", "my calendar",
        "calendar today", "today's calendar", "check calendar",
    ]),
    ('. calendar.add :title="meeting" :time="15:00"', [
        "add meeting at 3pm", "schedule meeting at 3pm",
        "put meeting on calendar at 3pm", "create meeting at 3",
        "meeting at 3pm", "add meeting at 15:00",
    ]),
    (". calendar.next", [
        "next event", "what is my next appointment", "next appointment",
        "upcoming event", "what's next", "next on calendar",
    ]),

    # MUSIC
    (". music.play", [
        "play music", "start music", "music please", "play some music",
        "start playing", "put on music", "music on",
    ]),
    (". music.pause", [
        "pause music", "stop music", "pause", "stop playing",
        "music off", "pause the music",
    ]),
    (". music.next", [
        "next song", "skip", "next track", "skip song", "play next",
        "next", "skip track",
    ]),
    (". music.previous", [
        "previous song", "go back", "previous track", "last song",
        "play previous", "back", "previous",
    ]),
    ('. music.play :genre="jazz"', [
        "play jazz", "jazz music", "play some jazz", "put on jazz",
        "start jazz", "jazz please",
    ]),
    ('. music.play :artist="beatles"', [
        "play beatles", "play the beatles", "put on beatles",
        "beatles music", "start beatles",
    ]),

    # NAVIGATION
    ('. navigate.to :destination="home"', [
        "navigate home", "directions home", "take me home",
        "go home", "route home", "get me home",
    ]),
    ('. navigate.to :destination="work"', [
        "navigate to work", "directions to work", "take me to work",
        "go to work", "route to work",
    ]),
    (". navigate.eta", [
        "how long to destination", "eta", "estimated time",
        "how long until arrival", "time to destination",
    ]),

    # SHOPPING
    ('. shopping.add :item="milk"', [
        "add milk to shopping list", "put milk on list",
        "add milk to my list", "shopping list add milk", "buy milk",
    ]),
    ('. shopping.add :item="bread"', [
        "add bread to shopping list", "put bread on list",
        "add bread to my list", "shopping list add bread", "buy bread",
    ]),
    (". shopping.list", [
        "show shopping list", "my shopping list", "shopping list",
        "what's on my list", "display shopping list",
    ]),
    (". shopping.clear", [
        "clear shopping list", "empty shopping list", "reset shopping list",
        "delete shopping list",
    ]),

    # LIGHTS/HOME
    (". lights.on", [
        "turn on lights", "lights on", "switch on lights",
        "enable lights", "turn on the lights", "light up",
    ]),
    (". lights.off", [
        "turn off lights", "lights off", "switch off lights",
        "disable lights", "turn off the lights",
    ]),
    ('. lights.on :room="bedroom"', [
        "turn on bedroom lights", "bedroom lights on",
        "switch on bedroom lights", "lights on in bedroom",
    ]),
    ('. thermostat.set :temp="72"', [
        "set temperature to 72", "thermostat 72", "temperature 72",
        "set thermostat to 72", "change temperature to 72",
    ]),
    (". thermostat.status", [
        "what is the temperature", "check temperature", "current temperature",
        "thermostat status", "show temperature",
    ]),

    # SYSTEM
    (". system.status", [
        "system status", "check system status", "status",
        "how is the system", "system check",
    ]),
    (". system.help", [
        "help", "show help", "what can you do", "commands",
        "help me", "list commands",
    ]),
    (". system.version", [
        "show version", "what version", "version", "version number",
        "check version",
    ]),
    ('. settings.volume :level="50"', [
        "set volume to 50", "volume 50", "volume to 50",
        "change volume to 50", "adjust volume 50",
    ]),
    ('. settings.brightness :level="100"', [
        "max brightness", "brightness 100", "brightness to 100",
        "set brightness to 100", "full brightness",
    ]),

    # CHAINED
    ('. time.now ; memory.store :text="current_time"', [
        "get time and save it", "check time and store it",
        "time and save", "get time then save",
    ]),
    ('. payment.balance ; message.send :to="alice" :text="balance checked"', [
        "check balance and notify alice", "balance then message alice",
        "check balance and tell alice",
    ]),

    # MEMORY WRITE variants
    ('. memory.write :key="name" :value="alice"', [
        "save my name as alice", "my name is alice", "set my name to alice",
        "store my name as alice", "remember my name is alice",
    ]),
    ('. memory.write :key="email" :value="test@example.com"', [
        "save my email as test@example.com", "my email is test@example.com",
        "set my email to test@example.com", "store my email as test@example.com",
    ]),
    ('. memory.delete :key="name"', [
        "delete my name", "remove my name", "erase my name",
        "forget my name", "clear my name",
    ]),

    # NOTIFY
    ('. notify.send :text="reminder"', [
        "send notification reminder", "notify reminder",
        "send reminder notification", "push notification reminder",
    ]),

    # SETTINGS
    ('. settings.language :value="english"', [
        "set language to english", "language english",
        "change language to english", "switch to english",
    ]),
    (". settings.show", [
        "show settings", "open settings", "settings", "my settings",
        "display settings", "check settings",
    ]),

    # PAYMENT REQUEST
    ('. payment.request :amount="10" :from="alice"', [
        "request 10 dollars from alice", "ask alice for 10 dollars",
        "get 10 from alice", "collect 10 from alice",
    ]),

    # =========================================================================
    # EXTRA PARAMETERIZED PATTERNS (reinforces :key="value" structure)
    # =========================================================================

    # Payment — many amounts and recipients
    ('. payment.send :amount="5" :to="eve"', [
        "send 5 to eve", "pay eve 5", "transfer 5 to eve",
        "give eve 5 dollars", "5 dollars to eve", "wire 5 to eve",
    ]),
    ('. payment.send :amount="20" :to="bob"', [
        "send 20 to bob", "pay bob 20", "transfer 20 to bob",
        "give bob 20 dollars", "20 dollars to bob", "wire 20 to bob",
    ]),
    ('. payment.send :amount="100" :to="david"', [
        "send 100 to david", "pay david 100", "transfer 100 to david",
        "give david 100 dollars", "100 dollars to david",
    ]),
    ('. payment.send :amount="25" :to="frank"', [
        "send 25 to frank", "pay frank 25", "transfer 25 to frank",
        "give frank 25", "25 to frank",
    ]),
    ('. payment.send :amount="75" :to="charlie"', [
        "send 75 to charlie", "pay charlie 75", "transfer 75 to charlie",
        "75 dollars to charlie",
    ]),
    ('. payment.send :amount="15" :to="mom"', [
        "send 15 to mom", "pay mom 15", "transfer 15 to mom",
        "give mom 15 dollars",
    ]),
    ('. payment.send :amount="200" :to="dad"', [
        "send 200 to dad", "pay dad 200", "transfer 200 to dad",
        "200 dollars to dad",
    ]),
    ('. payment.request :amount="50" :from="bob"', [
        "request 50 from bob", "ask bob for 50", "get 50 from bob",
        "collect 50 from bob",
    ]),
    ('. payment.request :amount="100" :from="charlie"', [
        "request 100 from charlie", "ask charlie for 100",
        "get 100 from charlie", "collect 100 from charlie",
    ]),

    # Memory store — many texts
    ('. memory.store :text="buy groceries"', [
        "store note buy groceries", "save buy groceries",
        "remember buy groceries", "note buy groceries",
    ]),
    ('. memory.store :text="call doctor"', [
        "store note call doctor", "save call doctor",
        "remember call doctor", "note call doctor",
    ]),
    ('. memory.store :text="pick up kids"', [
        "store note pick up kids", "save pick up kids",
        "remember pick up kids", "note pick up kids",
    ]),
    ('. memory.store :text="important meeting"', [
        "store note important meeting", "save important meeting",
        "remember important meeting", "note important meeting",
    ]),
    ('. memory.store :text="dentist appointment"', [
        "remember dentist appointment", "save dentist appointment",
        "store note dentist appointment", "note dentist appointment",
    ]),

    # Memory write — many keys and values
    ('. memory.write :key="name" :value="bob"', [
        "save my name as bob", "my name is bob", "set name to bob",
        "store my name as bob",
    ]),
    ('. memory.write :key="name" :value="kyle"', [
        "save my name as kyle", "my name is kyle", "set name to kyle",
        "store my name as kyle",
    ]),
    ('. memory.write :key="city" :value="tokyo"', [
        "save my city as tokyo", "my city is tokyo", "set city to tokyo",
        "store my city as tokyo",
    ]),
    ('. memory.write :key="city" :value="london"', [
        "save my city as london", "my city is london", "set city to london",
        "store my city as london",
    ]),
    ('. memory.write :key="city" :value="paris"', [
        "save my city as paris", "my city is paris", "set city to paris",
        "store my city as paris",
    ]),
    ('. memory.write :key="phone" :value="555-1234"', [
        "save my phone as 555-1234", "my phone is 555-1234",
        "set phone to 555-1234", "store my phone as 555-1234",
    ]),
    ('. memory.read :key="name"', [
        "read my name", "what is my name", "get my name", "show my name",
    ]),
    ('. memory.read :key="email"', [
        "read my email", "what is my email", "get my email", "show my email",
    ]),
    ('. memory.read :key="city"', [
        "read my city", "what is my city", "get my city", "show my city",
    ]),
    ('. memory.delete :key="email"', [
        "delete my email", "remove my email", "erase my email",
        "forget my email",
    ]),
    ('. memory.delete :key="phone"', [
        "delete my phone", "remove my phone", "erase my phone",
        "forget my phone",
    ]),

    # Alarm — many times
    ('. alarm.set :time="06:00"', [
        "set alarm for 6 am", "alarm at 6", "wake me up at 6",
        "alarm 6am", "6 am alarm",
    ]),
    ('. alarm.set :time="09:00"', [
        "set alarm for 9 am", "alarm at 9", "wake me up at 9",
        "alarm 9am", "9 am alarm",
    ]),
    ('. alarm.set :time="22:00"', [
        "set alarm for 10 pm", "alarm at 10 pm", "alarm 10pm",
        "10 pm alarm",
    ]),
    ('. alarm.delete :id="1"', [
        "delete alarm 1", "remove alarm 1", "cancel alarm 1",
    ]),
    ('. alarm.delete :id="2"', [
        "delete alarm 2", "remove alarm 2", "cancel alarm 2",
    ]),

    # Timer — many durations
    ('. timer.set :minutes="1"', [
        "set timer for 1 minute", "1 minute timer", "timer 1 minute",
    ]),
    ('. timer.set :minutes="15"', [
        "set timer for 15 minutes", "15 minute timer", "timer 15 minutes",
    ]),
    ('. timer.set :minutes="30"', [
        "set timer for 30 minutes", "30 minute timer", "timer 30 minutes",
    ]),
    ('. timer.set :minutes="60"', [
        "set timer for 60 minutes", "60 minute timer", "timer 1 hour",
        "set timer for 1 hour",
    ]),

    # Messages — many recipients and texts
    ('. message.send :to="charlie" :text="hey"', [
        "text charlie hey", "message charlie hey", "send charlie hey",
        "tell charlie hey",
    ]),
    ('. message.send :to="mom" :text="I love you"', [
        "text mom I love you", "message mom I love you",
        "tell mom I love you", "send mom I love you",
    ]),
    ('. message.send :to="dad" :text="call me"', [
        "text dad call me", "message dad call me",
        "tell dad call me", "send dad call me",
    ]),
    ('. message.read :from="alice"', [
        "read messages from alice", "show messages from alice",
        "alice's messages", "messages from alice",
    ]),
    ('. message.read :from="bob"', [
        "read messages from bob", "show messages from bob",
        "bob's messages", "messages from bob",
    ]),

    # Calls — many contacts
    ('. call.start :to="bob"', [
        "call bob", "phone bob", "dial bob", "ring bob",
    ]),
    ('. call.start :to="dad"', [
        "call dad", "phone dad", "dial dad", "ring dad",
    ]),
    ('. call.start :to="911"', [
        "call 911", "dial 911", "emergency call", "phone 911",
    ]),

    # Navigation — many destinations
    ('. navigate.to :destination="airport"', [
        "navigate to airport", "directions to airport",
        "take me to airport", "route to airport",
    ]),
    ('. navigate.to :destination="hospital"', [
        "navigate to hospital", "directions to hospital",
        "take me to hospital", "route to hospital",
    ]),
    ('. navigate.to :destination="school"', [
        "navigate to school", "directions to school",
        "take me to school", "route to school",
    ]),

    # Shopping — many items
    ('. shopping.add :item="eggs"', [
        "add eggs to shopping list", "buy eggs",
        "put eggs on list", "shopping list add eggs",
    ]),
    ('. shopping.add :item="butter"', [
        "add butter to shopping list", "buy butter",
        "put butter on list", "shopping list add butter",
    ]),
    ('. shopping.add :item="cheese"', [
        "add cheese to shopping list", "buy cheese",
        "put cheese on list", "shopping list add cheese",
    ]),
    ('. shopping.remove :item="milk"', [
        "remove milk from shopping list", "delete milk from list",
        "take milk off list",
    ]),
    ('. shopping.remove :item="bread"', [
        "remove bread from shopping list", "delete bread from list",
        "take bread off list",
    ]),

    # Lights — many rooms
    ('. lights.off :room="bedroom"', [
        "turn off bedroom lights", "bedroom lights off",
        "switch off bedroom lights",
    ]),
    ('. lights.on :room="kitchen"', [
        "turn on kitchen lights", "kitchen lights on",
        "switch on kitchen lights",
    ]),
    ('. lights.off :room="kitchen"', [
        "turn off kitchen lights", "kitchen lights off",
        "switch off kitchen lights",
    ]),
    ('. lights.on :room="living room"', [
        "turn on living room lights", "living room lights on",
        "switch on living room lights",
    ]),
    ('. lights.dim :level="50"', [
        "dim lights to 50", "set lights to 50 percent",
        "dim to 50", "lights 50 percent",
    ]),
    ('. lights.dim :level="25"', [
        "dim lights to 25", "set lights to 25 percent",
        "dim to 25", "lights 25 percent",
    ]),

    # Thermostat — many temps
    ('. thermostat.set :temp="68"', [
        "set temperature to 68", "thermostat 68", "temperature 68",
        "set thermostat to 68",
    ]),
    ('. thermostat.set :temp="75"', [
        "set temperature to 75", "thermostat 75", "temperature 75",
        "set thermostat to 75",
    ]),
    ('. thermostat.set :temp="65"', [
        "set temperature to 65", "thermostat 65", "temperature 65",
        "set thermostat to 65",
    ]),

    # Settings — many levels
    ('. settings.volume :level="0"', [
        "mute", "mute volume", "volume 0", "set volume to 0", "silence",
    ]),
    ('. settings.volume :level="100"', [
        "max volume", "volume 100", "set volume to 100",
        "volume to max", "full volume",
    ]),
    ('. settings.volume :level="75"', [
        "set volume to 75", "volume 75", "volume to 75",
    ]),
    ('. settings.brightness :level="50"', [
        "set brightness to 50", "brightness 50", "brightness to 50",
        "half brightness",
    ]),
    ('. settings.brightness :level="0"', [
        "brightness off", "screen off", "brightness 0",
        "set brightness to 0",
    ]),
    ('. settings.language :value="chinese"', [
        "set language to chinese", "language chinese",
        "change language to chinese", "switch to chinese",
    ]),
    ('. settings.language :value="arabic"', [
        "set language to arabic", "language arabic",
        "change language to arabic", "switch to arabic",
    ]),

    # Weather — more locations
    ('. weather.current :location="paris"', [
        "weather in paris", "paris weather",
        "what's the weather in paris", "show weather for paris",
    ]),
    ('. weather.current :location="berlin"', [
        "weather in berlin", "berlin weather",
        "what's the weather in berlin", "show weather for berlin",
    ]),
    ('. weather.current :location="sydney"', [
        "weather in sydney", "sydney weather",
        "what's the weather in sydney", "show weather for sydney",
    ]),
    ('. weather.forecast :location="tokyo"', [
        "forecast for tokyo", "tokyo forecast",
        "weekly forecast tokyo", "tokyo weather forecast",
    ]),
    ('. weather.forecast :location="london"', [
        "forecast for london", "london forecast",
        "weekly forecast london", "london weather forecast",
    ]),

    # Calendar — more events
    ('. calendar.add :title="lunch" :time="12:00"', [
        "add lunch at noon", "schedule lunch at 12",
        "put lunch on calendar at noon", "lunch at 12",
    ]),
    ('. calendar.add :title="call" :time="10:00"', [
        "add call at 10am", "schedule call at 10",
        "put call on calendar at 10", "call at 10am",
    ]),
    ('. calendar.add :title="gym" :time="18:00"', [
        "add gym at 6pm", "schedule gym at 6",
        "put gym on calendar at 6pm", "gym at 6pm",
    ]),
    ('. calendar.delete :id="1"', [
        "delete event 1", "remove event 1", "cancel event 1",
    ]),
    ('. calendar.delete :id="2"', [
        "delete event 2", "remove event 2", "cancel event 2",
    ]),

    # Music — more artists/songs
    ('. music.play :song="imagine"', [
        "play imagine", "play the song imagine", "put on imagine",
    ]),
    ('. music.play :playlist="favorites"', [
        "play my favorites", "play favorites playlist",
        "put on my favorites", "favorites playlist",
    ]),
    ('. music.play :genre="rock"', [
        "play rock", "rock music", "play some rock", "put on rock",
    ]),
    ('. music.play :genre="classical"', [
        "play classical", "classical music", "play some classical",
    ]),
    ('. music.play :artist="drake"', [
        "play drake", "put on drake", "drake music", "play some drake",
    ]),

    # Email
    ('. email.send :to="alice@example.com" :subject="hello" :body="hi there"', [
        "email alice at example.com subject hello body hi there",
        "send email to alice@example.com about hello",
    ]),

    # Web requests
    ('. web.request :url="https://example.com"', [
        "fetch example.com", "load example.com", "open example.com",
        "go to example.com", "get example.com",
    ]),
    ('. web.request :url="https://google.com"', [
        "open google", "go to google", "load google.com",
        "fetch google.com",
    ]),

    # Notify — more texts
    ('. notify.send :text="meeting soon"', [
        "notify meeting soon", "send notification meeting soon",
        "push notification meeting soon", "alert meeting soon",
    ]),
    ('. notify.send :text="time to leave"', [
        "notify time to leave", "send notification time to leave",
        "alert time to leave",
    ]),

    # Timezone
    ('. time.timezone :value="utc"', [
        "set timezone to utc", "timezone utc", "change timezone to utc",
    ]),
    ('. time.timezone :value="pst"', [
        "set timezone to pst", "timezone pst", "change timezone to pst",
    ]),
    ('. time.timezone :value="est"', [
        "set timezone to est", "timezone est", "change timezone to est",
    ]),

    # Schedule
    (". schedule.today", [
        "what is my schedule today", "today's schedule",
        "my schedule today", "schedule for today",
    ]),
    (". schedule.tomorrow", [
        "what is my schedule tomorrow", "tomorrow's schedule",
        "my schedule tomorrow", "schedule for tomorrow",
    ]),
]


def _generate_copy_training_samples() -> list[dict]:
    """Generate large-scale copy-training samples to teach exact value reproduction."""
    random.seed(42)  # deterministic
    samples = []

    names = ["alice", "bob", "john", "emma", "mike", "charlie", "david",
             "eve", "frank", "grace", "mom", "dad"]
    words = ["hello", "world", "test", "note", "data", "value", "reminder",
             "urgent", "meeting", "groceries", "homework", "birthday",
             "appointment", "password", "important", "tomorrow",
             "schedule", "weather", "balance", "transfer", "message",
             "settings", "navigate", "shopping", "calendar", "playlist"]
    long_tail = ["hello123", "testing123", "example_value", "alpha_beta",
                 "world2026", "user_name", "test_data", "my_note",
                 "backup_file", "config_v2", "session_id", "token_abc"]
    cities = ["tokyo", "london", "paris", "berlin", "sydney", "mumbai",
              "toronto", "cairo", "rome", "seoul", "dubai", "oslo"]

    # Numeric copying — payment.send with many amounts and recipients
    for amount in list(range(1, 101)) + list(range(150, 1001, 50)):
        name = random.choice(names)
        samples.append({
            "intent": f'. payment.send :amount="{amount}" :to="{name}"',
            "input": f"send {amount} dollars to {name}",
        })

    # Payment request — numeric copying
    for amount in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 500]:
        name = random.choice(names)
        samples.append({
            "intent": f'. payment.request :amount="{amount}" :from="{name}"',
            "input": f"request {amount} dollars from {name}",
        })

    # String copying — memory.store with many texts
    for word in words:
        samples.append({
            "intent": f'. memory.store :text="{word}"',
            "input": f"store note {word}",
        })
        samples.append({
            "intent": f'. memory.store :text="{word}"',
            "input": f"save {word}",
        })
        samples.append({
            "intent": f'. memory.store :text="{word}"',
            "input": f"remember {word}",
        })

    # Two-word phrases for memory
    phrase_pairs = [
        "buy groceries", "call doctor", "pick up kids", "important meeting",
        "dentist appointment", "pay rent", "walk dog", "clean house",
        "fix bug", "send report", "book flight", "cancel subscription",
    ]
    for phrase in phrase_pairs:
        samples.append({
            "intent": f'. memory.store :text="{phrase}"',
            "input": f"remember {phrase}",
        })
        samples.append({
            "intent": f'. memory.store :text="{phrase}"',
            "input": f"store note {phrase}",
        })

    # memory.write — many key/value combos
    for name in names:
        samples.append({
            "intent": f'. memory.write :key="name" :value="{name}"',
            "input": f"save my name as {name}",
        })
    for city in cities:
        samples.append({
            "intent": f'. memory.write :key="city" :value="{city}"',
            "input": f"save my city as {city}",
        })

    # Explicit echo/copy tasks — NUMBERS (reduced to balance with strings)
    for i in list(range(1, 101)) + list(range(150, 501, 50)):
        samples.append({
            "intent": f'. debug.echo :value="{i}"',
            "input": f"repeat {i}",
        })

    # Explicit echo/copy tasks — STRINGS (high density, ~1:1 with numbers)
    all_strings = words + list(cities) + list(names) + long_tail
    for word in all_strings:
        # repeat X → echo X (3 variations each to increase density)
        samples.append({
            "intent": f'. debug.echo :value="{word}"',
            "input": f"repeat {word}",
        })
        samples.append({
            "intent": f'. debug.echo :value="{word}"',
            "input": f"echo {word}",
        })
        samples.append({
            "intent": f'. debug.echo :value="{word}"',
            "input": f"say {word}",
        })

    # =========================================================================
    # CONTRAST PAIRS — teach prefix ≠ full word (CRITICAL for completion)
    # =========================================================================
    contrast_groups = [
        # (prefix, full) — model must learn these are DIFFERENT outputs
        ("hell", "hello"),
        ("hel", "hello"),
        ("worl", "world"),
        ("wor", "world"),
        ("tes", "test"),
        ("test", "testing"),
        ("testing", "testing123"),
        ("not", "note"),
        ("dat", "data"),
        ("val", "value"),
        ("valu", "value"),
        ("remind", "reminder"),
        ("urge", "urgent"),
        ("meet", "meeting"),
        ("groc", "groceries"),
        ("grocer", "groceries"),
        ("home", "homework"),
        ("birth", "birthday"),
        ("pass", "password"),
        ("import", "important"),
        ("tom", "tomorrow"),
        ("tomor", "tomorrow"),
        ("appoint", "appointment"),
        ("sched", "schedule"),
        ("bal", "balance"),
        ("trans", "transfer"),
        ("mess", "message"),
        ("navi", "navigate"),
        ("play", "playlist"),
        ("calen", "calendar"),
        ("alic", "alice"),
        ("charl", "charlie"),
        ("davi", "david"),
        ("fran", "frank"),
        ("grac", "grace"),
        ("tok", "tokyo"),
        ("lond", "london"),
        ("par", "paris"),
        ("berl", "berlin"),
        ("sydn", "sydney"),
        ("mumb", "mumbai"),
        ("toron", "toronto"),
    ]
    for prefix, full in contrast_groups:
        # The prefix IS a valid output when explicitly requested
        samples.append({
            "intent": f'. debug.echo :value="{prefix}"',
            "input": f"repeat {prefix}",
        })
        # The full word is ALSO a valid output — model must distinguish
        samples.append({
            "intent": f'. debug.echo :value="{full}"',
            "input": f"repeat {full}",
        })
        # Also reinforce in memory.store context
        samples.append({
            "intent": f'. memory.store :text="{prefix}"',
            "input": f"store note {prefix}",
        })
        samples.append({
            "intent": f'. memory.store :text="{full}"',
            "input": f"store note {full}",
        })

    # =========================================================================
    # END-OF-VALUE REINFORCEMENT — extended forms that MUST complete
    # =========================================================================
    extended_groups = [
        ("hello", "helloo", "hellooo"),
        ("world", "worldd", "worlddd"),
        ("test", "testt", "testtt"),
        ("value", "valuee", "valueee"),
        ("data", "dataa", "dataaa"),
        ("note", "notee", "noteee"),
    ]
    for group in extended_groups:
        for form in group:
            samples.append({
                "intent": f'. debug.echo :value="{form}"',
                "input": f"repeat {form}",
            })
            samples.append({
                "intent": f'. memory.store :text="{form}"',
                "input": f"store note {form}",
            })

    # Long-tail strings — echo + memory.store
    for word in long_tail:
        samples.append({
            "intent": f'. memory.store :text="{word}"',
            "input": f"store note {word}",
        })
        samples.append({
            "intent": f'. memory.store :text="{word}"',
            "input": f"save {word}",
        })
        samples.append({
            "intent": f'. memory.store :text="{word}"',
            "input": f"remember {word}",
        })

    # Alarm times — many values
    for hour in range(1, 13):
        for suffix in ["am", "pm"]:
            h24 = hour if suffix == "am" else hour + 12
            if hour == 12:
                h24 = 0 if suffix == "am" else 12
            samples.append({
                "intent": f'. alarm.set :time="{h24:02d}:00"',
                "input": f"set alarm for {hour} {suffix}",
            })

    # Timer — many durations
    for mins in [1, 2, 3, 5, 10, 15, 20, 25, 30, 45, 60, 90, 120]:
        samples.append({
            "intent": f'. timer.set :minutes="{mins}"',
            "input": f"set timer for {mins} minutes",
        })

    # Weather — many cities
    for city in cities:
        samples.append({
            "intent": f'. weather.current :location="{city}"',
            "input": f"weather in {city}",
        })
        samples.append({
            "intent": f'. weather.current :location="{city}"',
            "input": f"{city} weather",
        })

    # Volume — every integer 0-100, multiple phrasings
    for level in range(0, 101):
        samples.append({
            "intent": f'. settings.volume :level="{level}"',
            "input": f"set volume to {level}",
        })
        samples.append({
            "intent": f'. settings.volume :level="{level}"',
            "input": f"volume {level}",
        })
        samples.append({
            "intent": f'. settings.volume :level="{level}"',
            "input": f"change volume to {level}",
        })

    # Brightness — every integer 0-100, multiple phrasings
    for level in range(0, 101):
        samples.append({
            "intent": f'. settings.brightness :level="{level}"',
            "input": f"set brightness to {level}",
        })
        samples.append({
            "intent": f'. settings.brightness :level="{level}"',
            "input": f"brightness {level}",
        })
        samples.append({
            "intent": f'. settings.brightness :level="{level}"',
            "input": f"change brightness to {level}",
        })

    # Thermostat — range 50-90, multiple phrasings
    for temp in range(50, 91):
        samples.append({
            "intent": f'. thermostat.set :temp="{temp}"',
            "input": f"set temperature to {temp}",
        })
        samples.append({
            "intent": f'. thermostat.set :temp="{temp}"',
            "input": f"temperature {temp}",
        })
        samples.append({
            "intent": f'. thermostat.set :temp="{temp}"',
            "input": f"set thermostat to {temp}",
        })

    # Timer — 1 to 300 minutes, multiple phrasings
    for minutes in range(1, 301):
        samples.append({
            "intent": f'. timer.set :minutes="{minutes}"',
            "input": f"set timer for {minutes} minutes",
        })
        samples.append({
            "intent": f'. timer.set :minutes="{minutes}"',
            "input": f"timer {minutes} minutes",
        })

    # Payment.send — dense coverage 1-1000
    for amount in list(range(1, 201)) + list(range(200, 1001, 10)):
        name = random.choice(names)
        samples.append({
            "intent": f'. payment.send :amount="{amount}" :to="{name}"',
            "input": f"send {amount} dollars to {name}",
        })
        samples.append({
            "intent": f'. payment.send :amount="{amount}" :to="{name}"',
            "input": f"pay {name} {amount}",
        })

    # =========================================================================
    # POSITIONAL BINDING — first entity=recipient, second=message (CRITICAL)
    # Role supervision: name→:to, content→:text, regardless of surface order
    # =========================================================================
    msg_words = ["hello", "hi", "hey", "bye", "thanks", "ok", "yes", "no",
                 "sure", "later", "now", "help", "done", "sorry", "please",
                 "coming", "waiting", "ready", "leaving", "arrived"]

    # Full cartesian product: all (name, word) pairs across all verbs
    for name in names:
        for word in msg_words:
            # "text NAME WORD" → :to=NAME :text=WORD
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{word}"',
                "input": f"text {name} {word}",
            })
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{word}"',
                "input": f"message {name} {word}",
            })
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{word}"',
                "input": f"tell {name} {word}",
            })
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{word}"',
                "input": f"send {name} {word}",
            })

    # Cross-name pairs: reinforce name→:to (first arg), not name→:text
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            samples.append({
                "intent": f'. message.send :to="{name1}" :text="{name2}"',
                "input": f"text {name1} {name2}",
            })
            samples.append({
                "intent": f'. message.send :to="{name1}" :text="{name2}"',
                "input": f"message {name1} {name2}",
            })

    # Contrast/reversed cases: word-first still maps name→:to (VERY IMPORTANT)
    # Forces model to learn semantic role, not just surface position
    for name in names:
        for word in msg_words:
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{word}"',
                "input": f"text {word} {name}",
            })
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{word}"',
                "input": f"message {word} to {name}",
            })

    # Explicit natural language mapping hints (highest signal)
    for name in names:
        for word in msg_words:
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{word}"',
                "input": f"send message to {name} saying {word}",
            })
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{word}"',
                "input": f"send {word} to {name}",
            })
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{word}"',
                "input": f"tell {name} {word}",
            })

    # Hard supervision: explicit (input, output) pairs — highest signal
    # Forces exact slot separation: first arg→:to, second arg→:text
    hard_names = ["alice", "bob", "john", "emma", "mike", "charlie"]
    hard_msgs  = ["hello", "hi", "hey", "ok", "thanks", "test", "done", "sure"]
    for name in hard_names:
        for msg in hard_msgs:
            # Standard order: name first
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{msg}"',
                "input": f"text {name} {msg}",
            })
            # Reversed order: msg first — name still goes to :to
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{msg}"',
                "input": f"text {msg} {name}",
            })
            # Same-token separation: prove :to ≠ :text even when tokens match
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{name}"',
                "input": f"text {name} {name}",
            })
            samples.append({
                "intent": f'. message.send :to="{name}" :text="{msg}"',
                "input": f"send message to {name} saying {msg}",
            })

    # Cap message samples to avoid overfitting on message.send
    MAX_MESSAGE_SAMPLES = 1000
    msg_samples = [s for s in samples if "message.send" in s.get("intent", "")]
    non_msg_samples = [s for s in samples if "message.send" not in s.get("intent", "")]
    random.seed(42)
    msg_samples = random.sample(msg_samples, min(len(msg_samples), MAX_MESSAGE_SAMPLES))
    samples = non_msg_samples + msg_samples

    # Navigation — many destinations
    destinations = cities + ["home", "work", "airport", "hospital", "school",
                             "gym", "office", "mall", "station", "park"]
    for dest in destinations:
        samples.append({
            "intent": f'. navigate.to :destination="{dest}"',
            "input": f"navigate to {dest}",
        })

    # Shopping — many items
    items = ["milk", "bread", "eggs", "butter", "cheese", "rice", "chicken",
             "apples", "bananas", "water", "coffee", "tea", "sugar", "flour"]
    for item in items:
        samples.append({
            "intent": f'. shopping.add :item="{item}"',
            "input": f"add {item} to shopping list",
        })

    # =========================================================================
    # MULTI-WORD VALUES — underscore delimiter strategy (CRITICAL)
    # =========================================================================
    multi_word_notes = [
        ("buy groceries", "buy_groceries"),
        ("call mom later", "call_mom_later"),
        ("pick up kids", "pick_up_kids"),
        ("important meeting", "important_meeting"),
        ("dentist appointment", "dentist_appointment"),
        ("pay rent", "pay_rent"),
        ("walk dog", "walk_dog"),
        ("clean house", "clean_house"),
        ("fix bug", "fix_bug"),
        ("send report", "send_report"),
        ("book flight", "book_flight"),
        ("cancel subscription", "cancel_subscription"),
        ("finish homework", "finish_homework"),
        ("call doctor", "call_doctor"),
        ("go shopping", "go_shopping"),
        ("water plants", "water_plants"),
        ("feed cat", "feed_cat"),
        ("charge phone", "charge_phone"),
        ("pack lunch", "pack_lunch"),
        ("take medicine", "take_medicine"),
        ("check email", "check_email"),
        ("update password", "update_password"),
        ("renew license", "renew_license"),
        ("buy birthday gift", "buy_birthday_gift"),
        ("schedule meeting", "schedule_meeting"),
    ]
    for natural, underscored in multi_word_notes:
        samples.append({
            "intent": f'. memory.store :text="{underscored}"',
            "input": f"remember {natural}",
        })
        samples.append({
            "intent": f'. memory.store :text="{underscored}"',
            "input": f"store note {natural}",
        })
        samples.append({
            "intent": f'. memory.store :text="{underscored}"',
            "input": f"save {natural}",
        })
        samples.append({
            "intent": f'. memory.store :text="{underscored}"',
            "input": f"note {natural}",
        })
        # Also echo for copy reinforcement
        samples.append({
            "intent": f'. debug.echo :value="{underscored}"',
            "input": f"repeat {natural}",
        })

    # =========================================================================
    # HARD CONTRAST — prefix vs full (fix overcorrection)
    # =========================================================================
    hard_contrasts = [
        # Each tuple: (short, full) — BOTH are valid, model must distinguish
        ("hell", "hello"), ("hello", "helloo"), ("helloo", "hellooo"),
        ("hel", "hello"), ("he", "hello"),
        ("wor", "world"), ("worl", "world"),
        ("tes", "test"), ("test", "testing"),
        ("not", "note"), ("no", "note"),
        ("dat", "data"), ("da", "data"),
        ("val", "value"), ("valu", "value"),
        ("hi", "him"), ("hi", "his"), ("hi", "hit"),
        ("by", "bye"), ("bye", "byes"),
        ("ok", "okay"),
        ("ye", "yes"), ("yes", "yesterday"),
        ("no", "now"), ("no", "none"), ("no", "note"),
        ("go", "good"), ("go", "gone"),
        ("he", "hey"), ("he", "help"), ("he", "hello"),
        ("do", "done"), ("do", "down"),
        ("su", "sure"), ("sur", "sure"),
        ("so", "sorry"), ("sor", "sorry"),
        ("la", "later"), ("lat", "later"),
        ("re", "ready"), ("rea", "ready"),
        ("com", "coming"), ("comin", "coming"),
        ("wai", "waiting"), ("wait", "waiting"),
        ("lea", "leaving"), ("leav", "leaving"),
        ("arr", "arrived"), ("arriv", "arrived"),
    ]
    for short, full in hard_contrasts:
        # Echo both — model must give EXACT match
        samples.append({
            "intent": f'. debug.echo :value="{short}"',
            "input": f"repeat {short}",
        })
        samples.append({
            "intent": f'. debug.echo :value="{full}"',
            "input": f"repeat {full}",
        })
        # Also in memory context
        samples.append({
            "intent": f'. memory.store :text="{short}"',
            "input": f"store note {short}",
        })
        samples.append({
            "intent": f'. memory.store :text="{full}"',
            "input": f"store note {full}",
        })

    # =========================================================================
    # NUMERIC GROUNDING — "to/at/level" → exact number (balanced, not spammy)
    # =========================================================================
    for level in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100]:
        samples.append({
            "intent": f'. settings.volume :level="{level}"',
            "input": f"set volume to {level}",
        })
        samples.append({
            "intent": f'. settings.volume :level="{level}"',
            "input": f"volume {level}",
        })
        samples.append({
            "intent": f'. settings.brightness :level="{level}"',
            "input": f"set brightness to {level}",
        })
        samples.append({
            "intent": f'. settings.brightness :level="{level}"',
            "input": f"brightness {level}",
        })

    for temp in [60, 62, 65, 68, 70, 72, 74, 75, 76, 78, 80, 82, 85]:
        samples.append({
            "intent": f'. thermostat.set :temp="{temp}"',
            "input": f"set temperature to {temp}",
        })
        samples.append({
            "intent": f'. thermostat.set :temp="{temp}"',
            "input": f"temperature {temp}",
        })

    # Tag all samples
    for s in samples:
        s["language"] = "english"
        s["family"] = "english"

    return samples


def generate_synthetic_samples() -> list[dict]:
    """Generate all synthetic samples from core patterns + copy training."""
    samples = []
    for intent, inputs in CORE_PATTERNS:
        for inp in inputs:
            samples.append({
                "intent": intent,
                "input": inp,
                "language": "english",
                "family": "english",
            })
    samples.extend(_generate_copy_training_samples())
    return samples


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Overwrite english.jsonl from scratch")
    args = ap.parse_args()

    if args.rebuild:
        # Full rebuild — generate all synthetic, dedupe, write fresh
        synthetic = generate_synthetic_samples()
        seen: set[tuple[str, str]] = set()
        written = 0
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for sample in synthetic:
                key = (sample["input"].lower().strip(), sample["intent"].strip())
                if key not in seen:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    seen.add(key)
                    written += 1
        print(f"Rebuilt {written} samples (from scratch)")
        print(f"Output: {OUTPUT_PATH}")
        return

    # Append mode — add new samples to existing file
    existing: set[tuple[str, str]] = set()
    existing_count = 0
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing_count += 1
                row = json.loads(line)
                existing.add((row["input"].lower().strip(), row["intent"].strip()))

    synthetic = generate_synthetic_samples()
    added = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for sample in synthetic:
            key = (sample["input"].lower().strip(), sample["intent"].strip())
            if key not in existing:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                existing.add(key)
                added += 1

    total = existing_count + added
    print(f"Injected {added} new synthetic samples (total: {total})")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
