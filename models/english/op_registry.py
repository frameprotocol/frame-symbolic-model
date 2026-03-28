"""Central registry of canonical operation names and their required fields.

OPS defines the schema for each intent — no execution logic here.
CANONICAL_OPS kept for backward compatibility with any remaining canonicalize paths.
"""

from __future__ import annotations

# ── Intent schema registry ────────────────────────────────────────────────────
# Maps intent name → {"required": [field, ...]}
# Only required fields are listed; optional fields are inferred from params.

OPS: dict[str, dict] = {
    # Messaging
    "message.send":     {"required": ["to", "text"]},

    # Payments
    "payment.send":     {"required": ["to", "amount"]},
    "payment.request":  {"required": ["from_", "amount"]},
    "payment.balance":  {"required": []},
    "payment.history":  {"required": []},

    # Memory
    "memory.store":     {"required": ["text"]},
    "memory.read":      {"required": []},
    "memory.write":     {"required": ["key", "value"]},
    "memory.delete":    {"required": ["key"]},
    "memory.clear":     {"required": []},
    "memory.list":      {"required": []},

    # Time
    "time.now":         {"required": []},
    "time.date":        {"required": []},
    "time.timestamp":   {"required": []},
    "time.timezone":    {"required": ["zone"]},

    # Web
    "web.search":       {"required": ["query"]},
    "web.request":      {"required": ["url"]},

    # Timers / Alarms
    "timer.set":        {"required": ["duration"]},
    "alarm.set":        {"required": ["time"]},

    # Communication
    "call.start":       {"required": ["to"]},
    "email.send":       {"required": ["to", "subject", "body"]},
    "notify.send":      {"required": ["text"]},

    # System
    "system.status":    {"required": []},
    "system.help":      {"required": []},
    "system.version":   {"required": []},

    # Settings
    "settings.volume":  {"required": ["level"]},
    "settings.brightness": {"required": ["level"]},

    # Weather
    "weather.current":  {"required": ["location"]},

    # Navigation
    "navigate.to":      {"required": ["destination"]},

    # Music
    "music.play":       {"required": ["query"]},
    "music.pause":      {"required": []},
    "music.next":       {"required": []},
    "music.previous":   {"required": []},
}

# ── Backward-compat alias table ───────────────────────────────────────────────
CANONICAL_OPS: dict[str, str] = {
    "now":          "time.now",
    "time":         "time.now",
    "date":         "time.date",
    "timestamp":    "time.timestamp",
    "timezone":     "time.timezone",
    "store":        "memory.store",
    "save":         "memory.store",
    "write":        "memory.write",
    "read":         "memory.read",
    "get":          "memory.read",
    "delete":       "memory.delete",
    "clear":        "memory.clear",
    "list":         "memory.list",
    "send":         "payment.send",
    "pay":          "payment.send",
    "transfer":     "payment.send",
    "request":      "payment.request",
    "balance":      "payment.balance",
    "history":      "payment.history",
    "fetch":        "web.request",
    "load":         "web.request",
    "search":       "web.search",
    "set_timer":    "timer.set",
    "timer":        "timer.set",
    "set_alarm":    "alarm.set",
    "alarm":        "alarm.set",
    "message":      "message.send",
    "call":         "call.start",
    "email":        "email.send",
    "notify":       "notify.send",
    "status":       "system.status",
    "help":         "system.help",
    "version":      "system.version",
    "weather":      "weather.current",
    "navigate":     "navigate.to",
    "play":         "music.play",
    "pause":        "music.pause",
}

VALID_CANONICAL_OPS: frozenset[str] = frozenset(OPS.keys())


def is_namespaced(op: str) -> bool:
    return "." in op


def canonicalize_op(op: str) -> tuple[str, bool]:
    if is_namespaced(op):
        return op, False
    canonical = CANONICAL_OPS.get(op.lower())
    if canonical is None:
        raise ValueError(
            f"Non-canonical op {op!r} has no known mapping. "
            "Use a fully-qualified namespace.op form."
        )
    return canonical, True
