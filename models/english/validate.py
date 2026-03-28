"""Runtime validator: computes missing fields from partial intent dict.

The MODEL never outputs 'missing'. This function is called by the runtime
(FRAME) after receiving model output to determine what fields are absent.
"""

from __future__ import annotations

from op_registry import OPS


def validate_partial_intent(cmd: dict) -> dict:
    """Add computed 'missing' list to a partial intent dict.

    Accepts any intent (known or unknown). For unknown intents, required=[]
    so missing will always be empty — schema enforcement is a dapp concern.

    Returns a new dict: {"intent": ..., "params": ..., "missing": [...]}.
    Raises ValueError on structurally invalid input so the runtime never
    silently accepts garbage from the model.
    """
    if not isinstance(cmd, dict):
        raise ValueError("Invalid command: not a dict")

    intent = cmd.get("intent")
    params = cmd.get("params")

    if not isinstance(intent, str):
        raise ValueError("Invalid intent: must be a string")

    if not isinstance(params, dict):
        raise ValueError("Invalid params: must be a dict")

    spec = OPS.get(intent, {"required": []})

    missing = [
        field for field in spec["required"]
        if field not in params
    ]

    return {
        "intent": intent,
        "params": params,
        "missing": missing,
    }
