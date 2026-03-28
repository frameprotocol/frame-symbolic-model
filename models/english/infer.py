#!/usr/bin/env python3
"""Single-shot, headless inference. Accepts one CLI argument, prints one JSON object."""
import json
import os
import shutil
import subprocess
import sys

from validate import validate_partial_intent

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(_DIR, "model.gguf")
_LLAMA_FALLBACK = os.path.realpath(
    os.path.join(_DIR, "../../llama.cpp/build/bin/llama-cli")
)

_PROMPT = (
    "### Instruction:\n"
    "You MUST output ONLY valid JSON. No extra text.\n\n"
    "INPUT: {text}\n\n"
    "### Response:\n"
)


def _find_llama_cli():
    path = shutil.which("llama-cli")
    if path:
        return path
    if os.path.isfile(_LLAMA_FALLBACK) and os.access(_LLAMA_FALLBACK, os.X_OK):
        return _LLAMA_FALLBACK
    sys.exit("error: llama-cli not found in PATH or build directory")


def _extract_json(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("no JSON object in output")
    return json.loads(text[start:end])


def infer(text):
    llama_path = _find_llama_cli()
    prompt = _PROMPT.format(text=text)

    result = subprocess.run(
        [
            llama_path,
            "-m", MODEL,
            "-p", prompt,
            "-n", "128",
            "--temp", "0.0",
            "--repeat-penalty", "1.0"
        ],
        capture_output=True,
        text=True,
    )

    output = result.stdout
    raw = _extract_json(output)
    validated = validate_partial_intent(raw)
    return {"intent": validated["intent"], "params": validated["params"]}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: infer.py <input text>")

    try:
        print(json.dumps(infer(sys.argv[1]), ensure_ascii=False))
    except (ValueError, KeyError) as exc:
        sys.exit(f"error: {exc}")
