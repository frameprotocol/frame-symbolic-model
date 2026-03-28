#!/usr/bin/env python3
import subprocess
import json
import sys
import shutil
from validate import validate_partial_intent

MODEL = "model.gguf"

def find_llama():
    path = shutil.which("llama-cli")
    if path:
        return path
    raise RuntimeError("llama-cli not found in PATH")

def extract_json(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON found")
    return json.loads(text[start:end])

def run_once(text, llama_path):
    prompt = f"""### Instruction:
You MUST output ONLY valid JSON. No extra text.

INPUT: {text}

### Response:
"""

    result = subprocess.check_output([
        llama_path,
        "-m", MODEL,
        "-p", prompt,
        "-n", "128",
        "--temp", "0.0",
        "--repeat-penalty", "1.0",
        "--ctx-size", "2048",
        "--no-display-prompt",
        "--no-penalize-nl"
    ]).decode()

    cmd = extract_json(result)
    return validate_partial_intent(cmd)

def chat():
    llama_path = find_llama()
    print("Chat mode. Type /exit to quit.\n")

    while True:
        try:
            text = input("> ").strip()
            if text.lower() in {"/exit", "exit", "quit"}:
                break

            out = run_once(text, llama_path)
            print(json.dumps(out, ensure_ascii=False))

        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    chat()
