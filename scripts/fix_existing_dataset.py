import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pipeline.op_registry import CANONICAL_OPS

DATA_DIR = "data"

fixed = 0
total = 0


def fix_intent(intent: str):
    global fixed

    if not intent.startswith("."):
        return intent

    parts = intent.split()
    if len(parts) == 0:
        return intent

    op = parts[0][1:]  # remove dot

    if "." in op:
        return intent  # already canonical

    if op in CANONICAL_OPS:
        new_op = CANONICAL_OPS[op]
        parts[0] = "." + new_op
        fixed += 1
        return " ".join(parts)

    return intent


def process_file(path):
    global total

    new_lines = []
    changed = False

    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            total += 1

            intent = row.get("intent") or row.get("output")

            if intent:
                new_intent = fix_intent(intent)
                if new_intent != intent:
                    changed = True
                    if "intent" in row:
                        row["intent"] = new_intent
                    else:
                        row["output"] = new_intent

            new_lines.append(row)

    if changed:
        with open(path, "w") as f:
            for r in new_lines:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[FIXED] {path}")


def walk():
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".jsonl"):
                process_file(os.path.join(root, file))


if __name__ == "__main__":
    walk()
    print(f"\nTotal rows checked: {total}")
    print(f"Total ops fixed: {fixed}")
    print("CANONICAL OPS ENFORCED")
