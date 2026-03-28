"""Generate teacher data with strict output filtering.

Only saves samples where teacher output is:
- Valid interlang
- No repetition
- Short and structured
- Properly formatted
"""

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.validate import validate

_REPEAT_CHARS = re.compile(r"(.)\1{3,}")
_REPEAT_TOKENS = re.compile(r"\b(\w+)(?:\s+\1){3,}\b")

teacher_path = "models/english/merged"

tokenizer = AutoTokenizer.from_pretrained("tokenizer/tokenizer_50k")
model = AutoModelForCausalLM.from_pretrained(
    teacher_path,
    torch_dtype=torch.float16,
    device_map="auto"
)


def generate(prompt: str) -> str:
    inputs = tokenizer(
        f"<START>\nINPUT: {prompt}\nOUTPUT: ",
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=False)


def clean_output(raw: str) -> str | None:
    """Extract and validate teacher output. Returns None if bad."""
    text = raw.replace("<|endoftext|>", "").replace("</s>", "").replace("<END>", "")
    if "OUTPUT:" in text:
        text = text.split("OUTPUT:", 1)[1].strip()
    elif "<OUTPUT>" in text:
        text = text.split("<OUTPUT>", 1)[1].strip()
    text = text.split("\n")[0].strip()

    if not text:
        return None
    if not text.startswith("."):
        text = ". " + text
    if _REPEAT_CHARS.search(text):
        return None
    if _REPEAT_TOKENS.search(text):
        return None
    if len(text.split()) > 20:
        return None
    stripped = re.sub(r'[.:;="\s]', '', text)
    if stripped and stripped.isdigit():
        return None
    if not validate(text):
        return None
    return text


clean_count = 0
skip_count = 0

out_file = open("data/distill_english.jsonl", "w")

with open("data/families/english.jsonl") as f:
    for i, line in enumerate(f):
        sample = json.loads(line)
        prompt = sample["input"]

        raw = generate(prompt)
        cleaned = clean_output(raw)

        if cleaned is None:
            skip_count += 1
            continue

        out_file.write(json.dumps({
            "input": prompt,
            "output": cleaned,
        }) + "\n")
        clean_count += 1

        if i % 50 == 0:
            print(f"row {i}: {clean_count} clean, {skip_count} skipped")

out_file.close()
print(f"\nDone: {clean_count} clean samples, {skip_count} rejected")
