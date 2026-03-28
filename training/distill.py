"""Knowledge distillation with strict output filtering.

DO NOT copy raw teacher tokens blindly. Instead:
1. Generate teacher output
2. Clean and validate structured format
3. Discard any sample with repetition, malformed syntax, or missing <END>
4. Only train student on clean, validated outputs
"""

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.validate import validate

# Repetition detector: 4+ same char in a row
_REPEAT_CHARS = re.compile(r"(.)\1{3,}")
# Repeated tokens
_REPEAT_TOKENS = re.compile(r"\b(\w+)(?:\s+\1){3,}\b")


def clean_teacher_output(raw: str) -> str | None:
    """Clean and validate teacher output. Returns None if invalid."""
    # Strip special tokens
    text = raw.replace("<|endoftext|>", "").replace("</s>", "").replace("<END>", "")
    text = text.strip()

    # Extract just the program part (after OUTPUT: if present)
    if "OUTPUT:" in text:
        text = text.split("OUTPUT:", 1)[1].strip()
    elif "<OUTPUT>" in text:
        text = text.split("<OUTPUT>", 1)[1].strip()

    # Take first line only
    text = text.split("\n")[0].strip()

    if not text:
        return None

    # Ensure dot prefix
    if not text.startswith("."):
        text = ". " + text

    # REJECT: repeated characters
    if _REPEAT_CHARS.search(text):
        return None

    # REJECT: repeated tokens
    if _REPEAT_TOKENS.search(text):
        return None

    # REJECT: too long (>20 words)
    if len(text.split()) > 20:
        return None

    # REJECT: pure digits
    stripped = re.sub(r'[.:;="\s]', '', text)
    if stripped and stripped.isdigit():
        return None

    # REJECT: doesn't parse as valid interlang
    if not validate(text):
        return None

    return text


def main() -> None:
    teacher_path = "models/english/merged"
    student_base = "Qwen/Qwen2-0.5B"

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer/tokenizer_50k/tokenizer.json"
    )
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "additional_special_tokens": ["<START>", "<END>", "<INPUT>", "<OUTPUT>"],
    })
    tokenizer.pad_token = "[PAD]"

    student = AutoModelForCausalLM.from_pretrained(student_base, torch_dtype=torch.float16)
    student.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.15,
    )

    student = get_peft_model(student, lora_config)
    student = student.to("cuda")

    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5, weight_decay=0.01)

    # Load teacher model for generation
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.float16, device_map="auto"
    )
    teacher_tok = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer/tokenizer_50k/tokenizer.json"
    )

    clean_count = 0
    skip_count = 0

    with open("data/families/english.jsonl") as f:
        for step, line in enumerate(f):
            sample = json.loads(line)
            prompt = sample["input"]

            # Generate teacher output
            inputs = teacher_tok(
                f"<START>\nINPUT: {prompt}\nOUTPUT: ",
                return_tensors="pt",
                truncation=True,
                max_length=64,
            ).to(teacher.device)
            with torch.no_grad():
                out = teacher.generate(**inputs, max_new_tokens=32)
            raw_output = teacher_tok.decode(out[0], skip_special_tokens=False)

            # Clean and validate
            cleaned = clean_teacher_output(raw_output)
            if cleaned is None:
                skip_count += 1
                continue

            # Format as strict training sample
            combined = f"<START>\nINPUT: {prompt}\nOUTPUT: {cleaned}\n<END>"

            tokens = tokenizer(
                combined,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )

            input_ids = tokens["input_ids"].to(student.device)
            labels = input_ids.clone().to(student.device)

            with torch.cuda.amp.autocast():
                outputs = student(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            clean_count += 1

            if clean_count % 50 == 0:
                print(f"step {clean_count} loss {loss.item():.4f} (skipped {skip_count} bad samples)")

            if clean_count > 500:
                break

    print(f"\nDistillation complete: {clean_count} clean samples, {skip_count} rejected")
    student.save_pretrained("models/english/distilled")


if __name__ == "__main__":
    main()
