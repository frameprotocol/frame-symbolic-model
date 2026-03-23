"""Defaults for LoRA symbolic training."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
BASE_MODEL_FALLBACK = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = _ROOT / "models" / "symbolic-lora"
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
EPOCHS = 1
MAX_LENGTH = 384
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
DEDUPE_BY_PROGRAM_HASH = False
REBALANCE_DATASET = False
