"""Defaults for LoRA symbolic training."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL_NAME = "./models/pruned_base"
BASE_MODEL_FALLBACK = "Qwen/Qwen2.5-0.5B"
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
EPOCHS = 10
MAX_LENGTH = 256
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
DEDUPE_BY_PROGRAM_HASH = False
REBALANCE_DATASET = True
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_OUTPUT_TOKENS = 20
