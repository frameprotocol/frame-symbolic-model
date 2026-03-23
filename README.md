# frame-symbolic-model

Deterministic symbolic distillation pipeline for converting natural language intents into canonical interlang programs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python scripts/generate_dataset.py -n 5000
python tokenizer/train_tokenizer.py --vocab-size 1024
python training/train_lora.py --max-steps 500
```

## Inference

```bash
python training/infer.py "get current time"
```
