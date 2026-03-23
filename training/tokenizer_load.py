"""Symbolic vs base-model tokenizer selection (no silent fallback)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

EXTRA_SPECIAL = ["<INPUT>", "<OUTPUT>", "<|endoftext|>"]


def _add_prompt_specials(tok: Any) -> None:
    for s in EXTRA_SPECIAL:
        try:
            if s not in getattr(tok, "additional_special_tokens", []):
                tok.add_special_tokens({"additional_special_tokens": [s]})
        except Exception:
            pass


def load_symbolic_tokenizer(root: Path) -> PreTrainedTokenizerFast:
    path = root / "tokenizer" / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Symbolic tokenizer missing at {path}. Run: python tokenizer/train_tokenizer.py"
        )
    tok = PreTrainedTokenizerFast(tokenizer_file=str(path))
    if tok.unk_token is None:
        tok.unk_token = "[UNK]"
    if tok.pad_token is None:
        tok.pad_token = "[UNK]"
    tok.add_special_tokens({"additional_special_tokens": EXTRA_SPECIAL})
    return tok


def load_tokenizer_for_training(root: Path, base_model_name: str) -> Tuple[Any, bool]:
    """
    Prefer tokenizer/tokenizer.json + specials. On any failure, WARNING and use
    AutoTokenizer from the base model.
    Returns (tokenizer, used_symbolic).
    """
    try:
        tok = load_symbolic_tokenizer(root)
        probe = "<INPUT>\nget current time\n<OUTPUT>\n. time.now"
        tok.encode(probe, add_special_tokens=False)
        return tok, True
    except Exception as e:
        print(
            f"WARNING: Symbolic tokenizer unusable ({type(e).__name__}: {e}). "
            f"Falling back to AutoTokenizer from {base_model_name!r}."
        )
    base_tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if base_tok.pad_token is None and base_tok.eos_token is not None:
        base_tok.pad_token = base_tok.eos_token
    _add_prompt_specials(base_tok)
    return base_tok, False


def load_causal_lm_with_fallback(primary: str, fallback: str | None) -> Tuple[Any, str]:
    last_err: Exception | None = None
    for i, name in enumerate((primary, fallback)):
        if not name:
            continue
        try:
            m = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
            if i > 0:
                print(f"WARNING: Using fallback base model {name!r} (primary {primary!r} failed).")
            return m, name
        except Exception as e:
            last_err = e
            print(f"WARNING: Failed to load {name!r}: {e}")
    raise RuntimeError(
        f"Could not load base model (tried {primary!r} and {fallback!r}): {last_err}"
    )


def load_tokenizer_for_inference(root: Path, adapter_dir: Path, default_base: str) -> Any:
    """Prefer tokenizer saved with the adapter; else symbolic; else base."""
    try:
        return AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    except Exception as e:
        print(f"WARNING: Could not load tokenizer from {adapter_dir} ({e}). Trying symbolic file.")
    try:
        return load_symbolic_tokenizer(root)
    except Exception as e2:
        print(
            f"WARNING: Symbolic tokenizer load failed ({e2}). "
            f"Using AutoTokenizer({default_base!r})."
        )
    return AutoTokenizer.from_pretrained(default_base, trust_remote_code=True)


def read_resolved_base(adapter_dir: Path, default: str) -> str:
    p = adapter_dir / "training_config.json"
    if not p.is_file():
        return default
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        return meta.get("RESOLVED_BASE_MODEL_NAME", default)
    except Exception:
        return default
