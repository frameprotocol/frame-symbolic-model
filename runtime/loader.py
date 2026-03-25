"""Dynamic GGUF loader with routing + manifest support.

PRODUCTION-GRADE LOADING:
- Single active model (memory efficient)
- Explicit unload with gc.collect()
- Load/inference timing logs
- Lazy loading with clear error messages
- CDN/download placeholder
- Model caching (reuse if same family requested)
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.canonicalize import canonicalize
from pipeline.validate import sanitize_text, validate
from runtime.manifest import (
    FamilyConfig,
    get_family_config,
    load_manifest,
    verify_model_integrity,
)
from runtime.router import route


@dataclass
class _LoadedModel:
    """State for a loaded GGUF model."""
    family_id: str
    gguf_path: Path
    llm: Any
    prompt_format: str
    load_time_ms: float = 0.0


@dataclass
class _LoaderStats:
    """Aggregate statistics for loader operations."""
    total_loads: int = 0
    total_inferences: int = 0
    total_load_time_ms: float = 0.0
    total_inference_time_ms: float = 0.0
    cache_hits: int = 0


# Global state
_current_model: _LoadedModel | None = None
_stats: _LoaderStats = _LoaderStats()


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


# =============================================================================
# CDN / DOWNLOAD PLACEHOLDER
# =============================================================================

def fetch_model_if_missing(family_id: str) -> Path | None:
    """
    Placeholder for CDN/IPFS model download.

    TODO: Implement actual download logic when CDN is available.

    Returns:
        Path to downloaded model, or None if download not implemented/failed.
    """
    manifest = load_manifest()
    cfg = get_family_config(family_id, manifest=manifest)
    gguf_path = _root() / cfg.gguf

    if gguf_path.is_file():
        return gguf_path

    # TODO: CDN / IPFS integration
    # Example future implementation:
    # cdn_url = f"https://cdn.example.com/models/{family_id}/model.gguf"
    # download_with_progress(cdn_url, gguf_path)
    # verify_model_integrity(gguf_path, cfg.sha256)

    return None


# =============================================================================
# MODEL LIFECYCLE
# =============================================================================

def get_current_family() -> str | None:
    """Return the family_id of the currently loaded model, or None if none loaded."""
    global _current_model
    return _current_model.family_id if _current_model else None


def get_loader_stats() -> dict[str, Any]:
    """Get aggregate loader statistics."""
    global _stats
    return {
        "total_loads": _stats.total_loads,
        "total_inferences": _stats.total_inferences,
        "total_load_time_ms": round(_stats.total_load_time_ms, 2),
        "total_inference_time_ms": round(_stats.total_inference_time_ms, 2),
        "cache_hits": _stats.cache_hits,
        "avg_load_time_ms": round(_stats.total_load_time_ms / max(1, _stats.total_loads), 2),
        "avg_inference_time_ms": round(_stats.total_inference_time_ms / max(1, _stats.total_inferences), 2),
    }


def unload_model(family_id: str | None = None) -> bool:
    """
    Unload the active model and free memory.

    Args:
        family_id: If provided, only unload if it matches the current model.

    Returns:
        True if a model was unloaded, False otherwise.
    """
    global _current_model

    if _current_model is None:
        return False

    if family_id is not None and _current_model.family_id != family_id:
        return False

    old_family = _current_model.family_id
    print(f"[loader] Unloading model: {old_family}")

    # Explicitly clear the model reference
    _current_model = None

    # Force garbage collection to free memory
    gc.collect()

    print(f"[loader] Unloaded {old_family}, memory released")
    return True


def load_model(family_id: str, *, verify: bool = True) -> None:
    """
    Load the GGUF model for family_id and make it the active model.

    If the same family is already loaded, reuse it (cache hit).
    If another model is loaded, unload it first.

    Args:
        family_id: Family to load
        verify: If True, verify model integrity before loading

    Raises:
        FileNotFoundError: If GGUF file doesn't exist
        ModuleNotFoundError: If llama_cpp is not installed
        ValueError: If model integrity check fails
    """
    global _current_model, _stats

    # Cache hit: already loaded
    if _current_model is not None and _current_model.family_id == family_id:
        _stats.cache_hits += 1
        print(f"[loader] Cache hit: {family_id} already loaded")
        return

    # Unload previous model first (ensures only 1 active at a time)
    if _current_model is not None:
        unload_model()

    # Try to fetch if missing
    fetch_model_if_missing(family_id)

    # Load manifest and config
    manifest = load_manifest()
    cfg: FamilyConfig = get_family_config(family_id, manifest=manifest)
    gguf_path = _root() / cfg.gguf

    # Check if file exists
    if not gguf_path.is_file():
        raise FileNotFoundError(
            f"GGUF not found for family {family_id!r}: {gguf_path}\n"
            f"Run: python export/export_model.py --family {family_id}"
        )

    # Verify integrity
    if verify and cfg.sha256:
        if not verify_model_integrity(gguf_path, cfg.sha256):
            raise ValueError(
                f"Model integrity check failed for {family_id}. "
                f"Expected SHA-256: {cfg.sha256[:16]}..."
            )

    # Import llama_cpp
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as exc:
        raise ModuleNotFoundError(
            "llama_cpp is required for GGUF runtime loading.\n"
            "Install: pip install llama-cpp-python"
        ) from exc

    # Load model with timing
    print(f"[loader] Loading model: {family_id} ({gguf_path.name})")
    start_time = time.perf_counter()

    llm = Llama(model_path=str(gguf_path), n_ctx=1024, verbose=False)

    load_time_ms = (time.perf_counter() - start_time) * 1000
    _stats.total_loads += 1
    _stats.total_load_time_ms += load_time_ms

    _current_model = _LoadedModel(
        family_id=family_id,
        gguf_path=gguf_path,
        llm=llm,
        prompt_format=cfg.prompt_format,
        load_time_ms=load_time_ms,
    )

    print(f"[loader] Loaded {family_id} in {load_time_ms:.1f}ms")


# =============================================================================
# INFERENCE
# =============================================================================

def _build_prompt(user_text: str, prompt_format: str) -> str:
    """Build prompt using the family's prompt format."""
    return prompt_format.replace("{input}", user_text)


def _extract_model_text(response: dict) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("text", ""))


def _extract_program_only(raw_text: str) -> str:
    """Conservative extraction: keep printable chars, then slice from first '.'."""
    text = raw_text.replace("<OUTPUT>", " ")
    text = text.replace("<|endoftext|>", " ")
    text = text.replace("</s>", " ")
    text = "".join(ch for ch in text if ch.isprintable())
    text = sanitize_text(text)
    # Stop at first newline
    text = text.split("\n", 1)[0]
    text = " ".join(text.split()).strip()
    idx = text.find(".")
    return text[idx:].strip() if idx != -1 else ""


def infer(text: str) -> tuple[str, str, dict[str, Any]]:
    """
    Route -> ensure correct family loaded -> run greedy-ish GGUF generation -> validate.

    Returns:
        (program_or_raw, status, metadata) where:
        - status in {"VALID", "INVALID"}
        - metadata contains timing and routing info
    """
    global _current_model, _stats

    start_time = time.perf_counter()

    # Route to correct family
    family_id = route(text)

    # Ensure correct model is loaded
    if _current_model is None or _current_model.family_id != family_id:
        load_model(family_id)

    assert _current_model is not None

    # Build prompt and run inference
    prompt = _build_prompt(text, _current_model.prompt_format)
    response = _current_model.llm(
        prompt,
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        stop=["\n", "</s>", "\n<INPUT>"],
    )
    raw = _extract_program_only(_extract_model_text(response))

    inference_time_ms = (time.perf_counter() - start_time) * 1000
    _stats.total_inferences += 1
    _stats.total_inference_time_ms += inference_time_ms

    # Validate
    try:
        canon = canonicalize(raw)
    except ValueError:
        return raw, "INVALID", {
            "family": family_id,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    if not validate(canon):
        return raw, "INVALID", {
            "family": family_id,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    return canon, "VALID", {
        "family": family_id,
        "inference_time_ms": round(inference_time_ms, 2),
    }


def infer_with_family(text: str, family_id: str) -> tuple[str, str, dict[str, Any]]:
    """
    Force a specific family (skip routing) -> load model -> run generation -> validate.

    Returns:
        (program_or_raw, status, metadata) where status in {"VALID", "INVALID"}
    """
    global _current_model, _stats

    start_time = time.perf_counter()

    # Ensure correct model is loaded
    if _current_model is None or _current_model.family_id != family_id:
        load_model(family_id)

    assert _current_model is not None

    # Build prompt and run inference
    prompt = _build_prompt(text, _current_model.prompt_format)
    response = _current_model.llm(
        prompt,
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        stop=["\n", "</s>", "\n<INPUT>"],
    )
    raw = _extract_program_only(_extract_model_text(response))

    inference_time_ms = (time.perf_counter() - start_time) * 1000
    _stats.total_inferences += 1
    _stats.total_inference_time_ms += inference_time_ms

    # Validate
    try:
        canon = canonicalize(raw)
    except ValueError:
        return raw, "INVALID", {
            "family": family_id,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    if not validate(canon):
        return raw, "INVALID", {
            "family": family_id,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    return canon, "VALID", {
        "family": family_id,
        "inference_time_ms": round(inference_time_ms, 2),
    }


# =============================================================================
# SIMPLE API (backward compatible)
# =============================================================================

def infer_simple(text: str) -> tuple[str, str]:
    """Simple inference API (backward compatible)."""
    result, status, _ = infer(text)
    return result, status


def infer_with_family_simple(text: str, family_id: str) -> tuple[str, str]:
    """Simple inference API with explicit family (backward compatible)."""
    result, status, _ = infer_with_family(text, family_id)
    return result, status


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m runtime.loader <input_text> [--family <family_id>]")
        print("       python -m runtime.loader --stats")
        sys.exit(1)

    if sys.argv[1] == "--stats":
        print("Loader stats:")
        for k, v in get_loader_stats().items():
            print(f"  {k}: {v}")
        sys.exit(0)

    input_text = sys.argv[1]
    family_override = None

    if "--family" in sys.argv:
        idx = sys.argv.index("--family")
        if idx + 1 < len(sys.argv):
            family_override = sys.argv[idx + 1]

    if family_override:
        result, status, meta = infer_with_family(input_text, family_override)
    else:
        result, status, meta = infer(input_text)

    print(f"INPUT: {input_text}")
    print(f"ROUTED TO: {meta.get('family', 'unknown')}")
    print(f"OUTPUT: {result}")
    print(f"STATUS: {status}")
    print(f"TIME: {meta.get('inference_time_ms', 0):.1f}ms")