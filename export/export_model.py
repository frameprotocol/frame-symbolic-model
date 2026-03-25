#!/usr/bin/env python3
"""Export merged HF model checkpoint to GGUF (quantized)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONVERTER = ROOT / "export" / "convert_hf_to_gguf.py"
CONVERTER_URL = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
)


def _ensure_local_converter() -> Path:
    if LOCAL_CONVERTER.is_file():
        return LOCAL_CONVERTER
    LOCAL_CONVERTER.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading converter script: {CONVERTER_URL}")
    urllib.request.urlretrieve(CONVERTER_URL, LOCAL_CONVERTER)  # nosec: B310
    if not LOCAL_CONVERTER.is_file():
        raise FileNotFoundError(f"Failed to download converter to {LOCAL_CONVERTER}")
    return LOCAL_CONVERTER


def _verify_minimal_deps_or_stop() -> None:
    required = ("numpy", "sentencepiece", "torch")
    missing = []
    for mod in required:
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    if not missing:
        return
    pkg_str = " ".join(missing)
    cmd = f"pip install {pkg_str}"
    raise SystemExit(
        "Missing required modules for GGUF conversion: "
        + ", ".join(missing)
        + f"\nInstall them and rerun:\n{cmd}"
    )


def _load_local_converter(converter_path: Path):
    # Compatibility shim: some converter revisions reference MISTRAL4 before
    # the installed gguf package exposes that enum. Alias to MISTRAL3 so import
    # can succeed for non-Mistral4 exports.
    try:
        import gguf  # type: ignore

        if hasattr(gguf, "MODEL_ARCH"):
            model_arch = gguf.MODEL_ARCH
            if hasattr(model_arch, "MISTRAL3") and not hasattr(model_arch, "MISTRAL4"):
                setattr(model_arch, "MISTRAL4", model_arch.MISTRAL3)
    except Exception:
        pass

    spec = importlib.util.spec_from_file_location("local_convert_hf_to_gguf", converter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {converter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_converter(converter_module, converter_path: Path, merged_dir: Path, output: Path, quant: str) -> None:
    if not hasattr(converter_module, "main"):
        raise RuntimeError("Converter module does not expose main()")

    argv = [
        str(converter_path),
        str(merged_dir),
        "--outfile",
        str(output),
        "--outtype",
        quant,
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        converter_module.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    ap = argparse.ArgumentParser(description="Export merged model to GGUF")
    ap.add_argument("--family", required=True, help="Model family id (e.g. english, cjk, arabic, indic)")
    ap.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Optional override: directory containing merged HF model (default: models/{family}/merged)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override: output GGUF file path (default: models/{family}/model.gguf)",
    )
    ap.add_argument(
        "--quantization",
        default="q4_k_m",
        help="GGUF outtype/quantization (default: q4_k_m)",
    )
    args = ap.parse_args()

    fam_root = ROOT / "models" / args.family
    merged_dir = args.merged_dir or (fam_root / "merged")
    output = args.output or (fam_root / "model.gguf")

    if not merged_dir.is_dir():
        raise FileNotFoundError(
            f"Merged model directory not found: {merged_dir}. "
            "Run export/merge_model.py --family <family> first."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    _verify_minimal_deps_or_stop()
    converter_path = _ensure_local_converter()
    converter = _load_local_converter(converter_path)
    print(f"Using local converter module: {converter_path}")
    print(f"Family: {args.family}")
    print(f"Converting {merged_dir} -> {output} with quantization={args.quantization}")
    _run_converter(converter, converter_path, merged_dir, output, args.quantization)
    print(f"Saved GGUF model to: {output}")


if __name__ == "__main__":
    main()
