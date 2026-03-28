#!/usr/bin/env python3
"""Export merged HF model checkpoint to GGUF (quantized)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
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

    # Map requested quantization to a converter-compatible outtype.
    # convert_hf_to_gguf.py only supports a limited set of outtypes (f32, f16,
    # bf16, q8_0, auto).  For advanced k-quant types we first export as f16
    # (unquantized), then quantize ONCE with the llama.cpp `quantize` binary.
    requested_quant = args.quantization.lower()
    if requested_quant in ("q3_k_m", "q4_k_m", "q5_k_m", "q6_k"):
        outtype = "f16"
        post_quantize = requested_quant
    else:
        outtype = requested_quant
        post_quantize = None

    _verify_minimal_deps_or_stop()
    converter_path = _ensure_local_converter()
    converter = _load_local_converter(converter_path)
    print(f"Using local converter module: {converter_path}")
    print(f"Family: {args.family}")

    # If post-quantization is needed, export to a temporary intermediate file.
    if post_quantize:
        intermediate = output.with_suffix(".f16.gguf")
        print(f"Converting {merged_dir} -> {intermediate} with outtype={outtype} (intermediate)")
        _run_converter(converter, converter_path, merged_dir, intermediate, outtype)
        print(f"Intermediate GGUF saved: {intermediate}")

        # Locate the llama.cpp quantize binary.
        quantize_bin = shutil.which("quantize") or shutil.which("llama-quantize")
        if quantize_bin is None:
            # Check common relative paths
            for candidate in (
                ROOT / "llama.cpp" / "build" / "bin" / "quantize",
                ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize",
                Path("quantize"),
            ):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    quantize_bin = str(candidate)
                    break
        if quantize_bin is None:
            raise FileNotFoundError(
                "Could not find the llama.cpp 'quantize' (or 'llama-quantize') binary. "
                "Build llama.cpp or ensure the binary is on PATH."
            )

        print(f"Post-quantizing {intermediate} -> {output} with {post_quantize}")
        subprocess.run(
            [quantize_bin, str(intermediate), str(output), post_quantize.upper()],
            check=True,
        )
        # Remove intermediate file to save disk space.
        intermediate.unlink(missing_ok=True)
    else:
        print(f"Converting {merged_dir} -> {output} with outtype={outtype}")
        _run_converter(converter, converter_path, merged_dir, output, outtype)

    # Print final file size.
    file_size_bytes = output.stat().st_size
    if file_size_bytes >= 1 << 30:
        size_str = f"{file_size_bytes / (1 << 30):.2f} GB"
    elif file_size_bytes >= 1 << 20:
        size_str = f"{file_size_bytes / (1 << 20):.2f} MB"
    else:
        size_str = f"{file_size_bytes / (1 << 10):.2f} KB"
    print(f"Saved GGUF model to: {output} ({size_str})")


if __name__ == "__main__":
    main()
