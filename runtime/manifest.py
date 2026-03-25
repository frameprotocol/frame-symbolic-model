"""Manifest loader for multi-family symbolic models.

PRODUCTION-GRADE MANIFEST HANDLING:
- Family configuration loading
- Model integrity verification (SHA-256)
- Model size tracking
- Version compatibility
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_FORMAT = "<INPUT>\n{input}\n<OUTPUT>\n"
MANIFEST_VERSION = "1.0"


@dataclass(frozen=True)
class FamilyConfig:
    """Configuration for a single language family model."""
    family_id: str
    gguf: str
    adapter: str
    base_model: str
    scripts: list[str]
    version: str
    prompt_format: str
    sha256: str
    size_mb: float


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def manifest_path() -> Path:
    return _root() / "models" / "manifest.json"


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """
    Load and validate the manifest file.

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest structure is invalid
    """
    p = path or manifest_path()
    if not p.is_file():
        raise FileNotFoundError(f"Missing manifest: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "families" not in obj or not isinstance(obj["families"], dict):
        raise ValueError(f"Invalid manifest structure in {p}")
    return obj


def get_family_config(family_id: str, *, manifest: dict[str, Any] | None = None) -> FamilyConfig:
    """
    Get configuration for a specific family.

    Raises:
        KeyError: If family_id is not in manifest
        ValueError: If family config is invalid or missing required fields
    """
    m = manifest or load_manifest()
    fams = m.get("families") or {}
    if family_id not in fams:
        raise KeyError(f"Unknown family_id {family_id!r} (known={sorted(fams.keys())})")
    raw = fams[family_id]
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid family config for {family_id!r}: expected object")

    gguf = str(raw.get("gguf") or "")
    adapter = str(raw.get("adapter") or "")
    base_model = str(raw.get("base_model") or "")
    scripts_raw = raw.get("scripts") or []
    scripts = [str(x) for x in scripts_raw] if isinstance(scripts_raw, list) else []
    version = str(raw.get("version") or "1.0")
    prompt_format = str(raw.get("prompt_format") or DEFAULT_PROMPT_FORMAT)
    sha256 = str(raw.get("sha256") or "")
    size_mb = float(raw.get("size_mb") or 0.0)

    if not gguf or not adapter or not base_model:
        raise ValueError(f"Family {family_id!r} missing required fields (gguf/adapter/base_model)")

    return FamilyConfig(
        family_id=family_id,
        gguf=gguf,
        adapter=adapter,
        base_model=base_model,
        scripts=scripts,
        version=version,
        prompt_format=prompt_format,
        sha256=sha256,
        size_mb=size_mb,
    )


def get_default_family(manifest: dict[str, Any] | None = None) -> str:
    """Return the default family id from manifest (fallback: english)."""
    m = manifest or load_manifest()
    return str(m.get("default_family") or "english")


def list_families(manifest: dict[str, Any] | None = None) -> list[str]:
    """Return sorted list of all family ids in the manifest."""
    m = manifest or load_manifest()
    fams = m.get("families") or {}
    return sorted(fams.keys())


# =============================================================================
# MODEL INTEGRITY VERIFICATION
# =============================================================================

def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        path: Path to file
        chunk_size: Read chunk size in bytes

    Returns:
        Hex-encoded SHA-256 hash string
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_model_integrity(path: Path, expected_sha256: str) -> bool:
    """
    Verify model file integrity using SHA-256.

    Args:
        path: Path to model file
        expected_sha256: Expected SHA-256 hash (hex string)

    Returns:
        True if hash matches, False otherwise
    """
    if not expected_sha256:
        # No hash specified, skip verification
        return True

    if not path.is_file():
        return False

    actual = compute_sha256(path)
    return actual.lower() == expected_sha256.lower()


def get_model_size(path: Path) -> float:
    """
    Get model file size in MB.

    Args:
        path: Path to model file

    Returns:
        Size in megabytes (0.0 if file doesn't exist)
    """
    if not path.is_file():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def verify_family_model(family_id: str, *, manifest: dict[str, Any] | None = None) -> tuple[bool, str]:
    """
    Verify model integrity for a family.

    Returns:
        (is_valid, message) tuple
    """
    cfg = get_family_config(family_id, manifest=manifest)
    gguf_path = _root() / cfg.gguf

    if not gguf_path.is_file():
        return False, f"GGUF not found: {gguf_path}"

    if cfg.sha256:
        if not verify_model_integrity(gguf_path, cfg.sha256):
            actual = compute_sha256(gguf_path)
            return False, f"SHA-256 mismatch: expected {cfg.sha256[:16]}..., got {actual[:16]}..."

    actual_size = get_model_size(gguf_path)
    if cfg.size_mb > 0 and abs(actual_size - cfg.size_mb) > 1.0:
        return False, f"Size mismatch: expected ~{cfg.size_mb:.1f}MB, got {actual_size:.1f}MB"

    return True, f"Model verified: {gguf_path.name} ({actual_size:.1f}MB)"


# =============================================================================
# MANIFEST UPDATE HELPERS
# =============================================================================

def update_family_metadata(
    family_id: str,
    *,
    manifest_file: Path | None = None,
    compute_hash: bool = True,
    compute_size: bool = True,
) -> dict[str, Any]:
    """
    Update a family's sha256 and size_mb fields based on actual model file.

    Returns the updated family config dict.
    """
    p = manifest_file or manifest_path()
    m = load_manifest(p)
    cfg = get_family_config(family_id, manifest=m)
    gguf_path = _root() / cfg.gguf

    updates: dict[str, Any] = {}

    if compute_hash and gguf_path.is_file():
        updates["sha256"] = compute_sha256(gguf_path)

    if compute_size and gguf_path.is_file():
        updates["size_mb"] = round(get_model_size(gguf_path), 2)

    if updates:
        m["families"][family_id].update(updates)
        p.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")

    return m["families"][family_id]


# =============================================================================
# CLI / DEBUG
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        # Verify all families
        m = load_manifest()
        for fam in list_families(m):
            ok, msg = verify_family_model(fam, manifest=m)
            status = "✓" if ok else "✗"
            print(f"[{status}] {fam}: {msg}")
    elif len(sys.argv) > 1 and sys.argv[1] == "update":
        # Update metadata for specified family
        if len(sys.argv) < 3:
            print("Usage: python -m runtime.manifest update <family_id>")
            sys.exit(1)
        fam = sys.argv[2]
        result = update_family_metadata(fam)
        print(f"Updated {fam}:")
        print(json.dumps(result, indent=2))
    else:
        # List families
        m = load_manifest()
        print(f"Manifest version: {m.get('version', 'unknown')}")
        print(f"Default family: {get_default_family(m)}")
        print(f"Families: {', '.join(list_families(m))}")
