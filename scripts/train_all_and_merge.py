#!/usr/bin/env python3
"""Full pipeline: train LoRA, merge, export GGUF, and test all 9 families."""

import os
import subprocess
import sys

FAMILIES = [
    "english",
    "cjk",
    "arabic",
    "indic",
    "cyrillic",
    "greek",
    "hebrew",
    "southeast_asian",
    "ethiopic",
]

# Group mappings (first family in list is the representative model for the group)
GROUPS = {
    "latin": ["english"],
    "cjk": ["cjk"],
    "rtl": ["arabic", "hebrew"],
    "indic": ["indic"],
    "europe": ["cyrillic", "greek"],
    "sea": ["southeast_asian"],
    "africa": ["ethiopic"],
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print(f"\n>>> {cmd}\n")
    subprocess.run(cmd, shell=True, check=True, cwd=BASE_DIR)


def full_pipeline(family):
    print(f"\n{'='*60}")
    print(f"  {family.upper()}")
    print(f"{'='*60}")

    print(f"\n--- Training {family} ---")
    run(f"python training/train_lora.py --family {family}")

    print(f"\n--- Merging {family} ---")
    run(f"python export/merge_model.py --family {family}")

    print(f"\n--- Exporting {family} (q4_k_m) ---")
    run(f"python export/export_model.py --family {family} --quantization q4_k_m")

    print(f"\n--- Testing {family} ---")
    run(f"python export/test_model.py --family {family}")


def main():
    skip = set(sys.argv[1:]) if len(sys.argv) > 1 else set()

    for fam in FAMILIES:
        if fam in skip:
            print(f"\n========== SKIPPING {fam.upper()} (already done) ==========")
            continue
        full_pipeline(fam)

    # Update manifest with real sizes and hashes
    print("\n--- Updating manifest ---")
    run("python -c \"\nimport hashlib, json, os\nmanifest_path = 'models/manifest.json'\nwith open(manifest_path) as f:\n    manifest = json.load(f)\nfor family, cfg in manifest['families'].items():\n    gguf_path = cfg['gguf']\n    if os.path.isfile(gguf_path):\n        size_bytes = os.path.getsize(gguf_path)\n        cfg['size_mb'] = round(size_bytes / (1024*1024), 1)\n        sha = hashlib.sha256()\n        with open(gguf_path, 'rb') as gf:\n            for chunk in iter(lambda: gf.read(1<<20), b''):\n                sha.update(chunk)\n        cfg['sha256'] = sha.hexdigest()\n        print(f'{family}: {cfg[\\\"size_mb\\\"]}MB')\nwith open(manifest_path, 'w') as f:\n    json.dump(manifest, f, indent=2)\n    f.write('\\\\n')\nprint('Manifest updated.')\n\"")

    print(f"\n{'='*60}")
    print("  ALL FAMILIES COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
