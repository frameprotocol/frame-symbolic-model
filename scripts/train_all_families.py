#!/usr/bin/env python3
"""Train, merge, export, and test all 9 families on the pruned base model."""

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


def run(cmd):
    print(f"\n>>> {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def main():
    skip = set(sys.argv[1:]) if len(sys.argv) > 1 else set()

    for f in FAMILIES:
        if f in skip:
            print(f"\n========== SKIPPING {f.upper()} ==========")
            continue

        print(f"\n{'='*60}")
        print(f"  {f.upper()}")
        print(f"{'='*60}")

        run(f"python training/train_lora.py --family {f}")
        run(f"python export/merge_model.py --family {f}")
        run(f"python export/export_model.py --family {f} --quantization q3_k_m")
        run(f"python export/test_model.py --family {f}")

    print(f"\n{'='*60}")
    print("  ALL FAMILIES COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
