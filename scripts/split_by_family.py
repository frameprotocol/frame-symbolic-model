#!/usr/bin/env python3
"""Split multilingual dataset by language family.

Takes multilingual dataset and outputs per-family JSONL files.

Input: data/multilingual/multilingual_intents.jsonl
Output: 
  data/families/english.jsonl
  data/families/cjk.jsonl
  data/families/arabic.jsonl
  data/families/indic.jsonl
  data/families/cyrillic.jsonl
  data/families/greek.jsonl
  data/families/hebrew.jsonl
  data/families/southeast_asian.jsonl
  data/families/ethiopic.jsonl

Mapping:
  english -> english
  chinese/japanese/korean -> cjk
  arabic -> arabic
  hindi -> indic
  russian -> cyrillic
  greek -> greek
  hebrew -> hebrew
  thai -> southeast_asian
  amharic -> ethiopic
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MULTILINGUAL_DIR = ROOT / "data" / "multilingual"
FAMILIES_DIR = ROOT / "data" / "families"

# =============================================================================
# LANGUAGE -> FAMILY MAPPING
# =============================================================================

LANGUAGE_TO_FAMILY: dict[str, str] = {
    # Latin script
    "english": "english",
    "spanish": "english",
    "french": "english",
    "german": "english",
    "portuguese": "english",
    "italian": "english",
    "dutch": "english",
    "polish": "english",
    "romanian": "english",
    "turkish": "english",
    "vietnamese": "english",  # Latin script
    "indonesian": "english",
    "malay": "english",
    "tagalog": "english",
    "swahili": "english",
    
    # CJK
    "chinese": "cjk",
    "mandarin": "cjk",
    "cantonese": "cjk",
    "japanese": "cjk",
    "korean": "cjk",
    
    # Arabic script
    "arabic": "arabic",
    "persian": "arabic",
    "urdu": "arabic",
    "pashto": "arabic",
    "farsi": "arabic",
    
    # Indic scripts
    "hindi": "indic",
    "bengali": "indic",
    "tamil": "indic",
    "telugu": "indic",
    "marathi": "indic",
    "gujarati": "indic",
    "kannada": "indic",
    "malayalam": "indic",
    "punjabi": "indic",
    "nepali": "indic",
    "sinhala": "indic",
    
    # Cyrillic
    "russian": "cyrillic",
    "ukrainian": "cyrillic",
    "bulgarian": "cyrillic",
    "serbian": "cyrillic",
    "macedonian": "cyrillic",
    "belarusian": "cyrillic",
    "kazakh": "cyrillic",
    
    # Greek
    "greek": "greek",
    
    # Hebrew
    "hebrew": "hebrew",
    "yiddish": "hebrew",
    
    # Southeast Asian
    "thai": "southeast_asian",
    "khmer": "southeast_asian",
    "lao": "southeast_asian",
    "myanmar": "southeast_asian",
    "burmese": "southeast_asian",
    
    # Ethiopic
    "amharic": "ethiopic",
    "tigrinya": "ethiopic",
    "geez": "ethiopic",
}

# All supported families
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


def get_family(language: str) -> str:
    """Get family for a language, defaulting to english."""
    return LANGUAGE_TO_FAMILY.get(language.lower(), "english")


def split_by_family() -> dict[str, list[dict]]:
    """Split multilingual dataset by family."""
    input_path = MULTILINGUAL_DIR / "multilingual_intents.jsonl"
    
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run generate_multilingual.py first.")
        sys.exit(1)
    
    # Load multilingual data
    entries: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    
    # Split by family
    family_data: dict[str, list[dict]] = defaultdict(list)
    
    for entry in entries:
        intent = entry["intent"]
        inputs = entry.get("inputs", {})
        
        # For each language in the entry, add to appropriate family
        for lang, text in inputs.items():
            family = get_family(lang)
            
            # Create row for this family
            row = {
                "intent": intent,
                "input": text,
                "language": lang,
                "family": family,
            }
            
            # Avoid duplicates in same family
            family_data[family].append(row)
    
    return dict(family_data)


def deduplicate_family(rows: list[dict]) -> list[dict]:
    """Remove duplicate (intent, input) pairs within a family."""
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    
    for row in rows:
        key = (row["intent"], row["input"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    
    return unique


def main() -> None:
    FAMILIES_DIR.mkdir(parents=True, exist_ok=True)
    
    family_data = split_by_family()
    
    print("Family split results:")
    print("-" * 50)
    
    total = 0
    for family in FAMILIES:
        rows = family_data.get(family, [])
        rows = deduplicate_family(rows)
        
        output_path = FAMILIES_DIR / f"{family}.jsonl"
        
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        total += len(rows)
        print(f"  {family}: {len(rows)} examples -> {output_path}")
    
    print("-" * 50)
    print(f"Total: {total} examples across {len(FAMILIES)} families")


if __name__ == "__main__":
    main()
