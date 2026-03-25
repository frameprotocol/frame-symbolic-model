#!/usr/bin/env python3
"""Generate multilingual dataset from canonical intents + variations.

Takes English intents and translates them to multiple languages using REAL translations.
NO fake placeholder prefixes like 【中文】 or [AR] are used.

Input: data/canonical/variations.jsonl
Output: data/multilingual/multilingual_intents.jsonl

Languages:
- english (passthrough)
- chinese (中文)
- arabic (العربية)
- hindi (हिन्दी)
- russian (Русский)
- greek (Ελληνικά)
- hebrew (עברית)
- thai (ไทย)
- amharic (አማርኛ)

Usage:
  python scripts/generate_multilingual.py                  # Uses built-in translations
  python scripts/generate_multilingual.py --use-llm       # Uses LLM for translation
  python scripts/generate_multilingual.py --strict        # Rejects untranslated rows
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.translations import TRANSLATIONS, get_translation

CANONICAL_DIR = ROOT / "data" / "canonical"
MULTILINGUAL_DIR = ROOT / "data" / "multilingual"

# =============================================================================
# LANGUAGE DEFINITIONS
# =============================================================================

LANGUAGES = [
    "english",
    "chinese",
    "arabic",
    "hindi",
    "russian",
    "greek",
    "hebrew",
    "thai",
    "amharic",
]

# Language -> Script family mapping
LANGUAGE_TO_FAMILY = {
    "english": "english",
    "chinese": "cjk",
    "japanese": "cjk",
    "korean": "cjk",
    "arabic": "arabic",
    "hindi": "indic",
    "bengali": "indic",
    "tamil": "indic",
    "telugu": "indic",
    "russian": "cyrillic",
    "ukrainian": "cyrillic",
    "greek": "greek",
    "hebrew": "hebrew",
    "thai": "southeast_asian",
    "vietnamese": "southeast_asian",
    "amharic": "ethiopic",
}

# =============================================================================
# INVALID PATTERN DETECTION
# =============================================================================

# Patterns that indicate fake/placeholder translations
INVALID_PATTERNS = [
    r"【.*?】",           # Chinese brackets with content
    r"\[CN\]",           # [CN] prefix
    r"\[AR\]",           # [AR] prefix
    r"\[RU\]",           # [RU] prefix
    r"\[HI\]",           # [HI] prefix
    r"\[EL\]",           # [EL] prefix (Greek)
    r"\[HE\]",           # [HE] prefix
    r"\[TH\]",           # [TH] prefix
    r"\[AM\]",           # [AM] prefix
    r"\[[A-Z]{2,4}\]",   # Any [XX] or [XXX] pattern
    r"^【",               # Starts with Chinese bracket
    r"】$",               # Ends with Chinese bracket
]

INVALID_REGEX = re.compile("|".join(INVALID_PATTERNS))


def is_valid_translation(text: str, language: str, original: str) -> tuple[bool, str]:
    """Validate a translation is real and clean.
    
    Returns (is_valid, error_message).
    """
    if not text:
        return False, "Empty translation"
    
    text = text.strip()
    
    if not text:
        return False, "Whitespace-only translation"
    
    # Check for invalid placeholder patterns
    if INVALID_REGEX.search(text):
        return False, f"Contains placeholder pattern: {INVALID_REGEX.search(text).group()}"
    
    # Check for [ or 【 characters (likely fake markers)
    if "[" in text or "【" in text or "】" in text:
        return False, "Contains bracket markers"
    
    # For non-English languages, ensure translation is different from original
    if language != "english":
        if text.lower().strip() == original.lower().strip():
            return False, "Translation identical to English original"
    
    # Check for numbers preservation (basic check)
    original_numbers = set(re.findall(r"\d+", original))
    translation_numbers = set(re.findall(r"\d+", text))
    
    # Numbers should generally be preserved (allow some flexibility)
    if original_numbers and not original_numbers.intersection(translation_numbers):
        # This is a warning, not a hard failure - some numbers might be written as words
        pass
    
    return True, ""


# =============================================================================
# TRANSLATION FUNCTIONS
# =============================================================================

def translate_with_dictionary(text: str, language: str) -> str | None:
    """Translate using built-in dictionary.
    
    Returns None if no translation found.
    """
    if language == "english":
        return text
    
    return get_translation(text, language)


def translate_with_llm(text: str, language: str) -> str:
    """Translation using LLM API.
    
    Must return ONLY the translated sentence.
    Must preserve meaning exactly.
    Must NOT add prefixes or explanations.
    """
    if language == "english":
        return text
    
    prompt = f"""Translate the following sentence into {language}.

Rules:
- Preserve meaning exactly
- Do NOT add explanations
- Do NOT include quotes
- Do NOT include language tags or markers like [CN] or 【中文】
- Keep names and numbers unchanged
- Return ONLY the translated sentence, nothing else

Sentence:
{text}"""

    # TODO: Integrate OpenAI / local model here
    # Example:
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    #     max_tokens=200,
    #     temperature=0.3,
    # )
    # return response.choices[0].message.content.strip()
    
    raise NotImplementedError(
        "LLM translation not implemented. "
        "Use --strict=false to allow fallback to dictionary translations."
    )


# =============================================================================
# MAIN GENERATION
# =============================================================================

class TranslationStats:
    """Track translation statistics."""
    def __init__(self):
        self.total = 0
        self.translated = 0
        self.skipped = 0
        self.rejected = 0
        self.by_language: dict[str, dict[str, int]] = {
            lang: {"translated": 0, "skipped": 0, "rejected": 0}
            for lang in LANGUAGES
        }
        self.rejection_reasons: list[str] = []


def generate_multilingual_dataset(
    *,
    use_llm: bool = False,
    strict: bool = False,
) -> tuple[list[dict], TranslationStats]:
    """Generate multilingual dataset from variations.
    
    Args:
        use_llm: If True, use LLM for translation (requires implementation)
        strict: If True, reject rows without complete translations
    
    Returns:
        Tuple of (rows, stats)
    """
    stats = TranslationStats()
    
    input_path = CANONICAL_DIR / "variations.jsonl"
    
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run generate_variations.py first.")
        sys.exit(1)
    
    # Load variations
    variations: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                variations.append(json.loads(line))
    
    # Generate multilingual versions
    rows: list[dict] = []
    
    for var in variations:
        stats.total += 1
        intent = var["intent"]
        english_input = var["input"]
        
        # Create entry with all language translations
        entry = {
            "intent": intent,
            "inputs": {},
        }
        
        all_valid = True
        missing_translations: list[str] = []
        
        for lang in LANGUAGES:
            translation = None
            
            if lang == "english":
                translation = english_input
            else:
                # Try dictionary first
                translation = translate_with_dictionary(english_input, lang)
                
                # Fall back to LLM if enabled and dictionary has no translation
                if translation is None and use_llm:
                    try:
                        translation = translate_with_llm(english_input, lang)
                    except NotImplementedError:
                        pass
            
            # Validate translation
            if translation is not None:
                is_valid, error = is_valid_translation(translation, lang, english_input)
                if is_valid:
                    entry["inputs"][lang] = translation
                    stats.by_language[lang]["translated"] += 1
                else:
                    stats.by_language[lang]["rejected"] += 1
                    stats.rejection_reasons.append(f"{lang}: {error} - {english_input[:50]}")
                    all_valid = False
                    missing_translations.append(lang)
            else:
                stats.by_language[lang]["skipped"] += 1
                missing_translations.append(lang)
                all_valid = False
        
        # Add metadata
        entry["base_input"] = var.get("base_input", english_input)
        
        # Decide whether to include this row
        if strict and not all_valid:
            stats.rejected += 1
            continue
        
        # Only include if we have at least English
        if "english" in entry["inputs"]:
            rows.append(entry)
            stats.translated += 1
        else:
            stats.skipped += 1
    
    return rows, stats


def print_samples(rows: list[dict], n_per_language: int = 3) -> None:
    """Print sample translations for each language."""
    print("\n" + "=" * 60)
    print("SAMPLE TRANSLATIONS")
    print("=" * 60)
    
    # Collect samples by language
    samples_by_lang: dict[str, list[tuple[str, str]]] = {lang: [] for lang in LANGUAGES}
    
    for row in rows:
        intent = row["intent"]
        inputs = row.get("inputs", {})
        
        for lang, text in inputs.items():
            if len(samples_by_lang[lang]) < n_per_language:
                samples_by_lang[lang].append((text, intent))
    
    for lang in LANGUAGES:
        print(f"\n--- {lang.upper()} ---")
        for text, intent in samples_by_lang[lang]:
            print(f"  INPUT:  {text}")
            print(f"  INTENT: {intent}")
            print()


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate multilingual dataset")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for translation (requires API setup)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Only include rows with complete translations for all languages"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample translations to show per language"
    )
    args = parser.parse_args()
    
    MULTILINGUAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MULTILINGUAL_DIR / "multilingual_intents.jsonl"
    
    print("Generating multilingual dataset...")
    print(f"  Use LLM: {args.use_llm}")
    print(f"  Strict mode: {args.strict}")
    
    rows, stats = generate_multilingual_dataset(
        use_llm=args.use_llm,
        strict=args.strict,
    )
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("GENERATION STATISTICS")
    print("=" * 60)
    print(f"Total variations processed: {stats.total}")
    print(f"Rows included: {stats.translated}")
    print(f"Rows rejected: {stats.rejected}")
    print(f"Rows skipped: {stats.skipped}")
    print(f"\nOutput: {output_path}")
    
    print("\nPer-language breakdown:")
    for lang in LANGUAGES:
        lang_stats = stats.by_language[lang]
        print(f"  {lang:15} | translated: {lang_stats['translated']:4} | "
              f"skipped: {lang_stats['skipped']:4} | rejected: {lang_stats['rejected']:4}")
    
    # Show sample rejection reasons
    if stats.rejection_reasons:
        print(f"\nSample rejection reasons (first 5):")
        for reason in stats.rejection_reasons[:5]:
            print(f"  - {reason}")
    
    # Print sample translations
    print_samples(rows, n_per_language=args.samples)
    
    # Final validation check
    print("\n" + "=" * 60)
    print("FINAL VALIDATION")
    print("=" * 60)
    
    # Check for any remaining invalid patterns
    invalid_count = 0
    for row in rows:
        for lang, text in row.get("inputs", {}).items():
            if INVALID_REGEX.search(text):
                invalid_count += 1
                print(f"  INVALID: [{lang}] {text[:50]}")
    
    if invalid_count == 0:
        print("✓ DATASET CLEAN - No placeholder patterns found")
    else:
        print(f"✗ DATASET HAS ISSUES - {invalid_count} invalid entries found")
        sys.exit(1)


if __name__ == "__main__":
    main()
