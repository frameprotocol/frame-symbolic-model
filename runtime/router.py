"""Lightweight script-based router: text -> language-family model id.

PRODUCTION-GRADE ROUTING:
- Mixed-language handling (dominant script wins)
- Comprehensive script detection
- NEVER fails (always returns valid family)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# Default fallback family when no script is detected or for ambiguous cases.
DEFAULT_FAMILY = "english"

# Minimum threshold for a non-Latin script to be considered dominant.
# If non-Latin script chars / total_letters >= this, route to that script's family.
NON_LATIN_DOMINANCE_THRESHOLD = 0.30

# For Latin to be chosen, it must be >= this fraction of total letters.
LATIN_DOMINANCE_THRESHOLD = 0.50


@dataclass
class ScriptCount:
    """Count of characters in a specific script family."""
    name: str
    family: str
    count: int


def _count_range(text: str, start: int, end: int) -> int:
    """Count characters in text that fall within the unicode range [start, end]."""
    return sum(1 for ch in text if start <= ord(ch) <= end)


# =============================================================================
# SCRIPT COUNTERS
# =============================================================================

def _count_latin(text: str) -> int:
    """Basic Latin + Latin-1 Supplement + Latin Extended-A/B/C/D + Latin Extended Additional."""
    return (
        _count_range(text, 0x0041, 0x005A)   # A-Z
        + _count_range(text, 0x0061, 0x007A) # a-z
        + _count_range(text, 0x00C0, 0x00FF) # Latin-1 Supplement (À-ÿ)
        + _count_range(text, 0x0100, 0x017F) # Latin Extended-A
        + _count_range(text, 0x0180, 0x024F) # Latin Extended-B
        + _count_range(text, 0x1E00, 0x1EFF) # Latin Extended Additional
    )


def _count_han(text: str) -> int:
    """CJK Unified Ideographs + Extensions."""
    return (
        _count_range(text, 0x4E00, 0x9FFF)   # CJK Unified Ideographs
        + _count_range(text, 0x3400, 0x4DBF) # CJK Extension A
        + _count_range(text, 0x20000, 0x2A6DF) # CJK Extension B
        + _count_range(text, 0xF900, 0xFAFF)  # CJK Compatibility Ideographs
    )


def _count_hiragana(text: str) -> int:
    return _count_range(text, 0x3040, 0x309F)


def _count_katakana(text: str) -> int:
    return _count_range(text, 0x30A0, 0x30FF) + _count_range(text, 0x31F0, 0x31FF)


def _count_hangul(text: str) -> int:
    """Korean Hangul syllables + Jamo."""
    return (
        _count_range(text, 0xAC00, 0xD7AF)   # Hangul Syllables
        + _count_range(text, 0x1100, 0x11FF) # Hangul Jamo
        + _count_range(text, 0x3130, 0x318F) # Hangul Compatibility Jamo
    )


def _count_arabic(text: str) -> int:
    """Arabic + Arabic Supplement + Arabic Extended-A/B."""
    return (
        _count_range(text, 0x0600, 0x06FF)   # Arabic
        + _count_range(text, 0x0750, 0x077F) # Arabic Supplement
        + _count_range(text, 0x08A0, 0x08FF) # Arabic Extended-A
        + _count_range(text, 0xFB50, 0xFDFF) # Arabic Presentation Forms-A
        + _count_range(text, 0xFE70, 0xFEFF) # Arabic Presentation Forms-B
    )


def _count_devanagari(text: str) -> int:
    """Devanagari (Hindi, Sanskrit, Marathi, etc.)."""
    return _count_range(text, 0x0900, 0x097F) + _count_range(text, 0xA8E0, 0xA8FF)


def _count_tamil(text: str) -> int:
    return _count_range(text, 0x0B80, 0x0BFF)


def _count_telugu(text: str) -> int:
    return _count_range(text, 0x0C00, 0x0C7F)


def _count_bengali(text: str) -> int:
    return _count_range(text, 0x0980, 0x09FF)


def _count_gujarati(text: str) -> int:
    return _count_range(text, 0x0A80, 0x0AFF)


def _count_kannada(text: str) -> int:
    return _count_range(text, 0x0C80, 0x0CFF)


def _count_malayalam(text: str) -> int:
    return _count_range(text, 0x0D00, 0x0D7F)


def _count_punjabi(text: str) -> int:
    """Gurmukhi script (Punjabi)."""
    return _count_range(text, 0x0A00, 0x0A7F)


def _count_cyrillic(text: str) -> int:
    """Cyrillic + Cyrillic Supplement + Extended."""
    return (
        _count_range(text, 0x0400, 0x04FF)   # Cyrillic
        + _count_range(text, 0x0500, 0x052F) # Cyrillic Supplement
        + _count_range(text, 0x2DE0, 0x2DFF) # Cyrillic Extended-A
        + _count_range(text, 0xA640, 0xA69F) # Cyrillic Extended-B
    )


def _count_greek(text: str) -> int:
    """Greek + Greek Extended."""
    return _count_range(text, 0x0370, 0x03FF) + _count_range(text, 0x1F00, 0x1FFF)


def _count_hebrew(text: str) -> int:
    return _count_range(text, 0x0590, 0x05FF) + _count_range(text, 0xFB1D, 0xFB4F)


def _count_thai(text: str) -> int:
    return _count_range(text, 0x0E00, 0x0E7F)


def _count_lao(text: str) -> int:
    return _count_range(text, 0x0E80, 0x0EFF)


def _count_khmer(text: str) -> int:
    return _count_range(text, 0x1780, 0x17FF) + _count_range(text, 0x19E0, 0x19FF)


def _count_myanmar(text: str) -> int:
    return _count_range(text, 0x1000, 0x109F) + _count_range(text, 0xAA60, 0xAA7F)


def _count_ethiopic(text: str) -> int:
    """Ethiopic (Ge'ez, Amharic, Tigrinya, etc.)."""
    return (
        _count_range(text, 0x1200, 0x137F)   # Ethiopic
        + _count_range(text, 0x1380, 0x139F) # Ethiopic Supplement
        + _count_range(text, 0x2D80, 0x2DDF) # Ethiopic Extended
    )


def _count_letters(text: str) -> int:
    """Count all unicode letters (category L*)."""
    return sum(1 for ch in text if ch.isalpha())


# =============================================================================
# MAIN ROUTER
# =============================================================================

def _compute_script_distribution(text: str) -> List[ScriptCount]:
    """
    Compute character counts for all supported scripts.
    Returns list sorted by count (descending).
    """
    # CJK family (Han + Hiragana + Katakana + Hangul)
    han = _count_han(text)
    hiragana = _count_hiragana(text)
    katakana = _count_katakana(text)
    hangul = _count_hangul(text)
    cjk_total = han + hiragana + katakana + hangul

    # Indic family
    devanagari = _count_devanagari(text)
    tamil = _count_tamil(text)
    telugu = _count_telugu(text)
    bengali = _count_bengali(text)
    gujarati = _count_gujarati(text)
    kannada = _count_kannada(text)
    malayalam = _count_malayalam(text)
    punjabi = _count_punjabi(text)
    indic_total = devanagari + tamil + telugu + bengali + gujarati + kannada + malayalam + punjabi

    # Southeast Asian family
    thai = _count_thai(text)
    lao = _count_lao(text)
    khmer = _count_khmer(text)
    myanmar = _count_myanmar(text)
    sea_total = thai + lao + khmer + myanmar

    # Individual scripts
    arabic = _count_arabic(text)
    cyrillic = _count_cyrillic(text)
    greek = _count_greek(text)
    hebrew = _count_hebrew(text)
    ethiopic = _count_ethiopic(text)
    latin = _count_latin(text)

    counts = [
        ScriptCount("cjk", "cjk", cjk_total),
        ScriptCount("arabic", "arabic", arabic),
        ScriptCount("indic", "indic", indic_total),
        ScriptCount("cyrillic", "cyrillic", cyrillic),
        ScriptCount("greek", "greek", greek),
        ScriptCount("hebrew", "hebrew", hebrew),
        ScriptCount("southeast_asian", "southeast_asian", sea_total),
        ScriptCount("ethiopic", "ethiopic", ethiopic),
        ScriptCount("latin", "english", latin),
    ]

    # Sort by count descending
    return sorted(counts, key=lambda x: x.count, reverse=True)


def route(text: str) -> str:
    """
    Route input text to a family id using script heuristics.

    ALGORITHM:
    1. Count characters in each script range.
    2. Compute script distribution.
    3. Apply dominance rules:
       - If any non-Latin script has >= 30% of letters → route to that family
       - If Latin has >= 50% of letters → route to english
       - Otherwise → default to english

    EXAMPLES:
    - "get current time" → english (100% Latin)
    - "获取当前时间" → cjk (100% Han)
    - "send $10 to 张三" → english (Latin majority: "send to" > "张三")
    - "Москва — столица России" → cyrillic (Cyrillic majority)
    - "مرحبا بالعالم" → arabic (100% Arabic)

    GUARANTEES:
    - NEVER fails
    - ALWAYS returns a valid family id from the manifest
    - Empty/whitespace → english
    - Unknown script → english
    """
    t = text or ""
    if not t.strip():
        return DEFAULT_FAMILY

    total_letters = _count_letters(t)
    if total_letters == 0:
        # No letters at all (e.g., just numbers/punctuation)
        return DEFAULT_FAMILY

    distribution = _compute_script_distribution(t)

    # Find the dominant script
    for sc in distribution:
        if sc.count == 0:
            continue

        ratio = sc.count / total_letters

        # Special handling for Latin: needs higher threshold since it's the fallback
        if sc.family == "english":
            if ratio >= LATIN_DOMINANCE_THRESHOLD:
                return "english"
        else:
            # Non-Latin scripts: lower threshold to capture minority scripts
            if ratio >= NON_LATIN_DOMINANCE_THRESHOLD:
                return sc.family

    # If no script is dominant enough, default to english
    return DEFAULT_FAMILY


def route_with_details(text: str) -> Tuple[str, List[ScriptCount]]:
    """
    Route with full diagnostic info.
    Returns (family_id, script_distribution).
    """
    t = text or ""
    if not t.strip():
        return DEFAULT_FAMILY, []

    total_letters = _count_letters(t)
    if total_letters == 0:
        return DEFAULT_FAMILY, []

    distribution = _compute_script_distribution(t)
    family_id = route(t)

    return family_id, distribution


def is_mixed_script(text: str) -> bool:
    """
    Check if text contains multiple scripts.
    Returns True if more than one script family has characters.
    """
    t = text or ""
    if not t.strip():
        return False

    distribution = _compute_script_distribution(t)
    scripts_with_chars = [sc for sc in distribution if sc.count > 0]
    return len(scripts_with_chars) > 1


# =============================================================================
# CLI / DEBUG
# =============================================================================

if __name__ == "__main__":
    import sys

    test_cases = [
        "get current time",
        "获取当前时间",
        "send $10 to 张三",
        "مرحبا بالعالم",
        "Москва — столица России",
        "नमस्ते दुनिया",
        "שלום עולם",
        "สวัสดีโลก",
        "Γειά σου κόσμε",
        "ሰላም ዓለም",
        "Hello 世界",
        "123 456",
        "",
        "   ",
    ]

    if len(sys.argv) > 1:
        test_cases = [" ".join(sys.argv[1:])]

    for text in test_cases:
        family, dist = route_with_details(text)
        total = _count_letters(text)
        print(f"INPUT: {text!r}")
        print(f"  → FAMILY: {family}")
        print(f"  → LETTERS: {total}")
        if dist:
            top3 = [f"{sc.name}={sc.count}" for sc in dist[:3] if sc.count > 0]
            print(f"  → TOP SCRIPTS: {', '.join(top3) or 'none'}")
        print()
