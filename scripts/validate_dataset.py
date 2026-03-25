#!/usr/bin/env python3
"""Validate dataset integrity and consistency.

Checks:
1. Every input maps to valid interlang
2. No duplicate intents with conflicting outputs
3. No malformed syntax
4. All required fields present
5. Unicode handling correct
6. NO fake multilingual data (placeholder prefixes)
7. Language diversity (non-English should not equal English)
8. Numbers preservation across translations

Usage:
  python scripts/validate_dataset.py                    # Validate all
  python scripts/validate_dataset.py --family english   # Validate specific family
  python scripts/validate_dataset.py --canonical        # Validate canonical only
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.canonicalize import canonicalize
from pipeline.validate import validate

CANONICAL_DIR = ROOT / "data" / "canonical"
MULTILINGUAL_DIR = ROOT / "data" / "multilingual"
FAMILIES_DIR = ROOT / "data" / "families"

# =============================================================================
# FAKE MULTILINGUAL DETECTION PATTERNS
# =============================================================================

# Patterns that indicate fake/placeholder translations - MUST REJECT
FAKE_PATTERNS = [
    r"【中文】",          # Chinese marker
    r"【عربي】",          # Arabic marker
    r"【РУС】",           # Russian marker
    r"【हिंदी】",          # Hindi marker
    r"【ΕΛ】",            # Greek marker
    r"【עב】",            # Hebrew marker
    r"【ไทย】",           # Thai marker
    r"【አማ】",            # Amharic marker
    r"【.*?】",           # Any content in Chinese brackets
    r"\[CN\]",           # [CN] prefix
    r"\[AR\]",           # [AR] prefix
    r"\[RU\]",           # [RU] prefix
    r"\[HI\]",           # [HI] prefix
    r"\[EL\]",           # [EL] prefix
    r"\[HE\]",           # [HE] prefix
    r"\[TH\]",           # [TH] prefix
    r"\[AM\]",           # [AM] prefix
    r"\[[A-Z]{2,4}\]",   # Any [XX] or [XXX] pattern
    r"^【",               # Starts with Chinese bracket
    r"】$",               # Ends with Chinese bracket
]

FAKE_PATTERN_REGEX = re.compile("|".join(FAKE_PATTERNS))


class ValidationError(Exception):
    """Validation error with context."""
    def __init__(self, message: str, file: str = "", line: int = 0, data: dict | None = None):
        super().__init__(message)
        self.file = file
        self.line = line
        self.data = data or {}


class ValidationResult:
    """Holds validation results."""
    def __init__(self):
        self.errors: list[ValidationError] = []
        self.warnings: list[str] = []
        self.stats: dict[str, int] = defaultdict(int)
    
    def add_error(self, msg: str, file: str = "", line: int = 0, data: dict | None = None):
        self.errors.append(ValidationError(msg, file, line, data))
        self.stats["errors"] += 1
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        self.stats["warnings"] += 1
    
    @property
    def ok(self) -> bool:
        return len(self.errors) == 0
    
    def print_summary(self):
        if self.ok:
            print("✓ Validation PASSED - DATASET CLEAN")
        else:
            print("✗ Validation FAILED")
        
        print(f"\nStats:")
        for key, value in sorted(self.stats.items()):
            print(f"  {key}: {value}")
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                print(f"  ⚠ {w}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for e in self.errors[:20]:
                loc = f"{e.file}:{e.line}" if e.file else ""
                print(f"  ✗ {loc} {e}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more")


def load_jsonl(path: Path) -> list[tuple[int, dict]]:
    """Load JSONL file, returning (line_number, data) tuples."""
    rows: list[tuple[int, dict]] = []
    if not path.exists():
        return rows
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append((i, json.loads(line)))
                except json.JSONDecodeError as e:
                    rows.append((i, {"_error": str(e), "_raw": line}))
    
    return rows


def validate_intent(intent: str) -> tuple[bool, str]:
    """Validate a single intent string."""
    if not intent:
        return False, "Empty intent"
    
    # Check parse validity
    if not validate(intent):
        return False, "Failed validation"
    
    # Check canonicalization consistency
    try:
        canonical = canonicalize(intent)
        if canonical != intent:
            return False, f"Not canonical: expected {canonical!r}"
    except ValueError as e:
        return False, f"Canonicalization error: {e}"
    
    return True, ""


def check_fake_multilingual(text: str) -> tuple[bool, str]:
    """Check if text contains fake multilingual markers.
    
    Returns (is_fake, matched_pattern).
    """
    match = FAKE_PATTERN_REGEX.search(text)
    if match:
        return True, match.group()
    return False, ""


def check_language_diversity(text: str, language: str, english_text: str) -> tuple[bool, str]:
    """Check that non-English text is actually different from English.
    
    Returns (is_valid, error_message).
    """
    if language == "english":
        return True, ""
    
    if text.lower().strip() == english_text.lower().strip():
        return False, f"Translation identical to English for {language}"
    
    return True, ""


def check_numbers_preserved(text: str, original: str) -> tuple[bool, str]:
    """Check that numbers from original are preserved in translation.
    
    Returns (is_valid, warning_message).
    """
    original_numbers = set(re.findall(r"\d+", original))
    translation_numbers = set(re.findall(r"\d+", text))
    
    if original_numbers and not original_numbers.intersection(translation_numbers):
        missing = original_numbers - translation_numbers
        return False, f"Numbers not preserved: {missing}"
    
    return True, ""


def validate_canonical(result: ValidationResult) -> None:
    """Validate canonical intents dataset."""
    path = CANONICAL_DIR / "canonical_intents.jsonl"
    rows = load_jsonl(path)
    
    if not rows:
        result.add_warning(f"No data in {path}")
        return
    
    result.stats["canonical_total"] = len(rows)
    
    # Track intents to detect conflicts
    intent_to_inputs: dict[str, set[str]] = defaultdict(set)
    
    for line_num, data in rows:
        if "_error" in data:
            result.add_error(f"JSON parse error: {data['_error']}", str(path), line_num)
            continue
        
        # Check required fields
        if "intent" not in data:
            result.add_error("Missing 'intent' field", str(path), line_num, data)
            continue
        
        if "input" not in data:
            result.add_error("Missing 'input' field", str(path), line_num, data)
            continue
        
        intent = data["intent"]
        inp = data["input"]
        
        # Check for fake markers in input
        is_fake, pattern = check_fake_multilingual(inp)
        if is_fake:
            result.add_error(
                f"Fake multilingual marker in input: {pattern}",
                str(path), line_num, {"input": inp}
            )
            continue
        
        # Validate intent
        valid, err = validate_intent(intent)
        if not valid:
            result.add_error(f"Invalid intent: {err}", str(path), line_num, {"intent": intent})
            continue
        
        result.stats["canonical_valid"] += 1
        
        # Track for consistency check
        intent_to_inputs[intent].add(inp)
    
    # Check for diverse inputs per intent
    for intent, inputs in intent_to_inputs.items():
        if len(inputs) < 2:
            result.add_warning(f"Intent has only {len(inputs)} input(s): {intent[:50]}")


def validate_variations(result: ValidationResult) -> None:
    """Validate variations dataset."""
    path = CANONICAL_DIR / "variations.jsonl"
    rows = load_jsonl(path)
    
    if not rows:
        result.add_warning(f"No data in {path}")
        return
    
    result.stats["variations_total"] = len(rows)
    
    for line_num, data in rows:
        if "_error" in data:
            result.add_error(f"JSON parse error: {data['_error']}", str(path), line_num)
            continue
        
        if "intent" not in data or "input" not in data:
            result.add_error("Missing required field", str(path), line_num)
            continue
        
        intent = data["intent"]
        inp = data["input"]
        
        # Check for fake markers
        is_fake, pattern = check_fake_multilingual(inp)
        if is_fake:
            result.add_error(
                f"Fake multilingual marker: {pattern}",
                str(path), line_num, {"input": inp}
            )
            continue
        
        valid, err = validate_intent(intent)
        if not valid:
            result.add_error(f"Invalid intent: {err}", str(path), line_num, {"intent": intent})
            continue
        
        result.stats["variations_valid"] += 1


def validate_multilingual(result: ValidationResult) -> None:
    """Validate multilingual dataset."""
    path = MULTILINGUAL_DIR / "multilingual_intents.jsonl"
    rows = load_jsonl(path)
    
    if not rows:
        result.add_warning(f"No data in {path}")
        return
    
    result.stats["multilingual_total"] = len(rows)
    
    for line_num, data in rows:
        if "_error" in data:
            result.add_error(f"JSON parse error: {data['_error']}", str(path), line_num)
            continue
        
        if "intent" not in data:
            result.add_error("Missing 'intent' field", str(path), line_num)
            continue
        
        if "inputs" not in data or not isinstance(data["inputs"], dict):
            result.add_error("Missing or invalid 'inputs' field", str(path), line_num)
            continue
        
        intent = data["intent"]
        valid, err = validate_intent(intent)
        if not valid:
            result.add_error(f"Invalid intent: {err}", str(path), line_num, {"intent": intent})
            continue
        
        # Check each language translation
        inputs = data["inputs"]
        english_text = inputs.get("english", "")
        
        for lang, text in inputs.items():
            # Check for fake multilingual markers
            is_fake, pattern = check_fake_multilingual(text)
            if is_fake:
                result.add_error(
                    f"Fake multilingual marker in {lang}: {pattern}",
                    str(path), line_num, {"text": text[:50], "language": lang}
                )
                continue
            
            # Check language diversity
            is_diverse, err = check_language_diversity(text, lang, english_text)
            if not is_diverse:
                result.add_error(err, str(path), line_num, {"language": lang, "text": text[:50]})
                continue
            
            # Check numbers preservation (warning only)
            if english_text:
                is_preserved, warn = check_numbers_preserved(text, english_text)
                if not is_preserved:
                    result.add_warning(f"Line {line_num}, {lang}: {warn}")
        
        if not inputs:
            result.add_warning(f"Empty inputs at line {line_num}")
        
        result.stats["multilingual_valid"] += 1
        result.stats["multilingual_languages"] = max(
            result.stats.get("multilingual_languages", 0),
            len(inputs)
        )


def validate_family(family: str, result: ValidationResult) -> None:
    """Validate a specific family dataset."""
    path = FAMILIES_DIR / f"{family}.jsonl"
    rows = load_jsonl(path)
    
    if not rows:
        result.add_warning(f"No data for family {family}")
        return
    
    result.stats[f"family_{family}_total"] = len(rows)
    
    # Track for conflict detection
    input_to_intent: dict[str, str] = {}
    
    for line_num, data in rows:
        if "_error" in data:
            result.add_error(f"JSON parse error: {data['_error']}", str(path), line_num)
            continue
        
        if "intent" not in data or "input" not in data:
            result.add_error("Missing required field", str(path), line_num)
            continue
        
        intent = data["intent"]
        inp = data["input"]
        
        # Check for fake multilingual markers
        is_fake, pattern = check_fake_multilingual(inp)
        if is_fake:
            result.add_error(
                f"Fake multilingual marker: {pattern}",
                str(path), line_num, {"input": inp[:50]}
            )
            continue
        
        # Validate intent
        valid, err = validate_intent(intent)
        if not valid:
            result.add_error(f"Invalid intent: {err}", str(path), line_num, {"intent": intent})
            continue
        
        # Check for conflicts (same input -> different intents)
        inp_lower = inp.lower().strip()
        if inp_lower in input_to_intent:
            existing = input_to_intent[inp_lower]
            if existing != intent:
                result.add_error(
                    f"Conflict: input maps to multiple intents",
                    str(path),
                    line_num,
                    {"input": inp, "intent1": existing, "intent2": intent}
                )
        else:
            input_to_intent[inp_lower] = intent
        
        result.stats[f"family_{family}_valid"] = result.stats.get(f"family_{family}_valid", 0) + 1


def validate_all_families(result: ValidationResult) -> None:
    """Validate all family datasets."""
    families = [
        "english", "cjk", "arabic", "indic", "cyrillic",
        "greek", "hebrew", "southeast_asian", "ethiopic"
    ]
    
    for family in families:
        path = FAMILIES_DIR / f"{family}.jsonl"
        if path.exists():
            validate_family(family, result)


def scan_for_fake_markers(result: ValidationResult) -> None:
    """Scan all datasets for any remaining fake multilingual markers."""
    print("\nScanning for fake multilingual markers...")
    
    all_paths = list(CANONICAL_DIR.glob("*.jsonl"))
    all_paths.extend(MULTILINGUAL_DIR.glob("*.jsonl"))
    all_paths.extend(FAMILIES_DIR.glob("*.jsonl"))
    
    fake_count = 0
    
    for path in all_paths:
        rows = load_jsonl(path)
        for line_num, data in rows:
            if "_error" in data:
                continue
            
            # Check all string values in the data
            for key, value in data.items():
                if isinstance(value, str):
                    is_fake, pattern = check_fake_multilingual(value)
                    if is_fake:
                        fake_count += 1
                        result.add_error(
                            f"Fake marker found: {pattern}",
                            str(path), line_num, {key: value[:50]}
                        )
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str):
                            is_fake, pattern = check_fake_multilingual(subvalue)
                            if is_fake:
                                fake_count += 1
                                result.add_error(
                                    f"Fake marker found in {subkey}: {pattern}",
                                    str(path), line_num, {subkey: subvalue[:50]}
                                )
    
    result.stats["fake_markers_found"] = fake_count
    if fake_count == 0:
        print("  ✓ No fake multilingual markers found")
    else:
        print(f"  ✗ Found {fake_count} fake markers")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate dataset")
    parser.add_argument("--canonical", action="store_true", help="Validate canonical only")
    parser.add_argument("--variations", action="store_true", help="Validate variations only")
    parser.add_argument("--multilingual", action="store_true", help="Validate multilingual only")
    parser.add_argument("--family", type=str, help="Validate specific family")
    parser.add_argument("--all-families", action="store_true", help="Validate all families")
    parser.add_argument("--scan-fake", action="store_true", help="Scan for fake multilingual markers only")
    args = parser.parse_args()
    
    result = ValidationResult()
    
    # If no specific flags, validate everything
    validate_everything = not any([
        args.canonical, args.variations, args.multilingual,
        args.family, args.all_families, args.scan_fake
    ])
    
    print("Dataset Validation")
    print("=" * 60)
    
    if args.scan_fake:
        scan_for_fake_markers(result)
    else:
        if args.canonical or validate_everything:
            print("\nValidating canonical intents...")
            validate_canonical(result)
        
        if args.variations or validate_everything:
            print("\nValidating variations...")
            validate_variations(result)
        
        if args.multilingual or validate_everything:
            print("\nValidating multilingual dataset...")
            validate_multilingual(result)
        
        if args.family:
            print(f"\nValidating family: {args.family}...")
            validate_family(args.family, result)
        
        if args.all_families or validate_everything:
            print("\nValidating all families...")
            validate_all_families(result)
        
        # Always scan for fake markers
        if validate_everything:
            scan_for_fake_markers(result)
    
    print("\n" + "=" * 60)
    result.print_summary()
    
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
