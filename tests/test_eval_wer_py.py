#!/usr/bin/env python3
"""Scorer normalization must be Unicode-safe (#266).

The regression this pins: the ASCII-only character class split German
words at their umlauts ("schön" -> "sch n"), inflating non-English WER
with phantom errors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from eval_audio_wer import edit_distance, norm_words  # noqa: E402


def main() -> int:
    print("eval_audio_wer normalization:")

    cases = [
        ("Schön groß, weiß!", ["schön", "groß", "weiß"]),
        ("It's an A-B test.", ["it's", "an", "a", "b", "test"]),
        ("Ärzte über Straße", ["ärzte", "über", "straße"]),
        ("  double  spaces\tand\nlines ", ["double", "spaces", "and", "lines"]),
    ]
    for raw, want in cases:
        got = norm_words(raw)
        if got != want:
            print(f"  FAIL: norm_words({raw!r}) = {got}, want {want}")
            return 1
        print(f"  ✓ {raw!r} -> {got}")

    # WER sanity: one substitution in three words.
    d = edit_distance(norm_words("der graue Hund"), norm_words("der große Hund"))
    if d != 1:
        print(f"  FAIL: edit_distance = {d}, want 1")
        return 1
    print("  ✓ edit distance counts umlaut words as whole words")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
