"""Metaphone, for step three of entity resolution (CLAUDE.md §10.4).

Written out rather than pulled in as a dependency (§2.10): the algorithm is
small, fully specified, and adding a package for it would mean carrying
something unmaintained for ~120 lines of well-defined string rewriting.

Metaphone collapses English spellings to a consonant skeleton, which is
exactly what a transcript mishearing preserves. ``Barriss`` and ``Baras`` both
reduce to ``BRS``; ``Drog`` and ``Draahg`` both reduce to ``TRK``.

Its limits are real and worth stating: it is English-oriented, and it will not
connect two names that differ by a consonant. ``Valron`` -> ``FLRN`` and
``Vowrawn`` -> ``FRN`` do not match, which is precisely why the domain pack is
consulted first and this is only step three.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

__all__ = ["metaphone", "sounds_like", "similarity", "normalize_name"]

_VOWELS = frozenset("AEIOU")
_NON_ALPHA = re.compile(r"[^A-Z]")

#: Initial pairs whose first letter is silent.
_SILENT_INITIAL_PAIRS = ("AE", "GN", "KN", "PN", "WR")


def normalize_name(text: str) -> str:
    """Casefold and collapse whitespace, for dictionary lookups."""
    return " ".join(text.strip().lower().split())


def _is_vowel(word: str, index: int) -> bool:
    return 0 <= index < len(word) and word[index] in _VOWELS


def metaphone(text: str) -> str:
    """Metaphone key for a single word, or for each word of a phrase joined by spaces.

    Returns an empty string for input with no letters.
    """
    if " " in text.strip():
        return " ".join(filter(None, (metaphone(part) for part in text.split())))

    word = _NON_ALPHA.sub("", text.upper())
    if not word:
        return ""

    # Silent initial letters.
    if word[:2] in _SILENT_INITIAL_PAIRS:
        word = word[1:]
    elif word.startswith("X"):
        word = "S" + word[1:]
    elif word.startswith("WH"):
        word = "W" + word[2:]
    if not word:
        return ""

    out: list[str] = []
    length = len(word)
    index = 0

    while index < length:
        char = word[index]
        prev = word[index - 1] if index > 0 else ""
        nxt = word[index + 1] if index + 1 < length else ""
        after_next = word[index + 2] if index + 2 < length else ""

        # Collapse doubled letters, except CC which carries meaning.
        if char == prev and char != "C":
            index += 1
            continue

        if char in _VOWELS:
            # Vowels survive only in first position.
            if index == 0:
                out.append(char)
            index += 1
            continue

        if char == "B":
            # Silent in a final "MB".
            if not (index == length - 1 and prev == "M"):
                out.append("B")

        elif char == "C":
            if nxt == "I" and after_next == "A":
                out.append("X")
            elif nxt == "H":
                out.append("K" if prev == "S" else "X")
                index += 1  # consume the H
            elif nxt in ("I", "E", "Y"):
                if prev != "S":  # "SCI", "SCE", "SCY" already sounded as S
                    out.append("S")
            else:
                out.append("K")

        elif char == "D":
            if nxt == "G" and after_next in ("E", "Y", "I"):
                out.append("J")
                index += 2  # consume the G and the following vowel marker
                continue
            out.append("T")

        elif char == "G":
            if nxt == "H":
                # Silent unless the H starts a new sounded syllable.
                if not (index + 2 >= length or _is_vowel(word, index + 2)):
                    index += 2
                    continue
                out.append("K")
                index += 2
                continue
            if nxt == "N":
                index += 1
                continue  # "GN", "GNED"
            out.append("J" if nxt in ("I", "E", "Y") else "K")

        elif char == "H":
            # Sounded only between a vowel and a following vowel.
            if prev in _VOWELS and not _is_vowel(word, index + 1):
                index += 1
                continue
            if prev in ("C", "S", "P", "T", "G"):
                index += 1
                continue  # already handled by the preceding consonant
            out.append("H")

        elif char in ("F", "J", "L", "M", "N", "R"):
            out.append(char)

        elif char == "K":
            if prev != "C":
                out.append("K")

        elif char == "P":
            if nxt == "H":
                out.append("F")
                index += 1
            else:
                out.append("P")

        elif char == "Q":
            out.append("K")

        elif char == "S":
            if nxt == "H":
                out.append("X")
                index += 1
            elif nxt == "I" and after_next in ("O", "A"):
                out.append("X")
            else:
                out.append("S")

        elif char == "T":
            if nxt == "I" and after_next in ("O", "A"):
                out.append("X")
            elif nxt == "H":
                out.append("0")  # theta
                index += 1
            elif not (nxt == "C" and after_next == "H"):
                out.append("T")

        elif char == "V":
            out.append("F")

        elif char in ("W", "Y"):
            # Sounded only before a vowel.
            if _is_vowel(word, index + 1):
                out.append(char)

        elif char == "X":
            out.append("KS")

        elif char == "Z":
            out.append("S")

        index += 1

    return "".join(out)


def sounds_like(a: str, b: str) -> bool:
    """True when two names share a metaphone key. Empty keys never match."""
    key_a, key_b = metaphone(a), metaphone(b)
    return bool(key_a) and key_a == key_b


def similarity(a: str, b: str) -> float:
    """Normalised edit similarity in 0.0..1.0, for the fuzzy pass.

    Used to catch spelling drift that metaphone misses, such as a dropped
    doubled letter.
    """
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()
