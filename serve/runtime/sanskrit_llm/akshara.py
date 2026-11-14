"""
Akṣara (syllable) segmentation for Devanagari Sanskrit.

An *akṣara* is the fundamental unit of Sanskrit prosody: a consonant cluster
with its vowel, or an independent vowel, optionally closed by anusvāra,
candrabindu or visarga.  In the abugida this is exactly the unit that the
script draws as one glyph cluster:

    धर्मक्षेत्रे  ->  ध र्म क्षे त्रे
    saṃskṛtam   ->  सं स्कृ तम्

Splitting on akṣara boundaries matters for tokenisation: a tokenizer must never
cut between a consonant and its virama, or between a consonant and its matra,
because that destroys the syllable.  Everything downstream (BPE, the model's
feature embeddings, the UI's token visualiser) is defined on these units.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Devanagari character classes (Sanskrit-relevant subset)
# --------------------------------------------------------------------------

VIRAMA = "\u094d"
NUKTA = "\u093c"
ANUSVARA = "\u0902"
CANDRABINDU = "\u0901"
VISARGA = "\u0903"
AVAGRAHA = "\u093d"
OM = "\u0950"
DANDA = "\u0964"
DOUBLE_DANDA = "\u0965"

CONSONANTS = set(
    "कखगघङ" "चछजझञ" "टठडढण" "तथदधन" "पफबभम" "यरलव" "शषसह"
    "\u0934"          # ऴ  Vedic retroflex l
    "\u0958\u0959\u095a\u095b\u095c\u095d\u095e\u095f"  # nukta consonants
)
INDEPENDENT_VOWELS = set("अआइईउऊऋॠऌॡएऐओऔ")
MATRAS = set("\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0962\u0963\u0945\u0946\u0947\u0948\u0949\u094a\u094b\u094c\u094e\u094f")
MODIFIERS = {ANUSVARA, CANDRABINDU, VISARGA}
DIGITS = set("०१२३४५६७८९")

DEVANAGARI_LETTERS = CONSONANTS | INDEPENDENT_VOWELS | MATRAS | MODIFIERS | {VIRAMA, NUKTA}


def is_devanagari(ch: str) -> bool:
    cp = ord(ch)
    return 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF


def _is_letter(ch: str) -> bool:
    return ch in CONSONANTS or ch in INDEPENDENT_VOWELS


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """One segmented unit: an akṣara, a space, punctuation, a digit or a Latin run."""

    text: str
    kind: str          # 'consonantal' | 'vowel' | 'om' | 'digit' | 'punct' | 'space' | 'latin'

    @property
    def is_word_char(self) -> bool:
        return self.kind in ("consonantal", "vowel", "om", "digit", "latin")


def _consume_devanagari(text: str, i: int) -> int:
    """Return the end index of the akṣara starting at ``i``."""
    n = len(text)
    ch = text[i]

    # ॐ
    if ch == OM:
        return i + 1

    # independent vowel (+ modifiers)
    if ch in INDEPENDENT_VOWELS:
        j = i + 1
        while j < n and text[j] in MODIFIERS:
            j += 1
        return j

    # consonant cluster: (C virama)* C (matra | virama)? modifiers*
    if ch in CONSONANTS:
        j = i
        while j < n and text[j] in CONSONANTS:
            k = j + 1
            if k < n and text[k] == NUKTA:
                k += 1
            if k < n and text[k] == VIRAMA:
                j = k + 1                    # cluster continues
                continue
            # final consonant of the cluster
            if k < n and text[k] in MATRAS:
                k += 1
            elif k < n and text[k] == VIRAMA:
                k += 1
            while k < n and text[k] in MODIFIERS:
                k += 1
            return k
        # the cluster ended on a dangling virama (word-final, e.g. "तद् ")
        return j if j > i else i + 1

    return i + 1   # stray mark


def segment(text: str) -> list[Unit]:
    """Split text into akṣara-level units, preserving every character."""
    text = unicodedata.normalize("NFC", text)
    units: list[Unit] = []
    i, n = 0, len(text)

    while i < n:
        ch = text[i]

        if ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            units.append(Unit(text[i:j], "space"))
            i = j
            continue

        if ch in (DANDA, DOUBLE_DANDA) or ch in "।॥|!?.,;:—-–()[]\"'‘’“”…":
            units.append(Unit(ch, "punct"))
            i += 1
            continue

        if ch in DIGITS or (ch.isdigit() and ch.isascii()):
            units.append(Unit(ch, "digit"))
            i += 1
            continue

        if ch in CONSONANTS:
            end = _consume_devanagari(text, i)
            units.append(Unit(text[i:end], "consonantal"))
            i = end
            continue

        if ch in INDEPENDENT_VOWELS:
            end = _consume_devanagari(text, i)
            units.append(Unit(text[i:end], "vowel"))
            i = end
            continue

        if ch == OM:
            units.append(Unit(ch, "om"))
            i += 1
            continue

        if is_devanagari(ch):
            # stray mark / avagraha
            units.append(Unit(ch, "punct" if ch == AVAGRAHA else "consonantal"))
            i += 1
            continue

        # latin runs (transliterated text, numbers in ASCII words, etc.)
        if ch.isalpha() or ch in "'’":
            j = i
            while j < n and (text[j].isalpha() or text[j] in "'’"):
                j += 1
            units.append(Unit(text[i:j], "latin"))
            i = j
            continue

        units.append(Unit(ch, "punct"))
        i += 1

    return units


def aksharas(text: str) -> list[str]:
    """Convenience: the akṣara strings of ``text`` (spaces/punctuation dropped)."""
    return [u.text for u in segment(text) if u.kind in ("consonantal", "vowel", "om")]


# --------------------------------------------------------------------------
# Phonetic features
# --------------------------------------------------------------------------

FEATURE_NAMES = [
    "has_virama",       # conjunct / closed syllable
    "has_matra",        # dependent vowel
    "has_anusvara",     # nasalisation classes
    "has_visarga",
    "has_candrabindu",
    "is_vowel_initial",
    "is_punct",
    "is_digit",
    "has_space",        # token spans a word boundary (sandhi juncture)
    "is_latin",
]

N_FEATURES = len(FEATURE_NAMES)


def features(unit_text: str) -> tuple[float, ...]:
    """Return the Sanskrit-specific feature vector of a token.

    These features are fed to the model alongside the token embedding, so the
    network is told what kind of phonological object it is looking at instead of
    having to rediscover Devanagari orthography from scratch.
    """
    t = unit_text
    return (
        1.0 if VIRAMA in t else 0.0,
        1.0 if any(c in MATRAS for c in t) else 0.0,
        1.0 if ANUSVARA in t else 0.0,
        1.0 if VISARGA in t else 0.0,
        1.0 if CANDRABINDU in t else 0.0,
        1.0 if t and t[0] in INDEPENDENT_VOWELS else 0.0,
        1.0 if t and t[0] in ",;:.!?()[]\"'‘’“”—–" + DANDA + DOUBLE_DANDA else 0.0,
        1.0 if t and all(c in DIGITS or c.isdigit() for c in t) else 0.0,
        1.0 if any(c.isspace() for c in t) else 0.0,
        1.0 if t and t[0].isascii() and t[0].isalpha() else 0.0,
    )


def feature_matrix(text: str) -> list[list[float]]:
    """Per-character feature matrix for the UI visualiser."""
    return [list(features(ch)) for ch in text]
