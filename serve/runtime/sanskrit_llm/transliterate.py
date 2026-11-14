"""
Bidirectional Sanskrit transliteration: Devanagari <-> IAST.

This module is deliberately written from the grammar of Sanskrit rather than
from generic Unicode tables.  The conversion is not a character-per-character
mapping: Devanagari encodes the *syllable* (akṣara) in an abugida, so a
consonant carries an inherent /a/ that must be dropped with a virama (्) when
it belongs to a cluster, and replaced by a matra (ा ि ी ...) when it does not.

Two directions are provided:

    devanagari_to_iast("धर्मक्षेत्रे कुरुक्षेत्रे")  -> "dharmakṣetre kurukṣetre"
    iast_to_devanagari("dharmakṣetre kurukṣetre")    -> "धर्मक्षेत्रे कुरुक्षेत्रे"

The IAST reader also understands the conventions used by GRETIL and the
Digital Corpus of Sanskrit (ṁ as a variant anusvāra, ẖ/ḻ for Vedic ḷ, the
apostrophe for avagraha, | and || for daṇḍas).
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------

#: independent vowels: IAST -> Devanagari
INDEPENDENT_VOWELS = {
    "a": "अ", "ā": "आ", "i": "इ", "ī": "ई", "u": "उ", "ū": "ऊ",
    "ṛ": "ऋ", "ṝ": "ॠ", "r̥": "ऋ", "l̥": "ऌ", "ḹ": "ॡ",
    "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ",
}

#: dependent vowel signs (matras).  'a' is inherent, so it maps to "".
MATRAS = {
    "a": "", "ā": "ा", "i": "ि", "ī": "ी", "u": "ु", "ū": "ू",
    "ṛ": "ृ", "ṝ": "ॄ", "r̥": "ृ", "l̥": "ॢ", "ḹ": "ॣ",
    "e": "े", "ai": "ै", "o": "ो", "au": "ौ",
}

#: consonants: IAST -> Devanagari
CONSONANTS = {
    "k": "क", "kh": "ख", "g": "ग", "gh": "घ", "ṅ": "ङ",
    "c": "च", "ch": "छ", "j": "ज", "jh": "झ", "ñ": "ञ",
    "ṭ": "ट", "ṭh": "ठ", "ḍ": "ड", "ḍh": "ढ", "ṇ": "ण",
    "t": "त", "th": "थ", "d": "द", "dh": "ध", "n": "न",
    "p": "प", "ph": "फ", "b": "ब", "bh": "भ", "m": "म",
    "y": "य", "r": "र", "l": "ल", "v": "व",
    "ś": "श", "ṣ": "ष", "s": "स", "h": "ह",
    # Vedic / marginal Sanskrit sounds: ḷ is the retroflex lateral (ळ)
    "ḷ": "ळ", "ḻ": "ळ", "ẖ": "ह",
}

VOWEL_SIGNS_RE = "|".join(sorted(MATRAS, key=len, reverse=True))
CONSONANTS_RE = "|".join(sorted(CONSONANTS, key=len, reverse=True))

VIRAMA = "्"
ANUSVARA = "ं"
CANDRABINDU = "ँ"
VISARGA = "ः"
AVAGRAHA = "ऽ"
NUKTA = "़"
OM = "ॐ"
DANDA = "।"
DOUBLE_DANDA = "॥"

#: Devanagari -> IAST
DEV_CONSONANTS = {}
for _iast, _dev in CONSONANTS.items():
    DEV_CONSONANTS.setdefault(_dev, _iast)
DEV_CONSONANTS["ळ"] = "ḷ"
DEV_CONSONANTS["ऌ"] = "ḷ"
DEV_VOWELS = {v: k for k, v in INDEPENDENT_VOWELS.items()}
DEV_MATRAS = {v: k for k, v in MATRAS.items() if v}
# canonical IAST forms win over the ring-below variants used by some corpora
DEV_VOWELS["ऋ"], DEV_VOWELS["ॠ"], DEV_VOWELS["ऌ"] = "ṛ", "ṝ", "ḷ"
DEV_MATRAS["ृ"], DEV_MATRAS["ॄ"], DEV_MATRAS["ॢ"] = "ṛ", "ṝ", "ḷ"

DEVANAGARI_RANGE = (0x0900, 0x097F)
DEVANAGARI_EXT = (0xA8E0, 0xA8FF)
VEDIC_EXT = (0x1CD0, 0x1CFF)

DIGITS_DEV = "०१२३४५६७८९"
DIGITS_ASCII = "0123456789"


class TransliterationError(ValueError):
    """Raised when a string cannot be transliterated."""


# --------------------------------------------------------------------------
# Devanagari -> IAST
# --------------------------------------------------------------------------

def _is_devanagari(ch: str) -> bool:
    cp = ord(ch)
    return DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1] or DEVANAGARI_EXT[0] <= cp <= DEVANAGARI_EXT[1]


def devanagari_to_iast(text: str, *, keep_punctuation: bool = True) -> str:
    """Transliterate Devanagari Sanskrit into IAST.

    The inherent /a/ is emitted for every consonant that is *not* followed by a
    virama or a matra; that is exactly the rule the abugida encodes.
    """

    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    i, n = 0, len(text)

    while i < n:
        ch = text[i]

        # --- daṇḍas -------------------------------------------------------
        if ch == DANDA:
            out.append("|" if keep_punctuation else ".")
            i += 1
            continue
        if ch == DOUBLE_DANDA:
            out.append("||" if keep_punctuation else ".")
            i += 1
            continue

        # --- om -----------------------------------------------------------
        if ch == OM:
            out.append("oṃ")
            i += 1
            continue

        # --- avagraha -----------------------------------------------------
        if ch == AVAGRAHA:
            out.append("'")
            i += 1
            continue

        # --- digits -------------------------------------------------------
        if ch in DIGITS_DEV:
            out.append(DIGITS_ASCII[DIGITS_DEV.index(ch)])
            i += 1
            continue

        # --- consonants ---------------------------------------------------
        if ch in DEV_CONSONANTS:
            base = DEV_CONSONANTS[ch]
            i += 1
            # optional nukta is part of the consonant
            if i < n and text[i] == NUKTA:
                i += 1
            # vowel sign?
            if i < n and text[i] == VIRAMA:
                out.append(base)          # cluster member: no inherent vowel
                i += 1
                continue
            if i < n and text[i] in DEV_MATRAS:
                out.append(base + DEV_MATRAS[text[i]])
                i += 1
            else:
                out.append(base + "a")    # inherent vowel
            # syllable-final modifiers
            while i < n and text[i] in (ANUSVARA, CANDRABINDU, VISARGA):
                out.append({"ं": "ṃ", "ँ": "m̐", "ः": "ḥ"}[text[i]])
                i += 1
            continue

        # --- independent vowels -------------------------------------------
        if ch in DEV_VOWELS:
            out.append(DEV_VOWELS[ch])
            i += 1
            while i < n and text[i] in (ANUSVARA, CANDRABINDU, VISARGA):
                out.append({"ं": "ṃ", "ँ": "m̐", "ः": "ḥ"}[text[i]])
                i += 1
            continue

        # --- stray marks --------------------------------------------------
        if ch in (ANUSVARA, CANDRABINDU, VISARGA):
            out.append({"ं": "ṃ", "ँ": "m̐", "ः": "ḥ"}[ch])
            i += 1
            continue
        if ch == VIRAMA:
            i += 1
            continue
        if ch == NUKTA:
            i += 1
            continue

        # --- everything else passes through --------------------------------
        out.append(ch)
        i += 1

    return "".join(out)


# --------------------------------------------------------------------------
# IAST -> Devanagari
# --------------------------------------------------------------------------

#: normalisation of the many romanisation variants seen in real corpora
_VARIANT_FOLD = {
    "ṁ": "ṃ", "ṁ": "ṃ", "m̐": "m̐", "ṃ": "ṃ",
    "ē": "e", "ō": "o", "ĕ": "e", "ŏ": "o",
    "ô": "o", "â": "ā", "ê": "e", "î": "ī", "û": "ū",
}


_UPPER_FOLD = str.maketrans({
    **{c: c.lower() for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    **{c: c.lower() for c in "ĀĪŪṚṜḶḸŚṢṆṬḌṄÑṀḤ"},
})


def _strip_vedic_accents(text: str) -> str:
    """Remove Vedic pitch accents that have no IAST-ASCII equivalent here."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if not (VEDIC_EXT[0] <= ord(ch) <= VEDIC_EXT[1])
    )


def iast_to_devanagari(text: str, *, danda: bool = True, om: bool = True) -> str:
    """Transliterate an IAST / GRETIL romanisation into Devanagari.

    Handles conjuncts (the pending-consonant buffer), the inherent vowel,
    avagraha, daṇḍas, and GRETIL's extra marks.
    """

    text = unicodedata.normalize("NFC", text)
    text = _strip_vedic_accents(text)
    text = unicodedata.normalize("NFC", text)
    for src, dst in _VARIANT_FOLD.items():
        text = text.replace(src, dst)

    # Sanskrit romanisation is case-insensitive here (Śiva == śiva)
    text = text.translate(_UPPER_FOLD)

    out: list[str] = []
    pending: str | None = None   # a consonant waiting for its vowel
    i, n = 0, len(text)

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            out.append(pending + VIRAMA)
            pending = None

    while i < n:
        ch = text[i]

        # ---- punctuation: daṇḍas -----------------------------------------
        if ch == "|":
            if text[i:i + 2] == "||":
                flush_pending()
                out.append(DOUBLE_DANDA)
                i += 2
            else:
                flush_pending()
                out.append(DANDA)
                i += 1
            continue
        # DCS uses / and // as pāda / verse separators
        if ch == "/":
            if text[i:i + 2] == "//":
                flush_pending()
                out.append(DOUBLE_DANDA)
                i += 2
            else:
                flush_pending()
                out.append(DANDA)
                i += 1
            continue

        # ---- avagraha -----------------------------------------------------
        if ch in ("'", "’", "ʼ"):
            flush_pending()
            out.append(AVAGRAHA)
            i += 1
            continue

        # ---- digits -------------------------------------------------------
        if ch.isdigit() and ch in DIGITS_ASCII:
            flush_pending()
            out.append(DIGITS_DEV[DIGITS_ASCII.index(ch)])
            i += 1
            continue

        # ---- praṇava: standalone oṃ / om -------------------------------
        if (om and (text.startswith("oṃ", i) or text.startswith("oṁ", i))
                and (i == 0 or text[i - 1] in " .,;|/")
                and (i + 2 >= n or not text[i + 2].isalpha())):
            flush_pending()
            out.append(OM)
            i += 2
            continue

        # ---- syllabic liquids: r̥ -> ऋ, l̥ -> ऌ --------------------------
        if text.startswith("r̥", i) or text.startswith("l̥", i):
            cand = text[i:i + 2]
            if pending is not None:
                out.append(pending + MATRAS[cand])
                pending = None
            else:
                out.append(INDEPENDENT_VOWELS[cand])
            i += 2
            continue

        # ---- Vedic retroflex ḷ (ळ) vs vocalic ḷ ---------------------------
        if ch == "ḷ" and not text.startswith("ḷh", i):
            prev = text[i - 1] if i else " "
            nxt = text[i + 1] if i + 1 < len(text) else " "
            vowelish = set("aāiīuūeo")
            if prev in vowelish and (nxt in vowelish or nxt == " "):
                # īḷe: a real retroflex lateral consonant
                flush_pending()
                pending = "ळ"
            else:
                # kḷpta: the syllabic liquid
                if pending is not None:
                    out.append(pending + MATRAS["l̥"])
                    pending = None
                else:
                    out.append(INDEPENDENT_VOWELS["l̥"])
            i += 1
            continue

        # ---- multi-letter units (consonant digraphs, diphthongs) ----------
        matched = False
        for cand in (CONSONANTS_RE.split("|")):
            if cand and text.startswith(cand, i):
                flush_pending()
                pending = CONSONANTS[cand]
                i += len(cand)
                matched = True
                break
        if matched:
            continue

        for cand in ("ai", "au", "ā", "ī", "ū", "ṛ", "ṝ", "r̥", "l̥", "ḹ",
                     "e", "o", "a", "i", "u"):
            if text.startswith(cand, i):
                if pending is not None:
                    out.append(pending + MATRAS[cand])
                    pending = None
                else:
                    out.append(INDEPENDENT_VOWELS[cand])
                i += len(cand)
                matched = True
                break
        if matched:
            continue

        # ---- syllable modifiers -------------------------------------------
        if ch in ("ṃ", "m̐", "ḥ"):
            # a consonant with no vowel gets a virama before the modifier
            if pending is not None:
                out.append(pending + VIRAMA)
                pending = None
            out.append({"ṃ": ANUSVARA, "m̐": CANDRABINDU, "ḥ": VISARGA}[ch]
                       if ch != "m̐" else CANDRABINDU)
            i += 1
            continue

        # ---- anything else: boundary --------------------------------------
        flush_pending()
        out.append(ch)
        i += 1

    flush_pending()
    return "".join(out)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def iast_to_slp1(text: str) -> str:
    """IAST -> SLP1 (used by many Sanskrit digital libraries)."""
    table = {
        "ā": "A", "ī": "I", "ū": "U", "ṛ": "f", "ṝ": "F", "ḷ": "x", "ḹ": "X",
        "ṅ": "N", "ñ": "Y", "ṭ": "w", "ḍ": "q", "ṇ": "R", "ś": "S", "ṣ": "z",
        "ṃ": "M", "ḥ": "H", "ḻ": "L", "m̐": "~", "ē": "e", "ō": "o",
        "Ā": "A", "Ī": "I", "Ū": "U", "Ś": "S", "Ṣ": "z", "Ṇ": "R", "Ṭ": "w",
        "Ḍ": "q", "Ñ": "Y", "Ṅ": "N", "Ṛ": "f",
    }
    return "".join(table.get(ch, ch) for ch in text)


def slp1_to_iast(text: str) -> str:
    """SLP1 -> IAST."""
    table = {
        "A": "ā", "I": "ī", "U": "ū", "f": "ṛ", "F": "ṝ", "x": "ḷ", "X": "ḹ",
        "N": "ṅ", "Y": "ñ", "w": "ṭ", "q": "ḍ", "R": "ṇ", "S": "ś", "z": "ṣ",
        "M": "ṃ", "H": "ḥ", "L": "ḻ",
    }
    return "".join(table.get(ch, ch) for ch in text)


def detect_script(text: str) -> str:
    """Return 'devanagari', 'iast', 'slp1' or 'latin' for a string."""
    if any(_is_devanagari(ch) for ch in text):
        return "devanagari"
    if any(ch in "āīūṛṝṅñṭḍṇśṣṃḥḷ" for ch in text):
        return "iast"
    # SLP1 heuristics: capital letters in the middle of words + no Devanagari
    if re.search(r"[a-z][A-Z]", text) or re.search(r"[wqSNRz]", text):
        return "slp1"
    return "latin"


def to_devanagari(text: str) -> str:
    """Auto-detect the romanisation and convert to Devanagari."""
    script = detect_script(text)
    if script == "devanagari":
        return unicodedata.normalize("NFC", text)
    if script == "slp1":
        return iast_to_devanagari(slp1_to_iast(text))
    return iast_to_devanagari(text)


def normalize_devanagari(text: str) -> str:
    """Canonicalise a Devanagari string.

    * NFC composition
    * unify the two anusvāra spellings, ``।।`` -> ``॥``
    * drop zero-width joiners (invisible in Sanskrit)
    * collapse whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    text = text.replace("।।", "॥")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return text.strip()
