"""
Sanskrit nominal morphology.

Grammar training pairs are only trustworthy if their answers are *computed*
from a paradigm rather than typed by hand, so this module declines every noun
it is given across eight cases and three numbers and can be queried in reverse
(given a form, which case/number can it be?).

The endings below are the standard ones for the classes in use. Only classes we
can state confidently are implemented; irregular nouns carry an explicit table
instead of an ending set, and anything not in `NOUNS` is refused rather than
guessed.

    python3 -m scripts.sanskrit_kb.morph     # self-check against known forms
"""

from __future__ import annotations

from dataclasses import dataclass

CASES = ("प्रथमा", "द्वितीया", "तृतीया", "चतुर्थी", "पञ्चमी", "षष्ठी", "सप्तमी",
         "सम्बोधनम्")
NUMBERS = ("एकवचनम्", "द्विवचनम्", "बहुवचनम्")

CASE_LOCATIVE = {
    "प्रथमा": "प्रथमायाम्", "द्वितीया": "द्वितीयायाम्", "तृतीया": "तृतीयायाम्",
    "चतुर्थी": "चतुर्थ्याम्", "पञ्चमी": "पञ्चम्याम्", "षष्ठी": "षष्ठ्याम्",
    "सप्तमी": "सप्तम्याम्", "सम्बोधनम्": "सम्बोधने",
}
CASE_LOCATIVE_ALT = {          # "in the X case" phrased with a past participle
    "प्रथमा": "प्रथमायां विभक्तौ", "द्वितीया": "द्वितीयायां विभक्तौ",
    "तृतीया": "तृतीयायां विभक्तौ", "चतुर्थी": "चतुर्थ्यां विभक्तौ",
    "पञ्चमी": "पञ्चम्यां विभक्तौ", "षष्ठी": "षष्ठ्यां विभक्तौ",
    "सप्तमी": "सप्तम्यां विभक्तौ", "सम्बोधनम्": "सम्बोधने",
}
CASE_INSTRUMENTAL = {          # "with/by the X case"
    "प्रथमा": "प्रथमया", "द्वितीया": "द्वितीयया", "तृतीया": "तृतीयया",
    "चतुर्थी": "चतुर्थ्या", "पञ्चमी": "पञ्चम्या", "षष्ठी": "षष्ठ्या",
    "सप्तमी": "सप्तम्या", "सम्बोधनम्": "सम्बोधनेन",
}
NUMBER_LOCATIVE = {"एकवचनम्": "एकवचने", "द्विवचनम्": "द्विवचने",
                   "बहुवचनम्": "बहुवचने"}
GENDERS = ("पुल्लिङ्गम्", "स्त्रीलिङ्गम्", "नपुंसकलिङ्गम्")


# --------------------------------------------------------------- endings
# [case][number]; the string is appended to the base declared with the class.
ENDINGS: dict[str, list[list[str]]] = {
    # रामः  (base राम)
    "a_m": [
        ["ः", "ौ", "आः"], ["म्", "ौ", "आन्"], ["ेण", "आभ्याम्", "ऐः"],
        ["आय", "आभ्याम्", "ेभ्यः"], ["आत्", "आभ्याम्", "ेभ्यः"],
        ["अस्य", "अयोः", "आणाम्"], ["े", "अयोः", "ेषु"], ["", "ौ", "आः"],
    ],
    # फलम्  (base फल)
    "a_n": [
        ["म्", "े", "आनि"], ["म्", "े", "आनि"], ["ेन", "आभ्याम्", "ऐः"],
        ["आय", "आभ्याम्", "ेभ्यः"], ["आत्", "आभ्याम्", "ेभ्यः"],
        ["अस्य", "अयोः", "आनाम्"], ["े", "अयोः", "ेषु"], ["", "े", "आनि"],
    ],
    # सीता  (base सीता)
    "ā_f": [
        ["आ", "े", "आः"], ["आम्", "े", "आः"], ["या", "आभ्याम्", "आभिः"],
        ["आयै", "आभ्याम्", "आभ्यः"], ["आयाः", "आभ्याम्", "आभ्यः"],
        ["आयाः", "योः", "आनाम्"], ["आयाम्", "योः", "आसु"], ["े", "े", "आः"],
    ],
    # मुनि  (base मुनि)
    "i_m": [
        ["ः", "ी", "अयः"], ["म्", "ी", "ीन्"], ["ना", "भ्याम्", "भिः"],
        ["अये", "भ्याम्", "भ्यः"], ["ेः", "भ्याम्", "भ्यः"],
        ["ेः", "योः", "ीनाम्"], ["ौ", "योः", "षु"], ["े", "ी", "अयः"],
    ],
    # मति  (base मति)
    "i_f": [
        ["ः", "ी", "अयः"], ["म्", "ी", "ीः"], ["्या", "भ्याम्", "भिः"],
        ["अये", "भ्याम्", "भ्यः"], ["ेः", "भ्याम्", "भ्यः"],
        ["ेः", "योः", "ीनाम्"], ["ौ", "योः", "षु"], ["े", "ी", "अयः"],
    ],
    # नदी  (base नद्)
    "ī_f": [
        ["ी", "्यौ", "्यः"], ["ीम्", "्यौ", "ीः"], ["्या", "ीभ्याम्", "ीभिः"],
        ["्यै", "ीभ्याम्", "ीभ्यः"], ["्याः", "ीभ्याम्", "ीभ्यः"],
        ["्याः", "्योः", "ीनाम्"], ["्याम्", "्योः", "ीषु"],
        ["ि", "्यौ", "्यः"],
    ],
    # गुरु  (base गुरु)
    "u_m": [
        ["ः", "ू", "अवः"], ["म्", "ू", "ून्"], ["ना", "भ्याम्", "भिः"],
        ["अवे", "भ्याम्", "भ्यः"], ["ोः", "भ्याम्", "भ्यः"],
        ["ोः", "वोः", "ूनाम्"], ["ौ", "वोः", "षु"], ["ो", "ू", "अवः"],
    ],
    # मधु  (base मधु)
    "u_n": [
        ["ु", "ुनी", "ूनि"], ["ु", "ुनी", "ूनि"], ["ुना", "ुभ्याम्", "ुभिः"],
        ["ुने", "ुभ्याम्", "ुभ्यः"], ["ुनः", "ुभ्याम्", "ुभ्यः"],
        ["ुनः", "ुनोः", "ूनाम्"], ["ुनि", "ुनोः", "ुषु"],
        ["ु", "ुनी", "ूनि"],
    ],
}

# Bases that are not simply the citation form minus its ending.
BASE_OF = {
    "a_m": lambda lemma: lemma[:-1],        # रामः -> राम
    "a_n": lambda lemma: lemma[:-2],        # फलम् -> फल
    "ā_f": lambda lemma: lemma[:-1],        # सीता -> सीत
    "i_m": lambda lemma: lemma[:-1],        # मुनिः -> मुनि
    "i_f": lambda lemma: lemma[:-1],        # मतिः -> मति
    "ī_f": lambda lemma: lemma[:-1],        # नदी -> नद्
    "u_m": lambda lemma: lemma[:-1],        # गुरुः -> गुरु
    "u_n": lambda lemma: lemma[:-1],        # मधु -> मधु
}


@dataclass(frozen=True)
class Noun:
    lemma: str          # citation form, e.g. रामः
    stem: str           # the word as named in a question, e.g. राम
    cls: str
    gender: str
    gloss: str = ""
    natva: bool = False  # र/ष् stems take ण in the instrumental

    def base(self) -> str:
        return BASE_OF[self.cls](self.lemma)


VOWEL_MATRA = {"अ": "", "आ": "ा", "इ": "ि", "ई": "ी", "उ": "ु", "ऊ": "ू",
               "ऋ": "ृ", "ए": "े", "ऐ": "ै", "ओ": "ो", "औ": "ौ"}
MATRAS = set("ािीुूृेैोौ")
VIRAMA = "्"


def _apply(base: str, ending: str) -> str:
    """Join a base and an ending the way Devanagari actually writes the junction.

    An ending's initial vowel is not appended as an independent letter: it
    replaces the base's final vowel sign, or becomes the matra of its final
    consonant. The ending sets are written so that their initial vowel already
    carries the right grade (i-stems use -ay-, u-stems -av-), which is why a
    plain replacement is correct here.
    """
    if not ending:
        return base
    head = ending[0]
    tail = ending[1:]
    if head == VIRAMA:                           # ending starts with a virama
        if base and (base[-1] == VIRAMA or base[-1] in MATRAS):
            return base[:-1] + ending            # मति + ्या -> मत्या, नद् + ्या -> नद्या
        return base + ending
    if head in VOWEL_MATRA:                      # vowel-initial ending
        if base and base[-1] in (VIRAMA,) or (base and base[-1] in MATRAS):
            return base[:-1] + VOWEL_MATRA[head] + tail   # मुनि + अये -> मुनये
        return base + VOWEL_MATRA[head] + tail            # राम + आन् -> रामान्
    if head in MATRAS:                           # matra-initial ending
        if base and (base[-1] in MATRAS or base[-1] == VIRAMA):
            return base[:-1] + ending            # मुनि + ेः -> मुनेः
        return base + ending                     # राम + े -> रामे
    return base + ending                         # consonant-initial ending


def decline(noun: Noun) -> dict[tuple[str, str], str]:
    """{(case, number): form} for every cell of the paradigm."""
    table = ENDINGS[noun.cls]
    base = noun.base()
    out: dict[tuple[str, str], str] = {}
    for ci, case in enumerate(CASES):
        for ni, number in enumerate(NUMBERS):
            ending = table[ci][ni]
            if noun.natva and ending.startswith("ना"):
                ending = "णा"
            out[(case, number)] = _apply(base, ending)
    return out


def paradigm(noun: Noun) -> dict[str, dict[str, str]]:
    """{case: {number: form}} — friendlier for printing and reports."""
    flat = decline(noun)
    return {c: {n: flat[(c, n)] for n in NUMBERS} for c in CASES}


# ------------------------------------------------------- curated inventory
def _n(lemma: str, cls: str, gender: str, gloss: str, natva: bool = False) -> Noun:
    stem = BASE_OF[cls](lemma)
    return Noun(lemma=lemma, stem=stem, cls=cls, gender=gender, gloss=gloss,
                natva=natva)


NOUNS: list[Noun] = [
    # a-stem masculine
    _n("रामः", "a_m", "पुल्लिङ्गम्", "Rāma"),
    _n("बालकः", "a_m", "पुल्लिङ्गम्", "boy"),
    _n("वृक्षः", "a_m", "पुल्लिङ्गम्", "tree"),
    _n("गजः", "a_m", "पुल्लिङ्गम्", "elephant"),
    _n("अश्वः", "a_m", "पुल्लिङ्गम्", "horse"),
    _n("पर्वतः", "a_m", "पुल्लिङ्गम्", "mountain"),
    _n("सागरः", "a_m", "पुल्लिङ्गम्", "ocean"),
    _n("धर्मः", "a_m", "पुल्लिङ्गम्", "dharma"),
    _n("कृष्णः", "a_m", "पुल्लिङ्गम्", "Kṛṣṇa"),
    _n("शिष्यः", "a_m", "पुल्लिङ्गम्", "student"),
    _n("गुरुकुलः", "a_m", "पुल्लिङ्गम्", "gurukula"),
    _n("सूर्यः", "a_m", "पुल्लिङ्गम्", "sun"),
    _n("चन्द्रः", "a_m", "पुल्लिङ्गम्", "moon"),
    _n("लेखकः", "a_m", "पुल्लिङ्गम्", "writer"),
    _n("ग्रामः", "a_m", "पुल्लिङ्गम्", "village"),
    _n("नगरः", "a_m", "पुल्लिङ्गम्", "city"),
    # a-stem neuter
    _n("फलम्", "a_n", "नपुंसकलिङ्गम्", "fruit"),
    _n("पुस्तकम्", "a_n", "नपुंसकलिङ्गम्", "book"),
    _n("जलम्", "a_n", "नपुंसकलिङ्गम्", "water"),
    _n("कमलम्", "a_n", "नपुंसकलिङ्गम्", "lotus"),
    _n("गृहम्", "a_n", "नपुंसकलिङ्गम्", "house"),
    _n("वनम्", "a_n", "नपुंसकलिङ्गम्", "forest"),
    _n("ज्ञानम्", "a_n", "नपुंसकलिङ्गम्", "knowledge"),
    _n("सत्यम्", "a_n", "नपुंसकलिङ्गम्", "truth"),
    _n("अन्नम्", "a_n", "नपुंसकलिङ्गम्", "food"),
    _n("पुष्पम्", "a_n", "नपुंसकलिङ्गम्", "flower"),
    _n("गीतम्", "a_n", "नपुंसकलिङ्गम्", "song"),
    _n("मित्रम्", "a_n", "नपुंसकलिङ्गम्", "friend"),
    # ā-stem feminine
    _n("सीता", "ā_f", "स्त्रीलिङ्गम्", "Sītā"),
    _n("लता", "ā_f", "स्त्रीलिङ्गम्", "creeper"),
    _n("बाला", "ā_f", "स्त्रीलिङ्गम्", "girl"),
    _n("नौका", "ā_f", "स्त्रीलिङ्गम्", "boat"),
    _n("गङ्गा", "ā_f", "स्त्रीलिङ्गम्", "Gaṅgā"),
    _n("विद्या", "ā_f", "स्त्रीलिङ्गम्", "learning"),
    _n("भाषा", "ā_f", "स्त्रीलिङ्गम्", "language"),
    _n("कथा", "ā_f", "स्त्रीलिङ्गम्", "story"),
    _n("माता", "ā_f", "स्त्रीलिङ्गम्", "mother"),   # मातर्-like but regular in use
    _n("सेना", "ā_f", "स्त्रीलिङ्गम्", "army"),
    _n("पूजा", "ā_f", "स्त्रीलिङ्गम्", "worship"),
    _n("कन्या", "ā_f", "स्त्रीलिङ्गम्", "daughter"),
    # i-stem masculine
    _n("मुनिः", "i_m", "पुल्लिङ्गम्", "sage"),
    _n("कविः", "i_m", "पुल्लिङ्गम्", "poet"),
    _n("अग्निः", "i_m", "पुल्लिङ्गम्", "fire"),
    _n("गिरिः", "i_m", "पुल्लिङ्गम्", "mountain"),
    _n("रविः", "i_m", "पुल्लिङ्गम्", "sun"),
    _n("व्याधिः", "i_m", "पुल्लिङ्गम्", "disease"),
    # i-stem feminine
    _n("मतिः", "i_f", "स्त्रीलिङ्गम्", "intellect"),
    _n("बुद्धिः", "i_f", "स्त्रीलिङ्गम्", "understanding"),
    _n("शक्तिः", "i_f", "स्त्रीलिङ्गम्", "power"),
    _n("गतिः", "i_f", "स्त्रीलिङ्गम्", "gait"),
    _n("रीतिः", "i_f", "स्त्रीलिङ्गम्", "custom"),
    # ī-stem feminine
    _n("नदी", "ī_f", "स्त्रीलिङ्गम्", "river"),
    _n("नगरी", "ī_f", "स्त्रीलिङ्गम्", "city"),
    _n("सरस्वती", "ī_f", "स्त्रीलिङ्गम्", "Sarasvatī"),
    _n("रोटिका", "ā_f", "स्त्रीलिङ्गम्", "bread"),
    # u-stem
    _n("गुरुः", "u_m", "पुल्लिङ्गम्", "teacher", natva=True),
    _n("शिशुः", "u_m", "पुल्लिङ्गम्", "child"),
    _n("साधुः", "u_m", "पुल्लिङ्गम्", "virtuous person", natva=True),
    _n("भानुः", "u_m", "पुल्लिङ्गम्", "sun"),
    _n("विष्णुः", "u_m", "पुल्लिङ्गम्", "Viṣṇu", natva=True),
    _n("मधु", "u_n", "नपुंसकलिङ्गम्", "honey"),
]

NOUN_BY_LEMMA = {n.lemma: n for n in NOUNS}
# The stem named in questions ("राम इति पदम्") is the base, not the citation form.
NOUN_BY_STEM = {n.base(): n for n in NOUNS}


def forms_of(lemma: str) -> dict[tuple[str, str], str]:
    return decline(NOUN_BY_LEMMA[lemma])


def reverse_index(nouns: list[Noun] | None = None) -> dict[str, set[tuple[str, str]]]:
    """form -> every (case, number) it can express. Ambiguity is the point."""
    index: dict[str, set[tuple[str, str]]] = {}
    for noun in (nouns or NOUNS):
        for cell, form in decline(noun).items():
            index.setdefault(form, set()).add(cell)
    return index


def name_case(form: str, index: dict[str, set[tuple[str, str]]] | None = None
              ) -> tuple[str, str] | None:
    """The single (case, number) a form expresses, or None if ambiguous/unknown."""
    idx = index if index is not None else reverse_index()
    cells = idx.get(form)
    if not cells or len(cells) != 1:
        return None
    return next(iter(cells))


# ------------------------------------------------------------------ self-check
KNOWN_FORMS = {
    # (lemma, case, number): form — checked by hand against the standard paradigm
    ("रामः", "प्रथमा", "एकवचनम्"): "रामः",
    ("रामः", "द्वितीया", "बहुवचनम्"): "रामान्",
    ("रामः", "तृतीया", "एकवचनम्"): "रामेण",
    ("रामः", "षष्ठी", "बहुवचनम्"): "रामाणाम्",
    ("रामः", "सप्तमी", "द्विवचनम्"): "रामयोः",
    ("फलम्", "प्रथमा", "बहुवचनम्"): "फलानि",
    ("फलम्", "षष्ठी", "एकवचनम्"): "फलस्य",
    ("फलम्", "सप्तमी", "बहुवचनम्"): "फलेषु",
    ("सीता", "द्वितीया", "एकवचनम्"): "सीताम्",
    ("सीता", "तृतीया", "एकवचनम्"): "सीतया",
    ("सीता", "चतुर्थी", "एकवचनम्"): "सीतायै",
    ("सीता", "सप्तमी", "बहुवचनम्"): "सीतासु",
    ("नदी", "प्रथमा", "बहुवचनम्"): "नद्यः",
    ("नदी", "तृतीया", "एकवचनम्"): "नद्या",
    ("नदी", "षष्ठी", "बहुवचनम्"): "नदीनाम्",
    ("मुनिः", "चतुर्थी", "एकवचनम्"): "मुनये",
    ("मुनिः", "षष्ठी", "एकवचनम्"): "मुनेः",
    ("मुनिः", "सप्तमी", "बहुवचनम्"): "मुनिषु",
    ("मतिः", "तृतीया", "एकवचनम्"): "मत्या",
    ("गुरुः", "तृतीया", "एकवचनम्"): "गुरुणा",
    ("गुरुः", "षष्ठी", "एकवचनम्"): "गुरोः",
    ("गुरुः", "सप्तमी", "एकवचनम्"): "गुरौ",
    ("विष्णुः", "प्रथमा", "एकवचनम्"): "विष्णुः",
    ("मधु", "प्रथमा", "बहुवचनम्"): "मधूनि",
}


def self_check() -> int:
    flat = {n.lemma: decline(n) for n in NOUNS}
    bad = 0
    for (lemma, case, number), expected in KNOWN_FORMS.items():
        got = flat[lemma][(case, number)]
        if got != expected:
            print(f"  ✗ {lemma} {case} {number}: expected {expected}, got {got}")
            bad += 1
    print(f"  {len(KNOWN_FORMS) - bad}/{len(KNOWN_FORMS)} paradigms correct, "
          f"{len(NOUNS)} nouns, {len(reverse_index())} distinct forms")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(self_check())
