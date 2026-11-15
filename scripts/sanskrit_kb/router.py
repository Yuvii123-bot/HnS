"""
Grammar questions answered from the tables, not from the weights.

A 5.8 M-parameter model can memorise the paradigms it is drilled on and nothing
else: on 56 cells of ordinary nouns it was never trained on it scored 1/56, while
scoring 14/15 on the drilled ones. The tables already hold every form, so the
honest architecture is to answer from them and leave the model to converse.

Every route below is a question *shape*; the answer is looked up:

    राम इति शब्दस्य षष्ठी एकवचनं किम्?        -> रामस्य        (śabdapatha cell)
    गच्छति इति पदस्य धातुः कः?                -> गम्           (reverse tiṅ index)
    अश्वः इति शब्दस्य स्त्रीलिङ्गं किम्?       -> अश्वा         (liṅga pair)
    पठ् धातोः लट् प्रथमपुरुष एकवचनं किम्?      -> पठति         (tiṅ table)
    संस्कृते कति विभक्तयः?                    -> अष्टौ विभक्तयः (counted facts)

    python3 -m scripts.sanskrit_kb.router            # self-check on the benchmark
    python3 -m scripts.sanskrit_kb.router --text "…" # answer one question
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

try:                                     # repo layout
    from scripts.sanskrit_kb.lexicon import (CASES, CASE_INS, CASE_LOC, CELLS,
                                             LINGA, NUMBERS, NUM_LOC, PURUSHA,
                                             lexicon)
except ImportError:                      # bundled runtime (serve/runtime/sanskrit_kb)
    from .lexicon import (CASES, CASE_INS, CASE_LOC, CELLS, LINGA, NUMBERS,
                          NUM_LOC, PURUSHA, lexicon)

CASE_PATTERNS = {c: (c, CASE_LOC[c], CASE_INS[c]) for c in CASES}
NUM_PATTERN = {n: (n, NUM_LOC[n]) for n in NUMBERS}

#: counted facts about the Sanskrit grammatical system itself
COUNTS = [
    (("विभक्त",), "अष्टौ विभक्तयः।", "8 vibhaktis (prathamā … sambodhana)"),
    (("वचन",), "त्रीणि वचनानि।", "3 vacanas (eka, dvi, bahu)"),
    (("पुरुष",), "त्रयः पुरुषाः।", "3 puruṣas (prathama, madhyama, uttama)"),
    (("लिङ्ग",), "त्रीणि लिङ्गानि।", "3 liṅgas (puṃ, strī, napuṃsaka)"),
    (("काल", "लकार"), "त्रयः कालाः।", "3 kālas"),
    (("स्वर",), "त्रयः स्वराः।", "3 svaras (udātta, anudātta, svarita)"),
]
NUMBER_WORDS = {"एक": "एकम्", "द्वि": "द्वे", "त्रि": "त्रीणि", "चतुर्": "चत्वारि",
                "पञ्च": "पञ्च", "षष्": "षट्", "सप्त": "सप्त", "अष्ट": "अष्टौ",
                "नव": "नव", "दश": "दश"}

_QI = r"[?।]?\s*$"                       # a question ends with ? or ।
_STEM = r"(?P<stem>[^\s,।?]+)"
_CASE = r"(?P<case>" + "|".join(p for c in CASES for p in CASE_PATTERNS[c]) + ")"
_NUM = r"(?P<num>" + "|".join(p for n in NUMBERS for p in NUM_PATTERN[n]) + ")"
_ROOT = r"(?P<root>[^\s,।?]+)"


@dataclass
class Answer:
    text: str
    route: str
    evidence: str = ""
    confidence: str = "table"            # table | rule | none
    detail: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.text)


def _norm(text: str) -> str:
    """Whitespace-tidy and expand a final anusvāra to म् for matching only:
    the benchmark writes एकवचनं, the tables write एकवचनम्."""
    text = text.strip()
    text = text.replace("\u0902", "\u092e\u094d")
    return re.sub(r"\s+", " ", text)


def _case(pattern: str) -> str:
    for c, ps in CASE_PATTERNS.items():
        if pattern in ps:
            return c
    raise KeyError(pattern)


def _number(pattern: str) -> str:
    for n, ps in NUM_PATTERN.items():
        if pattern in ps:
            return n
    raise KeyError(pattern)


def _strip_stem(stem: str) -> str:
    """Question wording says "राम इति शब्दस्य" — the stem may carry a final visarga
    or an accusative ending from the quoted form (`रामम् इति पदस्य`)."""
    stem = stem.strip("'\"‘’“”")
    for suffix in (" इति", "िति"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.strip()


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
def answer_declension(text: str) -> Answer | None:
    """राम इति शब्दस्य षष्ठी एकवचनं किम्? / … षष्ठ्याम् एकवचने रूपं किम्?"""
    m = re.search(_STEM + r"\s+इति\s+(?:शब्दस्य|पदस्य|नाम्नः)\s+" + _CASE
                  + r"\s+" + _NUM, text)
    if not m:
        return None
    stem = _strip_stem(m.group("stem"))
    case, num = _case(m.group("case")), _number(m.group("num"))
    lex = lexicon()

    # the quoted word may be inflected (`अश्वः इति शब्दस्य`); take the stem from
    # the paradigm if the bare form is not a stem itself
    for cand in _stem_candidates(lex, stem):
        form = lex.cell(cand, case, num)
        if form:
            entry = lex.stem_entries(cand)[0]
            return Answer(f"{form}।", "subanta/cell",
                          f"{cand} ({LINGA.get(entry['linga'], '?')}) → {case} {num}",
                          "table", {"stem": cand, "case": case, "number": num,
                                    "linga": LINGA.get(entry["linga"], "")})
    return None


def _stem_candidates(lex, word: str) -> list[str]:
    """Turn whatever the user quoted into a stem the śabdapatha knows.

    `अश्वः इति शब्दस्य` quotes the nominative, `फलम्` the neuter nominative,
    `लता` is already a stem. Order matters: an inflected form is stripped before
    the reverse index is consulted, or `अश्वः` would resolve to whichever
    paradigm happens to contain that string first.
    """
    out = [word]
    for suffix in ("ः", "म्", "्", "ा", "ी"):
        if word.endswith(suffix) and len(word) > len(suffix):
            out.append(word[: -len(suffix)])
    # last resort: a paradigm whose cells contain the quoted form
    for other, entries in lex.stems.items():
        for entry in entries:
            if word in entry["cells"]:
                out.append(other)
                break
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def answer_feminine(text: str) -> Answer | None:
    """अश्वः इति शब्दस्य स्त्रीलिङ्गं किम्?"""
    m = re.search(_STEM + r"\s+इति\s+(?:शब्दस्य|पदस्य)\s+स्त्रीलिङ्ग", text)
    if not m:
        return None
    lex = lexicon()
    cands = _stem_candidates(lex, _strip_stem(m.group("stem")))
    # attested pairs first (अश्व → अश्वा is in the śabdapatha); a regular
    # derivation is only accepted when no attested pair exists (बालक → बालिका)
    for allow_rule in (False, True):
        for cand in cands:
            got = lex.feminine_of(cand, allow_rule=allow_rule)
            if got:
                cand_f, form, how = got
                return Answer(f"{form}।", "liṅga/feminine",
                              f"{cand} → {cand_f} ({how})",
                              "table" if how.startswith("attested") else "rule",
                              {"stem": cand, "feminine": cand_f})
    return None


def answer_dhatu(text: str) -> Answer | None:
    """गच्छति इति पदस्य धातुः कः?"""
    m = re.search(_STEM + r"\s+इति\s+(?:पदस्य|क्रियापदस्य)\s+धातुः", text)
    if not m:
        return None
    form = _strip_stem(m.group("form") if "form" in m.groupdict() else m.group("stem"))
    lex = lexicon()
    hits = lex.root_of_form(form)
    if not hits:
        return None
    root, rec = hits[0][0], hits[0][1]
    same = sorted({h[0] for h in hits})
    ev = f"{form} → {root} (gaṇa {rec['gana']}, {rec['artha']})"
    if len(same) > 1:
        ev += f"; also {', '.join(same[1:])}"
    return Answer(f"{root}।", "dhatu/reverse", ev, "table",
                  {"form": form, "root": root, "gana": rec["gana"],
                   "artha": rec["artha"], "alternatives": same[1:]})


def answer_tin(text: str) -> Answer | None:
    """पठ् धातोः लट् प्रथमपुरुष एकवचनं किम्?"""
    m = re.search(_ROOT + r"\s+धातोः\s+(?P<lakara>" + "|".join(
        ["लट्", "लिट्", "लुट्", "लृट्", "लोट्", "लङ्", "लिङ्", "लुङ्", "लृङ्"])
        + r")\s+(?P<purusha>" + "|".join(PURUSHA) + r")\s+" + _NUM, text)
    if not m:
        return None
    root = _strip_stem(m.group("root"))
    got = lexicon().tin(root, m.group("lakara"), m.group("purusha"), _number(m.group("num")))
    if not got:
        return None
    form, rec = got
    return Answer(f"{form}।", "tin/cell",
                  f"{root} (gaṇa {rec['gana']}) → {m.group('lakara')} "
                  f"{m.group('purusha')} {m.group('num')}", "table",
                  {"root": root, "lakara": m.group("lakara"),
                   "purusha": m.group("purusha"), "number": _number(m.group("num"))})


def answer_count(text: str) -> Answer | None:
    """संस्कृते कति विभक्तयः? — the counted facts about the system."""
    if "कति" not in text:
        return None
    for keys, reply, why in COUNTS:
        if any(k in text for k in keys):
            return Answer(reply, "meta/count", why, "table")
    m = re.search(r"([^\s]+)\s*(?:संख्या|सङ्ख्या)", text)
    if m:
        word = m.group(1)
        for stem, reply in NUMBER_WORDS.items():
            if stem in word:
                return Answer(f"{reply}।", "meta/number", f"{stem} → {reply}", "table")
    return None


def answer_linga_of(text: str) -> Answer | None:
    """राम इति शब्दस्य किं लिङ्गम्?"""
    m = re.search(_STEM + r"\s+इति\s+शब्दस्य\s+(?:किं|किम्)\s+लिङ्गम्", text)
    if not m:
        return None
    lex = lexicon()
    for cand in _stem_candidates(lex, _strip_stem(m.group("stem"))):
        linga = lex.linga_of(cand)
        if linga:
            return Answer(f"{linga}।", "liṅga/name", f"{cand} → {linga}", "table",
                          {"stem": cand, "linga": linga})
    return None


def answer_case_of(text: str) -> Answer | None:
    """रामाय इति पदं कस्यां विभक्तौ?  (recognition — the old phase-2 shape)"""
    m = re.search(_STEM + r"\s+इति\s+(?:पदं|शब्दः)\s+कस्यां\s+विभक्तौ", text)
    if not m:
        return None
    form = _strip_stem(m.group("stem"))
    lex = lexicon()
    for stem, entries in lex.stems.items():
        for entry in entries:
            if form in entry["cells"]:
                i = entry["cells"].index(form)
                case, num = CELLS[i]
                return Answer(f"{case} विभक्तौ, {num}।", "subanta/reverse",
                              f"{form} = {stem} {case} {num}", "table",
                              {"form": form, "stem": stem, "case": case, "number": num})
    return None


ROUTES = (answer_declension, answer_tin, answer_dhatu, answer_feminine,
          answer_linga_of, answer_case_of, answer_count)


def answer(text: str) -> Answer | None:
    """Best route for a question, or None when no table applies."""
    text = _norm(text)
    for route in ROUTES:
        got = route(text)
        if got:
            return got
    return None


BENCHMARK = [
    ("q038", "राम इति शब्दस्य चतुर्थी एकवचनं किम्?", ["रामाय"]),
    ("q039", "राम इति शब्दस्य तृतीया एकवचनं किम्?", ["रामेण"]),
    ("q040", "राम इति शब्दस्य षष्ठी एकवचनं किम्?", ["रामस्य"]),
    ("q041", "लता इति शब्दस्य चतुर्थी एकवचनं किम्?", ["लतायै"]),
    ("q042", "फलम् इति शब्दस्य प्रथमा बहुवचनं किम्?", ["फलानि"]),
    ("q043", "गच्छति इति पदस्य धातुः कः?", ["गम्"]),
    ("q044", "पठति इति पदस्य धातुः कः?", ["पठ्"]),
    ("q045", "करोति इति पदस्य धातुः कः?", ["कृ"]),
    ("q046", "अश्वः इति शब्दस्य स्त्रीलिङ्गं किम्?", ["अश्वा"]),
    ("q047", "बालकः इति शब्दस्य स्त्रीलिङ्गं किम्?", ["बालिका"]),
    ("q048", "पठ् धातोः लट् प्रथमपुरुष एकवचनं किम्?", ["पठति"]),
    ("q049", "संस्कृते कति विभक्तयः?", ["अष्टौ", "अष्ट"]),
    ("q050", "संस्कृते कति वचनानि?", ["त्रीणि", "त्रि"]),
    ("q051", "संस्कृते कति पुरुषाः?", ["त्रयः", "त्रि"]),
    ("q052", "संस्कृते कति लिङ्गानि?", ["त्रीणि", "त्रि"]),
]


def _self_check() -> int:
    import json
    from pathlib import Path
    missing = 0
    print("benchmark (15 held-out grammar questions)")
    for qid, prompt, accept in BENCHMARK:
        got = answer(prompt)
        text = got.text if got else ""
        ok = any(a in text for a in accept)
        missing += not ok
        print(f"  {'ok ' if ok else 'MISS'} {qid}  {prompt:<44} {text or '—':<22} {got.route if got else ''}")
    # and the same routes on words that are in no benchmark
    probes = [
        ("नर इति शब्दस्य षष्ठी बहुवचनं किम्?", "नराणाम्"),
        ("गज इति शब्दस्य सप्तमी एकवचनं किम्?", "गजे"),
        ("मुनि इति शब्दस्य द्वितीया बहुवचनं किम्?", "मुनीन्"),
        ("सीता इति शब्दस्य तृतीया एकवचनं किम्?", "सीतया"),
    ]
    print("unseen nouns (not in the benchmark, not in any drill)")
    for prompt, want in probes:
        got = answer(prompt)
        text = got.text if got else ""
        ok = want in text
        missing += not ok
        print(f"  {'ok ' if ok else 'MISS'} {prompt:<44} {text or '—'}")
    return missing


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--text":
        a = answer(" ".join(sys.argv[2:]))
        print(f"{a.text}  [{a.route}] {a.evidence}" if a else "no route matched")
    else:
        raise SystemExit(_self_check())
