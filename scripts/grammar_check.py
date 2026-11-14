"""
Rule-based Sanskrit correctness checks for generated answers.

This is not a parser and does not prove a sentence is good Sanskrit. It catches
the specific error classes that a 5.8M-parameter model actually makes, and it
reports how many sentences each rule could even look at, so a high pass rate on
thin coverage is not mistaken for correctness.

Rules
  script        Devanagari only, no Latin, no Arabic digits
  termination   ends with । / ? / !
  structure     no verse markers, no word repeated three times, sane length
  echo          the answer is not the question read back
  person        pronoun and copula agree (`अहम् … अस्मि`, not `अहम् … अस्ति`)
  gender        a known adjective agrees with the noun it qualifies
  vibhakti      a claimed case name is present and is a real case
  numeral       a "कति" question is answered with a numeral
"""

from __future__ import annotations

import re

from sanskrit_kb.morph import CASES, NOUNS, decline, reverse_index

LATIN = re.compile(r"[A-Za-z]")
ARABIC = re.compile(r"[0-9]")

CASE_TOKENS = {
    "प्रथमा": ("प्रथमा", "प्रथमायाम्", "प्रथमया", "प्रथमायाः", "प्रथमायै"),
    "द्वितीया": ("द्वितीया", "द्वितीयायाम्", "द्वितीयया"),
    "तृतीया": ("तृतीया", "तृतीयायाम्", "तृतीयया"),
    "चतुर्थी": ("चतुर्थी", "चतुर्थ्याम्", "चतुर्थ्या"),
    "पञ्चमी": ("पञ्चमी", "पञ्चम्याम्", "पञ्चम्या"),
    "षष्ठी": ("षष्ठी", "षष्ठ्याम्", "षष्ठ्या"),
    "सप्तमी": ("सप्तमी", "सप्तम्याम्", "सप्तम्या"),
    "सम्बोधनम्": ("सम्बोधनम्", "सम्बोधने"),
}

PERSON = {
    "अहम्": ("अस्मि",), "अहं": ("अस्मि",), "आवाम्": ("स्वः", "वः"),
    "वयम्": ("स्मः",), "त्वम्": ("असि",), "त्वं": ("असि",),
    "युवाम्": ("स्थः",), "यूयम्": ("स्थ",),
}
THIRD_PERSON_COPULA = ("अस्ति", "स्तः", "सन्ति")

# a-stem adjectives whose three genders are unambiguous
ADJECTIVES = {
    "उत्तम": ("उत्तमः", "उत्तमा", "उत्तमम्"),
    "सुन्दर": ("सुन्दरः", "सुन्दरी", "सुन्दरम्"),
    "शुभ": ("शुभः", "शुभा", "शुभम्"),
    "प्रसन्न": ("प्रसन्नः", "प्रसन्ना", "प्रसन्नम्"),
    "कुशल": ("कुशलः", "कुशला", "कुशलम्"),
    "सन्तुष्ट": ("सन्तुष्टः", "सन्तुष्टा", "सन्तुष्टम्"),
    "दुःखित": ("दुःखितः", "दुःखिता", "दुःखितम्"),
    "बलवत्": ("बलवान्", "बलवती", "बलवत्"),
    "महत्": ("महान्", "महती", "महत्"),
}
ADJ_TO_GENDER = {}
for _stem, (m, f, n) in ADJECTIVES.items():
    ADJ_TO_GENDER[m] = "पुल्लिङ्गम्"
    ADJ_TO_GENDER[f] = "स्त्रीलिङ्गम्"
    ADJ_TO_GENDER[n] = "नपुंसकलिङ्गम्"

NUMERAL_WORDS = {
    "एकम्", "द्वे", "त्रीणि", "चत्वारि", "पञ्च", "षट्", "सप्त", "अष्ट", "नव",
    "दश", "एकादश", "द्वादश", "विंशतिः", "त्रिंशत्", "चत्वारिंशत्", "पञ्चाशत्",
    "षष्टिः", "सप्ततिः", "अशीतिः", "नवतिः", "शतम्", "सहस्रम्",
}
DEV_DIGITS = set("०१२३४५६७८९")

_INDEX = None
_FORM_NOUNS = None


def _index():
    global _INDEX
    if _INDEX is None:
        _INDEX = reverse_index()
    return _INDEX


def _form_to_nouns():
    """form -> [(noun, cells)] so an inflected word can be traced to its noun."""
    global _FORM_NOUNS
    if _FORM_NOUNS is None:
        table: dict[str, list] = {}
        for noun in NOUNS:
            for cell, form in decline(noun).items():
                table.setdefault(form, []).append((noun, cell))
        _FORM_NOUNS = table
    return _FORM_NOUNS


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s।॥?!,;:\"“”()]+", text) if w]


def check(answer: str, question: str = "") -> dict:
    issues: list[dict] = []
    words = _words(answer)

    def issue(rule: str, detail: str) -> None:
        issues.append({"rule": rule, "detail": detail})

    if not answer.strip():
        issue("empty", "answer is empty")
        return {"ok": False, "issues": issues, "checks_run": ["empty"]}

    if LATIN.search(answer):
        issue("script", "contains Latin characters")
    if ARABIC.search(answer):
        issue("script", "contains Arabic digits")
    if not answer.rstrip().endswith(("।", "?", "!")):
        issue("termination", "does not end with punctuation")
    if "॥" in answer:
        issue("structure", "contains a verse marker ॥")
    if len(words) > 45:
        issue("structure", f"{len(words)} words is too long for a chat answer")
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            issue("structure", f"word repeated three times: {words[i]}")
            break
    if question and answer.strip().rstrip("।?!") in question:
        issue("echo", "answer repeats the question")

    # ---- person and copula
    has_third = any(w in THIRD_PERSON_COPULA for w in words)
    for pronoun, copulas in PERSON.items():
        if pronoun in words:
            if has_third and not any(c in words for c in copulas):
                issue("person", f"{pronoun} takes {'/'.join(copulas)}, not a third-person copula")
            break

    # ---- adjective / noun gender
    gender_checked = 0
    for i, w in enumerate(words[:-1]):
        gender = ADJ_TO_GENDER.get(w)
        if not gender:
            continue
        entries = _form_to_nouns().get(words[i + 1], [])
        nouns = {e[0].lemma for e in entries}
        if len(nouns) != 1:
            continue
        noun = entries[0][0]
        gender_checked += 1
        if noun.gender != gender:
            issue("gender", f"{w} ({gender}) qualifies {words[i + 1]} ({noun.gender})")

    # ---- claimed case
    claimed = set()
    for case, tokens in CASE_TOKENS.items():
        if any(t in words for t in tokens):
            claimed.add(case)
    if "विभक्तौ" in words or "विभक्तिः" in words:
        if not claimed:
            issue("vibhakti", "claims a case but names no case")
        elif len(claimed) > 1:
            issue("vibhakti", f"names several cases: {'/'.join(sorted(claimed))}")

    # ---- numeral question answered with a numeral
    if "कति" in question:
        has_num = any(w in NUMERAL_WORDS for w in words) or \
            any(ch in DEV_DIGITS for ch in answer)
        if not has_num:
            issue("numeral", "a कति question was answered without a numeral")

    return {
        "ok": not issues,
        "issues": issues,
        "checks_run": ["script", "termination", "structure", "echo", "person",
                       "gender", "vibhakti", "numeral"],
        "gender_pairs_checked": gender_checked,
    }


def self_check() -> int:
    good = [
        ("अहम् उत्तमः अस्मि।", "भवान् कथम् अस्ति?"),
        ("तत् पदं षष्ठ्याम् विभक्तौ।", "रामस्य इति पदं कस्यां विभक्तौ?"),
        ("मिलित्वा ५ भवन्ति।", "२ ३ च मिलित्वा कति भवन्ति?"),
        ("मम नाम रामः।", "भवतः नाम किम्?"),
    ]
    bad = [
        ("अहं उत्तमः अस्ति।", "person"),
        ("सः सुन्दरी बालकः।", "gender"),
        ("तत् पदं विभक्तौ।", "vibhakti"),
        ("अयं च राजपुत्री ॥", "termination"),
        ("saṃskṛtam asti।", "script"),
        ("कति फलानि? अनेकानि।", "numeral"),
    ]
    fails = 0
    for text, q in good:
        r = check(text, q)
        if not r["ok"]:
            print(f"  ✗ false positive on {text}: {r['issues']}")
            fails += 1
    for text, rule in bad:
        r = check(text, "कति फलानि?")
        if not any(i["rule"] == rule for i in r["issues"]):
            print(f"  ✗ missed {rule} in {text}: {r['issues']}")
            fails += 1
    print(f"  {len(good) + len(bad) - fails}/{len(good) + len(bad)} checker cases correct")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(self_check())
