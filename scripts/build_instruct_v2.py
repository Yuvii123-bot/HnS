#!/usr/bin/env python3
"""
Phase-2 Sanskrit instruction dataset.

Sources
  1. the curated facts from phase 1 (`scripts/build_instruct_data.py`)
  2. ~400 new curated facts (`scripts/sanskrit_kb/facts.py`)
  3. grammar pairs *computed* from the declension engine in
     `scripts/sanskrit_kb/morph.py` — these are correct by construction
  4. scripted multi-turn dialogues (`scripts/sanskrit_kb/dialogues.py`)
  5. numerals and arithmetic, in Devanagari with Sanskrit number words

Held out for evaluation and never trained on: a set of question frames, a
sample of facts, and a set of nouns (their whole paradigms).

    python3 scripts/build_instruct_v2.py --out data/instruct_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_instruct_data as phase1            # noqa: E402
from sanskrit_kb import dialogues as kb_dialogues   # noqa: E402
from sanskrit_kb import facts as kb_facts       # noqa: E402
from sanskrit_kb.morph import (CASES, CASE_INSTRUMENTAL, CASE_LOCATIVE,
                               GENDERS, NOUNS, NUMBERS, NOUN_BY_LEMMA,
                               NUMBER_LOCATIVE, decline, reverse_index)

DEV_DIGITS = "०१२३४५६७८९"


def dev(n: int) -> str:
    return "".join(DEV_DIGITS[int(d)] for d in str(n))


NUM_WORDS = [
    "एकम्", "द्वे", "त्रीणि", "चत्वारि", "पञ्च", "षट्", "सप्त", "अष्ट", "नव",
    "दश", "एकादश", "द्वादश", "त्रयोदश", "चतुर्दश", "पञ्चदश", "षोडश", "सप्तदश",
    "अष्टादश", "नवदश", "विंशतिः", "एकविंशतिः", "द्वाविंशतिः", "त्रयोविंशतिः",
    "चतुर्विंशतिः", "पञ्चविंशतिः", "षड्विंशतिः", "सप्तविंशतिः", "अष्टाविंशतिः",
    "नवविंशतिः", "त्रिंशत्", "एकत्रिंशत्", "द्वात्रिंशत्", "त्रयस्त्रिंशत्",
    "चतुस्त्रिंशत्", "पञ्चत्रिंशत्", "षट्त्रिंशत्", "सप्तत्रिंशत्",
    "अष्टात्रिंशत्", "नवत्रिंशत्", "चत्वारिंशत्", "एकचत्वारिंशत्",
    "द्विचत्वारिंशत्", "त्रिचत्वारिंशत्", "चतुश्चत्वारिंशत्",
    "पञ्चचत्वारिंशत्", "षट्चत्वारिंशत्", "सप्तचत्वारिंशत्",
    "अष्टचत्वारिंशत्", "नवचत्वारिंशत्", "पञ्चाशत्", "एकपञ्चाशत्",
    "द्विपञ्चाशत्", "त्रिपञ्चाशत्", "चतुःपञ्चाशत्", "पञ्चपञ्चाशत्",
    "षट्पञ्चाशत्", "सप्तपञ्चाशत्", "अष्टपञ्चाशत्", "नवपञ्चाशत्", "षष्टिः",
    "एकषष्टिः", "द्विषष्टिः", "त्रिषष्टिः", "चतुःषष्टिः", "पञ्चषष्टिः",
    "षट्षष्टिः", "सप्तषष्टिः", "अष्टषष्टिः", "नवषष्टिः", "सप्ततिः",
    "एकसप्ततिः", "द्विसप्ततिः", "त्रिसप्ततिः", "चतुःसप्ततिः",
    "पञ्चसप्ततिः", "षट्सप्ततिः", "सप्तसप्ततिः", "अष्टसप्ततिः",
    "नवसप्ततिः", "अशीतिः", "एकाशीतिः", "द्व्यशीतिः", "त्र्यशीतिः",
    "चतुरशीतिः", "पञ्चाशीतिः", "षडशीतिः", "सप्ताशीतिः", "अष्टाशीतिः",
    "नवाशीतिः", "नवतिः", "एकनवतिः", "द्विनवतिः", "त्रिनवतिः", "चतुर्नवतिः",
    "पञ्चनवतिः", "षण्णवतिः", "सप्तनवतिः", "अष्टनवतिः", "नवनवतिः", "शतम्",
]

# ---------------------------------------------------------------- question frames
# Two of these are reserved for evaluation and never appear in training.
FRAMES_TRAIN = [
    "{q}",
    "कृपया वदतु। {q}",
    "भवान् कृपया वदेत् — {q}",
    "मम मित्रं पृच्छति — {q}",
    "एकं प्रश्नं पृच्छामि — {q}",
]
FRAMES_EVAL = [
    "अध्यापकः पृच्छति — {q}",
    "शिष्यः पृच्छति — {q}",
]

# ---------------------------------------------------------------- evaluation hold-outs
# Nouns whose every form is absent from the fact base, so holding out their
# paradigms leaves nothing of them in training (tests/test_phase2.py enforces it).
HELD_OUT_NOUNS = {"गजः", "गुरुकुलः", "नौका", "सेना", "गिरिः", "रविः", "मधु"}
TRAIN_NOUNS = [n for n in NOUNS if n.lemma not in HELD_OUT_NOUNS]

EVAL_FACT_RATE = 0.10          # facts kept out of training entirely
# A held-out fact is only a fair question if its answer is determinate. "Where
# do you live?" has no correct answer from the text alone, so personal
# categories are excluded from the unseen-fact subset (they stay in paraphrase).
OBJECTIVE_CATEGORIES = {"dharma", "world", "glossary", "reasoning"}


def fact_hash(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def is_eval_fact(category: str, question: str) -> bool:
    return fact_hash(category + "|" + question) < EVAL_FACT_RATE


# ---------------------------------------------------------------- rows
class Builder:
    def __init__(self) -> None:
        self.slots: list[tuple] = []
        self.rows: list[str] = []
        self.cats: dict[str, str] = {}
        self.seen: set[str] = set()
        self.per_cat: dict[str, int] = {}
        self.eval: list[dict] = []

    def add(self, category: str, text: str) -> bool:
        text = " ".join(text.split())
        if text in self.seen:
            return False
        self.seen.add(text)
        self.rows.append(text)
        self.cats[text] = category
        self.per_cat[category] = self.per_cat.get(category, 0) + 1
        return True

    def pair(self, category: str, question: str, answer: str) -> bool:
        return self.add(category, f"<user> {question} <assistant> {answer}")


# ---------------------------------------------------------------- facts
def collect_facts() -> list[tuple[str, str, str]]:
    """(category, question, answer) from both phases, de-duplicated on the pair."""
    out: list[tuple[str, str, str]] = []
    for category, _fid, q, a in phase1.FACTS:
        out.append((category, q, a))
    for name, group in (("dharma", kb_facts.DHARMA), ("world", kb_facts.WORLD),
                        ("glossary", kb_facts.GLOSSARY),
                        ("conversation", kb_facts.CONVERSATION),
                        ("reasoning", kb_facts.REASONING)):
        for q, a in group:
            out.append((name, q, a))
    seen = set()
    unique = []
    for item in out:
        key = (item[1], item[2])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def render_facts(b: Builder) -> tuple[int, int]:
    facts = collect_facts()
    trained = held = 0
    for category, q, a in facts:
        if is_eval_fact(category, q):
            held += 1
            continue
        trained += 1
        for frame in FRAMES_TRAIN:
            b.pair(category, frame.format(q=q), a)
    return trained, held


def eval_from_facts(b: Builder) -> None:
    facts = collect_facts()
    rng = random.Random(7)
    held = [f for f in facts
            if is_eval_fact(f[0], f[1]) and f[0] in OBJECTIVE_CATEGORIES]
    known = [f for f in facts if not is_eval_fact(f[0], f[1])]
    sample = known if len(known) <= 520 else rng.sample(known, 520)
    for i, (category, q, a) in enumerate(held + sample):
        held_out_fact = is_eval_fact(category, q)
        frame = FRAMES_EVAL[i % len(FRAMES_EVAL)]
        item = {
            "id": f"fact{i:04d}",
            "category": category,
            "kind": "unseen-fact" if held_out_fact else "paraphrase",
            "prompt": frame.format(q=q),
            "reference": a,
            "expected_type": "sentence",
            "history": [],
        }
        b.eval.append(item)
    # a second pass with a different held-out frame on a sample of facts
    for i, (category, q, a) in enumerate(rng.sample(facts, 160)):
        frame = FRAMES_EVAL[(i + 1) % len(FRAMES_EVAL)]
        b.eval.append({
            "id": f"factx{i:04d}",
            "category": category,
            "kind": "paraphrase-2",
            "prompt": frame.format(q=q),
            "reference": a,
            "expected_type": "sentence",
            "history": [],
        })


# ---------------------------------------------------------------- grammar
def render_grammar(b: Builder) -> None:
    for noun in TRAIN_NOUNS:
        table = decline(noun)
        by_form: dict[str, list[tuple[str, str]]] = {}
        for cell, form in table.items():
            by_form.setdefault(form, []).append(cell)
        for (case, number), form in table.items():
            cells = by_form[form]
            for frame in FRAMES_TRAIN[:2]:
                if len(cells) == 1:                     # form names its case
                    b.pair("grammar", frame.format(q=f"{form} इति पदं कस्यां विभक्तौ?"),
                           f"तत् पदं {CASE_LOCATIVE[case]} विभक्तौ।")
                if len({c[1] for c in cells}) == 1:     # form names its number
                    b.pair("grammar", frame.format(q=f"{form} इति पदं किम् वचनम्?"),
                           f"{form} इति पदं {number}।")
            # production: case+number -> form, asked three ways
            b.pair("grammar",
                   f"{noun.stem} इति पदस्य {CASE_INSTRUMENTAL[case]} विभक्त्या "
                   f"{NUMBER_LOCATIVE[number]} किम् रूपम्?",
                   f"{noun.stem} इति पदस्य रूपं {form}।")
            b.pair("grammar",
                   f"{CASE_LOCATIVE[case]} विभक्तौ {NUMBER_LOCATIVE[number]} "
                   f"{noun.stem} इति पदं कथं भवति?",
                   f"{CASE_LOCATIVE[case]} विभक्तौ {NUMBER_LOCATIVE[number]} "
                   f"{noun.stem} इति पदं {form} भवति।")
            b.slots.append((noun.stem, case, number, form))
            # reverse recognition
            if len(cells) == 1:
                b.pair("grammar",
                       f"{CASE_LOCATIVE[case]} विभक्तौ {NUMBER_LOCATIVE[number]} "
                       f"{noun.stem} इति पदस्य रूपं किम्?",
                       f"{noun.stem} इति पदस्य रूपं {form}।")
        for form in (table[("प्रथमा", "एकवचनम्")], table[("प्रथमा", "बहुवचनम्")],
                     table[("द्वितीया", "एकवचनम्")]):
            b.pair("grammar", f"{form} इति पदं किम् लिङ्गम्?",
                   f"{form} इति पदं {noun.gender}।")


def render_grammar_cloze(b: Builder) -> None:
    """Use the paradigms in a sentence, where the case has a job to do."""
    TEMPLATES = {
        "तृतीया": "अहं {form} सह गच्छामि।",
        "चतुर्थी": "अहं {form} फलं ददामि।",
        "षष्ठी": "इदं {form} गृहम्।",
        "सप्तमी": "अहं {form} वसामि।",
    }
    ASK = {
        "तृतीया": "अहं {stem} इति पदेन सह गच्छामि। वाक्ये किं रूपं युक्तम्?",
        "चतुर्थी": "अहं {stem} इति पदाय फलं ददामि। वाक्ये किं रूपं युक्तम्?",
        "षष्ठी": "इदं {stem} इति पदस्य गृहम्। वाक्ये किं रूपं युक्तम्?",
        "सप्तमी": "अहं {stem} इति पदे वसामि। वाक्ये किं रूपं युक्तम्?",
    }
    for noun in TRAIN_NOUNS:
        table = decline(noun)
        for case, frame in TEMPLATES.items():
            for number in ("एकवचनम्", "बहुवचनम्"):
                form = table[(case, number)]
                if form == table[("प्रथमा", number)]:
                    continue                      # ambiguous with the citation form
                b.pair("grammar",
                       ASK[case].format(stem=noun.stem) +
                       f" ({NUMBER_LOCATIVE[number]})",
                       f"युक्तं रूपं {form}।")


def eval_grammar(b: Builder) -> None:
    for noun in NOUNS:
        if noun.lemma not in HELD_OUT_NOUNS:
            continue
        table = decline(noun)
        for (case, number), form in table.items():
            b.eval.append({
                "id": f"gram_{noun.stem}_{case}_{number}",
                "category": "grammar",
                "kind": "grammar-unseen-noun",
                "prompt": f"{form} इति पदं कस्यां विभक्तौ?",
                "reference": f"तत् पदं {CASE_LOCATIVE[case]} विभक्तौ।",
                "expected_type": "vibhakti",
                "history": [],
            })
        for case in ("तृतीया", "षष्ठी"):
            for number in NUMBERS:
                form = table[(case, number)]
                b.eval.append({
                    "id": f"gramprod_{noun.stem}_{case}_{number}",
                    "category": "grammar",
                    "kind": "grammar-production",
                    "prompt": (f"{noun.stem} इति पदस्य {CASE_INSTRUMENTAL[case]} "
                               f"विभक्त्या {NUMBER_LOCATIVE[number]} किम् रूपम्?"),
                    "reference": f"{noun.stem} इति पदस्य रूपं {form}।",
                    "expected_type": "form",
                    "history": [],
                })


# ---------------------------------------------------------------- numbers
def render_numbers(b: Builder, rng: random.Random) -> None:
    for n in range(1, 101):
        word = NUM_WORDS[n - 1]
        b.pair("numbers", f"सङ्ख्या {dev(n)} संस्कृतेन का?",
               f"सङ्ख्या {dev(n)} संस्कृतेन {word}।")
        b.pair("numbers", f"{word} इति सङ्ख्या अङ्केन का?",
               f"{word} इति सङ्ख्या अङ्केन {dev(n)}।")
        b.pair("numbers", f"{dev(n)} इति अङ्कस्य संस्कृतशब्दः कः?",
               f"{dev(n)} इति अङ्कस्य संस्कृतशब्दः {word}।")
        b.pair("numbers", f"{dev(n)} इति अङ्कं संस्कृतेन लिखतु।",
               f"{word} इति लिखामि।")
        b.pair("numbers", f"कः शब्दः {dev(n)} इति अङ्कं बोधयति?",
               f"{word} इति शब्दः {dev(n)} इति अङ्कं बोधयति।")
        b.pair("numbers", f"गृहे {dev(n)} जनाः सन्ति। संस्कृतेन कथं वदामि?",
               f"गृहे {word} जनाः सन्ति इति वदामि।")
    for _ in range(240):
        a, c = rng.randint(1, 20), rng.randint(1, 20)
        total = a + c
        b.pair("numbers",
               f"{dev(a)} {dev(c)} च मिलित्वा कति भवन्ति?",
               f"मिलित्वा {dev(total)} ({NUM_WORDS[total - 1]}) भवन्ति।")
    for _ in range(120):
        total = rng.randint(3, 20)
        part = rng.randint(1, total - 1)
        b.pair("numbers",
               f"{dev(total)} फलानि सन्ति। {dev(part)} खादितानि। कति अवशिष्टानि?",
               f"अवशिष्टानि {dev(total - part)} फलानि।")


def eval_numbers(b: Builder) -> None:
    for n in (12, 27, 34, 45, 58, 63, 76, 89, 96, 100):
        b.eval.append({
            "id": f"num_{n}",
            "category": "numbers",
            "kind": "number-word",
            "prompt": f"सङ्ख्या {dev(n)} संस्कृतेन का?",
            "reference": f"सङ्ख्या {dev(n)} संस्कृतेन {NUM_WORDS[n - 1]}।",
            "expected_type": "numeral",
            "history": [],
        })
    for a, c in ((7, 8), (9, 11), (13, 6), (15, 14), (18, 19)):
        b.eval.append({
            "id": f"sum_{a}_{c}",
            "category": "numbers",
            "kind": "arithmetic",
            "prompt": f"{dev(a)} {dev(c)} च मिलित्वा कति भवन्ति?",
            "reference": f"मिलित्वा {dev(a + c)} ({NUM_WORDS[a + c - 1]}) भवन्ति।",
            "expected_type": "numeral",
            "history": [],
        })


# ---------------------------------------------------------------- dialogues
def render_dialogues(b: Builder) -> None:
    for d in kb_dialogues.generate(max_per_script=12):
        ex = d["exchanges"]
        b.add(d["topic"], kb_dialogues.to_line(ex))
        # every prefix of the dialogue is its own training example: the model
        # must answer the *next* turn after any amount of prior context
        for cut in range(2, len(ex) + 1):
            b.add(d["topic"], kb_dialogues.to_line(ex[:cut]))
        for q, a in ex[: d["standalone"]]:
            b.pair(d["topic"], q, a)


def eval_dialogues(b: Builder) -> None:
    """Multi-turn prompts: the question only makes sense with its history."""
    # Every answer below is recoverable from the history: this measures whether
    # the model reads the conversation, not whether it can guess.
    pairs = [
        ([("भवतः नाम किम्?", "मम नाम रामः।")],
         "किं भवतः नाम रामः?"),
        ([("भवान् कुत्र वसति?", "अहं काश्यां निवसामि।")],
         "किं भवान् काश्यां निवसति?"),
        ([("कः भवतः गुरुः?", "मम गुरुः सोमशर्मा।")],
         "किं सोमशर्मा भवतः गुरुः?"),
        ([("किं भवान् संस्कृतं पठति?", "आम्, अहं संस्कृतं पठामि।"),
          ("कां भाषां पठति?", "अहं संस्कृतं पठामि।")],
         "किं भवान् संस्कृतं पठति?"),
        ([("भवतः ग्रामः कः?", "मम ग्रामः काशीसमीपे।")],
         "किं भवतः ग्रामः काशीसमीपे?"),
        ([("अद्य भवतः दिनं कथम्?", "अद्य मम दिनम् उत्तमम्।")],
         "किं भवतः दिनम् उत्तमम्?"),
        ([("किं भवान् जलम् अपिबत्?", "आम्, अहं जलम् अपिबम्।")],
         "किम् अहं जलम् अपिबम्?"),
        ([("भवती कुत्र वसति?", "अहं मथुरायां निवसामि।")],
         "किं भवती मथुरायां निवसति?"),
        ([("भवतः प्रियं भोजनम् किम्?", "मम प्रियं भोजनं भातम्।")],
         "किं भवतः प्रियं भोजनं भातम्?"),
        ([("भवान् कदा उत्तिष्ठति?", "अहं प्रातः पञ्चवादने उत्तिष्ठामि।")],
         "किं भवान् प्रातः उत्तिष्ठति?"),
        ([("कः भवतः गुरुः?", "मम गुरुः सोमशर्मा।")],
         "कः सोमशर्मा?"),
        ([("भवान् कुत्र गच्छति?", "अहं विद्यालयं गच्छामि।")],
         "किं भवान् विद्यालयं गच्छति?"),
        ([("भवतः भगिन्याः नाम किम्?", "मम भगिन्याः नाम गौरी।")],
         "किं तस्याः नाम गौरी?"),
        ([("भवान् कुत्र वसति?", "अहं नगरे निवसामि।")],
         "किं भवान् नगरे निवसति?"),
        ([("भवतः पिता किं करोति?", "मम पिता कृषकः।")],
         "किं भवतः पिता कृषकः?"),
        ([("भवान् संस्कृतं पठति किम्?", "आम्, अहं संस्कृतं पठामि।")],
         "किं भवान् संस्कृतं पठति?"),
        ([("कः भवतः मित्रम्?", "मम मित्रं रमेशः।")],
         "कः रमेशः?"),
        ([("भवान् किं खादति?", "अहं भातं खादामि।")],
         "किं भवान् भातं खादति?"),
        ([("भवतः गृहं कुत्र?", "मम गृहं नगर्याः मध्ये।")],
         "किं भवतः गृहं नगर्याः मध्ये?"),
        ([("भवान् कति वर्षाणि पठति?", "अहं दश वर्षाणि पठामि।")],
         "किं भवान् दश वर्षाणि पठति?"),
        ([("कः विद्यालये पाठयति?", "आचार्यः सोमशर्मा पाठयति।")],
         "किं सोमशर्मा पाठयति?"),
        ([("भवतः प्रियं पुस्तकम् किम्?", "मम प्रियं पुस्तकं गीता।")],
         "किं भवतः प्रियं पुस्तकं गीता?"),
        ([("भवान् किं प्रतिदिनं करोति?", "अहं प्रतिदिनम् उद्यानं गच्छामि।")],
         "किं भवान् उद्यानं गच्छति?"),
        ([("तौ कथम् स्तः?", "तौ सुखिनौ स्तः।")],
         "किं तौ सुखिनौ स्तः?"),
    ]
    REF = {
        0: "आम्, मम नाम रामः।", 1: "आम्, अहं काश्यां निवसामि।",
        2: "आम्, सोमशर्मा मम गुरुः।", 3: "आम्, अहं संस्कृतं पठामि।",
        4: "आम्, मम ग्रामः काशीसमीपे।", 5: "आम्, अद्य मम दिनम् उत्तमम्।",
        6: "आम्, भवान् जलम् अपिपः।", 7: "आम्, सा मथुरायां निवसति।",
        8: "आम्, मम प्रियं भोजनं भातम्।", 9: "आम्, अहं प्रातः पञ्चवादने उत्तिष्ठामि।",
        10: "आम्, मम गुरुः सोमशर्मा।", 11: "आम्, अहं विद्यालयं गच्छामि।",
        12: "आम्, तस्य नाम गौरी।", 13: "आम्, अहं नगरे निवसामि।",
        14: "आम्, मम पिता कृषकः।", 15: "आम्, अहं संस्कृतं पठामि।",
        16: "आम्, मम मित्रं रमेशः।", 17: "आम्, अहं भातं खादामि।",
        18: "आम्, मम गृहं नगर्याः मध्ये।", 19: "आम्, अहं दश वर्षाणि पठामि।",
        20: "आम्, सः विद्यालये पाठयति।", 21: "आम्, मम प्रियं पुस्तकं गीता।",
        22: "आम्, अहं प्रतिदिनम् उद्यानं गच्छामि।", 23: "आम्, तौ सुखिनौ स्तः।",
    }
    for i, (history, prompt) in enumerate(pairs):
        b.eval.append({
            "id": f"dlg_{i}",
            "category": "dialogue",
            "kind": "multi-turn",
            "prompt": prompt,
            "history": [{"user": u, "assistant": a} for u, a in history],
            "reference": REF.get(i, ""),
            "expected_type": "sentence",
        })


# ---------------------------------------------------------------- build
def build(out: Path) -> dict:
    rng = random.Random(11)
    b = Builder()
    trained_facts, held_facts = render_facts(b)
    render_grammar(b)
    render_grammar_cloze(b)
    render_numbers(b, rng)
    render_dialogues(b)
    eval_from_facts(b)
    eval_grammar(b)
    eval_numbers(b)
    eval_dialogues(b)

    # nothing in the evaluation set may appear as a training turn
    blocked = {item["prompt"].strip() for item in b.eval}
    kept, dropped = [], 0
    for row in b.rows:
        turns = [t.split("<assistant>")[0].strip() for t in row.split("<user>")[1:]]
        if any(t in blocked for t in turns):
            dropped += 1
            continue
        kept.append(row)

    # split by category-stratified hash so paraphrases of one fact stay together
    train, val = [], []
    for row in kept:
        content = row.split("<assistant>", 1)[-1].strip()
        h = fact_hash(content)
        (val if h < 0.03 else train).append(row)

    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    with (out / "train.jsonl").open("w", encoding="utf-8") as fh:
        for row in train:
            fh.write(json.dumps({"category": b.cats.get(row, "other"), "text": row},
                                ensure_ascii=False) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    with (out / "eval_set.jsonl").open("w", encoding="utf-8") as fh:
        for item in b.eval:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "pairs": len(kept),
        "train": len(train),
        "val": len(val),
        "dropped_eval_overlap": dropped,
        "facts": {"trained": trained_facts, "held_out": held_facts,
                  "total": trained_facts + held_facts},
        "nouns": {"trained": len(TRAIN_NOUNS), "held_out": len(HELD_OUT_NOUNS)},
        "dialogues": len(kb_dialogues.generate(max_per_script=12)),
        "per_category": dict(sorted(b.per_cat.items(), key=lambda kv: -kv[1])),
        "eval_items": len(b.eval),
        "eval_by_kind": {},
    }
    for item in b.eval:
        stats["eval_by_kind"][item["kind"]] = stats["eval_by_kind"].get(item["kind"], 0) + 1
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "instruct_v2")
    args = ap.parse_args()
    stats = build(args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}/train.txt, val.txt, eval_set.jsonl, stats.json")


if __name__ == "__main__":
    main()
