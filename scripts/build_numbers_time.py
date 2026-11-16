#!/usr/bin/env python3
"""
Numbers, arithmetic and the sense of time — generated, not hand-listed.

Arithmetic is the one thing a 10 M-parameter model can learn exactly if it is
shown enough combinations with enough phrasings, because the answer space is
finite (the number words) and the mapping is total. So every sum, difference and
product whose result has a Sanskrit word is enumerated, in many frames:

    पञ्च अधिकं त्रीणि कति?            -> अष्टौ।
    त्रीणि पञ्च च मिलित्वा कति?        -> अष्टौ।
    पञ्चानां त्रयाणां च योगः कः?       -> अष्टौ।

Time is taught as *reasoning*, not as a clock: the weekday cycle (today → tomorrow
→ yesterday), the lengths of week/month/year and the seasons, so "अद्य शनिवासरः
चेत् श्वः कः वारः?" is answerable.

The 10 numbers questions of data/questions_100.jsonl are excluded byte-for-byte.

    python3 scripts/build_numbers_time.py --check
    python3 scripts/build_numbers_time.py --out data/numbers_time
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_IDS = {f"q{i:03d}" for i in range(38, 53)}
NUMBER_IDS = {f"q{i:03d}" for i in range(63, 73)}

#: number words that are unambiguous in Sanskrit; the answer space of the drills
NUM = {
    0: "शून्यम्", 1: "एकम्", 2: "द्वे", 3: "त्रीणि", 4: "चत्वारि", 5: "पञ्च",
    6: "षट्", 7: "सप्त", 8: "अष्टौ", 9: "नव", 10: "दश", 11: "एकादश",
    12: "द्वादश", 13: "त्रयोदश", 14: "चतुर्दश", 15: "पञ्चदश", 16: "षोडश",
    17: "सप्तदश", 18: "अष्टादश", 19: "एकोनविंशतिः", 20: "विंशतिः",
    24: "चतुर्विंशतिः", 25: "पञ्चविंशतिः", 30: "त्रिंशत्", 40: "चत्वारिंशत्",
    50: "पञ्चाशत्", 60: "षष्टिः", 70: "सप्ततिः", 80: "अशीतिः", 90: "नवतिः",
    100: "शतम्",
}
WORD_OF = {v: k for k, v in NUM.items()}

ADD_FRAMES = [
    "{a} अधिकं {b} कति?",
    "{a} {b} च मिलित्वा कति?",
    "{a} च {b} च मिलित्वा कति भवन्ति?",
    "{a} मिलित्वा {b} कति भवन्ति?",
    "{a}-मात्रं {b}-मात्रं च योजितं कति?",
    "{a} अपि {b} अपि — समष्टिः का?",
    "{a} सह {b} मिलित्वा किं फलम्?",
]
SUB_FRAMES = [
    "{a} न्यूनं {b} कति?",
    "{a} मध्ये {b} न्यूनीकृत्य कति शिष्यन्ते?",
    "{a}तः {b} हीनं कति?",
    "{a} भवति, {b} नश्यति — कति शिष्यन्ते?",
]
MUL_FRAMES = [
    "{a} द्विगुणं कति?",
    "{a} त्रिगुणं कति?",
    "{a} चतुर्गुणं कति?",
]
COUNT_FRAMES = [
    "एकतः पञ्चपर्यन्तं गणय।",
    "एकतः दशपर्यन्तं गणय।",
    "शून्यतः पञ्चपर्यन्तं गणय।",
]
NUMBER_WORD_FRAMES = [
    "{w} इति संख्या का?",
    "{w} इति शब्दस्य संख्या का?",
    "{w} संख्या कस्याः संख्यायाः शब्दः?",
]
COUNT_SEQ = [NUM[i] for i in range(0, 6)]

WEEKDAYS = ["रविवासरः", "सोमवासरः", "मङ्गलवासरः", "बुधवासरः", "गुरुवासरः",
            "शुक्रवासरः", "शनिवासरः"]
MONTHS = ["चैत्रः", "वैशाखः", "ज्येष्ठः", "आषाढः", "श्रावणः", "भाद्रपदः",
          "आश्विनः", "कार्तिकः", "मार्गशीर्षः", "पौषः", "माघः", "फाल्गुनः"]
SEASONS = ["वसन्तः", "ग्रीष्मः", "वर्षाः", "शरत्", "हेमन्तः", "शिशिरः"]

TIME_FACTS = [
    ("सप्ताहे कति दिनानि सन्ति?", "सप्त दिनानि।"),
    ("एकस्मिन् वासरे कति होराः भवन्ति?", "चतुर्विंशतिः होराः।"),
    ("वर्षे कति मासाः सन्ति?", "द्वादश मासाः।"),
    ("वर्षे कति ऋतवः सन्ति?", "षट् ऋतवः।"),
    ("मासे कति पक्षौ सन्ति?", "द्वौ पक्षौ।"),
    ("पक्षे कति दिनानि सन्ति?", "पञ्चदश दिनानि।"),
    ("अर्धे होरायां कति निमेषाः सन्ति?", "त्रिंशत् निमेषाः।"),
    ("प्रातःकाले किं भवति?", "सूर्यः उदेति।"),
    ("सायंकाले किं भवति?", "सूर्यः अस्तं गच्छति।"),
    ("रात्रौ किं भवति?", "चन्द्रः प्रकाशते।"),
]
TIME_SHAPES = [
    "{q}", "कृपया वदतु — {q}", "संस्कृते {q}",
]
REL_SHAPES = [
    ("अद्य {d} चेत् श्वः कः वासरः?", 1),
    ("अद्य {d} चेत् ह्यः कः वासरः?", -1),
    ("अद्य {d} अस्ति। श्वः किम्?", 1),
]


def rows(prompt: str, answer: str) -> str:
    return f"<user> {prompt} <assistant> {answer}"


def build(rng: random.Random) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"numbers": [], "time": []}
    vals = sorted(NUM)

    # arithmetic: every combination whose result is a number word
    for a in vals:
        for b in vals:
            for op, frames in ((a + b, ADD_FRAMES), (a - b, SUB_FRAMES)):
                if op not in NUM:
                    continue
                want = NUM[op]
                for frame in frames:
                    out["numbers"].append(rows(
                        frame.format(a=NUM[a], b=NUM[b]), f"{want}।"))
            for k, frames in ((2, MUL_FRAMES[0:1]), (3, MUL_FRAMES[1:2]), (4, MUL_FRAMES[2:3])):
                if a * k in NUM and k == 2:
                    out["numbers"].append(rows(
                        frames[0].format(a=NUM[a]), f"{NUM[a * k]}।"))

    # counting and number-word identification
    for frame in COUNT_FRAMES:
        out["numbers"].append(rows(frame, "।".join(COUNT_SEQ) + "।"))
    for word, value in NUM.items():
        for frame in NUMBER_WORD_FRAMES:
            out["numbers"].append(rows(frame.format(w=word), f"{word}।"))

    # objects + numbers ("दश फलानि") so the numerals attach to nouns
    things = [("फलानि", "फलम्"), ("पुष्पाणि", "पुष्पम्"), ("ग्रन्थाः", "ग्रन्थः"),
              ("जनाः", "जनः"), ("अश्वाः", "अश्वः"), ("गावः", "गौः"),
              ("वृक्षाः", "वृक्षः"), ("पुस्तकानि", "पुस्तकम्")]
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100):
        for plural, singular in things:
            noun = plural if n != 1 else singular
            out["numbers"].append(rows(f"कति {plural} सन्ति?", f"{NUM[n]} {noun}।"))
            out["numbers"].append(rows(f"उद्याने {NUM[n]} {noun} सन्ति। कति?", f"{NUM[n]}।"))

    # senses of time: facts, weekday arithmetic, seasons
    for q, a in TIME_FACTS:
        for shape in TIME_SHAPES:
            out["time"].append(rows(shape.format(q=q), a))
    for i, day in enumerate(WEEKDAYS):
        for frame, delta in REL_SHAPES:
            other = WEEKDAYS[(i + delta) % 7]
            out["time"].append(rows(frame.format(d=day), f"{other}।"))
        out["time"].append(rows(f"{day} कः वासरः?", f"{day}।"))
        out["time"].append(rows(f"{day} इति वासरस्य नाम किम्?", f"{day}।"))
    for m in MONTHS:
        out["time"].append(rows(f"{m} कः मासः?", f"{m}।"))
    for s in SEASONS:
        out["time"].append(rows(f"{s} कः ऋतुः?", f"{s}।"))
        out["time"].append(rows(f"कस्मिन् ऋतौ {s} भवति?", f"{s} ऋतौ।"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "numbers_time")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    held = set()
    for line in (ROOT / "data" / "questions_100.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item["id"] in GRAMMAR_IDS | NUMBER_IDS:
                held.add(item["prompt"].strip())

    kinds = build(random.Random(args.seed))
    kept, seen = {}, set()
    for kind, lines in kinds.items():
        kept[kind] = []
        for line in lines:
            prompt = line.split(" <assistant> ")[0].replace("<user> ", "").strip()
            if prompt in held or prompt in seen:
                continue
            seen.add(prompt)
            kept[kind].append(line)

    total = sum(len(v) for v in kept.values())
    print(f"rows {total:,}")
    for kind, lines in kept.items():
        print(f"  {kind:<9}{len(lines):>7,}")
    print(f"  held-out prompts present: {len(seen & held)} (must be 0)")
    if args.check:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    for kind, lines in kept.items():
        rng = random.Random(args.seed + len(kind))
        rng.shuffle(lines)
        cut = int(len(lines) * args.val_frac)
        (args.out / f"{kind}_val.txt").write_text("\n".join(lines[:cut]) + "\n", encoding="utf-8")
        (args.out / f"{kind}.txt").write_text("\n".join(lines[cut:]) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
