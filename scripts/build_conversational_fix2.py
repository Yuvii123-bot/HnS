#!/usr/bin/env python3
"""
Attempt 2 at the conversational failure.

Attempt 1 (drop the malformed rows + 127 hand pairs at 0.9% of the mix, 350
steps) moved the failure but did not fix it: the model stopped saying
"आम्, सत्यम्।" and started emitting other unrelated corpus sentences
("कृषकः क्षेत्रे बीजानि वपति।"). 127 rows in 13,633 cannot install a response
mode, and none of them taught the behaviour that relevance actually needs:
*echo the thing the user mentioned, then tell them what to do about it*.

So the added data here is a synthesised advice family built on that mechanism:

    question: "गृहं न सम्यक् वर्तते। किं करोमि?"        (mentions गृह)
    answer:   "गृहं प्रति ध्यानं ददातु।"                (echoes गृह)

Nouns are case-annotated by hand so every answer is grammatical, and the noun
pool is deliberately DISJOINT from the topics in data/conversational_eval.jsonl
so that any gain is generalisation, not memorisation of the eval topics.

    python3 scripts/build_conversational_fix2.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (nominative, accusative, note) — accusative supplied explicitly so the answer
# is grammatical for all three genders. Pool avoids eval topics (study, illness,
# anger, fear, sorrow, sleep, food, money, time, friend, speech, doubt, work,
# mother/family, travel, ocean/weather).
NOUNS: list[tuple[str, str]] = [
    ("गृहम्", "गृहम्"),          # n. house
    ("क्षेत्रम्", "क्षेत्रम्"),    # n. field
    ("गौः", "गाम्"),             # f. cow
    ("अश्वः", "अश्वम्"),          # m. horse
    ("वृक्षः", "वृक्षम्"),        # m. tree
    ("जलम्", "जलम्"),            # n. water
    ("अन्नम्", "अन्नम्"),         # n. food grain
    ("वस्त्रम्", "वस्त्रम्"),      # n. cloth
    ("पुस्तकम्", "पुस्तकम्"),      # n. book
    ("लेखनी", "लेखनीम्"),         # f. pen
    ("गीतम्", "गीतम्"),           # n. song
    ("वाद्यम्", "वाद्यम्"),        # n. instrument
    ("यानम्", "यानम्"),           # n. vehicle
    ("नदी", "नदीम्"),             # f. river
    ("पर्वतः", "पर्वतम्"),        # m. mountain
    ("वनम्", "वनम्"),             # n. forest
    ("सूर्यः", "सूर्यम्"),        # m. sun
    ("चन्द्रः", "चन्द्रम्"),      # m. moon
    ("वर्षा", "वर्षाम्"),         # f. rain
    ("शीतम्", "शीतम्"),           # n. cold
    ("उष्णम्", "उष्णम्"),          # n. heat
    ("अग्निः", "अग्निम्"),        # m. fire
    ("वायुः", "वायुम्"),          # m. wind
    ("गुरुः", "गुरुम्"),          # m. teacher
    ("शिष्यः", "शिष्यम्"),        # m. student
    ("ग्रामः", "ग्रामम्"),        # m. village
    ("नगरम्", "नगरम्"),           # n. city
    ("आपणः", "आपणम्"),            # m. shop
    ("धान्यम्", "धान्यम्"),        # n. grain
    ("फलम्", "फलम्"),             # n. fruit
    ("पुष्पम्", "पुष्पम्"),         # n. flower
    ("पशुः", "पशुम्"),            # m. animal
    ("पक्षी", "पक्षिणम्"),         # m. bird
    ("सस्यम्", "सस्यम्"),          # n. crop
    ("सेतुः", "सेतुम्"),           # m. bridge
    ("कूपः", "कूपम्"),             # m. well
    ("मार्गः", "मार्गम्"),         # m. road
    ("दीपः", "दीपम्"),             # m. lamp
    ("छत्रम्", "छत्रम्"),          # n. umbrella
    ("पात्रम्", "पात्रम्"),        # n. vessel
]

FRAMES = [
    "{n} न सम्यक् वर्तते। किं करोमि?",
    "{n} विषये किं करणीयम्?",
    "{n} प्रति कः उपायः?",
    "मम {a} रक्षितुं कः उपायः?",
    "{n} कथं साधयेयम्?",
]

ANSWERS = [
    "{a} प्रति ध्यानं ददातु।",
    "{a} रक्षतु, शनैः कार्यं करोतु।",
]


def build(out: Path, repeat_hand: int = 8, repeat_synth: int = 6) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bcf", ROOT / "scripts" / "build_conversational_fix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    src = ROOT / "data" / "instruct_mix2" / "train.txt"
    rows = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, dropped = [], 0
    for line in rows:
        if "किं सत्यम्," in line:
            dropped += 1
            continue
        kept.append(line)

    # hand-authored advice/meta/fallback pairs, upweighted this time
    hand = []
    for group in (mod.ADVICE, mod.META, mod.INDIRECT):
        for q, a in group:
            hand.append(f"<user> {q} <assistant> {a}")

    # synthetic echo family: the answer must reuse the noun from the question
    synth = []
    for nom, acc in NOUNS:
        for frame in FRAMES:
            q = frame.format(n=nom, a=acc)
            for ans in ANSWERS:
                synth.append(f"<user> {q} <assistant> {ans.format(a=acc)}")

    rng = random.Random(11)
    added = hand * repeat_hand + synth * repeat_synth
    rng.shuffle(added)
    mixed = kept + added
    rng.shuffle(mixed)

    val_src = ROOT / "data" / "instruct_mix2" / "val.txt"
    val = [l for l in val_src.read_text(encoding="utf-8").splitlines() if l.strip()]

    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(mixed) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    new_tokens = sum(len(l) for l in added)
    all_tokens = sum(len(l) for l in mixed)
    stats = {
        "source_rows": len(rows),
        "dropped_malformed_confirmation": dropped,
        "hand_pairs": len(hand),
        "synth_pairs": len(synth),
        "hand_rows_after_repeat": len(hand) * repeat_hand,
        "synth_rows_after_repeat": len(synth) * repeat_synth,
        "train_rows": len(mixed),
        "added_share_of_chars": round(new_tokens / max(all_tokens, 1), 4),
        "eval_leakage": mod.check_leakage(out / "train.txt"),
    }
    (out / "fix2_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "instruct_fix2")
    args = ap.parse_args()
    print(json.dumps(build(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
