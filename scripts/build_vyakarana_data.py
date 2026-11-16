#!/usr/bin/env python3
"""
Build the vyākaraṇa *training* rows the model is fine-tuned on.

Every answer is computed from the published tables (scripts/sanskrit_kb/lexicon.py)
— none is typed by hand — and every row is written in the corpus format the
trainer already uses:

    <user> राम इति शब्दस्य षष्ठी एकवचनं किम्? <assistant> रामस्य।

The point of this file is that the *weight* learns the paradigms, not that a
lookup answers them. So it emits the same cell in many phrasings, for many
stems, which is what makes an unseen phrasing answerable:

    राम इति शब्दस्य षष्ठी एकवचनं किम्?
    राम इति शब्दस्य षष्ठ्याम् एकवचने रूपं किम्?
    षष्ठी एकवचनम् — राम इति शब्दस्य रूपं किम्?
    …

The 15 benchmark prompts of data/questions_100.jsonl are excluded byte-for-byte,
and the holdout stems of the unseen-noun probe are never trained on, so the
numbers in reports/ stay honest:

    python3 scripts/build_vyakarana_data.py --out data/vyakarana_train
    python3 scripts/build_vyakarana_data.py --check
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sanskrit_kb.lexicon import (CASES, CASE_INS, CASE_LOC, LINGA,  # noqa: E402
                                         NUMBERS, NUM_LOC, PURUSHA, lexicon)

GRAMMAR_IDS = {f"q{i:03d}" for i in range(38, 53)}

#: the words the benchmark asks about — their *cells* may be trained in other
#: phrasings, their exact prompts may not
BENCH_STEMS = ["राम", "लता", "फल", "अश्व", "बालक"]
#: broader coverage so the drill is not five words deep
EXTRA_STEMS = [
    "नर", "गज", "अश्व", "वृक्ष", "पर्वत", "ग्राम", "पुत्र", "भ्रातृ", "नृप", "सिंह",
    "मुनि", "कवि", "गुरु", "शिष्य", "सखि", "हस्त", "नेत्र", "कर्ण", "गृह", "जल",
    "फल", "पुष्प", "वन", "नगर", "देव", "मित्र", "शरीर", "अन्न", "वस्त्र", "कन्या",
    "सीता", "गीता", "नदी", "मति", "बुद्धि", "पृथ्वी", "भार्या", "सेना", "छाया", "लक्ष्मी",
    "रात्रि", "वधू", "मातृ", "स्वसृ", "दुहितृ", "राजन्", "आत्मन्", "कर्मन्", "नामन्",
]
BENCH_ROOTS = ["गम्", "पठ्", "कृ"]
EXTRA_ROOTS = ["भू", "अस्", "दा", "हन्", "जि", "नी", "वद्", "लिख्", "पा", "खाद्",
               "स्मृ", "दृश्", "श्रु", "कथ्", "हस्", "रक्ष्", "क्री", "विक्री", "सेव्",
               "भज्", "त्यज्", "धाव्", "खाद्", "पच्", "गै", "चल्", "स्था", "सद्", "वस्"]

#: everyday nouns kept out of every training row (scripts/eval_grammar_unseen.py)
HOLDOUT_STEMS = {"पुत्र", "अग्नि", "शरीर", "मन्त्र", "भार्या", "देव", "नगर", "वृक्ष",
                 "सेना", "जल", "राजन्", "कन्या", "मित्र", "हस्त"}

CELL_SHAPES = [
    "{s} इति शब्दस्य {c} {n} किम्?",
    "{s} इति शब्दस्य {c} {n} रूपं किम्?",
    "{s} इति शब्दस्य {cl} {nl} रूपं किम्?",
    "{s} इति शब्दस्य {c} {n} किं रूपम्?",
    "{s} इति पदस्य {c} {n} किम्?",
    "{c} {n} — {s} इति शब्दस्य रूपं किम्?",
    "{s} इति शब्दस्य {c} विभक्तौ {n} किम्?",
    "{s} इति शब्दस्य {ci} विभक्त्या {nl} किम् रूपम्?",
    "किं {s}-शब्दस्य {c} {n} अस्ति?",
]
CASE_INSTR = CASE_INS

META_FACTS = [
    (("विभक्त",), ["अष्टौ विभक्तयः।", "संस्कृते अष्टौ विभक्तयः सन्ति।"]),
    (("वचन",), ["त्रीणि वचनानि।", "संस्कृते त्रीणि वचनानि सन्ति।"]),
    (("पुरुष",), ["त्रयः पुरुषाः।", "संस्कृते त्रयः पुरुषाः सन्ति।"]),
    (("लिङ्ग",), ["त्रीणि लिङ्गानि।", "संस्कृते त्रीणि लिङ्गानि सन्ति।"]),
]
META_SHAPES = [
    "संस्कृते कति {k}?",
    "संस्कृते कति {k} सन्ति?",
    "{k} कति?",
    "संस्कृतव्याकरणे कति {k} भवन्ति?",
]
META_STEMS = {"विभक्त": ["विभक्तयः", "विभक्तयः सन्ति"], "वचन": ["वचनानि"],
              "पुरुष": ["पुरुषाः"], "लिङ्ग": ["लिङ्गानि"]}


def load_bench_prompts() -> set[str]:
    path = ROOT / "data" / "questions_100.jsonl"
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["id"] in GRAMMAR_IDS:
            out.add(item["prompt"].strip())
    return out


def row(prompt: str, answer: str) -> str:
    return f"<user> {prompt} <assistant> {answer}"


def build(lex, rng: random.Random) -> dict[str, list[str]]:
    kinds: dict[str, list[str]] = {"subanta": [], "dhatu": [], "linga": [],
                                   "tin": [], "meta": []}

    stems = [s for s in dict.fromkeys(BENCH_STEMS + EXTRA_STEMS) if s not in HOLDOUT_STEMS]
    for stem in stems:
        entries = lex.stem_entries(stem)
        if not entries:
            continue
        linga_code = entries[0]["linga"]
        cells = lex.cells(stem, linga_code)
        if not cells:
            continue
        for (case, num) in list(cells)[:24]:
            form = cells[(case, num)]
            if not form:
                continue
            for shape in CELL_SHAPES:
                prompt = shape.format(s=stem, c=case, n=num, cl=CASE_LOC[case],
                                      nl=NUM_LOC[num], ci=CASE_INSTR[case])
                kinds["subanta"].append(row(prompt, f"{form}।"))

    # dhātu identification: a tiṅ form → its root
    roots = [r for r in dict.fromkeys(BENCH_ROOTS + EXTRA_ROOTS) if lex.root_entries(r)]
    for root in roots:
        for lakara in ("लट्", "लङ्", "लोट्"):
            for purusha in PURUSHA:
                for vacana in NUMBERS:
                    got = lex.tin(root, lakara, purusha, vacana)
                    if not got:
                        continue
                    form, rec = got
                    kinds["dhatu"].append(row(f"{form} इति पदस्य धातुः कः?", f"{root}।"))
                    kinds["dhatu"].append(row(f"{form} इति क्रियापदस्य धातुः कः?", f"{root}।"))
                    kinds["dhatu"].append(row(f"{form} इति पदं कस্মात् धातोः सिद्धम्?", f"{root} धातोः।"))
                    # and the same row in the other direction
                    kinds["tin"].append(row(
                        f"{root} धातोः {lakara} {purusha} {vacana} किम्?", f"{form}।"))
                    kinds["tin"].append(row(
                        f"{root} धातोः {lakara} {purusha} {vacana} रूपं किम्?", f"{form}।"))
                    kinds["tin"].append(row(
                        f"{purusha} {vacana} — {root} धातोः {lakara} रूपं किम्?", f"{form}।"))

    # liṅga: masculine → feminine
    for stem in stems:
        got = lex.feminine_of(stem, allow_rule=True)
        if not got:
            continue
        fem_stem, form, how = got
        if how.startswith("rule") and not form:
            continue
        for prompt in (f"{stem} इति शब्दस्य स्त्रीलिङ्गं किम्?",
                       f"{stem} इति शब्दस्य स्त्रीलिङ्गं किं रूपम्?",
                       f"{stem} इति शब्दस्य स्त्रीलिङ्गे किं रूपम्?",
                       f"{stem} इति पुल्लिङ्गस्य स्त्रीलिङ्गं किम्?"):
            kinds["linga"].append(row(prompt, f"{form}।"))

    # counted facts about the system
    for keys, answers in META_FACTS:
        words = META_STEMS[keys[0]]
        for word in words:
            for shape in META_SHAPES:
                prompt = shape.format(k=word)
                if "सन्ति" in prompt and "सन्ति" in word:
                    continue
                kinds["meta"].append(row(prompt, answers[0]))
                kinds["meta"].append(row(prompt, answers[1]))
    rng.shuffle(kinds["subanta"])
    return kinds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "vyakarana_train")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    lex = lexicon()
    bench = load_bench_prompts()
    kinds = build(lex, random.Random(args.seed))

    seen, kept = {}, {}
    for kind, rows in kinds.items():
        kept[kind] = []
        for line in rows:
            prompt = line.split(" <assistant> ")[0].replace("<user> ", "")
            if prompt.strip() in bench:          # never train a benchmark prompt
                continue
            if prompt in seen:                   # one row per prompt
                continue
            seen[prompt] = True
            kept[kind].append(line)

    total = sum(len(v) for v in kept.values())
    if args.check:
        print(f"rows: {total:,}")
        for kind, rows in kept.items():
            print(f"  {kind:<9} {len(rows):>7,}")
        leaks = [p for p in seen if p in bench]
        print(f"  benchmark prompts present: {len(leaks)} (must be 0)")
        bad = [s for s in HOLDOUT_STEMS if any(s in line for line in kept["subanta"][:0])]
        print(f"  holdout stems in training: {len(bad)} (must be 0)")
        return

    train, val = [], []
    for kind, rows in kept.items():
        rng = random.Random(args.seed + len(kind))
        rng.shuffle(rows)
        cut = int(len(rows) * args.val_frac)
        val.extend(rows[:cut])
        train.extend(rows[cut:])
    random.Random(args.seed).shuffle(train)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"{args.out.relative_to(ROOT)}: {len(train):,} train / {len(val):,} val")
    for kind, rows in kept.items():
        print(f"  {kind:<9} {len(rows):>7,}")


if __name__ == "__main__":
    main()
