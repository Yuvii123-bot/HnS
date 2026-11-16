#!/usr/bin/env python3
"""
Write a 150 M-token training stream whose synthetic half is *fresh* every chunk.

The repository holds about 12 M characters of text (≈4.5 M tokens): instruction
mixes, the vyākaraṇa drills, the śabdapatha rendered as prose, the Hitopadeśa and
the sūtras. Training 150 M tokens on that alone would be ~33 epochs and would
memorise it. So the arithmetic, the paradigm drills and the weekday logic are
*regenerated* per chunk from the same tables that produce the answers — the
answers stay exactly right, the sentences are new:

| share | source                                             |
|-------|----------------------------------------------------|
|  0.24 | arithmetic & counting (fresh combinations)          |
|  0.14 | vyākaraṇa drills (fresh stem × cell × phrasing)     |
|  0.12 | time logic (fresh weekday/season questions)         |
|  0.20 | conversation replay (fixed: the instruction mixes)  |
|  0.16 | facts & world (fixed: the curated knowledge base)   |
|  0.14 | prose (śabdapatha sentences, Hitopadeśa, sūtras)    |

The 15 grammar prompts and the 10 numbers prompts of data/questions_100.jsonl are
filtered out of every chunk, byte for byte.

    python3 scripts/build_curriculum.py --target-tokens 150000000
    python3 scripts/prepare_data.py --data data/curriculum --out data/bin_150m
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sanskrit_kb.lexicon import (CASES, CASE_INS, CASE_LOC,  # noqa: E402
                                         NUM_LOC, NUMBERS, PURUSHA, lexicon)

HELD_IDS = {f"q{i:03d}" for i in list(range(38, 53)) + list(range(63, 73))}
SHARES = {"arith": 0.24, "grammar": 0.14, "time": 0.12, "replay": 0.20,
          "facts": 0.16, "prose": 0.14}

NUM = {0: "शून्यम्", 1: "एकम्", 2: "द्वे", 3: "त्रीणि", 4: "चत्वारि", 5: "पञ्च",
       6: "षट्", 7: "सप्त", 8: "अष्टौ", 9: "नव", 10: "दश", 11: "एकादश",
       12: "द्वादश", 13: "त्रयोदश", 14: "चतुर्दश", 15: "पञ्चदश", 16: "षोडश",
       17: "सप्तदश", 18: "अष्टादश", 19: "एकोनविंशतिः", 20: "विंशतिः",
       24: "चतुर्विंशतिः", 30: "त्रिंशत्", 40: "चत्वारिंशत्", 50: "पञ्चाशत्",
       60: "षष्टिः", 70: "सप्ततिः", 80: "अशीतिः", 90: "नवतिः", 100: "शतम्"}
ADD_F = ["{a} अधिकं {b} कति?", "{a} {b} च मिलित्वा कति?",
         "{a} च {b} च मिलित्वा कति भवन्ति?", "{a} मिलित्वा {b} कति भवन्ति?",
         "{a}-मात्रं {b}-मात्रं च योजितं कति?", "{a} सह {b} मिलित्वा किं फलम्?"]
SUB_F = ["{a} न्यूनं {b} कति?", "{a} मध्ये {b} न्यूनीकृत्य कति शिष्यन्ते?",
         "{a}तः {b} हीनं कति?"]
MUL_F = ["{a} द्विगुणं कति?", "{a} त्रिगुणं कति?", "{a} चतुर्गुणं कति?"]
OBJ = [("फलानि", "फलम्"), ("पुष्पाणि", "पुष्पम्"), ("ग्रन्थाः", "ग्रन्थः"),
       ("जनाः", "जनः"), ("अश्वाः", "अश्वः"), ("गावः", "गौः"), ("वृक्षाः", "वृक्षः"),
       ("पुस्तकानि", "पुस्तकम्"), ("मित्राणि", "मित्रम्"), ("गृहाणि", "गृहम्")]
NAMES = ["रामः", "सीता", "लक्ष्मणः", "हनूमान्", "अर्जुनः", "कृष्णः", "युधिष्ठिरः",
         "द्रौपदी", "भीमः", "नकुलः", "सहदेवः", "व्यासः", "वाल्मीकिः", "कालिदासः"]

WEEK = ["रविवासरः", "सोमवासरः", "मङ्गलवासरः", "बुधवासरः", "गुरुवासरः",
        "शुक्रवासरः", "शनिवासरः"]
SEASONS = ["वसन्तः", "ग्रीष्मः", "वर्षाः", "शरत्", "हेमन्तः", "शिशिरः"]

G_SHAPES = [
    "{s} इति शब्दस्य {c} {n} किम्?",
    "{s} इति शब्दस्य {c} {n} रूपं किम्?",
    "{s} इति शब्दस्य {cl} {nl} रूपं किम्?",
    "{s} इति शब्दस्य {c} {n} किं रूपम्?",
    "{s} इति पदस्य {c} {n} किम्?",
    "{c} {n} — {s} इति शब्दस्य रूपं किम्?",
    "{s} इति शब्दस्य {c} विभक्तौ {n} किम्?",
    "{s} इति शब्दस्य {ci} विभक्त्या {nl} किम् रूपम्?",
    "किं {s}-शब्दस्य {c} {n} अस्ति?",
    "{c} {n} रूपम् — {s} शब्दस्य किम्?",
]


def load_held() -> set[str]:
    held = set()
    for line in (ROOT / "data" / "questions_100.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item["id"] in HELD_IDS:
                held.add(item["prompt"].strip())
    return held


def read_lines(path: Path) -> list[str]:
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def arith_rows(rng: random.Random, n: int) -> list[tuple[str, str]]:
    vals = sorted(NUM)
    out = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.55:
            a, b = rng.choice(vals), rng.choice(vals)
            if a + b in NUM:
                out.append((rng.choice(ADD_F).format(a=NUM[a], b=NUM[b]), NUM[a + b]))
        elif kind < 0.80:
            a, b = rng.choice(vals), rng.choice(vals)
            if a - b in NUM:
                out.append((rng.choice(SUB_F).format(a=NUM[a], b=NUM[b]), NUM[a - b]))
        elif kind < 0.90:
            a, k = rng.choice(vals), rng.choice((2, 3, 4))
            if a * k in NUM:
                out.append((rng.choice(MUL_F).format(a=NUM[a]), NUM[a * k]))
        else:
            a, b = rng.choice(vals), rng.choice(vals)
            if a + b in NUM and a and b:
                plural, singular = rng.choice(OBJ)
                who = rng.choice(NAMES)
                noun = plural if a + b != 1 else singular
                q = rng.choice([
                    f"{who[:-1]}ः {NUM[a]} {plural} आसन्। {NUM[b]} {plural} अपि आगच्छन्। कति {plural}?",
                    f"उद्याने {NUM[a]} {plural} सन्ति। {NUM[b]} {plural} अधिकानि आनीतानि। कति?",
                ])
                out.append((q, f"{NUM[a + b]} {noun}"))
    return out


def grammar_rows(lex, rng: random.Random, n: int, held: set[str]) -> list[tuple[str, str]]:
    stems = [s for s, e in lex.stems.items() if e and e[0]["cells"]]
    out = []
    tries = 0
    while len(out) < n and tries < n * 6:
        tries += 1
        stem = rng.choice(stems)
        entry = lex.stem_entries(stem)[0]
        cells = dict(zip([(c, num) for c in CASES for num in NUMBERS], entry["cells"]))
        case, num = rng.choice(list(cells))
        form = cells[(case, num)]
        if not form:
            continue
        shape = rng.choice(G_SHAPES)
        prompt = shape.format(s=stem, c=case, n=num, cl=CASE_LOC[case],
                              nl=NUM_LOC[num], ci=CASE_INS[case])
        if prompt.strip() in held:
            continue
        out.append((prompt, form))
    return out


def time_rows(rng: random.Random, n: int) -> list[tuple[str, str]]:
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.6:
            i = rng.randrange(7)
            delta = rng.choice((1, -1))
            frame = ("अद्य {d} चेत् श्वः कः वासरः?" if delta > 0
                     else "अद्य {d} चेत् ह्यः कः वासरः?")
            out.append((frame.format(d=WEEK[i]), WEEK[(i + delta) % 7]))
        elif r < 0.8:
            s = rng.choice(SEASONS)
            out.append((rng.choice([f"{s} कः ऋतुः?", f"कस्मिन् ऋतौ {s} भवति?"]),
                        f"{s} ऋतुः।" if "कस्मिन्" in s else f"{s}।"))
        else:
            out.append((rng.choice(["सप्ताहे कति दिनानि सन्ति?",
                                    "वर्षे कति मासाः सन्ति?",
                                    "वर्षे कति ऋतवः सन्ति?",
                                    "एकस्मिन् वासरे कति होराः भवन्ति?"]),
                        rng.choice(["सप्त दिनानि।", "द्वादश मासाः।", "षट् ऋतवः।",
                                    "चतुर्विंशतिः होराः।"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "curriculum")
    ap.add_argument("--target-tokens", type=int, default=150_000_000)
    ap.add_argument("--chunk-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-tokens", type=int, default=40_000)
    args = ap.parse_args()

    lex = lexicon()
    held = load_held()
    rng = random.Random(args.seed)

    replay = []
    for p in ("data/instruct_mix2/train.txt", "data/greet/train.txt"):
        replay.extend(read_lines(ROOT / p))
    facts = read_lines(ROOT / "data" / "instruct_v2" / "train.txt")
    prose = (read_lines(ROOT / "data" / "lexicon_prose" / "train.txt")
             + read_lines(ROOT / "data" / "raw" / "hitopadesha" / "hitokale_deva_F.txt")
             + read_lines(ROOT / "data" / "vyakarana_train" / "train.txt"))
    print(f"fixed pools: replay {len(replay):,}  facts {len(facts):,}  prose {len(prose):,}")

    def size(lines: list[str]) -> int:
        return sum(len(l) for l in lines)

    args.out.mkdir(parents=True, exist_ok=True)
    chunks = max(1, args.target_tokens // args.chunk_tokens)
    written = 0
    val: list[str] = []
    for c in range(chunks):
        fh = (args.out / f"part_{c:03d}.txt").open("w", encoding="utf-8")
        budget = args.chunk_tokens * 2.6          # chars per token ≈ 2.6
        part: list[str] = []
        n_arith = int(budget * SHARES["arith"] / 40)
        for prompt, ans in arith_rows(rng, n_arith):
            if prompt.strip() not in held:
                part.append(f"<user> {prompt} <assistant> {ans}।")
        for prompt, ans in grammar_rows(lex, rng, int(budget * SHARES["grammar"] / 42),
                                        held):
            part.append(f"<user> {prompt} <assistant> {ans}।")
        for prompt, ans in time_rows(rng, int(budget * SHARES["time"] / 40)):
            part.append(f"<user> {prompt} <assistant> {ans}")
        for key, pool in (("replay", replay), ("facts", facts), ("prose", prose)):
            want, got = budget * SHARES[key], 0
            while got < want:
                line = pool[rng.randrange(len(pool))]
                part.append(line)
                got += len(line) + 1
        rng.shuffle(part)
        cut = int(len(part) * args.val_tokens / args.chunk_tokens) if c == 0 else 0
        val.extend(part[:cut])
        part = part[cut:]
        fh.write("\n".join(part) + "\n")
        fh.close()
        written += size(part)
        if (c + 1) % 4 == 0 or c == chunks - 1:
            print(f"  chunk {c + 1}/{chunks}  {written:,} chars "
                  f"(~{written / 2.6:,.0f} tokens)", flush=True)

    (args.out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    rel = args.out.relative_to(ROOT) if ROOT in args.out.resolve().parents else args.out
    print(f"{rel}: {written:,} chars ≈ {written / 2.6:,.0f} tokens in {chunks} parts")


if __name__ == "__main__":
    main()
