#!/usr/bin/env python3
"""
Build the basic-exchange training mix: greetings + arithmetic + grammar
production + vocabulary + time + multi-turn flows, with scrubbed replay.

Why this exists: the 100-exchange evaluation (`scripts/eval_exchanges.py`) found
greetings at 97 %, but numbers 1/6, word-grammar 1/8, time 2/4 and multi-turn
68 %. Those are *other* areas, and the earlier greeting rounds had squeezed them
out — the replay scrub even removed the last arithmetic rows, because a stray
arithmetic row had once answered a greeting. The fix is not to keep arithmetic
out but to teach it deliberately, in enough phrasings that no single row can
become an answer to "चलति तर्हि".

Every sentence here is written or generated from a paradigm table; nothing is
machine-translated. Generated grammar rows come from the declension tables below,
which use the standard paradigms (अकारान्त, आकारान्त, इकारान्त, उकारान्त).

    python3 scripts/build_exchange_data.py --repeat-greet 12 --repeat-exchange 6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CARDINAL = {1: "एकम्", 2: "द्वे", 3: "त्रीणि", 4: "चत्वारि", 5: "पञ्च", 6: "षट्",
            7: "सप्त", 8: "अष्ट", 9: "नव", 10: "दश", 11: "एकादश", 12: "द्वादश",
            13: "त्रयोदश", 14: "चतुर्दश", 15: "पञ्चदश", 16: "षोडश", 17: "सप्तदश",
            18: "अष्टादश", 19: "नवदश", 20: "विंशतिः", 21: "एकविंशतिः",
            24: "चतुर्विंशतिः", 25: "पञ्चविंशतिः", 27: "सप्तविंशतिः"}

CASES = ["प्रथमा", "द्वितीया", "तृतीया", "चतुर्थी", "पञ्चमी", "षष्ठी", "सप्तमी"]
NUMBERS = ["एकवचन", "द्विवचन", "बहुवचन"]

# standard paradigms: table[number][case]
PARADIGMS = {
    "राम": [["रामः", "रामम्", "रामेण", "रामाय", "रामात्", "रामस्य", "रामे"],
            ["रामौ", "रामौ", "रामाभ्याम्", "रामाभ्याम्", "रामाभ्याम्", "रामयोः", "रामयोः"],
            ["रामाः", "रामान्", "रामैः", "रामेभ्यः", "रामेभ्यः", "रामाणाम्", "रामेषु"]],
    "लता": [["लता", "लताम्", "लतया", "लतायै", "लतायाः", "लतायाः", "लतायाम्"],
            ["लते", "लते", "लताभ्याम्", "लताभ्याम्", "लताभ्याम्", "लतयोः", "लतयोः"],
            ["लताः", "लताः", "लताभिः", "लताभ्यः", "लताभ्यः", "लतानाम्", "लतासु"]],
    "फल": [["फलम्", "फलम्", "फलेन", "फलाय", "फलात्", "फलस्य", "फले"],
            ["फले", "फले", "फलाभ्याम्", "फलाभ्याम्", "फलाभ्याम्", "फलयोः", "फलयोः"],
            ["फलानि", "फलानि", "फलैः", "फलेभ्यः", "फलेभ्यः", "फलानाम्", "फलेषु"]],
    "मुनि": [["मुनिः", "मुनिम्", "मुनिना", "मुनये", "मुनेः", "मुनेः", "मुनौ"],
             ["मुनी", "मुनी", "मुनिभ्याम्", "मुनिभ्याम्", "मुनिभ्याम्", "मुन्योः", "मुन्योः"],
             ["मुनयः", "मुनीन्", "मुनिभिः", "मुनिभ्यः", "मुनिभ्यः", "मुनीनाम्", "मुनिषु"]],
    "गुरु": [["गुरुः", "गुरुम्", "गुरुणा", "गुरवे", "गोः", "गोः", "गुरौ"],
             ["गुरू", "गुरू", "गुरुभ्याम्", "गुरुभ्याम्", "गुरुभ्याम्", "गुर्वोः", "गुर्वोः"],
             ["गुरवः", "गुरून्", "गुरुभिः", "गुरुभ्यः", "गुरुभ्यः", "गुरूणाम्", "गुरुषु"]],
}

ROOTS = [("गच्छति", "गम्"), ("पठति", "पठ्"), ("भवति", "भू"), ("करोति", "कृ"),
         ("ददाति", "दा"), ("वदति", "वद्"), ("खादति", "खाद्"), ("पिबति", "पा"),
         ("शृणोति", "श्रु"), ("पश्यति", "दृश्"), ("लिखति", "लिख्"), ("नमति", "नम्"),
         ("हसति", "हस्"), ("गायति", "गै"), ("स्मरति", "स्मृ"), ("वसति", "वस्"),
         ("नीयते", "नी"), ("कथयति", "कथ्")]

SYNONYMS = [("जलम्", "उदकम्"), ("पुस्तकम्", "ग्रन्थः"), ("अग्निः", "वह्निः"),
            ("सूर्यः", "रविः"), ("चन्द्रः", "शशी"), ("वायुः", "पवनः"),
            ("वृक्षः", "तरुः"), ("गृहम्", "आलयः"), ("मित्रम्", "सखा"),
            ("धनम्", "वित्तम्"), ("राजा", "नृपः"), ("नदी", "सरित्"),
            ("पर्वतः", "शैलः"), ("समुद्रः", "सागरः"), ("पुष्पम्", "कुसुमम्"),
            ("हस्तः", "करः"), ("नेत्रम्", "चक्षुः"), ("मनः", "चित्तम्"),
            ("अश्वः", "तुरगः"), ("गजः", "कुञ्जरः")]

FEMININES = [("अश्वः", "अश्वा"), ("बालकः", "बालिका"), ("सिंहः", "सिंही"),
             ("मृगः", "मृगी"), ("वृकः", "वृकी"), ("अजः", "अजा"),
             ("मयूरः", "मयूरी"), ("हंसः", "हंसी"), ("कविः", "कवयित्री")]

# multi-turn flows: (turn1 user, turn1 assistant, turn2 user, turn2 assistant)
FLOWS = [
    ("नमस्ते", "नमस्ते! भवान् कथम् अस्ति?",
     "अहं कुशलः अस्मि", "साधु! शुभम् अस्तु।"),
    ("तव नाम किम्?", "मम नाम सनातनी।",
     "किं त्वं संस्कृतं जानासि?", "आम्। अहं संस्कृतं जानामि, संस्कृतं च वदामि।"),
    ("क्षम्यताम्", "न दोषः। क्षमा न अपेक्षिता।",
     "धन्यवादः", "स्वागतम्।"),
    ("धन्यवादः", "स्वागतम्।",
     "अहं गच्छामि", "गच्छतु। पुनः मिलामः।"),
    ("मम साहाय्यं कुरु", "साहाय्यं करोमि। किं पृच्छति भवान्?",
     "धन्यवादः", "स्वागतम्।"),
    ("किं भवती मानुषी?", "अहं मानुषी न, अहं भाषामयी अस्मि।",
     "तर्हि किं करोषि?", "अहं प्रश्नानाम् उत्तराणि ददामि, संस्कृतेन संवदामि।"),
    ("अहं न जानामि", "बोधः न जातः। पुनः वदतु, अहं विवृणोमि।",
     "पुनः वद, कृपया", "आम्, पुनः स्पष्टं वदामि।"),
    ("सुप्रभातम्", "सुप्रभातम्। शुभम् अस्तु।",
     "अद्य किं करोषि?", "अहं प्रश्नानाम् उत्तराणि ददामि, संस्कृतेन संवदामि।"),
    ("भवत्याः नाम किम्?", "मम नाम सनातनी।",
     "स्वपरिचयं ददातु", "अहं सनातनी, अक्षराधारिता संस्कृतभाषामयी। केवलं संस्कृतेन वदामि।"),
    ("शुभा रात्रिः", "शुभरात्रिः। शुभं स्वप्नम्।",
     "पुनः मिलामः", "पुनः मिलामः। नमस्ते।"),
    ("भवत्याः उत्तरं सुन्दरम्", "अनुगृहीता अस्मि। धन्यवादः।",
     "अधिकं वद", "आम्, अहं विवृणोमि। पृच्छतु, अहं शृणोमि।"),
    ("अहं सम्यक् अस्मि", "साधु! शुभम् अस्तु।",
     "तव नाम किम्?", "मम नाम सनातनी।"),
    ("किं भवती आङ्ग्लं जानाति?", "आङ्ग्लं न जानामि। मम भाषा संस्कृतम् एव।",
     "केवलं संस्कृतेन वद", "आम्, अहं केवलं संस्कृतेन वदामि।"),
    ("नमस्कारः", "नमस्कारः! किं पृच्छति भवान्?",
     "किं धर्मः?", "धर्मः सत्याचरणं लोकरक्षणं च।"),
    ("अहं आगतः अस्मि", "स्वागतम्। किं पृच्छति भवान्?",
     "अहं संस्कृतं पठितुम् इच्छामि", "आम्, अहं अपि संस्कृतं पठामि।"),
    ("भवान् कथम् अस्ति?", "अहं कुशलिनी अस्मि। धन्यवादः।",
     "अहं अपि कुशलः अस्मि", "साधु! शुभम् अस्तु।"),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[^\w\u0900-\u097F]+", "", s, flags=re.UNICODE).lower()


def addition_rows() -> list[str]:
    rows = []
    for a in range(1, 13):
        for b in range(1, 13):
            total = a + b
            if total > 20:
                continue
            verb = "भवतः" if total == 2 else "भवन्ति"
            ans = f"मिलित्वा {CARDINAL[total]} {verb}।"
            rows.append(f"<user> {CARDINAL[a]} अधिकं {CARDINAL[b]} कति? <assistant> {ans}")
            rows.append(f"<user> {CARDINAL[a]} {CARDINAL[b]} च मिलित्वा कति भवन्ति? <assistant> {ans}")
            rows.append(f"<user> {CARDINAL[a]} अपि {CARDINAL[b]} अपि मिलित्वा कति? <assistant> {ans}")
    return rows


def subtraction_rows() -> list[str]:
    rows = []
    for a in range(3, 21):
        for b in range(1, a):
            diff = a - b
            if diff < 1:
                continue
            ans = f"{CARDINAL[diff]} अवशिष्यति।"
            # "{a}तः {b} न्यूनम्" is the unambiguous ablative construction; the bare
            # "a न्यूनं b" reading is genuinely ambiguous in Sanskrit, so it is not taught
            rows.append(f"<user> {CARDINAL[a]}तः {CARDINAL[b]} न्यूनं कति? <assistant> {ans}")
            rows.append(f"<user> {CARDINAL[a]}तः {CARDINAL[b]} अपनीते कति अवशिष्यति? <assistant> {ans}")
    return rows


def counting_rows() -> list[str]:
    seq = " ".join(CARDINAL[i] for i in range(1, 11)) + "।"
    seq5 = " ".join(CARDINAL[i] for i in range(11, 21)) + "।"
    rows = [
        f"<user> एकतः दशपर्यन्तं गणय <assistant> {seq}",
        f"<user> दशपर्यन्तं गणनां कुरु <assistant> {seq}",
        f"<user> एकादशतः विंशतिपर्यन्तं गणय <assistant> {seq5}",
        f"<user> पञ्चानां संख्यां वद <assistant> पञ्च इति संख्या।",
    ]
    for n in (1, 2, 5, 10, 20):
        rows.append(f"<user> {CARDINAL[n]} इति संख्या कीदृशी? <assistant> {CARDINAL[n]} इति संख्या।")
    rows += [
        "<user> गणनां कुरु <assistant> एकम् द्वे त्रीणि चत्वारि पञ्च षट् सप्त अष्ट नव दश।",
        "<user> संख्याः कथं गण्यन्ते? <assistant> एकम् द्वे त्रीणि चत्वारि पञ्च षट् सप्त अष्ट नव दश।",
        "<user> प्रथमा संख्या का? <assistant> एकम् इति प्रथमा संख्या।",
    ]
    return rows


def time_rows() -> list[str]:
    return [
        "<user> कः समयः? <assistant> समयं न जानामि। घटिकां न पश्यामि।",
        "<user> कति वादनम्? <assistant> समयं न जानामि। घटिकां न पश्यामि।",
        "<user> समयं वद <assistant> समयं न जानामि। घटिकां न पश्यामि।",
        "<user> अद्य का तिथिः? <assistant> तिथिं न जानामि। पञ्चाङ्गं न पश्यामि।",
        "<user> अद्य कः वासरः? <assistant> वासरं न जानामि। पञ्चाङ्गं न पश्यामि।",
        "<user> श्वः का तिथिः? <assistant> तिथिं न जानामि। पञ्चाङ्गं न पश्यामि।",
        "<user> संवत्सरः कः? <assistant> संवत्सरं न जानामि। पञ्चाङ्गं न पश्यामि।",
        # the four phrasings above are used by the evaluation, so they are dropped
        # from training by the leak filter; these teach the same answer elsewhere
        "<user> समयं ब्रूहि <assistant> समयं न जानामि। घटिकां न पश्यामि।",
        "<user> घटिकां वद <assistant> समयं न जानामि। घटिकां न पश्यामि।",
        "<user> इदानीं कः समयः स्यात्? <assistant> समयं न जानामि। घटिकां न पश्यामि।",
        "<user> अद्यतनं दिनं किम्? <assistant> तिथिं न जानामि। पञ्चाङ्गं न पश्यामि।",
        "<user> कः दिवसः अद्य? <assistant> वासरं न जानामि। पञ्चाङ्गं न पश्यामि।",
        "<user> मासः कः चलति? <assistant> मासं न जानामि। पञ्चाङ्गं न पश्यामि।",
    ]


def grammar_rows() -> list[str]:
    rows = []
    for noun, table in PARADIGMS.items():
        for ni, number in enumerate(NUMBERS):
            for ci, case in enumerate(CASES):
                form = table[ni][ci]
                # phrasings deliberately different from the evaluation items
                rows.append(f"<user> {noun} इति नाम्नः {case} {number} रूपं ब्रूहि <assistant> {form} इति रूपम्।")
                rows.append(f"<user> {noun} शब्दस्य {case} {number} किम्? <assistant> {form}।")
    for present, root in ROOTS:
        rows.append(f"<user> {present} इति पदस्य मूलधातुः कः? <assistant> {root} इति धातुः।")
        rows.append(f"<user> {present} इति क्रियायाः धातुं वद <assistant> {root}।")
    for masc, fem in FEMININES:
        rows.append(f"<user> {masc} इति शब्दस्य स्त्रीरूपं किम्? <assistant> {fem}।")
        rows.append(f"<user> {masc} इत्यस्य स्त्रियां किं रूपम्? <assistant> {fem}।")
    return rows


def vocabulary_rows() -> list[str]:
    rows = []
    for word, syn in SYNONYMS:
        rows.append(f"<user> {word} इत्यस्य समानार्थं पदं किम्? <assistant> {word} इत्यस्य समानार्थं {syn}।")
        rows.append(f"<user> {word} इति पदस्य पर्यायः कः? <assistant> {syn} इति पर्यायः।")
        rows.append(f"<user> {word} इत्यस्य अर्थं वद <assistant> {word} इति {syn}।")
    return rows


def flow_rows() -> list[str]:
    rows = []
    for u1, a1, u2, a2 in FLOWS:
        rows.append(f"<user> {u1} <assistant> {a1} <user> {u2} <assistant> {a2}")
        rows.append(f"<user> {u2} <assistant> {a2}")
    return rows


def load_greet_rows() -> list[str]:
    path = ROOT / "data/greet/train.txt"
    if not path.exists():
        raise SystemExit("run scripts/build_greet_data.py first (data/greet/train.txt missing)")
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def scrub_replay(path: Path, banned: set[str], limit: int, seed: int = 7) -> tuple[list[str], dict]:
    """Same scrubbing the greeting builder uses, plus arithmetic dropped from replay.

    Arithmetic rows are dropped from *replay* on purpose: they are now taught from
    the curated block above, in many phrasings, so a single stray row can no
    longer become the answer to a greeting.
    """
    spec = importlib.util.spec_from_file_location("greet_builder", ROOT / "scripts/build_greet_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = random.Random(seed)
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(lines)
    kept, stats = [], {"persona": 0, "collision": 0, "eval_leak": 0, "fragment": 0, "arithmetic": 0}
    for line in lines:
        head, _, tail = line.partition("<assistant>")
        if ("॥" in tail or re.search(r"[०-९]", tail) or "र्व्" in tail
                or any(k in tail for k in ("मिलित्वा", "योगफलम्", "गुणितम्",
                                           "भागफलम्", "गणना", "अष्ट भवन्ति", "नव भवन्ति"))):
            stats["fragment"] += 1
            continue
        if any(k in tail for k in ("मिलित्वा", "अवशिष्यति", "गणय", "कति भवन्ति")):
            stats["arithmetic"] += 1
            continue
        if any(pat in tail for pat in mod.PERSONA_PATTERNS):
            stats["persona"] += 1
            continue
        m = re.match(r"<user> (.*?) <assistant>", line)
        if m and norm(m.group(1)) in banned:
            stats["collision"] += 1
            continue
        kept.append(line)
        if len(kept) >= limit:
            break
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/exchange_mix"))
    ap.add_argument("--replay", type=Path, default=Path("data/instruct_mix2/train.txt"))
    ap.add_argument("--replay-lines", type=int, default=6000)
    ap.add_argument("--repeat-greet", type=int, default=12)
    ap.add_argument("--repeat-exchange", type=int, default=6)
    ap.add_argument("--repeat-flow", type=int, default=6,
                    help="multi-turn flows are the conversational backbone; weight them separately")
    ap.add_argument("--skip", type=str, default="",
                    help="comma-separated blocks to leave out, e.g. arithmetic,grammar")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    greet = load_greet_rows()
    blocks = {
        "arithmetic": addition_rows() + subtraction_rows(),
        "counting": counting_rows(),
        "time": time_rows(),
        "grammar": grammar_rows(),
        "vocabulary": vocabulary_rows(),
        "flow": flow_rows(),
    }
    for name in [s.strip() for s in args.skip.split(",") if s.strip()]:
        if name not in blocks:
            raise SystemExit(f"unknown block {name!r}; choose from {sorted(blocks)}")
        print(f"skipping block: {name} ({len(blocks[name])} rows)")
        blocks[name] = []
    greet_keys = {norm(r) for r in greet}
    exchange = [row for rows in blocks.values() for row in rows]
    before = len(exchange)
    exchange = [r for r in exchange if norm(r) not in greet_keys]
    print(f"dropped {before - len(exchange)} exchange rows that a greeting row already covers")

    # no evaluation prompt may be reproduced in training
    eval_prompts: set[str] = set()
    for name in ("data/greet_eval.jsonl", "data/greet_eval_b.jsonl", "data/conv_eval_100.jsonl"):
        fp = ROOT / name
        if fp.exists():
            for line in fp.open(encoding="utf-8"):
                d = json.loads(line)
                eval_prompts.add(norm(d.get("prompt", "")))
                for turn in d.get("context", []):
                    eval_prompts.add(norm(turn))
    eval_prompts.discard("")

    def prompt_of(row: str) -> str:
        m = re.search(r"<user> (.*?) <assistant>", row)
        return norm(m.group(1)) if m else ""

    # Arithmetic and counting are template-generated, so a few generated items can
    # land exactly on an evaluation item. Those are dropped: the evaluation should
    # test whether the rule generalises to other numbers, not whether a sentence
    # was memorised. The remaining pairs still exercise every template.
    leaks = [r for r in exchange if prompt_of(r) and prompt_of(r) in eval_prompts]
    if leaks:
        exchange = [r for r in exchange if not (prompt_of(r) and prompt_of(r) in eval_prompts)]
        print(f"dropped {len(leaks)} generated rows that exactly reproduce an eval prompt: "
              f"{[r.split(' <assistant>')[0].replace('<user> ', '') for r in leaks[:4]]}")

    seen: dict[str, int] = {}
    for row in exchange:
        key = norm(row)
        seen[key] = seen.get(key, 0) + 1
    dupes = [k for k, v in seen.items() if v > 1]
    if dupes:
        raise SystemExit(f"duplicate training row: {dupes[0][:80]}")
    print(f"exchange rows: {len(exchange)}  {[(k, len(v)) for k, v in blocks.items()]}")

    banned = {prompt_of(r) for r in greet} | {prompt_of(r) for r in exchange}
    banned.discard("")
    replay, stats = scrub_replay(args.replay, banned, args.replay_lines, args.seed)
    print(f"replay: kept {len(replay)}, dropped {stats}")

    rng = random.Random(args.seed)
    flow = [r for r in blocks["flow"]]
    rest = [r for r in exchange if r not in flow]
    greet_block = greet * args.repeat_greet
    exchange_block = rest * args.repeat_exchange + flow * args.repeat_flow
    rng.shuffle(greet_block)
    rng.shuffle(exchange_block)
    mix = greet_block + exchange_block + replay
    rng.shuffle(mix)

    # a validation row must not also be a training row, or val loss is 0 by
    # memorisation and says nothing (round 13 hit exactly that)
    cand = [r for r in mix if r in set(greet) or r in set(exchange)]
    val = []
    for row in cand:
        if row not in val and len(val) < 16:
            val.append(row)
    # remove *every* copy, otherwise the remaining repeats still memorise the row
    val_set = set(val)
    mix = [r for r in mix if r not in val_set]
    print(f"val {len(val)} rows, held out from training entirely")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n".join(mix) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    def tokens(rows: list[str]) -> int:
        return int(sum(len(r) for r in rows) / 2.07)

    g, e, r = tokens(greet_block), tokens(exchange_block), tokens(replay)
    stats_out = {
        "greet_rows": len(greet), "greet_rows_in_mix": len(greet_block),
        "exchange_rows": len(exchange), "exchange_rows_in_mix": len(exchange_block),
        "per_block": {k: len(v) for k, v in blocks.items()},
        "replay_rows": len(replay), "mix_rows": len(mix),
        "share": {"greet": round(g / max(1, g + e + r), 4),
                  "exchange": round(e / max(1, g + e + r), 4),
                  "replay": round(r / max(1, g + e + r), 4)},
        "replay_scrub": stats,
    }
    (args.out / "mix_stats.json").write_text(json.dumps(stats_out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps(stats_out["share"], ensure_ascii=False), f"mix rows {len(mix)}")


if __name__ == "__main__":
    main()
