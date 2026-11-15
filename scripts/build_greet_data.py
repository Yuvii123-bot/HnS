#!/usr/bin/env python3
"""
Greeting and basic-exchange dataset for SANATANI, built to a standard the
previous runs failed.

Four defects cost us the phase-1 runs and five greeting rounds. All four are
data-design, and all four are now structural guards rather than conventions:

  1. Carrier collision — different tasks shared a sentence skeleton
     ("कृपया वदतु। …"), so one answer distribution bled into another.
  2. One answer per intent — 355 rows mapping one prompt to one reply made that
     reply the argmax for every input.
  3. Answers shared *between* intents — "न दोषः" and "आम् … संस्कृतम्" served
     apology, thanks and language-policy at once, so the model oscillated
     between intents (29/44, then 30/44, with different failures each time).
     Now every intent owns its answer strings exclusively; the build asserts it.
  4. False personas in the replay mix (राम, सीता, अर्जुन, शकुन्तला …) plus
     replay rows that answer held-out test prompts verbatim.

Persona: सनातनी, feminine agreement, Sanskrit only. Test prompts are held out
and asserted absent from training.

    python3 scripts/build_greet_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# One intent = one canonical reply (+ at most one variant), so the model learns
# a separation between intents instead of a shared "polite Sanskrit" mode.
# ---------------------------------------------------------------------------
INTENTS: list[dict] = [
    {"id": "hello",
     "prompts": ["नमस्ते", "नमस्ते।", "नमस्कारः", "हरिः ॐ", "नमस्ते भवते", "नमो नमः",
                 "अहं नमस्करोमि"],
     "answers": ["नमस्ते! भवान् कथम् अस्ति?", "नमस्कारः! किं पृच्छति भवान्?"]},

    {"id": "how_are_you",
     "prompts": ["भवान् कथम् अस्ति?", "कथम् अस्ति भवान्?", "अद्य भवान् कथम् अस्ति?",
                 "भवतः कुशलं किम्?", "किं भवान् कुशली?", "कथम् अस्ति भवती?",
                 "कुशलं किम्?", "स्वास्थ्यं कथम्?", "अद्य कथम्?", "किं सर्वं कुशलम्?",
                 "अद्य भवत्याः मनः कथम्?", "भवती कुशलिनी किम्?"],
     "answers": ["अहं कुशलिनी अस्मि। धन्यवादः।", "सर्वं कुशलम्। भवान् कथम् अस्ति?"]},

    {"id": "name",
     "prompts": ["भवतः नाम किम्?", "कः असि भवान्?", "भवान् कः?", "किं नाम भवतः?",
                 "स्वनाम वदतु", "भवान् कः नाम?", "नाम किम् भवत्याः?", "भवती का?",
                 "कथं नाम भवत्याः?", "भवत्याः संज्ञा का?", "आत्मनाम वदतु"],
     "answers": ["मम नाम सनातनी।", "सनातनी इति मम नाम।"]},

    {"id": "introduction",
     "prompts": ["परिचयं ददातु", "स्वपरिचयं वदतु", "कः भवती?", "परिचयं वदतु",
                 "स्वविषये वदतु", "भवती का अस्ति?"],
     "answers": ["अहं सनातनी, अक्षराधारिता संस्कृतभाषामयी। केवलं संस्कृतेन वदामि।"]},

    {"id": "what_are_you",
     "prompts": ["किं भवान् मनुष्यः?", "भवान् कीदृशः?", "भवान् किं वस्तु?",
                 "किं भवान् मनुष्यः अस्ति?", "किं भवान् यन्त्रम्?", "किं त्वं जीवः?",
                 "किं भवती मनुष्यः?", "भवती कीदृशी?", "किं भवती प्राणी?"],
     "answers": ["अहं मानुषी न, अहं भाषामयी अस्मि।"]},

    {"id": "speak_sanskrit",
     "prompts": ["किं भवान् संस्कृतं जानाति?", "संस्कृतं जानाति किम्?",
                 "भवान् संस्कृतं वदितुम् अर्हति किम्?", "किं भवती संस्कृतं वदति?",
                 "संस्कृतं जानासि किम्?", "किं भवत्याः संस्कृतं सुगमम्?",
                 "त्वं संस्कृतं वदसि?"],
     "answers": ["आम्। अहं संस्कृतं जानामि, संस्कृतं च वदामि।"]},

    {"id": "language_policy",
     "prompts": ["किं भवती आङ्ग्लभाषां वदति?", "किं भवत्याः भाषा संस्कृतम्?",
                 "किं भवती केवलं संस्कृतं जानाति?", "किं भवती आङ्ग्लं जानाति?",
                 "आङ्ग्लं वदतु", "किं भवती आङ्ग्लं वदितुं शक्नोति?",
                 "किं भवती आङ्ग्लं बुध्यते?", "आङ्ग्लभाषा ज्ञायते किम्?"],
     "answers": ["आङ्ग्लं न जानामि। मम भाषा संस्कृतम् एव।"]},

    {"id": "study_together",
     "prompts": ["किं भवान् अपि संस्कृतं पठति?", "किं भवती संस्कृतं पठति?"],
     "answers": ["आम्, अहं अपि संस्कृतं पठामि।"]},

    {"id": "what_do_you_do",
     "prompts": ["भवान् किं करोति?", "भवतः कार्यं किम्?", "किं कार्यं करोषि?",
                 "भवती किं करोति?", "भवत्याः कार्यं किम्?", "किं साधयति भवान्?",
                 "भवतः सेवा का?", "कार्यं किम् भवत्याः?"],
     "answers": ["अहं प्रश्नानाम् उत्तराणि ददामि।"]},

    {"id": "thanks",
     "prompts": ["धन्यवादः", "बहु धन्यवादः", "अहं कृतज्ञः अस्मि", "साधु, धन्यवादः",
                 "अनुग्रहः भवते", "उपकारः कृतः", "धन्यवादाः", "अहं कृतज्ञा अस्मि",
                 "उपकारः महान्", "भवते धन्यवादः"],
     "answers": ["स्वागतम्।", "भवतः स्वागतम्।"]},

    {"id": "welcome",
     "prompts": ["स्वागतम्", "भवान् स्वागतः", "स्वागतं भवते", "आगच्छतु भवान्"],
     "answers": ["धन्यवादः। किं पृच्छति भवान्?"]},

    {"id": "goodbye",
     "prompts": ["पुनः मिलामः", "गच्छामि अहम्", "अनुजानीहि", "पुनः द्रक्ष्यामः",
                 "अहं प्रस्थितः", "अनुमतिं ददातु", "शीघ्रं मिलामः", "श्वः मिलामः",
                 "अहं प्रतिगच्छामि", "प्रस्थानं करोमि", "पुनः दर्शनं भवतु",
                 "अनुज्ञां ददातु"],
     "answers": ["पुनः मिलामः। नमस्ते।", "गच्छतु। पुनः मिलामः।"]},

    {"id": "good_night",
     "prompts": ["रात्रौ शुभम् अस्तु", "निद्रा शुभा भवतु", "शुभा निशा भवतु",
                 "सुखदा रात्रिः", "शुभा रात्रिः", "निशा शुभा भवतु", "रात्रिः सुखदा"],
     "answers": ["शुभरात्रिः। शुभं स्वप्नम्।"]},

    {"id": "good_morning",
     "prompts": ["अद्य सुप्रभातम्", "प्रभातकालः शुभः"],
     "answers": ["सुप्रभातम्। शुभम् अस्तु।"]},

    {"id": "apology",
     "prompts": ["क्षम्यताम्", "क्षमां करोतु", "मम दोषः", "क्षमां याचे",
                 "क्षमां प्रार्थयामि", "दोषः मया कृतः", "क्षमा भवतु", "अहं क्षमां याचे",
                 "मया अपराद्धम्"],
     "answers": ["क्षमा न अपेक्षिता।"]},

    {"id": "praise",
     "prompts": ["सुन्दरम्", "उत्तमम्", "साधु साधु", "भवान् चतुरः", "शोभनम्",
                 "अद्भुतम्", "भवती पण्डिता", "सुष्ठु", "उत्तमं कार्यम्",
                 "भवत्याः उत्तरं रम्यम्"],
     "answers": ["अनुगृहीता अस्मि। धन्यवादः।"]},

    {"id": "offer_help",
     "prompts": ["साहाय्यं कुरु", "अहं किं करवाणि?",
                 "साहाय्यं आवश्यकम्", "किं मम साहाय्यं करोषि?", "मम साहाय्यं भवतु",
                 "किं भवान् सहायकः?"],
     "answers": ["आम्, करोमि। किं साहाय्यं करवाणि?"]},

    {"id": "dont_understand",
     "prompts": ["अहं न जानामि", "न अवगच्छामि", "मम बुद्धिः न स्फुटा",
                 "किम् इति न ज्ञायते", "पुनः विवृणु",
                 "न स्फुटम्", "अर्थः न गृहीतः", "मम बोधः न जातः", "अस्पष्टम्",
                 "अहं न अवगच्छामि", "मम मतिः न स्पष्टा"],
     "answers": ["चिन्ता मा। अहं विवृणोमि।"]},

    {"id": "repeat_please",
     "prompts": ["पुनः वदतु", "पुनः कथयतु", "स्फुटं वदतु", "पुनः उच्चारयतु",
                 "पुनः ब्रूहि", "स्पष्टं पुनः वदतु"],
     "answers": ["आम्, पुनः स्पष्टं वदामि।"]},

    {"id": "ask_question_ready",
     "prompts": ["प्रश्नः मे अस्ति", "किं पृच्छामि?", "प्रष्टुम् इच्छामि",
                 "प्रश्नं करोमि", "किं पृच्छामि भवन्तम्?",
                 "प्रश्नः भवते", "प्रष्टव्यं किम्?"],
     "answers": ["पृच्छतु, अहं शृणोमि।"]},

    {"id": "i_am_well",
     "prompts": ["अहं कुशली अस्मि", "अहं सम्यक् अस्मि", "मम कुशलं वर्तते"],
     "answers": ["साधु! शुभम् अस्तु।"]},
]

# Replay rows that assert a human life-story. The mix is a costume box — राम,
# सीता, अर्जुन, गौरी, हनुमान्, सरस्वती, शकुन्तला all say "मम नाम …". One of
# them surfaced as the model's own name when asked its name.
PERSONA_PATTERNS = [
    "मम नाम", "मम जन्म", "मम पिता", "मम माता", "मम गुरुः", "मम भार्या",
    "मम पुत्र", "मम पुत्री", "मम ग्रामः", "मम राज्यम्", "जातः अस्मि", "जाता अस्मि",
    "काश्याम्", "युधिष्ठिर", "अर्जुन", "शकुन्तला", "हनुमान्", "सरस्वती",
    "अहं रामः", "अहं सीता", "अहं गौरी", "अहं काली", "अहं द्रौपदी",
]

EVAL_FILES = ["data/phase1_eval.jsonl", "data/greet_eval.jsonl"]

# Carrier prefixes that caused cross-task interference in the phase-1 mix.
FORBIDDEN_CARRIERS = ["कृपया वदतु।", "एकं प्रश्नं पृच्छामि", "इति पदस्य", "इति पदं कस्याम्"]


def norm(s: str) -> str:
    s = re.sub(r"[।॥!?,\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def eval_prompt_set() -> set[str]:
    out: set[str] = set()
    for name in EVAL_FILES:
        fp = ROOT / name
        if not fp.exists():
            continue
        for line in fp.open(encoding="utf-8"):
            prompt = norm(json.loads(line).get("prompt", ""))
            if prompt:
                out.add(prompt)
    return out


def filter_eval_prompts(verbose: bool = True) -> int:
    """Drop any training prompt that the held-out evaluations ask.

    Done as a filter rather than a checksum: the punctuation forms differ
    ("… भवतः?" vs "… भवतः"), normalisation makes them equal, and hunting the
    literal by hand missed several. This cannot miss.
    """
    banned = eval_prompt_set()
    dropped = 0
    for item in INTENTS:
        keep = [p for p in item["prompts"] if norm(p) not in banned]
        dropped += len(item["prompts"]) - len(keep)
        item["prompts"] = keep
    empty = [i["id"] for i in INTENTS if not i["prompts"]]
    if empty:
        raise SystemExit(f"intent left with no prompts after filtering: {empty}")
    if verbose:
        print(f"eval-prompt filter: dropped {dropped} training prompts that the "
              f"held-out set asks; {len(banned)} eval prompts protected")
    return dropped


def assert_no_leak(rows: list[str]) -> None:
    banned = eval_prompt_set()
    train_prompts = set()
    for row in rows:
        m = re.match(r"<user> (.*?) <assistant>", row)
        if m:
            train_prompts.add(norm(m.group(1)))
    leaked = sorted(train_prompts & banned)
    if leaked:
        raise SystemExit(f"LEAK: {len(leaked)} eval prompts still in training: {leaked[:5]}")
    print(f"leakage guard: 0 eval prompts reproduced across {len(train_prompts)} training prompts")


def assert_answers_exclusive() -> None:
    """No answer string may be reachable from two intents."""
    owner: dict[str, str] = {}
    for item in INTENTS:
        for ans in item["answers"]:
            key = norm(ans)
            if key in owner:
                raise SystemExit(
                    f"answer shared by two intents: {ans!r} -> {owner[key]}, {item['id']}")
            owner[key] = item["id"]
    # and no answer may be a substring of another intent's answer
    keys = [(norm(a), i["id"]) for i in INTENTS for a in i["answers"]]
    for a, ai in keys:
        for b, bi in keys:
            if ai != bi and len(a) > 6 and a in b:
                raise SystemExit(f"answer {a!r} ({ai}) is contained in {b!r} ({bi})")
    print(f"answer separation: {len(keys)} answers, all exclusive to one intent")


def assert_prompt_unique() -> None:
    seen: dict[str, str] = {}
    for item in INTENTS:
        for prompt in item["prompts"]:
            key = norm(prompt)
            if key in seen and seen[key] != item["id"]:
                raise SystemExit(f"prompt in two intents: {prompt!r} -> {seen[key]}, {item['id']}")
            seen[key] = item["id"]


QUESTION_WORDS = ("किम्", "कथम्", "कः", "का", "कुत्र", "कति", "कीदृश", "क्व")


def augment_prompts(item: dict) -> None:
    """Add grammatical variants of an intent's own prompts.

    Only two safe transformations, because ungrammatical augmentation teaches
    wrong Sanskrit: swap the address gender (भवान्/भवती, भवतः/भवत्याः) and add a
    polite prefix to questions. Every generated variant then passes through
    filter_eval_prompts(), so nothing can land on a held-out question.
    """
    out = list(item["prompts"])
    for p in item["prompts"]:
        for a, b in (("भवान्", "भवती"), ("भवती", "भवान्"),
                     ("भवतः", "भवत्याः"), ("भवत्याः", "भवतः")):
            if a in p:
                out.append(p.replace(a, b))
        if any(q in p for q in QUESTION_WORDS):
            out.append("कृपया " + p)
    seen, uniq = set(), []
    for p in out:
        k = norm(p)
        if k not in seen:
            seen.add(k); uniq.append(p)
    item["prompts"] = uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/greet"))
    ap.add_argument("--mix-out", type=Path, default=Path("data/greet_mix"))
    ap.add_argument("--replay", type=Path, default=Path("data/instruct_mix2/train.txt"))
    ap.add_argument("--replay-lines", type=int, default=6000)
    ap.add_argument("--repeat", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    assert_answers_exclusive()
    before = sum(len(i["prompts"]) for i in INTENTS)
    for item in INTENTS:
        augment_prompts(item)
    after = sum(len(i["prompts"]) for i in INTENTS)
    print(f"paraphrase augmentation: {before} prompts -> {after}")
    filter_eval_prompts()
    assert_prompt_unique()

    rows: list[str] = []
    per_intent: dict[str, int] = {}
    for item in INTENTS:
        for prompt in item["prompts"]:
            for answer in item["answers"]:
                rows.append(f"<user> {prompt} <assistant> {answer}")
        per_intent[item["id"]] = len(item["prompts"]) * len(item["answers"])

    for carrier in FORBIDDEN_CARRIERS:
        for row in rows:
            if carrier in row.split("<assistant>")[0]:
                raise SystemExit(f"forbidden carrier prefix in prompt: {row[:60]}")

    assert_no_leak(rows)
    rng.shuffle(rows)

    val, train = [], []
    seen: set[str] = set()
    for row in rows:
        intent = next(i["id"] for i in INTENTS
                      if any(f"<user> {p} " in row for p in i["prompts"]))
        if intent not in seen and len(val) < len(INTENTS):
            val.append(row); seen.add(intent)
        else:
            train.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    # ---- training mix: greet block + filtered replay ----
    greet_prompts = {norm(p) for item in INTENTS for p in item["prompts"]}
    eval_prompts: set[str] = set()
    for name in EVAL_FILES:
        fp = ROOT / name
        if fp.exists():
            for line in fp.open(encoding="utf-8"):
                prompt = norm(json.loads(line).get("prompt", ""))
                if prompt:
                    eval_prompts.add(prompt)

    replay: list[str] = []
    dropped_persona = dropped_collision = dropped_eval_leak = dropped_corpus_fragment = 0
    if args.replay.exists():
        lines = [l.strip() for l in args.replay.open(encoding="utf-8") if l.strip()]
        rng.shuffle(lines)
        kept: list[str] = []
        for line in lines:
            head, _, tail = line.partition("<assistant>")
            # corpus fragments ("न र्व्१,३.३ ॥") leaked into greetings as answers
            if "॥" in tail or re.search(r"[०-९]", tail) or "र्व्" in tail:
                dropped_corpus_fragment += 1
                continue
            if any(pat in tail for pat in PERSONA_PATTERNS):
                dropped_persona += 1
                continue
            m = re.match(r"<user> (.*?) <assistant>", line)
            if m and norm(m.group(1)) in greet_prompts:
                dropped_collision += 1
                continue
            if m and norm(m.group(1)) in eval_prompts:
                dropped_eval_leak += 1
                continue
            kept.append(line)
            if len(kept) >= args.replay_lines:
                break
        replay = kept
    print(f"replay scrub: dropped {dropped_persona} false-persona rows, "
          f"{dropped_collision} rows contradicting a greet prompt, "
          f"{dropped_eval_leak} rows answering a held-out eval prompt, "
          f"{dropped_corpus_fragment} corpus fragments, kept {len(replay)}")

    greet_block = train * args.repeat
    rng.shuffle(greet_block)
    mix = greet_block + replay
    rng.shuffle(mix)

    args.mix_out.mkdir(parents=True, exist_ok=True)
    (args.mix_out / "train.txt").write_text("\n".join(mix) + "\n", encoding="utf-8")
    (args.mix_out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    def est_tokens(rs: list[str]) -> int:
        return int(sum(len(r) for r in rs) / 2.07)

    greet_tok = est_tokens(greet_block)
    replay_tok = est_tokens(replay)
    stats = {
        "greet_rows": len(train),
        "greet_val_rows": len(val),
        "greet_rows_after_repeat": len(greet_block),
        "repeat": args.repeat,
        "replay_rows": len(replay),
        "mix_rows": len(mix),
        "per_intent_rows": per_intent,
        "est_greet_tokens": greet_tok,
        "est_replay_tokens": replay_tok,
        "greet_share_of_tokens": round(greet_tok / max(1, greet_tok + replay_tok), 4),
        "answers_per_intent": {i["id"]: len(i["answers"]) for i in INTENTS},
        "prompts_per_intent": {i["id"]: len(i["prompts"]) for i in INTENTS},
        "forbidden_carriers_absent": True,
        "replay_dropped_persona_rows": dropped_persona,
        "replay_dropped_prompt_collisions": dropped_collision,
        "replay_dropped_eval_leaks": dropped_eval_leak,
        "replay_dropped_corpus_fragments": dropped_corpus_fragment,
        "answers_exclusive_to_one_intent": True,
        "eval_prompts_held_out": True,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    (args.mix_out / "mix_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items()
                      if k in ("greet_rows", "mix_rows", "per_intent_rows",
                               "greet_share_of_tokens", "est_greet_tokens")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
