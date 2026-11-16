#!/usr/bin/env python3
"""The 500-question exam, run automatically when the 250M budget is spent.

Pass mark, set by the user: **more than 400 correct out of 500**. If the run
falls short, this script launches the 100M-token extension (350M tokens total,
85,449 steps) on its own and writes the decision into the report.

The battery is 100 items in each of five domains:

| domain | source | how it is scored |
|---|---|---|
| conversation | `data/questions_100.jsonl` (100 exchanges, 25 of them the held-out prompts) | an accepted phrase appears in the reply |
| arithmetic | generated here from the same number table the curriculum uses | the expected number word appears as a whole word |
| grammar | 100 rows sampled from `data/vyakarana_train/train.txt` | the expected form appears as a whole word |
| time | weekday arithmetic computed here | the expected weekday appears as a whole word |
| knowledge | short factual turns from `data/instruct_v2/train.txt` | the answer's key content word appears |

Honesty note carried into the report: only the 25 prompts that the curriculum
filters out byte-for-byte are truly held out. The other 475 items are drawn
from the same domains as training data, so they measure *behaviour*, not
generalisation. Both numbers are reported.

    python3 scripts/post_run_250.py                 # full exam, extends if <400
    python3 scripts/post_run_250.py --limit 5 --no-extend   # smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from sanskrit_llm.data import token_feature_table
from sanskrit_llm.model import load_checkpoint
from sanskrit_llm.tokenizer import AksharaTokenizer

import build_numbers_time as bnt  # NUM, WEEKDAYS, SEASONS, frames

OUT = ROOT / "artifacts" / "run250"
REPORTS = ROOT / "reports"
PASS_MARK = 400
EXTEND_STEPS = 85449          # 350,000,000 tokens / 4096
TOKENS_PER_STEP = 4096
FUNCTION_WORDS = {"इत्यस्य", "इत्यस्मात्", "अस्ति", "भवति", "भवन्ति", "इति", "किम्",
                  "किं", "तत्र", "अत्र", "एतत्", "तत्", "यत्", "सः", "सा", "ते", "वा",
                  "अपि", "च", "हि", "एव", "उत", "अथ", "तथा", "यथा", "कृते", "कृत्य",
                  "विषये", "मम", "तव", "अस्य", "अस्मिन्", "एतेषाम्"}

DANDA = "।॥?!.,;:\"'()[]{}—–-"


# ---------------------------------------------------------------- scoring

def norm(text: str) -> str:
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹0123456789]", "", text)
    for ch in DANDA:
        text = text.replace(ch, " ")
    return re.sub(r"\s+", " ", text).strip()


def whole_word_hit(reply: str, accept: list[str]) -> bool:
    words = norm(reply).split()
    for cand in accept:
        cw = norm(cand).split()
        if not cw:
            continue
        for i in range(len(words) - len(cw) + 1):
            if words[i:i + len(cw)] == cw:
                return True
    return False


def loose_hit(reply: str, accept: list[str]) -> bool:
    r = norm(reply)
    return any(norm(a) and norm(a) in r for a in accept)


# ------------------------------------------------------------ question sets

def load_jsonl_questions() -> list[dict]:
    rows = []
    path = ROOT / "data" / "questions_100.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        rows.append({
            "domain": "conversation", "prompt": q["prompt"],
            "accept": q["accept"], "held_out": q["id"] in
            ({f"q{i:03d}" for i in range(38, 53)} | {f"q{i:03d}" for i in range(63, 73)}),
            "id": q["id"],
        })
    return rows


def arith_questions(rng: random.Random, n: int = 100) -> list[dict]:
    num = bnt.NUM
    keys = sorted(num)
    out: list[dict] = []
    ops = ["+", "-", "×"]
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        op = rng.choice(ops)
        if op == "+":
            a, b = rng.randint(0, 60), rng.randint(0, 40)
            res = a + b
        elif op == "-":
            a, b = rng.randint(10, 90), rng.randint(0, 9)
            res = a - b
        else:
            a, k = rng.randint(0, 20), rng.choice([2, 3, 4])
            res = a * k
        if res not in keys or a not in keys or (op != "x" and b not in keys):
            continue
        if op == "+":
            prompt = f"{num[a]} + {num[b]} = किम्?"
        elif op == "-":
            prompt = f"{num[a]} - {num[b]} = किम्?"
        else:
            prompt = f"{num[a]} × {num[k]} = किम्?"
        out.append({"domain": "arithmetic", "prompt": prompt,
                    "accept": [num[res]], "held_out": False,
                    "id": f"a{len(out):03d}"})
    # number-word identification, the other half of the number skill
    while len(out) < n:
        v = rng.choice([v for v in keys if v > 10])
        frame = rng.choice(bnt.NUMBER_WORD_FRAMES)
        prompt = frame.format(word=num[v])
        out.append({"domain": "arithmetic", "prompt": prompt,
                    "accept": [num[v]], "held_out": False,
                    "id": f"a{len(out):03d}"})
    return out


def grammar_questions(rng: random.Random, n: int = 100) -> list[dict]:
    path = ROOT / "data" / "vyakarana_train" / "train.txt"
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if "<assistant>" in l]
    rng.shuffle(lines)
    out = []
    for line in lines:
        if len(out) >= n:
            break
        q, _, a = line.partition("<assistant>")
        q = q.replace("<user>", "").strip()
        a = a.strip()
        if not q or not a:
            continue
        forms = [f.strip(DANDA + " ") for f in re.split(r"[-–/]", a) if f.strip(DANDA + " ")]
        if not forms:
            continue
        out.append({"domain": "grammar", "prompt": q, "accept": list(dict.fromkeys([a.strip(DANDA + " ")] + forms)),
                    "held_out": False, "id": f"g{len(out):03d}"})
    return out


def time_questions(rng: random.Random, n: int = 100) -> list[dict]:
    wd = bnt.WEEKDAYS
    out: list[dict] = []
    frames = [
        ("अद्य {a}वासरः अस्ति। श्वः कः वासरः भविष्यति?", 1),
        ("अद्य {a}वासरः अस्ति। ह्यः कः वासरः आसीत्?", -1),
        ("अद्य {a}वासरः। परश्वः कः वासरः भविष्यति?", 2),
        ("{a}वासरात् अनन्तरं कः वासरः आगच्छति?", 1),
    ]
    while len(out) < n:
        i = rng.randrange(7)
        frame, delta = rng.choice(frames)
        day = wd[i].replace("वासरः", "")     # frames add the word themselves
        out.append({"domain": "time", "prompt": frame.format(a=day),
                    "accept": [wd[(i + delta) % 7]], "held_out": False,
                    "id": f"t{len(out):03d}"})
    while len(out) < n:
        s = rng.choice(bnt.SEASONS)
        out.append({"domain": "time",
                    "prompt": f"कः ऋतुः {s} इति कथ्यते?", "accept": [s],
                    "held_out": False, "id": f"t{len(out):03d}"})
    return out


def knowledge_questions(rng: random.Random, n: int = 100) -> list[dict]:
    path = ROOT / "data" / "instruct_v2" / "train.txt"
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if "<assistant>" in l]
    rng.shuffle(lines)
    out = []
    for line in lines:
        if len(out) >= n:
            break
        q, _, a = line.partition("<assistant>")
        q = q.replace("<user>", "").strip()
        a = a.strip()
        if not (8 < len(a) < 90) or not q:
            continue
        words = [w for w in norm(a).split()
                 if len(w) >= 4 and w not in FUNCTION_WORDS]
        if not words:
            continue
        # the answer's first two content words: a fair target for a one-line reply
        keys = list(dict.fromkeys(words[:2]))
        out.append({"domain": "knowledge", "prompt": q, "accept": keys,
                    "held_out": False, "id": f"k{len(out):03d}"})
    return out


def build_battery() -> list[dict]:
    rng = random.Random(250)
    items = load_jsonl_questions()
    items += arith_questions(rng)
    items += grammar_questions(rng)
    items += time_questions(rng)
    items += knowledge_questions(rng)
    return items


# --------------------------------------------------------------- inference

@torch.no_grad()
def ask(model, tok, feat, prompt: str, *, max_new: int = 48) -> str:
    ids = tok.encode(f"<user> {prompt} <assistant>")
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, feat, max_new_tokens=max_new, temperature=1.0,
                         top_k=1, top_p=1.0, repetition_penalty=1.0,
                         eos_id=tok.eos_id)
    new = out[0, len(ids):].tolist()
    return tok.decode(new)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=OUT / "model.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-extend", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    REPORTS.mkdir(exist_ok=True)
    if not args.checkpoint.exists():
        print(f"no checkpoint at {args.checkpoint}")
        return 1

    tok = AksharaTokenizer.load(ROOT / "artifacts" / "tokenizer.json")
    model, _tok, meta = load_checkpoint(args.checkpoint)
    feat = token_feature_table(tok)
    print(f"exam on {args.checkpoint} (step {meta.get('step')}, "
          f"{model.num_params():,} params)")

    items = build_battery()
    if args.limit:
        items = items[: args.limit]

    results = []
    for i, item in enumerate(items, 1):
        reply = ask(model, tok, feat, item["prompt"])
        ok = (whole_word_hit(reply, item["accept"]) or loose_hit(reply, item["accept"])
              if item["domain"] == "conversation" else whole_word_hit(reply, item["accept"]))
        results.append({**item, "reply": reply.strip(), "correct": bool(ok)})
        if i % 50 == 0:
            print(f"  {i}/{len(items)}  running score "
                  f"{sum(r['correct'] for r in results)}/{len(results)}", flush=True)

    total = len(results)
    correct = sum(r["correct"] for r in results)
    by_domain: dict[str, dict] = {}
    for r in results:
        d = by_domain.setdefault(r["domain"], {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(r["correct"])
    held = [r for r in results if r["held_out"]]
    held_correct = sum(r["correct"] for r in held)
    frac = correct / max(1, total)

    verdict = "not scored" if args.limit else (
        "PASS" if correct > PASS_MARK else "SHORT — 100M extension")
    print(f"\nscore {correct}/{total} ({frac:.1%})  held-out {held_correct}/{len(held)}")
    for d, v in sorted(by_domain.items()):
        print(f"  {d:13s} {v['correct']:3d}/{v['n']}")
    print(f"verdict: {verdict}")

    if not args.limit:
        payload = {
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "step": meta.get("step"), "params": model.num_params(),
            "total": total, "correct": correct, "fraction": round(frac, 4),
            "pass_mark": PASS_MARK, "verdict": verdict, "by_domain": by_domain,
            "held_out": {"n": len(held), "correct": held_correct,
                         "ids": [r["id"] for r in held if not r["correct"]]},
            "caveat": "only the 25 filtered prompts are strictly held out; the rest "
                      "are same-domain behaviour checks",
            "items": results,
        }
        (REPORTS / "run250_exam500.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rows = "\n".join(f"| {d} | {v['correct']} | {v['n']} |"
                         for d, v in sorted(by_domain.items()))
        (REPORTS / "run250_exam500.md").write_text(
            f"""# 500-question exam

checkpoint `{args.checkpoint.name}` at step **{meta.get('step')}**

**{correct} / {total} correct ({frac:.1%})** — pass mark {PASS_MARK}. Verdict: **{verdict}**

| domain | correct | asked |
|---|---|---|
{rows}

Strictly held out (the 25 prompts filtered from training): **{held_correct}/{len(held)}**.

{item_failures(results)}
""", encoding="utf-8")

        if correct <= PASS_MARK and not args.no_extend:
            hook = ROOT / "scripts" / "post_run_350.py"
            print(f"below pass mark -> extending to 350M tokens "
                  f"({EXTEND_STEPS} steps), hook {hook.name}")
            cmd = [sys.executable, str(ROOT / "scripts" / "supervise_run.py"),
                   "--steps", str(EXTEND_STEPS),
                   "--budget-tokens", str(EXTEND_STEPS * TOKENS_PER_STEP),
                   "--out", str(OUT), "--bin-dir", str(ROOT / "data" / "bin_250m"),
                   "--batch", "16", "--ctx", "256", "--d-model", "288",
                   "--layers", "8", "--heads", "8", "--d-ff", "880",
                   "--lr", "5e-4", "--warmup", "500", "--threads", "2",
                   "--eval-every", "4000", "--ckpt-every", "2000",
                   "--hook", str(hook)]
            with open(OUT / "extend.out", "ab") as fh:
                subprocess.Popen(["setsid"] + cmd, cwd=str(ROOT), stdout=fh,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            (OUT / "EXTENDING").write_text(json.dumps(
                {"from_step": meta.get("step"), "to_step": EXTEND_STEPS,
                 "reason": f"{correct}/{total} below {PASS_MARK}"}, indent=2))
    return 0


def item_failures(results: list[dict], n: int = 12) -> str:
    bad = [r for r in results if not r["correct"]][:n]
    if not bad:
        return "No failures to show."
    out = ["", "## First failures", "", "| domain | prompt | expected | reply |", "|---|---|---|---|"]
    for r in bad:
        out.append(f"| {r['domain']} | {r['prompt'][:60]} | "
                   f"{r['accept'][0][:25]} | {(r['reply'] or '—')[:60]} |")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
