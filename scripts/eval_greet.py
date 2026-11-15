#!/usr/bin/env python3
"""
Score greeting / basic-exchange behaviour on the held-out set.

A greeting has many correct answers, so scoring is intent-level: each item
carries an `accept` list and passes if any listed word or phrase occurs in the
reply. Exact-string matching would punish a correct but differently phrased
greeting, which is the opposite of what we want to measure.

The prompts in data/greet_eval.jsonl do not occur in training — asserted by
scripts/build_greet_data.py at build time — so this measures generalisation to
unseen phrasings, not recall.

    python3 scripts/eval_greet.py --checkpoint artifacts/x1_final/best.pt --label base
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from sanskrit_llm.data import token_feature_table
from sanskrit_llm.model import load_checkpoint, stream_tokens
from sanskrit_llm.tokenizer import AksharaTokenizer

SPECIAL = {"<eos>", "<user>", "<assistant>", "<pad>", "<bos>", "<unk>", "<system>"}


def norm(s: str) -> str:
    s = re.sub(r"[।॥!?,\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def generate(model, tok, feat, prompt: str, max_new: int = 40) -> str:
    ids = torch.tensor([tok.encode(f"<user> {prompt} <assistant>")], dtype=torch.long)
    out = ""
    with torch.no_grad():
        for tid in stream_tokens(model, feat, ids, max_new_tokens=max_new, temperature=0.0,
                                 top_k=40, top_p=1.0, repetition_penalty=1.15, eos_id=None):
            piece = tok.vocab[tid]
            if piece in SPECIAL or "\n" in piece:
                break
            out += piece
    return out.strip()


def score(answer: str, accept: list[str]) -> bool:
    n = norm(answer)
    return any(norm(a) and norm(a) in n for a in accept)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--eval", type=Path, default=Path("data/greet_eval.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model, _, _ = load_checkpoint(args.checkpoint)
    model.eval()
    tok = AksharaTokenizer.load("artifacts/tokenizer.json")
    feat = token_feature_table(tok)

    items = [json.loads(l) for l in args.eval.open(encoding="utf-8")]
    results, passed = [], 0
    by_intent: dict[str, list[bool]] = {}

    for item in items:
        answer = generate(model, tok, feat, item["prompt"])
        ok = score(answer, item["accept"])
        passed += ok
        by_intent.setdefault(item["intent"], []).append(ok)
        results.append({**item, "answer": answer, "pass": ok})
        if args.verbose or not ok:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {item['id']:4s} {item['intent']:18s} {item['prompt'][:34]:34s} -> {answer[:46]}")

    total = len(items)
    acc = passed / total
    print(f"\n{args.label}: {passed}/{total} = {acc:.1%} intent accuracy")
    print("per intent:")
    for intent in sorted(by_intent):
        vals = by_intent[intent]
        print(f"  {intent:20s} {sum(vals)}/{len(vals)}")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"greet_{args.label}.json"
    out.write_text(json.dumps({
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "accuracy": round(acc, 4),
        "passed": passed,
        "total": total,
        "per_intent": {k: [sum(v), len(v)] for k, v in by_intent.items()},
        "failures": [{"id": r["id"], "intent": r["intent"], "prompt": r["prompt"],
                      "answer": r["answer"]} for r in results if not r["pass"]],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if acc == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
