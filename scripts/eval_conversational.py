#!/usr/bin/env python3
"""
Score a checkpoint on held-out Sanskrit *conversational* questions.

This is the class that failed in production: advice-seeking and meta questions
("किं करणीयम्?", "भवान् कः असि?"). None of these 40 prompts or their answers
appear in any training row; topics are disjoint from the corrective data.

Reported separately, because they fail differently from fact questions:
  content   strict + F1 against the reference
  boundary  the model stopped with <eos> instead of rolling into another turn
  grammar   rule checks from scripts/grammar_check.py
  format    punctuated, no verse, no template leakage

    python3 scripts/eval_conversational.py --checkpoint artifacts/model.pt --label before
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import grammar_check                                                   # noqa: E402
from eval_semantic import content_f1, distinctive_terms, load_checkpoint, strict_ok  # noqa: E402
from sanskrit_llm.data import token_feature_table                      # noqa: E402
from sanskrit_llm.model import stream_tokens                           # noqa: E402
from sanskrit_llm.tokenizer import AksharaTokenizer                    # noqa: E402

SPECIAL = {"<eos>", "<user>", "<assistant>", "<pad>", "<bos>", "<unk>", "<system>"}


def generate(model, tok, feat, prompt: str, max_new: int = 40) -> tuple[str, str]:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    text, stop = "", "budget"
    with torch.no_grad():
        for tid in stream_tokens(model, feat, ids, max_new_tokens=max_new, temperature=0.0,
                                 top_k=40, top_p=1.0, repetition_penalty=1.15, eos_id=None):
            piece = tok.vocab[tid]
            if piece in SPECIAL:
                stop = piece
                break
            if "\n" in piece or piece.strip() == "॥":
                stop = "newline"
                break
            text += piece
    return text.strip(), stop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--eval-set", type=Path,
                    default=ROOT / "data" / "conversational_eval.jsonl")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tokenizer.json")
    args = ap.parse_args()

    tok = AksharaTokenizer.load(args.tokenizer)
    feat = token_feature_table(tok)
    model = load_checkpoint(args.checkpoint)
    items = [json.loads(l) for l in args.eval_set.read_text(encoding="utf-8").splitlines() if l.strip()]
    musts = distinctive_terms([i["reference"] for i in items])

    rows = []
    for item in items:
        answer, stop = generate(model, tok, feat, f"<user> {item['prompt']} <assistant>")
        strict = strict_ok(item["reference"], answer, musts.get(item["reference"], []))
        f1 = content_f1(item["reference"], answer)
        g = grammar_check.check(answer, item["prompt"])
        boundary = stop == "<eos>"
        fmt = bool(answer) and answer.endswith(("।", "?", "!")) and "॥" not in answer \
            and "\n" not in answer
        rows.append({
            "id": item["id"], "topic": item["topic"], "prompt": item["prompt"],
            "reference": item["reference"], "answer": answer, "stop": stop,
            "strict_ok": strict, "content_f1": f1, "grammar_ok": g["ok"],
            "grammar_issues": g["issues"], "boundary_ok": boundary, "format_ok": fmt,
        })

    n = len(rows)
    summary = {
        "n": n,
        "strict_ok": round(sum(r["strict_ok"] for r in rows) / n, 4),
        "content_f1": round(sum(r["content_f1"] for r in rows) / n, 4),
        "grammar_ok": round(sum(r["grammar_ok"] for r in rows) / n, 4),
        "boundary_ok": round(sum(r["boundary_ok"] for r in rows) / n, 4),
        "format_ok": round(sum(r["format_ok"] for r in rows) / n, 4),
    }
    by_topic = {}
    for topic in sorted({r["topic"] for r in rows}):
        sub = [r for r in rows if r["topic"] == topic]
        by_topic[topic] = {
            "n": len(sub),
            "strict_ok": round(sum(r["strict_ok"] for r in sub) / len(sub), 4),
            "boundary_ok": round(sum(r["boundary_ok"] for r in sub) / len(sub), 4),
        }

    out = ROOT / "reports" / f"conv_{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "checkpoint": str(args.checkpoint),
                               "summary": summary, "by_topic": by_topic, "rows": rows},
                              ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n=== {args.label} === {n} held-out conversational questions")
    for k, v in summary.items():
        if k != "n":
            print(f"  {k:<12} {v:.1%}")
    for topic, m in by_topic.items():
        print(f"    {topic:<16} n={m['n']:<3} strict {m['strict_ok']:.0%}  boundary {m['boundary_ok']:.0%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
