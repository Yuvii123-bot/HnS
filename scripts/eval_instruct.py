#!/usr/bin/env python3
"""
Measure an instruction-tuned (or untuned) checkpoint on held-out Sanskrit.

Two kinds of number, deliberately separated:

  * teacher-forced loss / perplexity on held-out instruction pairs — objective,
    but only says "how likely is the reference answer", not "is it good";
  * generated-answer metrics against 38 evaluation prompts that appear nowhere
    in the training data: keyword coverage, topical overlap, format compliance,
    echo rate, and variety.

Nothing here is a judgement of semantic correctness — the script says what it
measures and does not pretend otherwise.

    python3 scripts/eval_instruct.py --checkpoint artifacts/model.pt --label before
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sanskrit_llm.data import TokenDataset, token_feature_table
from sanskrit_llm.model import ModelConfig, SanskritLM, stream_tokens
from sanskrit_llm.tokenizer import AksharaTokenizer

LATIN = re.compile(r"[A-Za-z]")
DEVANAGARI = re.compile(r"[\u0900-\u097f]")
# function words: too common to indicate that an answer is on topic
STOP = set("""अपि इति एव च हि तु वा यत् तत् यः सः सा ते त्वम् अहम् मम मया
भवान् भवतः भवती किम् कः का कस्मात् अस्ति सन्ति अस्मि आसीत् न एवम् इति
एषः एषा एतत् तत्र अत्र यदा तदा कुतः यतः तर्हि""".split())


def load_checkpoint(path: Path, tok: AksharaTokenizer) -> SanskritLM:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**blob["config"])
    model = SanskritLM(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s।॥?!,]+", text) if w and w not in STOP]


def teacher_forced(model: SanskritLM, feat, tok, bin_path: Path,
                   *, windows: int = 48, batch: int = 8) -> dict:
    if not bin_path.exists():
        return {}
    ds = TokenDataset(bin_path, feat, 256)
    rng = np.random.default_rng(0)
    n = len(ds.data) - 257
    idx = rng.choice(n, size=min(windows, n), replace=False)
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for start in range(0, len(idx), batch):
            chunk = idx[start:start + batch]
            x = np.stack([ds.data[i:i + 256] for i in chunk])
            y = np.stack([ds.data[i + 1:i + 257] for i in chunk])
            xb = torch.from_numpy(x).long()
            yb = torch.from_numpy(y).long()
            feats = ds.feat_table[xb]
            logits, loss, _ = model(xb, feats, targets=yb)
            losses.append(float(loss))
            correct += int((logits.argmax(-1) == yb).sum())
            total += int(yb.numel())
    mean = sum(losses) / len(losses)
    return {"loss": round(mean, 4), "ppl": round(float(np.exp(mean)), 2),
            "token_acc": round(correct / total, 4), "windows": len(losses)}


def generate(model: SanskritLM, feat, tok, prompt: str, *, seed: int,
             temperature: float = 0.0, max_new_tokens: int = 40) -> str:
    torch.manual_seed(seed)
    text = f"<user> {prompt} <assistant>"
    ids = tok.encode(text)
    ctx = model.cfg.max_seq_len
    if len(ids) >= ctx - 8:
        ids = ids[-(ctx - 8):]
    stop = {tok.pad_id, tok.bos_id, tok.unk_id,
            tok.special_id("user"), tok.special_id("assistant"), tok.special_id("system")}
    out = ""
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for token_id in stream_tokens(
                model, feat, x, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=40, top_p=0.95,
                repetition_penalty=1.15, eos_id=tok.eos_id):
            if token_id in stop:
                break
            piece = tok.vocab[token_id]
            if piece.startswith("<") or LATIN.search(piece):
                break
            if piece == "<eos>" or piece.startswith("\n"):
                break                                  # one-line answers only
            out += piece
            if len(out) >= 8 and (piece.strip() == "॥" or out.count("।") >= 2):
                break
    return out.strip()


def evaluate(model, feat, tok, prompts, *, seed_base: int = 1000,
             temperature: float = 0.0) -> dict:
    rows = []
    for i, item in enumerate(prompts):
        answer = generate(model, feat, tok, item["prompt"], seed=seed_base + i,
                          temperature=temperature)
        expected = item.get("expected", [])
        pw, aw = set(words(item["prompt"])), set(words(answer))
        overlap = len(pw & aw) / max(1, len(pw))
        rows.append({
            "category": item["category"],
            "prompt": item["prompt"],
            "answer": answer,
            "keyword_hit": any(k in answer for k in expected) if expected else False,
            "overlap": round(overlap, 3),
            "empty": not answer.strip(),
            "latin": bool(LATIN.search(answer)),
            "devanagari": bool(DEVANAGARI.search(answer)),
            "verse_marker": "॥" in answer,
            "arabic_digit": bool(re.search(r"[0-9]", answer)),
            "ends_punct": answer.endswith(("।", "?", "!")),
            "chars": len(answer),
        })

    n = len(rows)
    per_cat: dict[str, dict] = {}
    for r in rows:
        c = per_cat.setdefault(r["category"], {"n": 0, "keyword_hit": 0, "overlap": 0.0, "empty": 0})
        c["n"] += 1
        c["keyword_hit"] += int(r["keyword_hit"])
        c["overlap"] += r["overlap"]
        c["empty"] += int(r["empty"])
    for c in per_cat.values():
        c["keyword_hit"] = round(c["keyword_hit"] / c["n"], 3)
        c["overlap"] = round(c["overlap"] / c["n"], 3)
        c["empty"] = round(c["empty"] / c["n"], 3)

    answers = [r["answer"] for r in rows]
    summary = {
        "n_prompts": n,
        "keyword_hit_rate": round(sum(r["keyword_hit"] for r in rows) / n, 3),
        "mean_prompt_overlap": round(sum(r["overlap"] for r in rows) / n, 3),
        "empty_rate": round(sum(r["empty"] for r in rows) / n, 3),
        "latin_rate": round(sum(r["latin"] for r in rows) / n, 3),
        "devanagari_rate": round(sum(r["devanagari"] for r in rows) / n, 3),
        "verse_marker_rate": round(sum(r["verse_marker"] for r in rows) / n, 3),
        "arabic_digit_rate": round(sum(r["arabic_digit"] for r in rows) / n, 3),
        "ends_punct_rate": round(sum(r["ends_punct"] for r in rows) / n, 3),
        "unique_answer_rate": round(len(set(answers)) / n, 3),
        "mean_answer_chars": round(sum(r["chars"] for r in rows) / n, 1),
        "per_category": per_cat,
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    ap.add_argument("--prompts", type=Path, default=Path("data/instruct/eval_prompts.jsonl"))
    ap.add_argument("--label", type=str, default="model")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (reproducible capability probe); the UI samples at 0.85")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tok = AksharaTokenizer.load(args.tokenizer)
    feat = token_feature_table(tok)
    model = load_checkpoint(args.checkpoint, tok)

    prompts = [json.loads(l) for l in args.prompts.read_text(encoding="utf-8").splitlines() if l.strip()]

    report = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "params": model.num_params(),
        "instruct_val": teacher_forced(model, feat, tok, Path("data/bin_instruct/val.bin")),
        "verse_val": teacher_forced(model, feat, tok, Path("data/bin_clean/val.bin")),
    }
    report["temperature"] = args.temperature
    report.update(evaluate(model, feat, tok, prompts, temperature=args.temperature))

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.temperature == 0 else f"_t{args.temperature:g}"
    out = args.out / f"eval_{args.label}{suffix}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]
    print(f"\n=== {args.label} ===  params {report['params']:,}")
    print(f"  instruct val loss {report['instruct_val'].get('loss')} "
          f"(ppl {report['instruct_val'].get('ppl')}) | "
          f"verse val loss {report['verse_val'].get('loss')} "
          f"(ppl {report['verse_val'].get('ppl')})")
    print(f"  keyword hit {s['keyword_hit_rate']:.1%} | prompt overlap {s['mean_prompt_overlap']:.1%} "
          f"| empty {s['empty_rate']:.1%} | verse-marker {s['verse_marker_rate']:.1%}")
    print(f"  devanagari {s['devanagari_rate']:.1%} | ends with punctuation {s['ends_punct_rate']:.1%} "
          f"| unique answers {s['unique_answer_rate']:.1%} | mean {s['mean_answer_chars']} chars")
    print("\n  per category:")
    for c, m in sorted(s["per_category"].items()):
        print(f"    {c:<13} n={m['n']:<3} keyword {m['keyword_hit']:.0%}  overlap {m['overlap']:.0%}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
