#!/usr/bin/env python3
"""
Phase-2 evaluation: meaning, relevance, grammar, instruction-following.

Keyword coverage (phase 1) only asked whether a word appeared. This harness
scores four separate things on held-out questions and reports them apart, so a
model cannot look good by being fluent but empty:

  content  F1 of stemmed content words against the reference answer
  type     the answer is the right *kind* (a numeral for कति, a case name for
           कस्यां विभक्तौ, a nominal form for a paradigm question)
  grammar  the answer passes the rule checks in scripts/grammar_check.py
  format   one or two sentences, punctuated, no verse, no echo, no repetition

composite := 0.45·content + 0.15·type + 0.25·grammar + 0.15·format

The weights are a judgement call and are stated rather than hidden. Nothing
here understands Sanskrit; the limits are spelled out in the report.

    python3 scripts/eval_semantic.py --checkpoint artifacts/model.pt --label baseline
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import grammar_check                                  # noqa: E402
from sanskrit_llm.data import token_feature_table     # noqa: E402
from sanskrit_llm.model import ModelConfig, SanskritLM, stream_tokens  # noqa: E402
from sanskrit_llm.tokenizer import AksharaTokenizer   # noqa: E402

LATIN = re.compile(r"[A-Za-z]")
STOP = set("""अपि इति एव च हि तु वा यत् तत् यः सः सा ते त्वम् अहम् मम मया
भवान् भवतः भवती किम् कः का कस्मात् अस्ति सन्ति अस्मि आसीत् न एषः एषा एतत्
तत्र अत्र यदा तदा कुतः यतः तर्हि मह्यं तस्य तस्याः एतानि अयम् इयम्""".split())
NUMERALS = set("""एकम् द्वे त्रीणि चत्वारि पञ्च षट् सप्त अष्ट नव दश एकादश द्वादश
त्रयोदश चतुर्दश पञ्चदश षोडश सप्तदश अष्टादश नवदश विंशतिः शतम् सहस्रम्""".split())
DEV_DIGITS = set("०१२३४५६७८९")


def load_checkpoint(path: Path) -> SanskritLM:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = SanskritLM(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s।॥?!,;:\"“”()]+", text) if w]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOP and len(w) > 1]


def _prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def same_word(a: str, b: str) -> bool:
    """Same lemma, different inflection — compared on a shared stem."""
    if a == b:
        return True
    return _prefix(a, b) >= max(4, min(len(a), len(b)) - 2)


def content_f1(reference: str, answer: str) -> float:
    ref = content_words(reference)
    got = content_words(answer)
    if not ref or not got:
        return 0.0
    used = [False] * len(got)
    hits = 0
    for r in ref:
        for i, g in enumerate(got):
            if not used[i] and same_word(r, g):
                used[i] = True
                hits += 1
                break
    if not hits:
        return 0.0
    precision = hits / len(got)
    recall = hits / len(ref)
    return round(2 * precision * recall / (precision + recall), 4)


def distinctive_terms(references: list[str], max_df: float = 0.06) -> dict[str, list[str]]:
    """Per reference: the words that actually carry its content.

    Template words ("पदं", "विभक्तौ", "इति") show up in many references; a match
    on them says nothing. Anything appearing in more than `max_df` of the
    reference answers is dropped, leaving the terms a correct answer must
    actually produce.
    """
    df: dict[str, int] = {}
    per_item = [content_words(r) for r in references]
    for ws in per_item:
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    limit = max(3, int(max_df * max(len(references), 1)))
    out = {}
    for ref, ws in zip(references, per_item):
        keep = []
        for w in ws:
            if df.get(w, 0) <= limit or len(w) >= 6:
                if not any(same_word(w, k) for k in keep):
                    keep.append(w)
        out[ref] = keep[:4]
    return out


def strict_ok(reference: str, answer: str, must: list[str]) -> bool:
    """Every distinctive term of the reference appears in the answer."""
    if not must:
        return False
    got = content_words(answer)
    return all(any(same_word(m, g) for g in got) for m in must)


def type_ok(expected: str, answer: str, question: str) -> bool:
    ws = words(answer)
    if expected == "numeral":
        return any(w in NUMERALS for w in ws) or any(c in DEV_DIGITS for c in answer)
    if expected == "vibhakti":
        stems = ("प्रथम", "द्वितीय", "तृतीय", "चतुर्थ", "पञ्चम", "षष्ठ", "सप्तम")
        return any(any(w.startswith(s) for s in stems) for w in ws)
    if expected == "form":
        return len(ws) >= 1 and not all(w in question for w in ws)
    # free sentence: must be more than a single bare word
    return len(ws) >= 2


def format_ok(answer: str, question: str) -> bool:
    ws = words(answer)
    if not ws:
        return False
    if not answer.rstrip().endswith(("।", "?", "!")):
        return False
    if len(ws) > 30 or "\n" in answer or "॥" in answer:
        return False
    if LATIN.search(answer):
        return False
    if any(ws[i] == ws[i + 1] == ws[i + 2] for i in range(len(ws) - 2)):
        return False
    if answer.strip().rstrip("।?!").strip() == question.strip().rstrip("।?!").strip():
        return False
    return True


def build_prompt(item: dict) -> str:
    parts = []
    for turn in item.get("history", []):
        parts.append(f"<user> {turn['user']} <assistant> {turn['assistant']}")
    parts.append(f"<user> {item['prompt']} <assistant>")
    return " ".join(parts)


def generate(model, tok, feat, prompt: str, max_new: int = 42) -> str:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = ""
    with torch.no_grad():
        for tid in stream_tokens(model, feat, ids, max_new_tokens=max_new,
                                 temperature=0.0, top_k=40, top_p=1.0,
                                 repetition_penalty=1.05, eos_id=tok.eos_id):
            piece = tok.vocab[tid]
            if piece in ("<eos>", "<pad>", "<bos>", "<unk>"):
                break
            if piece.startswith("<") and piece.endswith(">"):
                break
            if "\n" in piece or piece.strip() == "॥":
                break
            if LATIN.search(piece):
                break
            out += piece
            if out.count("।") >= 2 and out.rstrip().endswith("।"):
                break
    return out.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tokenizer.json")
    ap.add_argument("--eval-set", type=Path, default=ROOT / "data" / "instruct_v2" / "eval_set.jsonl")
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = every item")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    tok = AksharaTokenizer.load(args.tokenizer)
    feat = token_feature_table(tok)
    model = load_checkpoint(args.checkpoint)

    items = [json.loads(l) for l in args.eval_set.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]

    musts = distinctive_terms([i["reference"] for i in items])
    rows = []
    for n, item in enumerate(items):
        answer = generate(model, tok, feat, build_prompt(item))
        f1 = content_f1(item["reference"], answer)
        strict = strict_ok(item["reference"], answer, musts.get(item["reference"], []))
        t_ok = type_ok(item["expected_type"], answer, item["prompt"])
        g = grammar_check.check(answer, item["prompt"])
        f_ok = format_ok(answer, item["prompt"])
        composite = 0.40 * strict + 0.15 * f1 + 0.15 * t_ok + 0.20 * g["ok"] + 0.10 * f_ok
        rows.append({
            "id": item["id"], "category": item["category"], "kind": item["kind"],
            "prompt": item["prompt"], "reference": item["reference"],
            "answer": answer, "content_f1": f1, "strict_ok": strict,
            "must_contain": musts.get(item["reference"], []), "type_ok": t_ok,
            "grammar_ok": g["ok"], "grammar_issues": g["issues"],
            "format_ok": f_ok, "composite": round(composite, 4),
        })
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(items)}", flush=True)

    def agg(subset: list[dict]) -> dict:
        n = len(subset) or 1
        return {
            "n": len(subset),
            "strict_ok": round(sum(r["strict_ok"] for r in subset) / n, 4),
            "content_f1": round(sum(r["content_f1"] for r in subset) / n, 4),
            "type_ok": round(sum(r["type_ok"] for r in subset) / n, 4),
            "grammar_ok": round(sum(r["grammar_ok"] for r in subset) / n, 4),
            "format_ok": round(sum(r["format_ok"] for r in subset) / n, 4),
            "composite": round(sum(r["composite"] for r in subset) / n, 4),
        }

    by_kind = {}
    for kind in sorted({r["kind"] for r in rows}):
        by_kind[kind] = agg([r for r in rows if r["kind"] == kind])
    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        by_cat[cat] = agg([r for r in rows if r["category"] == cat])

    grammar_rules: dict[str, int] = {}
    for r in rows:
        for issue in r["grammar_issues"]:
            grammar_rules[issue["rule"]] = grammar_rules.get(issue["rule"], 0) + 1

    result = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "n_items": len(rows),
        "summary": agg(rows),
        "by_kind": by_kind,
        "by_category": by_cat,
        "grammar_rule_failures": dict(sorted(grammar_rules.items(), key=lambda kv: -kv[1])),
        "weights": {"strict": 0.40, "content_f1": 0.15, "type": 0.15,
                    "grammar": 0.20, "format": 0.10},
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"sem_{args.label}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"\n=== {args.label} ===  {len(rows)} held-out questions")
    for key in ("strict_ok", "content_f1", "type_ok", "grammar_ok", "format_ok", "composite"):
        print(f"  {key:<11} {result['summary'][key]:.1%}")
    print("  by kind:")
    for kind, m in by_kind.items():
        print(f"    {kind:<22} n={m['n']:<4} strict {m['strict_ok']:.0%}  content {m['content_f1']:.0%}  "
              f"grammar {m['grammar_ok']:.0%}  composite {m['composite']:.0%}")
    if grammar_rules:
        print("  grammar rule failures:", grammar_rules)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
