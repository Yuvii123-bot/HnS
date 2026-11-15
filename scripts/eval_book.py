#!/usr/bin/env python3
"""
Score what the model actually took from Hitopadeśa.

Two kinds of item:
  complete_line — the first half of a held-out line is given and the rest is
                  expected; scored by character similarity, so a paraphrase of
                  the same continuation still counts if it is close.
  recite_new    — asked for a verse, any line of the book counts (substring
                  match), which measures whether the book's language is
                  available at all rather than a single memorised verse.

    python3 scripts/eval_book.py --checkpoint artifacts/book2/latest.pt --label book2
    python3 scripts/eval_book.py --bundle serve --label deployed
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPECIAL = {"<eos>", "<user>", "<assistant>", "<pad>", "<bos>", "<unk>", "<system>"}
LATIN = re.compile(r"[A-Za-z]")


def norm(s: str) -> str:
    s = re.sub(r"[।॥!?,\--]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def backend_from_args(args):
    if args.checkpoint:
        import torch
        from sanskrit_llm.data import token_feature_table
        from sanskrit_llm.model import load_checkpoint, stream_tokens
        from sanskrit_llm.tokenizer import AksharaTokenizer

        model, _, _ = load_checkpoint(args.checkpoint)
        model.eval()
        tok = AksharaTokenizer.load(args.tokenizer)
        feat = token_feature_table(tok)

        def reply(prompt: str, max_new: int = 48) -> str:
            ids = torch.tensor([tok.encode(f"<user> {prompt} <assistant>")])
            out = ""
            with torch.no_grad():
                for tid in stream_tokens(model, feat, ids, max_new_tokens=max_new,
                                         temperature=0.0, top_k=40, top_p=1.0,
                                         repetition_penalty=1.15, eos_id=None):
                    piece = tok.vocab[tid]
                    if piece in SPECIAL or "\n" in piece:
                        break
                    out += piece
            return out.strip()
        return reply, str(args.checkpoint)

    import numpy as np
    sys.path.insert(0, str(args.bundle / "runtime"))
    from sanskrit_llm.numpy_lm import Runtime

    rt = Runtime(args.bundle)
    tok = rt.tokenizer
    stop = {tok.pad_id, tok.bos_id, tok.unk_id, tok.special_id("user"),
            tok.special_id("assistant"), tok.special_id("system")}

    def reply(prompt: str, max_new: int = 48) -> str:
        ids = tok.encode(f"<user> {prompt} <assistant>")
        out = ""
        for tid, line_end in rt.model.sample(
            np.array(ids), temperature=0.0, top_k=40, top_p=1.0,
            repetition_penalty=1.15, max_new_tokens=max_new, max_lines=2,
            eos_id=tok.eos_id, feat_table=rt.feat_table, stop_token_ids=stop,
        ):
            if line_end:
                break
            piece = tok.vocab[tid]
            if piece in SPECIAL or piece.startswith("<") or LATIN.search(piece):
                break
            out += piece
        return out.strip()
    return reply, str(args.bundle)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    ap.add_argument("--probe", type=Path, default=Path("data/book_probe.jsonl"))
    ap.add_argument("--book", type=Path, default=Path("data/book/train.txt"))
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    reply, source = backend_from_args(args)
    items = [json.loads(l) for l in args.probe.open(encoding="utf-8")]
    book_text = norm((args.book).read_text(encoding="utf-8"))

    rows, passed = [], 0
    for item in items:
        out = reply(item["prompt"])
        kind = item["kind"]
        if kind in ("recall_trained", "recall_heldout"):
            ratio = difflib.SequenceMatcher(None, norm(out), norm(item["target"])).ratio()
            ok = ratio >= args.threshold
            detail = f"sim {ratio:.2f}"
        elif kind == "fact":
            n = norm(out)
            ok = norm(item["accept"]) in n or n in norm(item["accept"])
            detail = "fact ok" if ok else "wrong fact"
        else:  # recite / narrate: any 18-character stretch of a book line counts
            n = norm(out)
            ok = any(n[i:i + 18] in book_text for i in range(0, max(1, len(n) - 18)))
            detail = "from book" if ok else "not from book"
        passed += ok
        rows.append({**item, "reply": out, "passed": bool(ok), "detail": detail})
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {item['id']} {item['kind']:13s} -> {out[:58]}   ({detail})")

    total = len(rows)
    summary = {"label": args.label, "source": source, "items": total, "passed": passed,
               "accuracy": round(passed / total, 4),
               "by_kind": {}}
    for kind in ("recall_trained", "recall_heldout", "recite", "narrate", "fact"):
        sub = [r for r in rows if r["kind"] == kind]
        if sub:
            summary["by_kind"][kind] = {"n": len(sub), "pass": sum(r["passed"] for r in sub),
                                        "rate": round(sum(r["passed"] for r in sub) / len(sub), 3)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"book_{args.label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n{args.label}: {passed}/{total} = {100*passed/total:.1f}%  {summary['by_kind']}")
    print(f"wrote {args.out / f'book_{args.label}.json'}")


if __name__ == "__main__":
    main()
