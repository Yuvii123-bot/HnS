#!/usr/bin/env python3
"""
Score the 15 grammar questions of data/questions_100.jsonl (q038–q052).

Two answer paths are scored side by side, and both are held to the same rule:

    --router                 the table router (scripts/sanskrit_kb/router.py)
    --bundle serve           the deployed NumPy bundle
    --checkpoint x.pt        a PyTorch checkpoint

The benchmark prompts are *held out*: no training row in this repository
contains them (see scripts/build_vyakarana_data.py, which filters them from
train and val byte-for-byte). Scoring a checkpoint therefore measures whether
the model can produce the form, not whether it remembers the string.

    python3 scripts/eval_grammar.py --router --label router
    python3 scripts/eval_grammar.py --bundle serve --label deployed

Scoring is reported two ways, because the question set was written with a
substring rule and strictness matters:

    strict   an accepted answer appears as a whole word of the reply
    loose    an accepted answer appears anywhere in the reply

q049 accepts both `अष्ट` and `अष्टौ` (the stem and its inflected form), which is
what makes the two counts differ for a correct answer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRAMMAR_IDS = [f"q{i:03d}" for i in range(38, 53)]


def norm(text: str) -> str:
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹0123456789]", "", text)
    return re.sub(r"\s+", " ", text).strip()


DANDA = "।॥?!.,;:"


def score(reply: str, accept: list[str]) -> tuple[bool, bool]:
    """strict = the accepted form is a whole word; loose = it appears at all.

    The danda and the question mark are Devanagari code points, so they are
    removed before the word-boundary test — otherwise `रामाय।` would look like
    one long word and no answer could ever score strictly.
    """
    reply = norm(reply)
    loose = any(a in reply for a in accept)
    words = reply
    for ch in DANDA:
        words = words.replace(ch, " ")
    strict = any(re.search(rf"(?<![\u0900-\u097f]){re.escape(a)}(?![\u0900-\u097f])", words)
                 for a in accept)
    return strict, loose or strict


def load_items(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [it for it in items if it["id"] in GRAMMAR_IDS]


def model_reply(args) -> callable:
    from scripts.eval_exchanges import NumpyBackend, TorchBackend  # local import: slow
    if args.bundle:
        backend = NumpyBackend(args.bundle)
    else:
        backend = TorchBackend(args.checkpoint, args.tokenizer)
    return lambda prompt: backend.reply([], prompt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", action="store_true", help="score the table router")
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    ap.add_argument("--eval", type=Path, default=Path("data/questions_100.jsonl"))
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not (args.router or args.bundle or args.checkpoint):
        ap.error("choose --router, --bundle or --checkpoint")

    items = load_items(args.eval)
    if len(items) != 15:
        sys.exit(f"FAIL: expected 15 grammar items in {args.eval}, found {len(items)}")

    if args.router:
        from scripts.sanskrit_kb.router import answer
        reply_of = lambda p: (a.text if (a := answer(p)) else "")  # noqa: E731
        source = "router"
    else:
        reply_of = model_reply(args)
        source = str(args.bundle or args.checkpoint)

    rows, strict_n, loose_n = [], 0, 0
    for it in items:
        text = reply_of(it["prompt"])
        strict, loose = score(text, it["accept"])
        strict_n += strict
        loose_n += loose
        rows.append({"id": it["id"], "prompt": it["prompt"], "accept": it["accept"],
                     "reply": text, "strict": strict, "loose": loose})
        if args.verbose:
            mark = "ok  " if strict else ("~   " if loose else "MISS")
            print(f"  {mark} {it['id']} {it['prompt'][:44]:<46} {text[:34]}")

    n = len(items)
    print(f"\n  {args.label}: {source}")
    print(f"    grammar test15   {strict_n}/{n} strict  ({loose_n}/{n} loose)")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"label": args.label, "source": source,
              "test15": {"strict": strict_n, "loose": loose_n, "n": n},
              "records": rows}
    path = (args.out / f"grammar_{args.label}.json").resolve()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    wrote {path.relative_to(ROOT) if ROOT in path.parents else path}")


if __name__ == "__main__":
    main()
