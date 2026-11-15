#!/usr/bin/env python3
"""
Generalisation probe: cells of ordinary nouns that appear in no drill.

The 15 benchmark questions use five words (राम, लता, फलम्, अश्व, बालक). A model can
score 15/15 on them by memorising those five paradigms and still know no
declension — which is exactly what the 5.8 M checkpoints did (1/56 on this probe).
This script asks for four cells of fourteen everyday nouns held out of the
inventory, so the two answer paths can be compared on the same question:

    python3 scripts/eval_grammar_unseen.py --router
    python3 scripts/eval_grammar_unseen.py --bundle serve
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_grammar import score  # noqa: E402

#: everyday nouns kept out of every training run and out of the benchmark
HELDOUT_NOUNS = [
    ("पुत्र", "P"), ("अग्नि", "P"), ("शरीर", "N"), ("मन्त्र", "P"),
    ("भार्या", "S"), ("देव", "P"), ("नगर", "N"), ("वृक्ष", "P"),
    ("सेना", "S"), ("जल", "N"), ("राजन्", "P"), ("कन्या", "S"),
    ("मित्र", "N"), ("हस्त", "P"),
]
CELLS = [("प्रथमा", "एकवचनम्"), ("द्वितीया", "बहुवचनम्"),
         ("तृतीया", "एकवचनम्"), ("षष्ठी", "बहुवचनम्")]
CASE_LOC = {"प्रथमा": "प्रथमायाम्", "द्वितीया": "द्वितीयायाम्", "तृतीया": "तृतीयायाम्",
            "चतुर्थी": "चतुर्थ्याम्", "पञ्चमी": "पञ्चम्याम्", "षष्ठी": "षष्ठ्याम्",
            "सप्तमी": "सप्तम्याम्"}
NUM_LOC = {"एकवचनम्": "एकवचने", "द्विवचनम्": "द्विवचने", "बहुवचनम्": "बहुवचने"}


def probes() -> list[dict]:
    from scripts.sanskrit_kb.lexicon import lexicon
    lex = lexicon()
    out = []
    for stem, linga in HELDOUT_NOUNS:
        for case, num in CELLS:
            form = lex.cell(stem, case, num, linga)
            if not form:
                continue
            prompt = f"{stem} इति शब्दस्य {case} {num} किम्?"
            out.append({"prompt": prompt, "accept": [form], "stem": stem,
                        "case": case, "number": num,
                        "locative": f"{stem} इति शब्दस्य {CASE_LOC[case]} "
                                    f"{NUM_LOC[num]} रूपं किम्?"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", action="store_true")
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    items = probes()
    if args.router:
        from scripts.sanskrit_kb.router import answer
        reply_of = lambda p: (a.text if (a := answer(p)) else "")  # noqa: E731
        source = "router"
    else:
        from scripts.eval_exchanges import NumpyBackend, TorchBackend
        backend = NumpyBackend(args.bundle) if args.bundle else TorchBackend(
            args.checkpoint, args.tokenizer)
        reply_of = lambda p: backend.reply([], p)  # noqa: E731
        source = str(args.bundle or args.checkpoint)

    rows, hits, hits_locative = [], 0, 0
    for it in items:
        text = reply_of(it["prompt"])
        strict, loose = score(text, it["accept"])
        hits += loose
        if args.router:
            text_loc = reply_of(it["locative"])
            hits_locative += score(text_loc, it["accept"])[1]
        rows.append({**it, "reply": text, "hit": loose})
        if args.verbose:
            print(f"  {'ok ' if loose else 'MISS'} {it['prompt'][:46]:<48} {text[:30]}")

    n = len(items)
    print(f"\n  {args.label}: {source}")
    print(f"    unseen nouns      {hits}/{n} ({hits/n:.1%})")
    if args.router:
        print(f"    locative phrasing {hits_locative}/{n} ({hits_locative/n:.1%})")
    args.out.mkdir(parents=True, exist_ok=True)
    path = (args.out / f"grammar_unseen_{args.label}.json").resolve()
    path.write_text(json.dumps({"label": args.label, "source": source,
                                "hits": hits, "n": n, "records": rows},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    wrote {path.relative_to(ROOT) if ROOT in path.parents else path}")


if __name__ == "__main__":
    main()
