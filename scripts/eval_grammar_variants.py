#!/usr/bin/env python3
"""
Does the model know the paradigm, or just the benchmark string?

The 15 benchmark questions are held out, but a model can still pass them by
memorising "after षष्ठी एकवचनं किम्? say रामस्य". This probe asks for the *same
cells* in 9 different phrasings — some trained, some not — and reports how many of
them the model answers. A checkpoint that learned the cell generalises across
phrasing; one that learned the prompt does not.

    python3 scripts/eval_grammar_variants.py --bundle serve --label deployed
    python3 scripts/eval_grammar_variants.py --checkpoint artifacts/vyak20/model.pt --label vyak20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_grammar import score  # noqa: E402
from scripts.sanskrit_kb.lexicon import (CASE_INS, CASE_LOC, NUM_LOC,  # noqa: E402
                                         lexicon)

#: the cells the benchmark asks for
CELLS = [
    ("राम", "चतुर्थी", "एकवचनम्"), ("राम", "तृतीया", "एकवचनम्"),
    ("राम", "षष्ठी", "एकवचनम्"), ("लता", "चतुर्थी", "एकवचनम्"),
    ("फल", "प्रथमा", "बहुवचनम्"),
]
SHAPES = [
    ("{s} इति शब्दस्य {c} {n} किम्?", "benchmark"),
    ("{s} इति शब्दस्य {cl} {nl} रूपं किम्?", "locative"),
    ("{s} इति शब्दस्य {ci} विभक्त्या {nl} किम् रूपम्?", "instrumental"),
    ("{c} {n} — {s} इति शब्दस्य रूपं किम्?", "fronted case"),
    ("{s} इति पदस्य {c} {n} किम्?", "pada"),
    ("किं {s}-शब्दस्य {c} {n} अस्ति?", "yes/no frame"),
]
#: conjugation / root / liṅga / count probes
OTHER = [
    ("{f} इति पदस्य धातुः कः?", None, "dhatu", "गच्छति"),
    ("{r} धातोः लट् प्रथमपुरुष एकवचनं किम्?", None, "tin", "पठ्"),
    ("{s} इति शब्दस्य स्त्रीलिङ्गं किम्?", None, "linga", "अश्व"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.bundle:
        from scripts.eval_exchanges import NumpyBackend
        backend = NumpyBackend(args.bundle)
    else:
        from scripts.eval_exchanges import TorchBackend
        backend = TorchBackend(args.checkpoint, args.tokenizer)
    lex = lexicon()

    by_shape: dict[str, list[bool]] = {}
    rows = []
    for stem, case, num in CELLS:
        want = lex.cell(stem, case, num)
        for shape, name in SHAPES:
            prompt = shape.format(s=stem, c=case, n=num, cl=CASE_LOC[case],
                                  nl=NUM_LOC[num], ci=CASE_INS[case])
            reply = backend.reply([], prompt)
            ok = score(reply, [want])[1]
            rows.append({"prompt": prompt, "want": want, "reply": reply, "ok": ok,
                         "shape": name})
            by_shape.setdefault(name, []).append(ok)
            if args.verbose:
                print(f"  {'ok ' if ok else 'MISS'} [{name:<14}] {prompt[:44]:<46} {reply[:22]}")

    extra = [
        ("गच्छति इति पदस्य धातुः कः?", ["गम्"], "dhatu"),
        ("पठ् धातोः लट् प्रथमपुरुष एकवचनं किम्?", ["पठति"], "tin"),
        ("अश्व इति शब्दस्य स्त्रीलिङ्गं किम्?", ["अश्वा"], "linga"),
        ("संस्कृते कति विभक्तयः?", ["अष्टौ", "अष्ट", "सप्त"], "count"),
    ]
    for prompt, want, name in extra:
        reply = backend.reply([], prompt)
        ok = score(reply, want)[1]
        rows.append({"prompt": prompt, "want": want, "reply": reply, "ok": ok,
                     "shape": name})
        by_shape.setdefault(name, []).append(ok)
        if args.verbose:
            print(f"  {'ok ' if ok else 'MISS'} [{name:<14}] {prompt[:44]:<46} {reply[:22]}")

    total = sum(r["ok"] for r in rows)
    print(f"\n  {args.label}: {args.checkpoint or args.bundle}")
    print(f"    phrasings of the same cells   {total}/{len(rows)}")
    for name, oks in by_shape.items():
        print(f"      {name:<16} {sum(oks)}/{len(oks)}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = (args.out / f"grammar_variants_{args.label}.json").resolve()
    path.write_text(json.dumps({"label": args.label, "hits": total, "n": len(rows),
                                "by_shape": {k: [sum(v), len(v)] for k, v in by_shape.items()},
                                "records": rows}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"    wrote {path.relative_to(ROOT) if ROOT in path.parents else path}")


if __name__ == "__main__":
    main()
