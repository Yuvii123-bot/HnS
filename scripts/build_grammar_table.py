#!/usr/bin/env python3
"""
Pack the vyākaraṇa tables into one file the serverless function can load.

The raw sources are 24 MB of JSON and stay out of the repository; this
compresses the *useful* part — every stem's 24 subanta cells and every root's
tiṅ forms in the ten lakāras — into a single gzip that ships with `serve/`.

    python3 scripts/build_grammar_table.py            # -> serve/grammar_table.json.gz
    python3 scripts/build_grammar_table.py --check    # how big / what is in it
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sanskrit_kb.lexicon import ALL_LAKARA_KEYS, lexicon  # noqa: E402

DEFAULT_OUT = ROOT / "serve" / "grammar_table.json.gz"
FORMAT = 2


def build(out: Path) -> dict:
    lex = lexicon()
    stems: dict[str, list] = {}
    for word, entries in lex.stems.items():
        # one entry per (stem, liṅga): a stem can be both masculine and neuter
        stems[word] = [[e["linga"], e["artha"], ";".join(e["cells"])] for e in entries]

    roots: dict[str, list] = {}
    for root, recs in lex.roots.items():
        packed = []
        for rec in recs:
            table = lex.tin_tables.get(rec["index"], {})
            row = {"g": rec["gana"], "a": rec["artha"], "i": rec["index"],
                   "c": int(rec.get("common", False))}
            for key in ALL_LAKARA_KEYS:
                forms = table.get(key)
                if forms:
                    row[key] = forms
            packed.append(row)
        roots[root] = packed

    blob = {"format": FORMAT, "stems": stems, "roots": roots,
            "credit": "Vyākaraṇa data © ashtadhyayi.com — ashtadhyayi-com/data"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"))
    return blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        if not args.out.exists():
            sys.exit(f"FAIL: {args.out} missing — run without --check")
        with gzip.open(args.out, "rt", encoding="utf-8") as fh:
            blob = json.load(fh)
        n_cells = sum(e[2].count(";") + 1 for v in blob["stems"].values() for e in v)
        print(f"{args.out.relative_to(ROOT)}  {args.out.stat().st_size/1e6:.2f} MB")
        print(f"  stems {len(blob['stems']):,}   roots {len(blob['roots']):,}")
        print(f"  subanta cells packed: {n_cells:,}")
        return

    blob = build(args.out)
    print(f"wrote {args.out.relative_to(ROOT)}  "
          f"{args.out.stat().st_size/1e6:.2f} MB  "
          f"({len(blob['stems']):,} stems, {len(blob['roots']):,} roots)")


if __name__ == "__main__":
    main()
