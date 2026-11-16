#!/usr/bin/env python3
"""
Turn the paradigm tables into Sanskrit prose the model can read as a book.

The śabdapatha and the dhātupāṭha are lexicons: 9,007 stems with gender, gloss and
all 24 cells; 2,259 roots with gaṇa, pada and gloss. Rendered as sentences they
become a few million characters of true, checkable Sanskrit that teaches both
vocabulary and morphology at once — the closest thing to "a book on Sanskrit"
that can be generated without inventing facts:

    अश्वः पुल्लिङ्गः। अस्य अर्थः अश्वः, वामी, अश्वबालः। प्रथमायां विभक्तौ
    एकवचने अश्वः, द्विवचने अश्वौ, बहुवचने अश्वाः। षष्ठ्यां एकवचने अस्य,
    बहुवचने अश्वानाम्।

    गम् धातुः प्रथमगणे परस्मैपदे च। अस्य अर्थः गतौ। लटि प्रथमपुरुषे
    एकवचने गच्छति, द्विवचने गच्छतः, बहुवचने गच्छन्ति।

    python3 scripts/build_lexicon_prose.py --out data/lexicon_prose
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sanskrit_kb.lexicon import (CASES, LINGA, NUMBERS,  # noqa: E402
                                         PURUSHA, lexicon)

PADA = {"P": "परस्मैपदे", "A": "आत्मनेपदे", "U": "उभयपदे", "": ""}


def stem_paragraph(lex, stem: str, cells: dict, linga: str, artha: str) -> str:
    g = LINGA.get(linga, "")
    head = f"{cells[('प्रथमा', 'एकवचनम्')]} {g}।"
    gloss = f" अस्य अर्थः {artha}।" if artha else ""
    parts = [head + gloss]
    for case in CASES[:7]:
        trio = [cells.get((case, n), "") for n in NUMBERS]
        if not all(trio):
            continue
        parts.append(f"{case} विभक्तौ एकवचने {trio[0]}, द्विवचने {trio[1]}, बहुवचने {trio[2]}।")
    return " ".join(parts)


def root_paragraphs(lex, root: str, rec: dict) -> list[str]:
    out = []
    gana = rec.get("gana", "")
    pada = PADA.get(rec.get("pada", ""), "")
    artha = rec.get("artha", "")
    head = f"{root} धातुः"
    if gana:
        head += f" {gana}गणे"
    if pada:
        head += f" {pada}"
    head += " च।"
    if artha:
        head += f" अस्य अर्थः {artha}।"
    out.append(head)
    for rec_, cells in lex.tin_cells(root, "लट्"):
        if rec_ is not rec:
            continue
        for p in range(3):
            trio = cells[p * 3:p * 3 + 3]
            out.append(f"लटि {PURUSHA[p]} एकवचने {trio[0]}, द्विवचने {trio[1]}, "
                       f"बहुवचने {trio[2]}।")
        break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "lexicon_prose")
    ap.add_argument("--stems", type=int, default=0, help="limit stems (0 = all)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.005)
    args = ap.parse_args()

    lex = lexicon()
    lines: list[str] = []
    stems = sorted(lex.stems)
    if args.stems:
        random.Random(args.seed).shuffle(stems)
        stems = stems[: args.stems]
    for stem in stems:
        for entry in lex.stem_entries(stem):
            cells = dict(zip(
                [(c, n) for c in CASES for n in NUMBERS], entry["cells"]))
            para = stem_paragraph(lex, stem, cells, entry["linga"], entry["artha"])
            for sentence in para.split("। "):
                if sentence.strip():
                    lines.append(sentence.strip().rstrip("।") + "।")

    for root, recs in sorted(lex.roots.items()):
        for rec in recs[:1]:
            lines.extend(root_paragraphs(lex, root, rec))

    rng = random.Random(args.seed)
    rng.shuffle(lines)
    cut = int(len(lines) * args.val_frac)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n".join(lines[cut:]) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text("\n".join(lines[:cut]) + "\n", encoding="utf-8")
    chars = sum(len(l) for l in lines[cut:])
    print(f"{args.out.relative_to(ROOT)}: {len(lines) - cut:,} lines, "
          f"{chars:,} chars (~{chars / 2.6:,.0f} tokens)")
    print(f"  sample: {lines[0][:110]}")


if __name__ == "__main__":
    main()
