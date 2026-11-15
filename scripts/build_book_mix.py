#!/usr/bin/env python3
"""
Compose the two Hitopadeśa training stages.

Stage 1 (pretrain): the book's own lines, repeated, with a slice of the existing
verse corpus as replay so the model does not lose its earlier register.

Stage 2 (instruct): instruction rows derived from the book, plus the greeting
block (to hold the 43/44) and a filtered replay. Deliberately *not* the exchange
mix from rounds 13-15: that composition measured worse than the deployed model,
so building on it would import its regression.

    python3 scripts/build_book_mix.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path, n: int | None = None) -> list[str]:
    rows = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if n is not None and len(rows) > n:
        rows = rows[:n]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", type=Path, default=Path("data/book/train.txt"))
    ap.add_argument("--book-instr", type=Path, default=Path("data/book_instr/train.txt"))
    ap.add_argument("--greet", type=Path, default=Path("data/greet/train.txt"))
    ap.add_argument("--corpus", type=Path, default=Path("data/instruct_mix2/train.txt"),
                    help="source of raw verse replay: the plain (non-instruction) lines "
                         "inside the instruction mix, which are the GRETIL verse samples")
    ap.add_argument("--replay", type=Path, default=Path("data/instruct_mix2/train.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/book_mix"))
    ap.add_argument("--stage1-repeat", type=int, default=6)
    ap.add_argument("--stage2-book-repeat", type=int, default=8)
    ap.add_argument("--stage2-greet-repeat", type=int, default=14)
    ap.add_argument("--stage2-raw-repeat", type=int, default=0,
                    help="also include the book's own raw lines in stage 2, for register")
    ap.add_argument("--corpus-lines", type=int, default=4000)
    ap.add_argument("--replay-lines", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    book = load("book", args.book)
    # raw verse replay: plain Devanagari lines (no <user> markup) in the mix
    corpus = [l for l in load("corpus", args.corpus)
              if "<user>" not in l and re.search(r"[\u0900-\u097F]", l) and len(l) > 25]
    rng.shuffle(corpus)
    corpus = corpus[:args.corpus_lines]
    print(f"verse replay lines available: {len(corpus)}")

    stage1 = book * args.stage1_repeat + corpus
    rng.shuffle(stage1)

    book_instr = load("book_instr", args.book_instr)
    greet = load("greet", args.greet)

    spec = importlib.util.spec_from_file_location("exb", ROOT / "scripts/build_exchange_data.py")
    exb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exb)
    banned = set()
    for row in book_instr + greet:
        m = re.search(r"<user> (.*?) <assistant>", row)
        if m:
            banned.add(exb.norm(m.group(1)))
    replay, stats = exb.scrub_replay(args.replay, banned, args.replay_lines, args.seed)

    stage2 = (book_instr * args.stage2_book_repeat
              + book * args.stage2_raw_repeat
              + greet * args.stage2_greet_repeat + replay)
    rng.shuffle(stage2)

    def tokens(rows: list[str]) -> int:
        return int(sum(len(r) for r in rows) / 2.07)

    for name, rows, val_src in (("stage1", stage1, corpus),
                                ("stage2", stage2, book_instr)):
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "train.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        val = val_src[:16]
        (d / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
        print(f"{name}: {len(rows)} rows, ~{tokens(rows):,} tokens")

    print("replay scrub:", json.dumps(stats, ensure_ascii=False))
    print(f"stage2 shares: book_instr {tokens(book_instr*args.stage2_book_repeat)/max(1,tokens(stage2)):.2%}, "
          f"raw {tokens(book*args.stage2_raw_repeat)/max(1,tokens(stage2)):.2%}, "
          f"greet {tokens(greet*args.stage2_greet_repeat)/max(1,tokens(stage2)):.2%}, "
          f"replay {tokens(replay)/max(1,tokens(stage2)):.2%}")


if __name__ == "__main__":
    main()
