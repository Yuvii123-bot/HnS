#!/usr/bin/env python3
"""
Mix the vyākaraṇa drill rows with the instruction replay set.

Grammar rows are short and repetitive; the replay set is what keeps the model
conversational. The mix reports the grammar share in *tokens*, not rows, because
that is the number that predicts what the fine-tune will do to conversation.

    python3 scripts/build_vyakarana_mix.py --grammar-repeat 4 --replay-repeat 1
    python3 scripts/prepare_data.py --data data/vyakarana_mix --out data/bin_vyak
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def chars(path: Path) -> int:
    return len(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", type=Path,
                    default=ROOT / "data" / "vyakarana_train" / "train.txt")
    ap.add_argument("--grammar-val", type=Path,
                    default=ROOT / "data" / "vyakarana_train" / "val.txt")
    ap.add_argument("--replay", type=Path, nargs="+",
                    default=[ROOT / "data" / "instruct_mix2" / "train.txt",
                             ROOT / "data" / "greet" / "train.txt"])
    ap.add_argument("--grammar-repeat", type=int, default=4)
    ap.add_argument("--replay-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "vyakarana_mix")
    args = ap.parse_args()

    grammar = args.grammar.read_text(encoding="utf-8").splitlines()
    replay: list[str] = []
    for path in args.replay:
        replay.extend(path.read_text(encoding="utf-8").splitlines())
    grammar = [g for g in grammar if g.strip()]
    replay = [r for r in replay if r.strip()]

    rows = grammar * args.grammar_repeat + replay * args.replay_repeat
    random.Random(args.seed).shuffle(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    val = [v for v in args.grammar_val.read_text(encoding="utf-8").splitlines() if v.strip()]
    (args.out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    g_chars = sum(len(g) for g in grammar) * args.grammar_repeat
    r_chars = sum(len(r) for r in replay) * args.replay_repeat
    share = g_chars / (g_chars + r_chars)
    stats = {"rows": len(rows), "grammar_rows": len(grammar) * args.grammar_repeat,
             "replay_rows": len(replay) * args.replay_repeat,
             "grammar_share_chars": round(share, 4),
             "grammar_repeat": args.grammar_repeat,
             "replay_repeat": args.replay_repeat}
    (args.out / "mix_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
