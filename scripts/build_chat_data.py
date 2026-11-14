#!/usr/bin/env python3
"""
Build chat-template data for the format-adaptation phase.

The corpus is verse, not dialogue, so these pairs teach the model the
``<user> … <assistant> …`` *structure* (where a turn ends, that an answer
follows a prompt, that answers are verse-shaped).  They do not teach
question answering — that would need real Sanskrit dialogue data, which does
not exist at scale.  Being explicit about that matters.
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/train.txt"))
    ap.add_argument("--val-corpus", type=Path, default=Path("data/val.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/chat"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    def pairs(lines: list[str]) -> list[str]:
        out = []
        i = 0
        while i + 1 < len(lines):
            user, assistant = lines[i], lines[i + 1]
            if 8 < len(user) < 220 and 8 < len(assistant) < 220:
                out.append(f"<user> {user} <assistant> {assistant}")
            i += 2
        return out

    train = pairs([l.strip() for l in args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()])
    val = pairs([l.strip() for l in args.val_corpus.read_text(encoding="utf-8").splitlines() if l.strip()])
    rng.shuffle(train); rng.shuffle(val)

    (args.out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"chat pairs: train {len(train):,}  val {len(val):,}")
    print("example:", train[0][:160])

if __name__ == "__main__":
    main()
