#!/usr/bin/env python3
"""
Build the training mixture for instruction tuning.

Two problems with training on instruction pairs alone, both measured in
reports/ before this script existed:

  1. category imbalance — 663 glossary pairs against 38 greetings, so the model
     drifts toward "X इत्यस्य पर्यायः Y।" and answers the wrong question;
  2. catastrophic forgetting — verse/continuation loss went 3.84 -> 6.77,
     i.e. the model that could continue Sanskrit verse stopped being able to.

So: oversample the small categories, and replay a share of the original
Sanskrit corpus alongside the instructions.  The replay share is reported, not
hidden, and "replayed" lines are verbatim corpus lines.

    python3 scripts/mix_instruct_data.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

# how many times each category is repeated (1 = as authored)
WEIGHTS = {
    "greeting": 6,
    "numbers": 6,
    "grammar": 2,
    "reasoning": 3,
    "conversation": 2,
    "dharma": 2,
    "world": 2,
    "glossary": 1,
    "dialogue": 1,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", type=Path, default=Path("data/instruct"))
    ap.add_argument("--verse", type=Path, default=Path("data/clean/train.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/instruct_mix"))
    ap.add_argument("--replay-per", type=int, default=3,
                    help="one corpus line is replayed for every N instruction pairs")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in (args.instruct / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    counts = collections.Counter(r["category"] for r in rows)

    balanced: list[str] = []
    for r in rows:
        balanced.extend([r["text"]] * WEIGHTS.get(r["category"], 1))

    verse = [l.strip() for l in args.verse.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(verse)
    n_replay = len(balanced) // args.replay_per
    replay = verse[:n_replay]

    # interleave: after every `replay_per` instruction pairs insert one verse line
    mixed: list[str] = []
    v = 0
    for i, line in enumerate(balanced):
        mixed.append(line)
        if (i + 1) % args.replay_per == 0 and v < len(replay):
            mixed.append(replay[v])
            v += 1
    mixed.extend(replay[v:])
    rng.shuffle(mixed)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n".join(mixed) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text((args.instruct / "val.txt").read_text(encoding="utf-8"),
                                      encoding="utf-8")

    # token-level composition, measured not guessed
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sanskrit_llm.tokenizer import AksharaTokenizer
    tok = AksharaTokenizer.load("artifacts/tokenizer.json")
    instr_tokens = sum(len(tok.encode(r["text"])) for r in rows for _ in range(WEIGHTS.get(r["category"], 1)))
    verse_tokens = sum(len(tok.encode(l)) for l in replay)
    total = instr_tokens + verse_tokens

    stats = {
        "instruction_pairs_authored": len(rows),
        "instruction_pairs_after_balancing": len(balanced),
        "category_weights": WEIGHTS,
        "authored_counts": dict(counts.most_common()),
        "verse_lines_replayed": len(replay),
        "tokens": {"instruction": instr_tokens, "verse_replay": verse_tokens,
                   "total": total},
        "verse_share_of_training_tokens": round(verse_tokens / total, 4),
        "mean_pair_tokens": round(instr_tokens / max(1, len(balanced)), 1),
    }
    (args.out / "mix_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
