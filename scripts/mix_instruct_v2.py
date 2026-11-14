#!/usr/bin/env python3
"""
Balance the phase-2 dataset and replay the verse corpus.

Grammar pairs are correct by construction and therefore easy to generate in
bulk — 4,921 of them, which would swamp everything else if used as-is. Each
category is capped or lifted to a target count here, and a slice of the
original verse corpus is replayed so instruction tuning does not erase the
model's continuation ability (phase 1 measured that forgetting precisely).

    python3 scripts/mix_instruct_v2.py --out data/instruct_mix2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# target number of instruction rows per category after balancing
TARGETS = {
    "grammar": 2400,
    "conversation": 1265,
    "glossary": 1310,
    "dharma": 989,
    "world": 827,
    "reasoning": 640,
    "numbers": 552,
    "greeting": 480,
}
REPLAY_FRACTION = 0.22          # share of training *tokens* taken by verse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "instruct_v2")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "clean" / "train.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "instruct_mix2")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in (args.data / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cat: dict[str, list[str]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["text"])

    kept: list[str] = []
    report: dict[str, dict] = {}
    for cat, items in sorted(by_cat.items()):
        target = TARGETS.get(cat, len(items))
        multi = [t for t in items if t.count("<user>") >= 2]
        single = [t for t in items if t.count("<user>") < 2]
        if cat == "grammar":
            # production ("किम् रूपम्", "कथं भवति", cloze) is what the model gets
            # wrong; keep all of it and sample only from the easier recognition rows
            hard = [t for t in single if any(k in t for k in
                    ("किम् रूपम्", "कथं भवति", "युक्तम्"))]
            easy = [t for t in single if t not in set(hard)]
            room = max(target - len(multi) - len(hard), 0)
            chosen = multi + hard + rng.sample(easy, min(room, len(easy)))
            kept.extend(chosen)
            report[cat] = {"source": len(items), "used": len(chosen),
                           "production_kept": len(hard), "multi_turn_kept": len(multi),
                           "note": f"subsampled {len(items)} -> {len(chosen)}"}
            continue
        if len(items) > target:
            # dialogs carry the context a single question lacks: keep them all
            room = max(target - len(multi), 0)
            chosen = multi + rng.sample(single, min(room, len(single))) if multi else \
                rng.sample(items, target)
            kept.extend(chosen)
            report[cat] = {"source": len(items), "used": len(chosen),
                           "multi_turn_kept": len(multi),
                           "note": f"subsampled {len(items)} -> {len(chosen)}"}
            continue
        if len(items) > target:
            chosen = rng.sample(items, target)
            note = f"subsampled {len(items)} -> {target}"
        elif len(items) < target:
            reps = target // len(items) + 1
            pool = (items * reps)[:target]
            rng.shuffle(pool)
            chosen = pool
            note = f"repeated {len(items)} -> {target}"
        else:
            chosen = items
            note = "kept as-is"
        kept.extend(chosen)
        report[cat] = {"source": len(items), "used": len(chosen), "note": note}

    token_est = lambda s: len(s) / 2.05          # chars per token, measured
    instr_tokens = sum(token_est(r) for r in kept)
    need = instr_tokens * REPLAY_FRACTION / (1 - REPLAY_FRACTION)
    verse_pool = [l for l in args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(verse_pool)
    replay: list[str] = []
    got = 0.0
    for line in verse_pool:
        if got >= need:
            break
        replay.append(line)
        got += token_est(line)

    rng.shuffle(kept)
    val_rows = [l for l in (args.data / "val.txt").read_text(encoding="utf-8").splitlines()
                if l.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n".join(kept + replay) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text("\n".join(val_rows) + "\n", encoding="utf-8")
    total = instr_tokens + got
    stats = {
        "instruction_rows": len(kept),
        "verse_replay_lines": len(replay),
        "estimated_tokens": {"instruction": round(instr_tokens), "verse": round(got),
                             "total": round(total)},
        "verse_share_of_training_tokens": round(got / max(total, 1), 4),
        "per_category": report,
    }
    (args.out / "mix_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}/train.txt ({len(kept) + len(replay)} lines)")


if __name__ == "__main__":
    main()
