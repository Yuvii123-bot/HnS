#!/usr/bin/env python3
"""
Train the akṣara-BPE tokenizer on the Devanagari corpus.

    python scripts/train_tokenizer.py --vocab-size 4096 --sample 3000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sanskrit_llm.tokenizer import AksharaTokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/train.txt"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/tokenizer.json"))
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--sample", type=int, default=3_000_000,
                    help="characters of corpus to train the merges on")
    ap.add_argument("--min-pair-freq", type=int, default=4)
    args = ap.parse_args()

    text = args.train.read_text(encoding="utf-8")
    if args.sample and len(text) > args.sample:
        text = text[: args.sample]
    print(f"training on {len(text):,} characters of Devanagari")

    t0 = time.time()
    tok = AksharaTokenizer().train(
        text, vocab_size=args.vocab_size, min_pair_freq=args.min_pair_freq, verbose=True)
    dt = time.time() - t0

    tok.save(args.out)
    stats = tok.stats()
    stats["train_seconds"] = round(dt, 1)
    stats["train_chars"] = len(text)
    stats["sample_merges"] = [tok.vocab[m[2]] for m in tok.merges[-25:]]
    (args.out.parent / "tokenizer_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in stats.items() if k != "sample_merges"},
                     ensure_ascii=False, indent=2))

    # ---- compression report -------------------------------------------
    sample = args.train.read_text(encoding="utf-8")[-200_000:]
    ids = tok.encode(sample)
    print(f"\ncompression: {len(sample):,} chars -> {len(ids):,} tokens "
          f"({len(sample) / max(len(ids), 1):.2f} chars/token)")
    demo = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
    print("\nsegmentation demo:")
    print(" ", demo)
    print(" ", " | ".join(tok.tokenize(demo)))
    assert tok.decode(tok.encode(demo)) == demo, "round-trip failed"
    print("  round-trip: exact")


if __name__ == "__main__":
    main()
