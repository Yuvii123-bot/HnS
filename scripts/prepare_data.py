#!/usr/bin/env python3
"""Encode the Devanagari corpus into flat token arrays for training."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sanskrit_llm.data import encode_corpus
from sanskrit_llm.tokenizer import AksharaTokenizer

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/bin"))
    args = ap.parse_args()
    tok = AksharaTokenizer.load(args.tokenizer)
    t0 = time.time()
    stats = encode_corpus({"train": args.data / "train.txt",
                           "val": args.data / "val.txt"}, tok, args.out)
    print(f"encoded in {time.time() - t0:.1f}s -> {stats}")

if __name__ == "__main__":
    main()
