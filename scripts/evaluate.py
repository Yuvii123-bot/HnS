#!/usr/bin/env python3
"""Perplexity / bits-per-akṣara evaluation on the held-out split."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sanskrit_llm.data import TokenDataset, token_feature_table
from sanskrit_llm.model import load_checkpoint
from sanskrit_llm.trainer import evaluate

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    ap.add_argument("--bin", type=Path, default=Path("data/bin/val.bin"))
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batches", type=int, default=32)
    args = ap.parse_args()

    model, tok, _ = load_checkpoint(args.checkpoint)
    if tok is None:
        print("tokenizer.json not found next to checkpoint"); return
    feat = token_feature_table(tok)
    ds = TokenDataset(args.bin, feat, args.ctx)
    loss = evaluate(model, ds, batches=args.batches, batch_size=8)
    print(json.dumps({
        "val_loss": round(loss, 4),
        "perplexity": round(math.exp(min(loss, 20)), 2),
        "bits_per_token": round(loss / math.log(2), 3),
        "chars_per_token": 2.07,
        "bits_per_character": round(loss / math.log(2) / 2.07, 3),
    }, indent=2))

if __name__ == "__main__":
    main()
