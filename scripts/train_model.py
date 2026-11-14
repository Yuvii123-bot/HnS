#!/usr/bin/env python3
"""Train the Sanskrit LM from scratch.

    python scripts/train_model.py --steps 3000 --batch 16 --ctx 256
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sanskrit_llm.data import TokenDataset, token_feature_table
from sanskrit_llm.model import ModelConfig, SanskritLM
from sanskrit_llm.tokenizer import AksharaTokenizer
from sanskrit_llm.trainer import train

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    ap.add_argument("--bin-dir", type=Path, default=Path("data/bin"))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=688)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="residual/attention/embedding dropout; default 0.0 keeps prior runs identical")
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--reset-steps", action="store_true",
                    help="restart the step counter (for a fine-tuning phase)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AksharaTokenizer.load(args.tokenizer)
    feat = token_feature_table(tok)

    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads, d_ff=args.d_ff,
                      max_seq_len=512, dropout=args.dropout)
    model = SanskritLM(cfg)
    print(f"model: {model.num_params():,} parameters  vocab={cfg.vocab_size}")

    train_ds = TokenDataset(args.bin_dir / "train.bin", feat, args.ctx)
    val_ds = TokenDataset(args.bin_dir / "val.bin", feat, args.ctx)
    print(f"train tokens: {len(train_ds.data):,}  val tokens: {len(val_ds.data):,}")

    summary = train(model, train_ds, val_ds, tok, steps=args.steps,
                    batch_size=args.batch, lr=args.lr, warmup=args.warmup,
                    out_dir=args.out, resume=args.resume,
                    reset_steps=args.reset_steps, threads=args.threads,
                    eval_every=args.eval_every, ckpt_every=args.ckpt_every)
    print(summary)

if __name__ == "__main__":
    main()
