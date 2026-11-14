#!/usr/bin/env python3
"""
Export the trained checkpoint into a torch-free deployment bundle.

    artifacts/model.pt  ->  serve/model.npz        (float32 weights)
                            serve/tokenizer.json   (vocabulary + merges)
                            serve/meta.json        (config + training record)

The bundle is what the Vercel serverless function loads; it carries no PyTorch.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    ap.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    ap.add_argument("--summary", type=Path, default=Path("artifacts/train_summary.json"))
    ap.add_argument("--corpus", type=Path, default=Path("data/clean/corpus_stats.json"))
    ap.add_argument("--out", type=Path, default=Path("serve"))
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = blob["config"]
    state = blob["state_dict"]

    args.out.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.detach().numpy().astype(np.float32) for k, v in state.items()}
    np.savez(args.out / "model.npz", **arrays)
    shutil.copy(args.tokenizer, args.out / "tokenizer.json")

    # stage a torch-free copy of the runtime modules inside the bundle, so the
    # serverless function needs a single includeFiles glob ("serve/**")
    runtime_dir = args.out / "runtime" / "sanskrit_llm"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    for mod in ("akshara.py", "tokenizer.py", "transliterate.py", "numpy_lm.py"):
        shutil.copy(root / "sanskrit_llm" / mod, runtime_dir / mod)
    (runtime_dir / "__init__.py").write_text(
        '"""Torch-free subset of sanskrit_llm, staged for serverless deployment."""\n'
        "from .akshara import N_FEATURES, aksharas, features, segment\n"
        "from .tokenizer import AksharaTokenizer\n"
        "from .transliterate import normalize_devanagari\n"
        "from .numpy_lm import NumpyLM, Runtime\n"
        "__all__ = [\"AksharaTokenizer\", \"NumpyLM\", \"Runtime\", \"N_FEATURES\",\n"
        "           \"segment\", \"aksharas\", \"features\", \"normalize_devanagari\"]\n",
        encoding="utf-8")

    meta = {
        "config": config,
        "params": int(sum(v.size for v in arrays.values())),
        "train_summary": json.loads(args.summary.read_text()) if args.summary.exists() else {},
        "corpus": json.loads(args.corpus.read_text()) if args.corpus.exists() else {},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                        encoding="utf-8")

    size = (args.out / "model.npz").stat().st_size
    print(f"wrote {args.out}/model.npz  ({size / 1e6:.1f} MB, {len(arrays)} tensors, "
          f"{meta['params']:,} params)")
    print(f"wrote {args.out}/tokenizer.json, {args.out}/meta.json")


if __name__ == "__main__":
    main()
