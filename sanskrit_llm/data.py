"""Corpus -> token-id binaries, batching, and per-token feature tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import akshara
from .tokenizer import AksharaTokenizer


def token_feature_table(tok: AksharaTokenizer) -> torch.Tensor:
    """(vocab, N_FEATURES) lookup table of akṣara features for every token."""
    table = np.zeros((tok.vocab_size, akshara.N_FEATURES), dtype=np.float32)
    for i, token in enumerate(tok.vocab):
        table[i] = akshara.features(token)
    return torch.from_numpy(table)


def encode_corpus(
    files: dict[str, Path],
    tokenizer: AksharaTokenizer,
    out_dir: Path,
    *,
    progress_every: int = 20000,
) -> dict:
    """Encode text files into flat uint16 token arrays (one '<eos>' per line)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    eos = tokenizer.eos_id

    for name, path in files.items():
        ids: list[int] = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines):
            if not line.strip():
                continue
            ids.extend(tokenizer.encode(line))
            ids.append(eos)
            if progress_every and n % progress_every == 0 and n:
                print(f"  [{name}] {n:,}/{len(lines):,} lines -> {len(ids):,} tokens",
                      flush=True)
        arr = np.asarray(ids, dtype=np.uint16)
        (out_dir / f"{name}.bin").write_bytes(arr.tobytes())
        stats[name] = {"tokens": int(arr.size), "lines": len(lines),
                       "bytes": int(arr.nbytes)}
        print(f"  [{name}] wrote {arr.size:,} tokens "
              f"({arr.nbytes / 1e6:.1f} MB)", flush=True)

    (out_dir / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class TokenDataset:
    """Random contiguous windows over a flat token array."""

    def __init__(self, bin_path: Path, feat_table: torch.Tensor, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.feat_table = feat_table
        self.block_size = block_size
        self.rng = np.random.default_rng(1234)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)

    def sample_batch(self, batch_size: int):
        ix = self.rng.integers(0, len(self) - 1, size=batch_size)
        x = np.stack([self.data[i: i + self.block_size] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i: i + self.block_size + 1][1:] for i in ix]).astype(np.int64)
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        feats = self.feat_table[xt]
        return xt, yt, feats

    def sequential_batches(self, batch_size: int, stride: int | None = None):
        """Deterministic pass (used for evaluation)."""
        stride = stride or self.block_size
        n = len(self)
        for start in range(0, n - self.block_size, stride * batch_size):
            starts = [start + k * stride for k in range(batch_size)]
            starts = [s for s in starts if s + self.block_size + 1 <= len(self.data)]
            if not starts:
                break
            x = np.stack([self.data[s: s + self.block_size] for s in starts]).astype(np.int64)
            y = np.stack([self.data[s + 1: s + self.block_size + 1] for s in starts]).astype(np.int64)
            xt, yt = torch.from_numpy(x), torch.from_numpy(y)
            yield xt, yt, self.feat_table[xt]
