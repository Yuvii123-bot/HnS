"""
Torch-free inference for the Sanskrit LM (NumPy only).

Vercel's Python functions have a 250 MB uncompressed bundle limit, and a PyTorch
wheel alone is several times that.  Since this model is only 5.8M parameters we
don't need PyTorch at inference time: this module re-implements the exact same
forward pass (RoPE, RMSNorm, SwiGLU, causal attention) in NumPy, with a KV cache
so each new token costs one cheap step instead of a full re-forward.

`tests/test_numpy_lm.py` checks that logits match the PyTorch implementation to
within 1e-4, so the deployed model is the trained model, not an approximation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Reuse the torch-free parts of the package (tokenizer/akṣara import stdlib only).
from .akshara import N_FEATURES, features
from .tokenizer import AksharaTokenizer

EPS = 1e-6


def _rmsnorm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)) * w


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class NumpyLM:
    """Decoder-only transformer forward pass + sampling, in NumPy."""

    def __init__(self, npz_path: str | Path, config: dict):
        blob = np.load(npz_path)
        self.w = {k: blob[k].astype(np.float32) for k in blob.files}
        self.cfg = config
        self.d_model = int(config["d_model"])
        self.n_layers = int(config["n_layers"])
        self.n_heads = int(config["n_heads"])
        self.d_ff = int(config["d_ff"])
        self.max_seq_len = int(config["max_seq_len"])
        self.vocab_size = int(config["vocab_size"])
        self.head_dim = self.d_model // self.n_heads

        inv_freq = 1.0 / (10000.0 ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        freqs = np.outer(np.arange(self.max_seq_len), inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        self.cos = np.cos(emb).astype(np.float32)
        self.sin = np.sin(emb).astype(np.float32)

    # ------------------------------------------------------------------
    def num_params(self) -> int:
        return int(sum(v.size for v in self.w.values()))

    @staticmethod
    def _rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, offset: int) -> np.ndarray:
        """x: (H, T, head_dim)"""
        T = x.shape[1]
        c = cos[offset: offset + T][None, :, :]
        s = sin[offset: offset + T][None, :, :]
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        rotated = np.concatenate([-x2, x1], axis=-1)
        return x * c + rotated * s

    # ------------------------------------------------------------------
    def forward(self, ids: np.ndarray, feats: np.ndarray, past: list | None = None):
        """ids: (T,) int64, feats: (T, N_FEATURES). Returns logits (T, V), new past."""
        ids = np.asarray(ids, dtype=np.int64)
        T = ids.shape[0]
        H, hd = self.n_heads, self.head_dim

        x = self.w["token_emb.weight"][ids] + feats @ self.w["feat_proj.weight"].T
        offset = 0 if past is None else past[0][0].shape[1]
        new_past: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(self.n_layers):
            p = f"blocks.{i}."
            h = _rmsnorm(x, self.w[p + "norm1.weight"])

            qkv = h @ self.w[p + "attn.qkv.weight"].T
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, H, hd).transpose(1, 0, 2)
            k = k.reshape(T, H, hd).transpose(1, 0, 2)
            v = v.reshape(T, H, hd).transpose(1, 0, 2)

            q = self._rope(q, self.cos, self.sin, offset)
            k = self._rope(k, self.cos, self.sin, offset)

            if past is not None:
                pk, pv = past[i]
                k = np.concatenate([pk, k], axis=1)
                v = np.concatenate([pv, v], axis=1)
            new_past.append((k, v))

            scores = np.einsum("htd,hTd->htT", q, k) / np.sqrt(hd)
            if past is None or T > 1:
                Tq, Tk = q.shape[1], k.shape[1]
                mask = np.triu(np.ones((Tq, Tk), dtype=bool), k=1)
                if past is not None:      # only the new rows are queries
                    mask = np.triu(np.ones((Tq, Tk), dtype=bool), k=Tk - Tq + 1)
                scores = np.where(mask, -np.inf, scores)
            att = _softmax(scores)
            y = np.einsum("htT,hTd->htd", att, v)
            y = y.transpose(1, 0, 2).reshape(T, self.d_model)

            x = x + y @ self.w[p + "attn.out.weight"].T

            h2 = _rmsnorm(x, self.w[p + "norm2.weight"])
            a = h2 @ self.w[p + "mlp.w1.weight"].T
            b = h2 @ self.w[p + "mlp.w3.weight"].T
            x = x + (_silu(a) * b) @ self.w[p + "mlp.w2.weight"].T

        x = _rmsnorm(x, self.w["norm_f.weight"])
        logits = x @ self.w["token_emb.weight"].T      # tied embeddings
        return logits, new_past

    # ------------------------------------------------------------------
    def sample(
        self,
        ids: np.ndarray,
        *,
        temperature: float = 0.85,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        max_new_tokens: int = 90,
        max_lines: int = 2,
        eos_id: int | None = None,
        feat_table: np.ndarray,
        stop_token_ids: set[int] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """Yield (token_id, is_line_end) tuples, mirroring the torch sampler."""
        rng = rng or np.random.default_rng()
        ids = np.asarray(ids, dtype=np.int64)[-self.max_seq_len:]
        logits, past = self.forward(ids, feat_table[ids])
        cur_len = ids.shape[0]
        produced: list[int] = []
        lines = 0

        for _ in range(max_new_tokens):
            if cur_len >= self.max_seq_len:
                return
            step = logits[-1].astype(np.float64).copy()

            if repetition_penalty and repetition_penalty != 1.0:
                for tid in set(produced):
                    step[tid] /= repetition_penalty

            if temperature <= 0:
                nxt = int(np.argmax(step))
            else:
                step = step / max(temperature, 1e-5)
                if top_k and top_k > 0:
                    k = min(top_k, step.shape[0])
                    kth = np.partition(step, -k)[-k]
                    step = np.where(step < kth, -np.inf, step)
                probs = _softmax(step)
                if top_p and 0 < top_p < 1.0:
                    order = np.argsort(-probs)
                    sorted_p = probs[order]
                    cum = np.cumsum(sorted_p)
                    cut = np.searchsorted(cum, top_p) + 1
                    keep = order[:cut]
                    p = probs[keep]
                    p = p / p.sum()
                    nxt = int(rng.choice(keep, p=p))
                else:
                    nxt = int(rng.choice(step.shape[0], p=probs))

            if eos_id is not None and nxt == eos_id:
                # <eos> marks a line break in this corpus
                lines += 1
                yield nxt, True
                if lines >= max_lines:
                    return
                logits, past = self.forward(
                    np.array([nxt]), feat_table[[nxt]], past)
                cur_len += 1
                continue

            if stop_token_ids and nxt in stop_token_ids:
                return

            yield nxt, False
            produced.append(nxt)

            logits, past = self.forward(np.array([nxt]), feat_table[[nxt]], past)
            cur_len += 1


# ----------------------------------------------------------------------
# Loading the deployment bundle
# ----------------------------------------------------------------------

def token_feature_table(tok: AksharaTokenizer) -> np.ndarray:
    table = np.zeros((tok.vocab_size, N_FEATURES), dtype=np.float32)
    for i, token in enumerate(tok.vocab):
        table[i] = features(token)
    return table


class Runtime:
    """Everything the serverless function needs, loaded once per container."""

    def __init__(self, serve_dir: str | Path = "serve"):
        self.dir = Path(serve_dir)
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.tokenizer = AksharaTokenizer.load(self.dir / "tokenizer.json")
        self.feat_table = token_feature_table(self.tokenizer)
        self.model = NumpyLM(self.dir / "model.npz", meta["config"])
        self.summary = meta.get("train_summary", {})
        self.corpus = meta.get("corpus", {})

    def info(self) -> dict:
        return {
            "ready": True,
            "model": {"params": self.model.num_params(), "config": self.model.cfg},
            "tokenizer": self.tokenizer.stats(),
            "training": {"summary": self.summary},
            "corpus": self.corpus,
            "tokens_seen_total": self.summary.get("total_tokens_seen", 0),
            "runtime": "numpy",
        }
