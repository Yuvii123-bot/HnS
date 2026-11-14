"""
A compact decoder-only transformer built for Sanskrit.

Design notes specific to this project
-------------------------------------
* **Akṣara features.**  Every token carries a small phonological fingerprint
  (does it contain a virama? a matra? an anusvāra? does it span a word
  boundary?).  A linear projection of those features is *added* to the token
  embedding, so the network is told what kind of Sanskrit object it is reading
  instead of having to infer Devanagari orthography from scratch.
* **RoPE** positional encoding, tied input/output embeddings, RMSNorm,
  SwiGLU feed-forward — a modern, efficient stack that fits on a CPU.
* Small enough (≈6M parameters) to train from scratch on a couple of CPU
  cores, which is what makes this project reproducible in this sandbox.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .akshara import N_FEATURES


@dataclass
class ModelConfig:
    vocab_size: int = 4000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 688          # SwiGLU inner size (~8/3 * d)
    max_seq_len: int = 256
    dropout: float = 0.0
    n_features: int = N_FEATURES
    tie_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def precompute_rope(head_dim: int, max_len: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)              # (max_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (max_len, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               pos_offset: int = 0) -> torch.Tensor:
    """x: (B, H, T, head_dim); positions continue from ``pos_offset``."""
    T = x.size(-2)
    cos = cos[pos_offset: pos_offset + T].unsqueeze(0).unsqueeze(0)
    sin = sin[pos_offset: pos_offset + T].unsqueeze(0).unsqueeze(0)
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, pos_offset=0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin, pos_offset)
        k = apply_rope(k, cos, sin, pos_offset)

        new_kv = None
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        if use_cache:
            new_kv = (k, v)

        # Regularisation happens on the residual branch in Block (GPT-2 style).
        # Passing dropout_p>0 here makes PyTorch take the math path that
        # materialises the full (B, H, T, T) attention matrix, which OOMs a
        # 3 GB CPU box at batch 16 / ctx 256. Inference is unaffected either
        # way, since Block's nn.Dropout is a no-op in eval mode.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=(past_kv is None),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y), new_kv


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, pos_offset=0):
        a, kv = self.attn(self.norm1(x), cos, sin, past_kv, use_cache, pos_offset)
        x = x + self.dropout(a)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, kv


class SanskritLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.feat_proj = nn.Linear(cfg.n_features, cfg.d_model, bias=False)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        cos, sin = precompute_rope(cfg.d_model // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, feats, targets=None, past_kv=None, use_cache=False,
                pos_offset=0):
        B, T = idx.shape
        x = self.token_emb(idx) + self.feat_proj(feats)
        x = self.emb_drop(x)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            pk = past_kv[i] if past_kv is not None else None
            x, kv = block(x, self.rope_cos, self.rope_sin, pk, use_cache, pos_offset)
            new_kvs.append(kv)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss, (new_kvs if use_cache else None)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        feat_table: torch.Tensor,
        *,
        max_new_tokens: int = 120,
        temperature: float = 0.9,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Sample a continuation (returns ids, prompt included)."""
        self.eval()
        for token_id in stream_tokens(
            self, feat_table, idx, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, eos_id=eos_id,
        ):
            idx = torch.cat([idx, torch.tensor([[token_id]])], dim=1)
        return idx


# --------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------

def save_checkpoint(path, model: SanskritLM, tokenizer, extra: dict | None = None) -> None:
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": model.cfg.to_dict(),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }, path)
    # human-readable sidecar
    side = path.with_suffix(".json")
    side.write_text(json.dumps({
        "config": model.cfg.to_dict(),
        "params": model.num_params(),
        "vocab_size": tokenizer.vocab_size if tokenizer else model.cfg.vocab_size,
        "extra": extra or {},
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def load_checkpoint(path, map_location="cpu"):
    from .tokenizer import AksharaTokenizer
    from pathlib import Path
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    cfg = ModelConfig.from_dict(blob["config"])
    model = SanskritLM(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    tok_path = Path(path).parent / "tokenizer.json"
    tok = AksharaTokenizer.load(tok_path) if tok_path.exists() else None
    return model, tok, blob.get("extra", {})


# --------------------------------------------------------------------------
# Sampling with a KV cache (shared by the API's streaming endpoint)
# --------------------------------------------------------------------------

@torch.no_grad()
def stream_tokens(
    model: SanskritLM,
    feat_table: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    eos_id: int | None = None,
):
    """Yield generated token ids one at a time."""
    model.eval()
    cfg = model.cfg
    idx = prompt_ids[:, -cfg.max_seq_len:]
    feats = feat_table[idx]

    logits, _, past = model(idx, feats, use_cache=True)
    cur_len = idx.size(1)

    for _ in range(max_new_tokens):
        if cur_len >= cfg.max_seq_len:
            return
        step_logits = logits[:, -1, :].float()

        if repetition_penalty and repetition_penalty != 1.0:
            for b in range(prompt_ids.size(0)):
                uniq = torch.unique(prompt_ids[b])
                step_logits[b, uniq] /= repetition_penalty

        if temperature <= 0:
            nxt = step_logits.argmax(-1, keepdim=True)
        else:
            step_logits = step_logits / max(temperature, 1e-5)
            if top_k and top_k > 0:
                k = min(top_k, step_logits.size(-1))
                kth = torch.topk(step_logits, k, dim=-1).values[:, -1:]
                step_logits = torch.where(
                    step_logits < kth,
                    torch.full_like(step_logits, -float("inf")),
                    step_logits)
            probs = F.softmax(step_logits, dim=-1)
            if top_p and 0 < top_p < 1.0:
                sp, si = torch.sort(probs, descending=True, dim=-1)
                cum = sp.cumsum(-1)
                sp[cum - sp > top_p] = 0.0
                sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-9)
                nxt = si.gather(-1, torch.multinomial(sp, 1))
            else:
                nxt = torch.multinomial(probs, 1)

        token_id = int(nxt[0, 0])
        if eos_id is not None and token_id == eos_id:
            return
        yield token_id

        prompt_ids = torch.cat([prompt_ids, nxt], dim=1)
        logits, _, past_new = model(
            nxt, feat_table[nxt], past_kv=past, use_cache=True, pos_offset=cur_len)
        past = past_new
        cur_len += 1
