"""
Inference facade: prompt construction, streaming generation, token inspection.

The chat interface speaks to this class; it hides the tokenizer/model plumbing
and adds the Sanskrit-specific behaviour a chat client needs:

* auto-transliteration (the user may type IAST, the model is trained on
  Devanagari),
* two prompt modes — ``chat`` uses the ``<user>``/``<assistant>`` template the
  model is fine-tuned on, ``continuation`` simply lets it continue a verse,
* verse-aware stopping (stop at the end of a line or a ``॥`` daṇḍa),
* an akṣara-level breakdown of everything that passes through.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch

from . import akshara
from .data import token_feature_table
from .model import SanskritLM, stream_tokens
from .tokenizer import AksharaTokenizer
from .transliterate import normalize_devanagari


def _has_latin(text: str) -> bool:
    """The model is Sanskrit-only: romanised or English input is refused."""
    return any(("a" <= ch.lower() <= "z") for ch in text)


@dataclass
class GenParams:
    mode: str = "chat"             # "chat" | "continuation"
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    max_new_tokens: int = 90
    max_lines: int = 2
    stop_at_verse_end: bool = True
    max_chars: int = 0   # chat: cut runaway answers that drift into verse


@dataclass
class TokenInfo:
    text: str
    kind: str
    features: list[float]


def classify_token(text: str) -> str:
    units = akshara.segment(text)
    kinds = {u.kind for u in units}
    if "space" in kinds and len(units) > 1:
        return "juncture"
    if "space" in kinds:
        return "space"
    if "punct" in kinds:
        return "punct"
    if "digit" in kinds:
        return "digit"
    if "latin" in kinds:
        return "latin"
    if akshara.VIRAMA in text:
        return "conjunct"
    if text and text[0] in akshara.INDEPENDENT_VOWELS:
        return "vowel"
    if any(c in akshara.MATRAS for c in text):
        return "syllable"
    return "consonant"


class SanskritEngine:
    """Loads a checkpoint and tokenizer, and serves generations."""

    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        self.dir = Path(artifacts_dir)
        self.lock = threading.Lock()
        self.model: SanskritLM | None = None
        self.tokenizer: AksharaTokenizer | None = None
        self.feat_table: torch.Tensor | None = None
        self.meta: dict = {}
        self.reload()

    # ------------------------------------------------------------------
    def reload(self) -> bool:
        """(Re)load the newest checkpoint + tokenizer. Returns True on success."""
        tok_path = self.dir / "tokenizer.json"
        if not tok_path.exists():
            return False
        tok = AksharaTokenizer.load(tok_path)

        # prefer the finished model, then the best-eval checkpoint written
        # during a training run still in progress
        model_path = self.dir / "model.pt"
        if not model_path.exists():
            model_path = self.dir / "best.pt"
        if not model_path.exists():
            self.tokenizer, self.feat_table = tok, token_feature_table(tok)
            return False

        blob = torch.load(model_path, map_location="cpu", weights_only=False)
        from .model import ModelConfig
        cfg = ModelConfig.from_dict(blob["config"])
        model = SanskritLM(cfg)
        model.load_state_dict(blob["state_dict"])
        model.eval()

        extra = dict(blob.get("extra", {}))
        extra["checkpoint"] = model_path.name
        for sidecar in ("train_summary.json", "metrics.json"):
            p = self.dir / sidecar
            if p.exists():
                extra[sidecar.replace(".json", "")] = json.loads(p.read_text())

        with self.lock:
            self.model, self.tokenizer = model, tok
            self.feat_table = token_feature_table(tok)
            self.meta = extra
        return True

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    # ------------------------------------------------------------------
    def info(self) -> dict:
        tok = self.tokenizer
        info = {
            "ready": self.ready,
            "model": {
                "params": self.model.num_params() if self.model else 0,
                "config": self.model.cfg.to_dict() if self.model else {},
            },
            "tokenizer": tok.stats() if tok else {},
            "training": {
                "summary": self.meta.get("train_summary", {}),
                "last_point": (self.meta.get("metrics") or [{}])[-1],
            },
            "tokens_seen_total": (self.meta.get("train_summary", {}) or {}).get(
                "total_tokens_seen", 0),
        }
        for candidate in (self.dir.parent / "data" / "clean" / "corpus_stats.json",
                          self.dir.parent / "data" / "corpus_stats.json"):
            if candidate.exists():
                info["corpus"] = json.loads(candidate.read_text())
                break
        return info

    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> dict:
        """Full akṣara/token breakdown of an input string (Devanagari only)."""
        tok = self.tokenizer
        if tok is None:
            return {"error": "शब्दकोशः न लब्धः"}
        if _has_latin(text):
            # अनुवादः न भवति — रोमन्-लिपिः न स्वीक्रियते
            return {"error": "केवलं देवनागरी स्वीक्रियते", "tokens": [],
                    "devanagari": "", "n_tokens": 0, "n_aksharas": 0,
                    "chars_per_token": 0}
        devanagari = normalize_devanagari(text)
        ids = tok.encode(devanagari)
        tokens = [
            {
                "id": i,
                "text": tok.vocab[i],
                "kind": classify_token(tok.vocab[i]),
                "features": [round(f, 3) for f in akshara.features(tok.vocab[i])],
            }
            for i in ids
        ]
        return {
            "devanagari": devanagari,
            "aksharas": akshara.aksharas(devanagari),
            "n_aksharas": len(akshara.aksharas(devanagari)),
            "n_tokens": len(tokens),
            "chars_per_token": round(len(devanagari) / max(len(tokens), 1), 2),
            "tokens": tokens,
        }

    # ------------------------------------------------------------------
    def build_prompt(self, message: str, mode: str = "chat",
                     history: list[dict] | None = None) -> tuple[str, str]:
        """Returns (devanagari_prompt_text, template_string).

        `history` is the preceding conversation as [{"user":…, "assistant":…}].
        Multi-turn training data teaches the model to read it, so the template
        includes it whenever the caller supplies some.
        """
        dev = normalize_devanagari(message)
        if mode == "continuation":
            # प्रवाहः: the model simply continues the verse, no template at all
            return dev, dev
        turns = []
        for turn in (history or [])[-4:]:
            u = normalize_devanagari(str(turn.get("user", "")))
            a = normalize_devanagari(str(turn.get("assistant", "")))
            if u and a:
                turns.append(f"<user> {u} <assistant> {a}")
        turns.append(f"<user> {dev} <assistant>")
        return dev, " ".join(turns)

    def generate_stream(self, message: str, params: GenParams,
                        history: list[dict] | None = None) -> Iterator[dict]:
        """Yield events: start, token, done."""
        if not self.ready:
            yield {"event": "error",
                   "message": "भाषामयः अद्यापि प्रशिक्ष्यते — किञ्चित् क्षणं प्रतीक्षस्व।"}
            return
        if _has_latin(message):
            yield {"event": "error",
                   "message": "केवलं देवनागर्यां लिखतु। अनुवादः न क्रियते।"}
            return
        assert self.model is not None and self.tokenizer is not None

        dev, templated = self.build_prompt(message, params.mode, history)
        prompt_ids = torch.tensor([self.tokenizer.encode(templated)], dtype=torch.long)
        ctx = self.model.cfg.max_seq_len
        if prompt_ids.size(1) >= ctx - 4:
            prompt_ids = prompt_ids[:, -(ctx - 8):]

        yield {"event": "start", "prompt": templated, "devanagari_input": dev}

        out_ids: list[int] = []
        text_so_far = ""
        lines = 0
        with self.lock, torch.no_grad():
            for token_id in stream_tokens(
                self.model,
                self.feat_table,
                prompt_ids,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                repetition_penalty=params.repetition_penalty,
                eos_id=None,
            ):
                piece = self.tokenizer.vocab[token_id]
                out_ids.append(token_id)

                # <eos> marks a line end in this corpus: render it as a newline
                if piece == "<eos>":
                    lines += 1
                    text_so_far += "\n"
                    yield {"event": "token", "text": "\n", "token_id": token_id,
                           "kind": "space"}
                    if lines >= params.max_lines:
                        break
                    continue
                # other structural specials are not text
                if piece.startswith("<") and piece.endswith(">"):
                    break
                # Sanskrit only: never emit Latin characters
                if any(("a" <= ch.lower() <= "z") for ch in piece):
                    break

                text_so_far += piece
                yield {
                    "event": "token",
                    "text": piece,
                    "token_id": token_id,
                    "kind": classify_token(piece),
                }
                if params.stop_at_verse_end and len(out_ids) >= 8:
                    if piece.strip() == "॥" or text_so_far.count("।") >= 3:
                        break
                # A chat answer that has stopped making its point drifts into the
                # verse corpus. Cut it at a word boundary instead of rambling.
                if params.max_chars and len(text_so_far) >= params.max_chars:
                    if piece and piece[-1] in " ।?!":
                        break

        yield {
            "event": "done",
            "text": text_so_far,
            "script": "devanagari",
            "n_tokens": len(out_ids),
            "params": params.__dict__,
        }
