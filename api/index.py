#!/usr/bin/env python3
"""
Vercel serverless entrypoint — the Sanskrit-only model, without PyTorch.

Vercel's Python runtime installs `requirements.txt` (NumPy only) and invokes
`handler` (a BaseHTTPRequestHandler) for every request routed to /api/*.
Inference runs on `sanskrit_llm.numpy_lm`, which reproduces the trained
transformer exactly in NumPy (verified against PyTorch to 1.5e-5), reading the
weights from `serve/model.npz`.

Routes
------
GET  /api/health
GET  /api/info
POST /api/tokenize
POST /api/chat        (newline-delimited JSON events)
POST /api/reload      (no-op: the bundle is immutable per deployment)

Devanagari only: Latin input is refused, never translated.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
# On Vercel the torch-free runtime ships inside serve/runtime; locally we fall
# back to the repository package.
_RUNTIME_PKG = ROOT / "serve" / "runtime"
sys.path.insert(0, str(_RUNTIME_PKG if _RUNTIME_PKG.exists() else ROOT))

import numpy as np  # noqa: E402

from sanskrit_llm import akshara  # noqa: E402
from sanskrit_llm.numpy_lm import Runtime  # noqa: E402
from sanskrit_llm.transliterate import normalize_devanagari  # noqa: E402

SERVE_DIR = Path(os.environ.get("SERVE_DIR", str(ROOT / "serve"))).resolve()
_RUNTIME: Runtime | None = None
_LOAD_ERROR: str | None = None

LATIN = re.compile(r"[A-Za-z]")
MAX_NEW_TOKENS_CAP = 120
MAX_MESSAGE = 2000


def runtime() -> Runtime | None:
    """Load the model bundle once per warm container."""
    global _RUNTIME, _LOAD_ERROR
    if _RUNTIME is None and _LOAD_ERROR is None:
        try:
            _RUNTIME = Runtime(SERVE_DIR)
        except Exception as exc:  # pragma: no cover - deployment diagnostics
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    return _RUNTIME


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def classify_token(text: str) -> str:
    kinds = {u.kind for u in akshara.segment(text)}
    if "space" in kinds and len(text) > 1:
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


def do_tokenize(text: str) -> dict:
    rt = runtime()
    if rt is None:
        return {"error": "शब्दकोशः न लब्धः"}
    if LATIN.search(text):
        return {"error": "केवलं देवनागरी स्वीक्रियते", "tokens": [], "devanagari": "",
                "n_tokens": 0, "n_aksharas": 0, "chars_per_token": 0}
    dev = normalize_devanagari(text)
    ids = rt.tokenizer.encode(dev)
    tokens = [{
        "id": i,
        "text": rt.tokenizer.vocab[i],
        "kind": classify_token(rt.tokenizer.vocab[i]),
        "features": [round(f, 3) for f in akshara.features(rt.tokenizer.vocab[i])],
    } for i in ids]
    ak = akshara.aksharas(dev)
    return {
        "devanagari": dev,
        "aksharas": ak,
        "n_aksharas": len(ak),
        "n_tokens": len(tokens),
        "chars_per_token": round(len(dev) / max(len(tokens), 1), 2),
        "tokens": tokens,
    }


def do_chat(payload: dict) -> list[dict]:
    """Run generation and return the full event list (NDJSON body)."""
    message = str(payload.get("message", ""))[:MAX_MESSAGE]
    if not message.strip():
        return [{"event": "error", "message": "रिक्तं सन्देशम्।"}]
    if LATIN.search(message):
        return [{"event": "error",
                 "message": "केवलं देवनागर्यां लिखतु। अनुवादः न क्रियते।"}]

    dev = normalize_devanagari(message)
    mode = payload.get("mode", "chat")
    mode = mode if mode in ("chat", "continuation") else "chat"

    rt = runtime()
    if rt is None:
        return [{"event": "error",
                 "message": "भाषामयी न लभ्यते। " + (_LOAD_ERROR or "")}]

    temperature = float(np.clip(payload.get("temperature", 0.85), 0.0, 2.0))
    top_k = int(np.clip(payload.get("top_k", 40), 0, 500))
    top_p = float(np.clip(payload.get("top_p", 0.95), 0.1, 1.0))
    rep = float(np.clip(payload.get("repetition_penalty", 1.15), 1.0, 2.0))
    max_new = int(np.clip(payload.get("max_new_tokens", 90), 4, MAX_NEW_TOKENS_CAP))
    max_lines = int(np.clip(payload.get("max_lines", 2), 1, 4))

    if mode == "continuation":
        prompt = dev
    else:
        # multi-turn: replay the prior turns in the same template they were
        # trained on, so the model can see what it already said
        turns = payload.get("history") or []
        context = ""
        for turn in (turns[-6:] if isinstance(turns, list) else []):
            if not isinstance(turn, dict):
                continue
            user = str(turn.get("user", "")).strip()[:MAX_MESSAGE]
            reply = str(turn.get("assistant", "")).strip()[:MAX_MESSAGE]
            if user and reply and not LATIN.search(user):
                context += f"<user> {user} <assistant> {reply} "
        prompt = f"{context}<user> {dev} <assistant>"

    tok = rt.tokenizer
    ids = tok.encode(prompt)
    ctx = rt.model.max_seq_len
    if len(ids) >= ctx - 8:
        ids = ids[-(ctx - 8):]

    events: list[dict] = [{"event": "start", "prompt": prompt,
                           "devanagari_input": dev}]

    text = ""
    n = 0
    started = time.time()
    try:
        for token_id, is_line_end in rt.model.sample(
            np.array(ids),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep,
            max_new_tokens=max_new,
            # Corpus lines are complete training examples. Chat must not
            # continue into the next example after its end-of-sequence marker.
            # Continuation mode retains the existing multiline behaviour.
            max_lines=1 if mode == "chat" else max_lines,
            eos_id=tok.eos_id,
            feat_table=rt.feat_table,
            stop_token_ids={tok.pad_id, tok.bos_id, tok.unk_id,
                            tok.special_id("user"), tok.special_id("assistant"),
                            tok.special_id("system")},
        ):
            n += 1
            if is_line_end:
                if mode == "chat":
                    break
                text += "\n"
                events.append({"event": "token", "text": "\n",
                               "token_id": token_id, "kind": "space"})
                continue
            piece = tok.vocab[token_id]
            # the model must speak Sanskrit only: stop if it starts emitting
            # template markup or Latin characters
            if piece.startswith("<") or LATIN.search(piece):
                break
            text += piece
            events.append({"event": "token", "text": piece,
                           "token_id": token_id, "kind": classify_token(piece)})
            if len(text) >= 8 and (piece.strip() == "॥" or text.count("।") >= 3):
                break
    except Exception as exc:  # pragma: no cover
        events.append({"event": "error", "message": f"उत्पादने दोषः: {exc}"})
        return events

    events.append({
        "event": "done",
        "text": text,
        "script": "devanagari",
        "n_tokens": n,
        "seconds": round(time.time() - started, 3),
        "params": {"mode": mode, "temperature": temperature, "top_k": top_k,
                   "top_p": top_p, "repetition_penalty": rep,
                   "max_new_tokens": max_new, "max_lines": max_lines},
    })
    return events


def do_info() -> dict:
    rt = runtime()
    if rt is None:
        return {"ready": False, "error": _LOAD_ERROR,
                "tokenizer": {}, "model": {"params": 0, "config": {}},
                "training": {"summary": {}}}
    return rt.info()


# ----------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------

def do_selftest() -> dict:
    """Run one short fixed generation so a live deployment can be verified.

    Deterministic (temperature 0) and tiny, so it is cheap enough to leave in
    production as a monitoring probe.
    """
    rt = runtime()
    if rt is None:
        return {"ok": False, "error": _LOAD_ERROR}
    started = time.time()
    events = do_chat({"message": "धर्मः कः", "max_new_tokens": 12, "temperature": 0.0})
    done = events[-1]
    return {
        "ok": done.get("event") == "done",
        "output": done.get("text", ""),
        "tokens": done.get("n_tokens"),
        "params": rt.info().get("model", {}).get("params"),
        "seconds": round(time.time() - started, 3),
    }


class handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _ndjson(self, events: list[dict]) -> None:
        body = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
        self._send(200, body.encode("utf-8"), "application/x-ndjson; charset=utf-8")

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            data = json.loads(raw or b"{}")
            # a body of `null` (or a bare list) must not take the handler down
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    def _resolve_path(self) -> str:
        """Logical route for this request.

        vercel.json rewrites /api/<route> to /api/index?route=<route>, so on
        Vercel the original path arrives as a query parameter; locally the
        request path is already correct.
        """
        raw = self.path or "/"
        path, _, query = raw.partition("?")
        path = path.rstrip("/") or "/"
        for pair in query.split("&"):
            key, _, value = pair.partition("=")
            if key == "route":
                route = unquote(value).strip("/")
                return f"/api/{route}" if route else "/api"
        return path

    # ------------------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:  # noqa: N802
        path = self._resolve_path()
        if path in ("/api/health", "/api"):
            self._json({"ok": True, "model_ready": runtime() is not None,
                        "runtime": "numpy"})
        elif path == "/api/info":
            self._json(do_info())
        elif path == "/api/selftest":
            self._json(do_selftest())
        elif path in ("", "/"):
            self._json({"service": "संस्कृतम्", "see": "/"}, 200)
        else:
            self._json({"error": "not found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self._resolve_path()
        payload = self._body()
        if path == "/api/chat":
            self._ndjson(do_chat(payload))
        elif path == "/api/tokenize":
            self._json(do_tokenize(str(payload.get("text", ""))[:4000]))
        elif path == "/api/reload":
            self._json({"reloaded": True, "model_ready": runtime() is not None})
        else:
            self._json({"error": "not found", "path": path}, 404)

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))
