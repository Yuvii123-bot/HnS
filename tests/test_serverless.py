#!/usr/bin/env python3
"""
Test the Vercel serverless handler exactly as deployed: NumPy only, no PyTorch.

Torch and FastAPI are blocked from importing during this test, so it fails loudly
if the deployed code path ever grows a dependency that Vercel cannot install.

    python3 tests/test_serverless.py
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"


class Blocker:
    """Import hook that simulates the serverless environment (no torch)."""

    BLOCKED = {"torch", "fastapi", "uvicorn", "pydantic"}

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.BLOCKED:
            raise ImportError(f"'{fullname}' is blocked in the serverless test")
        return None


def start_server():
    sys.meta_path.insert(0, Blocker())
    for mod in [m for m in list(sys.modules) if m.split(".")[0] in Blocker.BLOCKED]:
        del sys.modules[mod]
    index = importlib.import_module("api.index")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), index.handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, index


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return r.status, json.loads(r.read())


def post(path: str, payload: dict):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = r.read().decode()
    try:
        parsed = json.loads(body)
        # a one-event NDJSON body is itself valid JSON: normalise to a list
        return r.status, parsed if isinstance(parsed, dict) and "event" not in parsed else [parsed]
    except json.JSONDecodeError:
        return r.status, [json.loads(l) for l in body.splitlines() if l.strip()]


def main() -> int:
    server, index = start_server()
    # prove the environment really is torch-free
    assert "torch" not in sys.modules, "torch leaked into the serverless path"
    failures = 0

    def check(name, fn):
        nonlocal failures
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")

    def health():
        status, body = get("/api/health")
        assert status == 200 and body["model_ready"] is True
        assert body["runtime"] == "numpy"

    def info():
        status, body = get("/api/info")
        assert body["model"]["params"] > 5_000_000
        assert body["tokenizer"]["vocab_size"] == 4000
        assert body["runtime"] == "numpy"

    def tokenize():
        status, body = post("/api/tokenize", {"text": "धर्मक्षेत्रे कुरुक्षेत्रे"})
        assert body["n_aksharas"] > 5, body
        assert body["tokens"][0]["text"] == "धर्म"
        assert body["tokens"][0]["kind"] == "conjunct"

    def latin_refused():
        status, body = post("/api/tokenize", {"text": "dharma"})
        assert "error" in body
        status, events = post("/api/chat", {"message": "what is dharma"})
        assert isinstance(events, list) and events[0]["event"] == "error"
        assert "देवनागर्" in events[0]["message"]  # केवलं देवनागर्यां लिखतु

    def chat_streams_sanskrit():
        status, events = post("/api/chat", {
            "message": "रामः कः आसीत्", "max_new_tokens": 24, "mode": "chat"})
        kinds = [e["event"] for e in events]
        assert kinds[0] == "start" and kinds[-1] == "done", kinds
        text = events[-1]["text"]
        assert text.strip(), "model produced nothing"
        # output must be Devanagari, never Latin
        assert not any("a" <= ch.lower() <= "z" for ch in text), text

    check("health", health)
    check("info", info)
    check("tokenize (akṣara split)", tokenize)
    check("latin input refused", latin_refused)
    check("chat answers in Sanskrit", chat_streams_sanskrit)

    server.shutdown()
    print("\nall serverless checks passed" if not failures else f"\n{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
