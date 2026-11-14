#!/usr/bin/env python3
"""
API smoke tests.

Uses FastAPI's TestClient when available; otherwise talks to a server on
$HNS_API (default http://localhost:8000) with urllib, so it can also be run
against a live preview.

    python3 tests/test_api.py
"""

from __future__ import annotations

import json

s_dumps = json.dumps
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = os.environ.get("HNS_API", "http://localhost:8000")


def _client():
    try:
        from fastapi.testclient import TestClient  # noqa: F401
        from api.main import app
        return TestClient(app), True
    except Exception:
        return None, False


class UrllibClient:
    def get(self, path):
        with urllib.request.urlopen(BASE + path, timeout=20) as r:
            return r.status, json.loads(r.read() or b"{}")

    def post(self, path, payload=None, json=None, **_):
        payload = payload if payload is not None else json
        data = s_dumps(payload).encode()
        req = urllib.request.Request(BASE + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
        # streaming endpoints return newline-delimited JSON
        try:
            return r.status, json.loads(body)
        except json.JSONDecodeError:
            return r.status, [json.loads(l) for l in body.splitlines() if l.strip()]


def _resp_json(resp):
    try:
        return resp.json()
    except Exception:
        return [json.loads(l) for l in resp.text.splitlines() if l.strip()]


def test_health(client, native):
    resp = client.get("/api/health")
    body = _resp_json(resp) if native else resp[1]
    assert "model_ready" in body


def test_info(client, native):
    resp = client.get("/api/info")
    body = _resp_json(resp) if native else resp[1]
    assert body["tokenizer"]["vocab_size"] > 100
    assert body["model"]["params"] >= 0


def test_tokenize(client, native):
    resp = client.post("/api/tokenize", json={"text": "धर्मक्षेत्रे कुरुक्षेत्रे"})
    body = _resp_json(resp) if native else resp[1]
    assert body["devanagari"] == "धर्मक्षेत्रे कुरुक्षेत्रे"
    assert body["n_tokens"] > 0
    assert all("kind" in t and "features" in t for t in body["tokens"])


def test_latin_is_refused(client, native):
    """The model is Sanskrit-only: romanised input must be rejected, not translated."""
    resp = client.post("/api/tokenize", json={"text": "saṃskṛtam"})
    body = _resp_json(resp) if native else resp[1]
    assert "error" in body, "latin input should be refused"
    assert body["n_tokens"] == 0


def test_devanagari_akshara_split(client, native):
    resp = client.post("/api/tokenize", json={"text": "संस्कृतम्"})
    body = _resp_json(resp) if native else resp[1]
    assert body["aksharas"] == ["सं", "स्कृ", "त", "म्"]
    assert body["n_aksharas"] == 4


def test_chat_stream_shape(client, native):
    payload = {"message": "धर्मः किम् ?", "max_new_tokens": 6, "mode": "continuation"}
    resp = client.post("/api/chat", json=payload)
    events = _resp_json(resp) if native else resp[1]
    if isinstance(events, dict):
        events = [events]
    kinds = {e["event"] for e in events}
    assert kinds & {"start", "token", "done", "error"}
    if "error" in kinds:
        print("  (model not trained yet — chat returns an error event, as designed)")


def main() -> int:
    native_client, native = _client()
    client = native_client if native else UrllibClient()
    print(f"testing via {'TestClient' if native else BASE}\n")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(client, native)
            print(f"  ✓ {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
