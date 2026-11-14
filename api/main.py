#!/usr/bin/env python3
"""
FastAPI backend for the Sanskrit chat interface.

Endpoints
---------
GET  /                 the chat UI
GET  /api/info         model + tokenizer + training metadata
GET  /api/health       liveness / readiness
POST /api/chat         streaming generation (newline-delimited JSON)
POST /api/tokenize     akṣara + token breakdown of a string
POST /api/reload       re-read the checkpoint from disk (after a training run)

Run
---
    python api/main.py            # http://0.0.0.0:8000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sanskrit_llm.akshara import aksharas
from sanskrit_llm.inference import GenParams, SanskritEngine
from sanskrit_llm.transliterate import normalize_devanagari

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "public"
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))

app = FastAPI(title="Saṃskṛta-LLM", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SanskritEngine(ARTIFACTS)


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    mode: str = "chat"
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    max_new_tokens: int = 90
    stop_at_verse_end: bool = True


class TextRequest(BaseModel):
    text: str = Field(..., max_length=4000)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model_ready": engine.ready}


@app.get("/api/info")
def info() -> dict:
    return engine.info()


@app.post("/api/reload")
def reload_model() -> dict:
    ok = engine.reload()
    return {"reloaded": ok, "model_ready": engine.ready}


@app.post("/api/tokenize")
def tokenize(req: TextRequest) -> dict:
    return engine.tokenize(req.text)


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    params = GenParams(
        mode=req.mode if req.mode in ("chat", "continuation") else "chat",
        temperature=max(0.0, min(2.0, req.temperature)),
        top_k=max(0, min(500, req.top_k)),
        top_p=max(0.1, min(1.0, req.top_p)),
        repetition_penalty=max(1.0, min(2.0, req.repetition_penalty)),
        max_new_tokens=max(4, min(240, req.max_new_tokens)),
        stop_at_verse_end=req.stop_at_verse_end,
    )

    dev = normalize_devanagari(req.message)
    if not dev:
        def empty():
            yield json.dumps({"event": "error", "message": "empty message"}) + "\n"
        return StreamingResponse(empty(), media_type="application/x-ndjson")

    def event_stream():
        try:
            for event in engine.generate_stream(dev, params):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # surface errors to the UI instead of a dead stream
            yield json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------

if (WEB / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")


@app.get("/")
def index():
    index_file = WEB / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"error": "web/index.html missing"}, status_code=404)


@app.get("/favicon.ico")
def favicon():
    icon = WEB / "static" / "favicon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    return JSONResponse({}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
