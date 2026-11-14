#!/usr/bin/env python3
"""Sample from a checkpoint from the command line (sanity checks)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sanskrit_llm.inference import GenParams, SanskritEngine

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", default=["धर्मक्षेत्रे कुरुक्षेत्रे"])
    ap.add_argument("--mode", default="continuation", choices=["chat", "continuation"])
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    engine = SanskritEngine(args.artifacts)
    if not engine.ready:
        print("no checkpoint found in", args.artifacts); return
    prompt = " ".join(args.prompt)
    for i in range(args.n):
        params = GenParams(mode=args.mode, temperature=args.temperature,
                           top_k=args.top_k, max_new_tokens=args.max_tokens)
        text = ""
        for evt in engine.generate_stream(prompt, params):
            if evt["event"] == "token":
                text += evt["text"]
            elif evt["event"] == "error":
                text = evt["message"]
        print(f"\n── नमूना {i + 1} ──\n{text}")

if __name__ == "__main__":
    main()
