#!/usr/bin/env python3
"""
Does the tuned model still write verse?

Instruction tuning measurably damaged continuation in phase 1 (verse val loss
3.84 -> 5.04), so every candidate is measured here before shipping: teacher
forced loss on held-out corpus lines, plus samples from fixed prompts at the
app's sampling settings.

    python3 scripts/eval_continuation.py --checkpoint artifacts/model.pt --label baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sanskrit_llm.data import TokenDataset, token_feature_table  # noqa: E402
from sanskrit_llm.model import ModelConfig, SanskritLM, stream_tokens  # noqa: E402
from sanskrit_llm.tokenizer import AksharaTokenizer  # noqa: E402

PROMPTS = ["धर्मक्षेत्रे कुरुक्षेत्रे", "अथ राजा", "यदा सूर्यः उदेति",
           "ततः कृत्स्नम्", "सत्यमेव जयते", "कर्मण्येवाधिकारः"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--bin-dir", type=Path, default=ROOT / "data" / "bin_clean")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tokenizer.json")
    args = ap.parse_args()

    tok = AksharaTokenizer.load(args.tokenizer)
    feat = token_feature_table(tok)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SanskritLM(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()

    ds = TokenDataset(str(args.bin_dir / "val.bin"), feat, 256)
    losses = []
    with torch.no_grad():
        for xb, yb, f in ds.sequential_batches(8):
            _, loss, _ = model(xb, f, yb)
            losses.append(float(loss))
            if len(losses) >= 30:
                break
    verse_loss = sum(losses) / len(losses)

    samples = []
    for prompt in PROMPTS:
        torch.manual_seed(0)
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        out = ""
        with torch.no_grad():
            for tid in stream_tokens(model, feat, ids, max_new_tokens=32,
                                     temperature=0.8, top_k=40, top_p=0.95,
                                     repetition_penalty=1.15, eos_id=tok.eos_id):
                piece = tok.vocab[tid]
                if piece in ("<eos>", "<pad>", "<bos>", "<unk>"):
                    break
                out += piece
        samples.append({"prompt": prompt, "continuation": out.strip()})

    result = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "verse_val_loss": round(verse_loss, 4),
        "verse_perplexity": round(float(torch.exp(torch.tensor(verse_loss))), 2),
        "mean_continuation_chars": round(sum(len(s["continuation"]) for s in samples) / len(samples), 1),
        "samples": samples,
    }
    out = ROOT / "reports" / f"cont_{args.label}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, ensure_ascii=False, indent=2))
    for s in samples:
        print(f"  {s['prompt']:<22} {s['continuation']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
