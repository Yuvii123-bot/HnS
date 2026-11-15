#!/usr/bin/env python3
"""
Rebuild a trainable PyTorch checkpoint from the deployed NumPy bundle.

The bundle in serve/ is the only copy of the currently deployed weights that
travels with the repository -- the .pt files are gitignored. When training has to
continue from the deployed model, this reconstructs it exactly:

    python3 scripts/import_bundle.py --bundle serve --out artifacts/deployed/latest.pt

The forward pass is then checked against the bundle's own NumPy runtime on a few
prompts; a mismatch means the import is wrong, not that the model changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sanskrit_llm.model import ModelConfig, SanskritLM
from sanskrit_llm.tokenizer import AksharaTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=Path("serve"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/deployed/latest.pt"))
    ap.add_argument("--check", action="store_true", default=True)
    args = ap.parse_args()

    meta = json.loads((args.bundle / "meta.json").read_text(encoding="utf-8"))
    cfg = ModelConfig.from_dict(meta["config"])
    model = SanskritLM(cfg)

    npz = np.load(args.bundle / "model.npz")
    state = {k: torch.from_numpy(npz[k].copy()) for k in npz.files}

    expected = set(model.state_dict())
    got = set(state)
    missing, extra = expected - got, got - expected
    if missing or extra:
        raise SystemExit(f"tensor mismatch: missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}")

    model.load_state_dict(state, strict=True)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(),
                "optimizer": None, "step": 0,
                "extra": {"imported_from": str(args.bundle),
                          "note": "deployed NumPy bundle, imported back for further training"}},
               args.out)
    tok = AksharaTokenizer.load(args.bundle / "tokenizer.json")
    tok.save(args.out.parent / "tokenizer.json")
    print(f"wrote {args.out}  ({sum(p.numel() for p in model.parameters()):,} params, "
          f"{len(state)} tensors)")

    if args.check:
        # same weights must give the same continuation through both runtimes
        from sanskrit_llm.data import token_feature_table
        from sanskrit_llm.model import stream_tokens
        from sanskrit_llm.numpy_lm import Runtime

        sys.path.insert(0, str(args.bundle / "runtime"))
        rt = Runtime(args.bundle)
        feat = token_feature_table(tok)
        SPECIAL = {"<eos>", "<user>", "<assistant>", "<pad>", "<bos>", "<unk>", "<system>"}
        stop = {tok.pad_id, tok.bos_id, tok.unk_id, tok.special_id("user"),
                tok.special_id("assistant"), tok.special_id("system")}

        def torch_reply(prompt: str) -> str:
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
            out = ""
            with torch.no_grad():
                for tid in stream_tokens(model, feat, ids, max_new_tokens=40,
                                         temperature=0.0, top_k=40, top_p=1.0,
                                         repetition_penalty=1.15, eos_id=None):
                    piece = tok.vocab[tid]
                    if piece in SPECIAL or "\n" in piece:
                        break
                    out += piece
            return out.strip()

        def numpy_reply(prompt: str) -> str:
            ids = tok.encode(prompt)
            out = ""
            for tid, line_end in rt.model.sample(
                np.array(ids), temperature=0.0, top_k=40, top_p=1.0,
                repetition_penalty=1.15, max_new_tokens=40, max_lines=2,
                eos_id=tok.eos_id, feat_table=rt.feat_table, stop_token_ids=stop,
            ):
                if line_end:
                    break
                piece = tok.vocab[tid]
                if piece in SPECIAL or piece.startswith("<"):
                    break
                out += piece
            return out.strip()

        agree = 0
        probes = ["<user> नमस्ते <assistant>", "<user> तव नाम किम्? <assistant>",
                  "<user> धर्मः कः? <assistant>", "<user> पञ्च अधिकं त्रीणि कति? <assistant>"]
        for p in probes:
            a, b = torch_reply(p), numpy_reply(p)
            same = a[:40] == b[:40]
            agree += same
            print(f"  {'same' if same else 'DIFF'}  {p[:38]:40s} torch={a[:34]!r} numpy={b[:34]!r}")
        print(f"runtime agreement on {agree}/{len(probes)} probes")
        if agree < len(probes):
            raise SystemExit("import does not reproduce the deployed model — stop and investigate")


if __name__ == "__main__":
    main()
