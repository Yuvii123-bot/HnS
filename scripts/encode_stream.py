#!/usr/bin/env python3
"""
Encode a directory of text parts into one flat uint16 token array, streaming.

`prepare_data.py` loads a whole file into memory, which cannot work for a 150 M
token stream on a 3 GB machine. This reads line by line and appends.

    python3 scripts/encode_stream.py --dir data/curriculum --out data/bin_150m
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sanskrit_llm.tokenizer import AksharaTokenizer  # noqa: E402


def encode_file(path: Path, fh, tok, eos: int) -> int:
    buf = array("H")
    total = 0
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            buf.extend(tok.encode(line))
            buf.append(eos)
            if len(buf) >= 200_000:
                fh.write(buf.tobytes())
                total += len(buf)
                buf = array("H")
    if buf:
        fh.write(buf.tobytes())
        total += len(buf)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tokenizer.json")
    args = ap.parse_args()

    tok = AksharaTokenizer.load(args.tokenizer)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, pattern in (("train", "part_*.txt"), ("val", "val.txt")):
        paths = sorted(args.dir.glob(pattern))
        if not paths:
            continue
        total = 0
        with (args.out / f"{split}.bin").open("wb") as fh:
            for path in paths:
                n = encode_file(path, fh, tok, tok.eos_id)
                total += n
                print(f"  {path.name}  {n:,} tokens  (total {total:,})", flush=True)
        stats[split] = {"tokens": total, "bytes": total * 2, "parts": len(paths)}
        print(f"[{split}] {total:,} tokens -> {(args.out / f'{split}.bin')}", flush=True)
    (args.out / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
