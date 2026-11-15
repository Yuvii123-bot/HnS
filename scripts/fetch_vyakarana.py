#!/usr/bin/env python3
"""
Fetch the published vyākaraṇa tables this project computes its grammar from.

Everything the grammar layer knows is read out of these files; nothing is
hand-typed. They come from https://github.com/ashtadhyayi-com/data, the data
layer behind ashtadhyayi.com, whose README grants use "provided that appropriate
credits are mentioned wherever applicable". There is no licence file in that
repository, so this note is the credit:

    Vyākaraṇa data © ashtadhyayi.com — https://ashtadhyayi.com

The files total ~24 MB, which is why they are fetched rather than committed:

    data/raw/vyakarana/shabda_data2.txt        śabdapatha, 9,007 stems x 24 cells
    data/raw/vyakarana/dhatu_data.txt          dhātupāṭha, 2,259 roots
    data/raw/vyakarana/dhatuforms_kartari.txt  tiṅanta, every root x lakāra (vidyut)
    data/raw/vyakarana/sutraani_laghukaumudi.txt  the Laghusiddhāntakaumudī text

    python3 scripts/fetch_vyakarana.py            # ~24 MB, needs `gh` logged in
    python3 scripts/fetch_vyakarana.py --check    # verify what is on disk
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "vyakarana"

REPO = "ashtadhyayi-com/data"
FILES = {
    "shabda_data2.txt": "shabda/data2.txt",
    "dhatu_data.txt": "dhatu/data.txt",
    "dhatuforms_kartari.txt": "dhatu/dhatuforms_vidyut_shuddha_kartari.txt",
    "sutraani_laghukaumudi.txt": "sutraani/laghukaumudi.txt",
}
MIN_BYTES = {
    "shabda_data2.txt": 5_000_000,
    "dhatu_data.txt": 1_000_000,
    "dhatuforms_kartari.txt": 5_000_000,
    "sutraani_laghukaumudi.txt": 200_000,
}


def fetch(name: str, remote: str) -> None:
    dest = OUT / name
    if dest.exists() and dest.stat().st_size >= MIN_BYTES[name]:
        print(f"  {name:<30} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  {name:<30} fetching {remote} …", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        subprocess.run(
            ["gh", "api", "-H", "Accept: application/vnd.github.raw",
             f"repos/{REPO}/contents/{remote}"],
            stdout=fh, check=True,
        )
    size = dest.stat().st_size
    if size < MIN_BYTES[name]:
        sys.exit(f"FAIL: {name} is {size} bytes, expected >= {MIN_BYTES[name]}")
    print(f"  {name:<30} {size/1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify the files")
    args = ap.parse_args()

    print(f"vyākaraṇa sources  ->  {OUT.relative_to(ROOT)}")
    if args.check:
        missing = [n for n in FILES if not (OUT / n).exists()]
        if missing:
            sys.exit(f"FAIL: missing {missing} — run without --check to fetch")
        for n in FILES:
            print(f"  {n:<30} {(OUT / n).stat().st_size/1e6:.1f} MB")
        print("  credit: Vyākaraṇa data © ashtadhyayi.com")
        return

    for name, remote in FILES.items():
        fetch(name, remote)
    (OUT.parent / "CREDIT.txt").write_text(
        "Vyākaraṇa data © ashtadhyayi.com — https://ashtadhyayi.com\n"
        "Source: https://github.com/ashtadhyayi-com/data\n"
        "Used under that repository's README: free to use with credit.\n",
        encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
