#!/usr/bin/env python3
"""
Build a Devanagari Sanskrit corpus from GRETIL and DCS source texts.

Both source corpora are romanised (IAST / GRETIL conventions), so every line is
cleaned of editorial apparatus (verse numbers, page headers, URLs) and then
transliterated into Devanagari with :mod:`sanskrit_llm.transliterate`.

Outputs
-------
data/train.txt   one verse / sentence per line, Devanagari
data/val.txt     1% held out
data/corpus_stats.json

Usage
-----
    python scripts/build_corpus.py --max-chars 8000000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sanskrit_llm.transliterate import (  # noqa: E402
    detect_script, iast_to_devanagari, normalize_devanagari, iast_to_slp1,
)

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

#: editorial references: [1.001.01a], (2.4), {3}, ⟨4⟩, "1.2.3"
REF_BRACKETS = re.compile(r"\[[^\]]{0,24}\]|\([^)]{0,18}\)|\{[^}]{0,18}\}|⟨[^⟩]{0,18}⟩")
URL = re.compile(r"https?://\S+|www\.\S+")
PAGE_NOISE = re.compile(r"^\s*(GRETIL|Source|Text from|Input by|Proofread|page|Page)\b.*$", re.M)
MULTISPACE = re.compile(r"[ \t\u00a0]+")
LEADING_NUM = re.compile(r"^\s*\d+[.)]\s*")

#: characters we accept in romanised source text (IAST + ASCII)
IAST_OK = set(
    "abcdefghijklmnopqrstuvwxyz"
    "āīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻẖḷ"
    "m̐'’"
    " .,;:!?/|-\u2013\u2014()[]{}<>\"“”‘’…*#&%+=_0123456789\n\t"
)
IAST_OK |= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IAST_OK |= set("ĀĪŪṚṜḶḸŚṢṆṬḌṄÑṀḤ")

DEV_LETTER = re.compile(r"[\u0900-\u097F]")
DEV_LETTERS = re.compile(r"[\u0915-\u0939\u0905-\u0914\u093E-\u094C\u0901-\u0903\u094D]")


def looks_romanised_sanskrit(text: str) -> bool:
    """Reject Dravidian / Tibetan / English matter that sneaks into the corpora."""
    bad = set(text) - IAST_OK
    if bad:
        return False
    # needs a decent share of IAST-specific letters to count as romanised Sanskrit
    iast_marks = sum(1 for c in text if c in "āīūṛṝṃḥṅñṭḍṇśṣḷēō")
    return iast_marks >= max(3, len(text) // 200)


def clean_line(line: str) -> str:
    line = line.replace("\u00ad", "")
    line = URL.sub(" ", line)
    line = REF_BRACKETS.sub(" ", line)
    line = LEADING_NUM.sub("", line)
    line = line.replace("--", "—").replace("_", "")
    return MULTISPACE.sub(" ", line).strip()


def to_devanagari_line(line: str) -> str:
    script = detect_script(line)
    if script == "devanagari":
        return normalize_devanagari(line)
    if script == "slp1":
        from sanskrit_llm.transliterate import slp1_to_iast
        return normalize_devanagari(iast_to_devanagari(slp1_to_iast(line)))
    return normalize_devanagari(iast_to_devanagari(line))


#: characters that only appear in editorial apparatus, never in a verse
JUNK_CHARS = set("%#=<>*+&_~^{}[\\]|@$`")


DEV_HYPHEN = re.compile(r"(?<=[\u0915-\u0939\u093e-\u094c])[-\u2013](?=[\u0915-\u0939])")


def strip_editorial_hyphens(line: str) -> str:
    """GRETIL hyphenates compound members editorially; Sanskrit writes them joined."""
    prev = None
    while prev != line:
        prev = line
        line = DEV_HYPHEN.sub("", line)
    return line


def accept_devanagari(line: str, min_letters: int = 12) -> bool:
    """Accept only lines that look like clean Sanskrit verse or prose."""
    letters = DEV_LETTERS.findall(line)
    if len(letters) < min_letters:
        return False
    if any(c in JUNK_CHARS for c in line):
        return False
    # editorial numbers (verse / page / footnote) must not survive
    if re.search(r"[०-९\d]{2,}", line):
        return False
    # Śrautasūtra-style "word.word.word" notation and commentary colons
    if line.count(".") >= 2 or ":" in line:
        return False
    # no Latin letters should remain after transliteration
    if any(c.isascii() and c.isalpha() for c in line):
        return False
    non_space = [c for c in line if not c.isspace()]
    if not non_space:
        return False
    dev_ratio = len(DEV_LETTER.findall(line)) / len(non_space)
    if dev_ratio < 0.62:
        return False
    # GRETIL hyphenates compound splits; too many means an editorial line
    if line.count("-") > max(2, len(line) // 18):
        return False
    # must contain at least one real word (2+ akṣaras)
    return len(re.findall(r"[\u0915-\u0939\u0905-\u0914][\u093e-\u094c\u0901-\u0903\u094d]*[\u0915-\u0939]", line)) >= 2


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def iter_source_files(raw: Path, rng: random.Random) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    gretil = raw / "gretil"
    for p in sorted(gretil.rglob("*.txt")):
        rel = p.relative_to(gretil)
        category = rel.parts[0] if len(rel.parts) > 1 else "gretil"
        files.append((f"gretil/{category}", p))
    dcs = raw / "dcs" / "corpora"
    for p in sorted(dcs.glob("*.txt")):
        files.append(("dcs", p))
    rng.shuffle(files)
    # interleave categories so a truncated build stays genre-balanced
    by_cat: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for item in files:
        by_cat[item[0]].append(item)
    ordered: list[tuple[str, Path]] = []
    while any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat]:
                ordered.append(by_cat[cat].pop())
    return ordered


def build(raw: Path, out_dir: Path, max_chars: int, seed: int = 13) -> dict:
    rng = random.Random(seed)
    files = iter_source_files(raw, rng)

    seen: set[str] = set()
    per_source: Counter = Counter()
    char_count = 0
    skipped_files = 0
    lines_out: list[str] = []

    for source, path in files:
        if char_count >= max_chars:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            skipped_files += 1
            continue

        # quick file-level reject
        sample = text[:4000]
        if not re.search(r"[\u0900-\u097F]", sample) and not looks_romanised_sanskrit(
                re.sub(r"\[[^\]]*\]", "", sample)):
            skipped_files += 1
            continue

        kept = 0
        for raw_line in PAGE_NOISE.sub("", text).splitlines():
            line = clean_line(raw_line)
            if not line:
                continue
            dev = strip_editorial_hyphens(to_devanagari_line(line))
            if not accept_devanagari(dev):
                continue
            if dev in seen:
                continue
            seen.add(dev)
            lines_out.append(dev)
            char_count += len(dev)
            kept += 1
            if char_count >= max_chars:
                break
        per_source[source] += kept

    rng.shuffle(lines_out)
    n_val = max(200, len(lines_out) // 100)
    val, train = lines_out[:n_val], lines_out[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    stats = {
        "train_lines": len(train),
        "val_lines": len(val),
        "devanagari_chars": char_count,
        "unique_lines": len(seen),
        "sources": dict(per_source.most_common()),
        "skipped_files": skipped_files,
        "files_seen": len(files),
    }
    (out_dir / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--max-chars", type=int, default=8_000_000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    stats = build(args.raw, args.out, args.max_chars, args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
