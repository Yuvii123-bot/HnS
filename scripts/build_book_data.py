#!/usr/bin/env python3
"""
Turn one public-domain Sanskrit book into training data.

Book: Nārāyaṇa's *Hitopadeśa*, M. R. Kale's edition — the traditional first
reader for Sanskrit: short prose stories with dialogue, in simple sentences,
interspersed with moral verses. Public domain (Nārāyaṇa is mediaeval; Kale's
edition is 1896/1912). Devanagari transcription from
github.com/funderburkjim/hitopadesha-kale (hitokale_deva_F.txt), which carries
no explicit licence, so treat the transcription as unlicensed and the underlying
text as public domain.

Outputs:
  data/book/train.txt,val.txt        cleaned prose/verse lines for continued pretraining
  data/book_instr/train.txt          instruction rows derived from the book
  data/book_probe.jsonl              held-out items: no line used here appears in training

Two derived instruction families, both grounded in the text rather than invented:
  * recitation — "नीतिवचनं वद" / "हितोपदेशतः श्लोकं वद" answered with a real verse
  * narration  — "कथां वद" answered with a real prose line
A held-out tail of the book supplies the probe, so the probe measures what the
model learned about the book rather than what it was shown.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")
PAGE_NO = re.compile(r"^[०-९]+$")
VERSE_NO = re.compile(r"\s*॥\s*[०-९]+\s*॥\s*$")

RECITE_PROMPTS = ["नीतिवचनं वद", "हितोपदेशतः श्लोकं वद", "सुभाषितं वद",
                  "उपदेशं वद", "नीतिं वद", "श्लोकं वद", "धर्मवचनं वद"]
NARRATE_PROMPTS = ["कथां वद", "हितोपदेशात् कथां वद", "किं कथयसि?", "कथां श्रावय"]

BOOK_FACTS = [
    ("हितोपदेशः कः?", "हितोपदेशः नीतिशास्त्रग्रन्थः, नारायणेन रचितः।"),
    ("हितोपदेशं केन रचितः?", "हितोपदेशः नारायणेन रचितः।"),
    ("हितोपदेशे किम् अस्ति?", "हितोपदेशे कथाः नीतिवचनानि च सन्ति।"),
    ("हितोपदेशः कीदृशः ग्रन्थः?", "हितोपदेशः नीतिशास्त्रग्रन्थः नारायणेन रचितः।"),
]


def clean_lines(path: Path) -> list[str]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or PAGE_NO.match(s) or LATIN.search(s):
            continue
        # keep the verse-final marker: it is how a verse is told from prose later
        s = VERSE_NO.sub(" ॥", s)
        s = re.sub(r"^[०-९]+\s*", "", s)          # leading page numerals
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s+([।॥])", r"\1", s)
        if len(DEV.findall(s)) < 20:
            continue
        if not s.endswith(("।", "॥")):
            s += " ।"
        out.append(s)
    return out


def is_verse(line: str) -> bool:
    return line.rstrip().endswith("॥")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/hitopadesha/hitokale_deva_F.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/book"))
    ap.add_argument("--instr-out", type=Path, default=Path("data/book_instr"))
    ap.add_argument("--probe", type=Path, default=Path("data/book_probe.jsonl"))
    ap.add_argument("--probe-lines", type=int, default=260,
                    help="trailing lines held out of training entirely")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--curate", type=int, default=0,
                    help="keep only the first N verses and N prose lines as instruction "
                         "targets; 0 keeps everything. A 9.9M model fits ~400 short "
                         "mappings, not 2500 long ones, so curating is the honest option.")
    args = ap.parse_args()

    lines = clean_lines(args.source)
    # verses and maxims recur in Hitopadeśa; without de-duplication a held-out
    # line can also sit in the training half, which would fake the probe
    seen: set[str] = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    print(f"de-duplication: {len(lines)} -> {len(unique)} lines")
    lines = unique
    head, tail = lines[:-args.probe_lines], lines[-args.probe_lines:]
    print(f"book: {len(lines)} usable lines ({len(head)} train / {len(tail)} held out)")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n".join(head) + "\n", encoding="utf-8")
    rng = random.Random(args.seed)
    val = rng.sample(head, 32)
    (args.out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    # ---- derived instruction data (from the training side only) ----
    verses = [l for l in head if is_verse(l)]
    prose = [l for l in head if not is_verse(l)]
    if len(verses) < 60:  # guard: if verse tagging ever collapses, say so loudly
        raise SystemExit(f"verse tagging found only {len(verses)} verses; inspect the cleaner")
    print(f"verse lines {len(verses)}, prose lines {len(prose)}")
    rng.shuffle(verses)
    rng.shuffle(prose)
    if args.curate:
        verses, prose = verses[:args.curate], prose[:args.curate]

    seen_lines: set[str] = set()
    rows: list[str] = []
    for i, verse in enumerate(verses):
        rows.append(f"<user> {RECITE_PROMPTS[i % len(RECITE_PROMPTS)]} <assistant> {verse}")
    for i, line in enumerate(prose):
        rows.append(f"<user> {NARRATE_PROMPTS[i % len(NARRATE_PROMPTS)]} <assistant> {line}")
    # the completion format is taught on training lines, so a completion test is
    # testing recall of the book, not recall of a format it never saw
    # every usable line becomes a completion pair: recall of the book is what
    # "fed it the book" should mean, and half the book is only 5 words a line
    pool = verses + prose if args.curate else [l for l in head if len(l.split()) >= 4]
    complete_src = [l for l in pool if len(l.split()) >= 4]
    for line in complete_src:
        w = line.split()
        cut = len(w) // 2
        rows.append(f"<user> पूर्णं कुरु — {' '.join(w[:cut])} <assistant> {' '.join(w[cut:])}")
    for q, a in BOOK_FACTS:
        for _ in range(20):
            rows.append(f"<user> {q} <assistant> {a}")

    # every probe line must be absent from training
    seen_lines |= {r.split("<assistant>")[-1].strip() for r in rows}
    held = set(tail)
    leaks = [r for r in rows if r.split("<assistant>")[-1].strip() in held]
    if leaks:
        raise SystemExit(f"instruction row reproduces a held-out line: {leaks[0][:80]}")

    args.instr_out.mkdir(parents=True, exist_ok=True)
    (args.instr_out / "train.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (args.instr_out / "val.txt").write_text("\n".join(rows[:8]) + "\n", encoding="utf-8")
    print(f"instruction rows: {len(rows)} "
          f"({min(280, len(verses))} verses, {min(120, len(prose))} prose, {len(BOOK_FACTS)} facts)")

    # ---- probe: complete a held-out verse, and recite something new ----
    # completion probe: show the first half of a held-out line, expect the rest.
    # Splitting on a word boundary keeps the prompt grammatical.
    def halves(pool, n, kind):
        out = []
        for line in pool:
            w = line.split()
            if len(w) < 6:
                continue
            cut = len(w) // 2
            second = " ".join(w[cut:])
            if len(DEV.findall(second)) < 12:
                continue
            out.append({"kind": kind, "prompt": f"पूर्णं कुरु — {' '.join(w[:cut])}",
                        "target": second, "full": line})
            if len(out) >= n:
                break
        return out

    probe = []
    # memorisation: lines the model was trained on
    # lines that were in the pretraining text (6x) but were never used as a
    # completion pair, so this tests absorption of the book, not of one prompt
    # every line is now a completion pair, so "trained" recall is train-set
    # recall by construction; it is reported separately from held-out recall
    step = max(1, len(complete_src) // 24)
    absorbed = complete_src[::step]
    probe += halves(absorbed, 12, "recall_trained")
    # generalisation: lines held out of training entirely
    probe += halves(tail, 12, "recall_heldout")
    for i in range(4):
        probe.append({"kind": "recite", "prompt": RECITE_PROMPTS[i],
                      "target": None})
    for i in range(2):
        probe.append({"kind": "narrate", "prompt": NARRATE_PROMPTS[i], "target": None})
    for q, a in BOOK_FACTS[:4]:
        probe.append({"kind": "fact", "prompt": q, "accept": a, "target": None})
    for j, item in enumerate(probe):
        item["id"] = f"b{j+1:02d}"
    args.probe.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in probe) + "\n",
                          encoding="utf-8")
    print(f"probe: {len(probe)} items written to {args.probe}")


if __name__ == "__main__":
    main()
