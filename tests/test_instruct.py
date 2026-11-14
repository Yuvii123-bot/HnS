#!/usr/bin/env python3
"""
Guarantees for the instruction-tuning pipeline.

These are the properties the dataset and the measurement harness must keep;
they are cheap, so they run in CI rather than being checked by hand.

    python3 tests/test_instruct.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LATIN = re.compile(r"[A-Za-z]")
ARABIC_DIGIT = re.compile(r"[0-9]")


def load(path: Path) -> list[str]:
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def turns(line: str) -> list[str]:
    return [t.split("<assistant>")[0].strip() for t in line.split("<user>")[1:]]


def main() -> int:
    failures = 0

    def check(name, fn):
        nonlocal failures
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")

    data = ROOT / "data" / "instruct"
    train = load(data / "train.txt")
    val = load(data / "val.txt")
    stats = json.loads((data / "stats.json").read_text(encoding="utf-8"))
    evals = [json.loads(l) for l in load(data / "eval_prompts.jsonl")]

    def size():
        assert stats["pairs"] >= 2000, stats["pairs"]
        assert stats["base_facts"] >= 700, stats["base_facts"]
        assert len(train) + len(val) == stats["pairs"]

    def devanagari_only():
        for line in train + val:
            plain = line.replace("<user>", "").replace("<assistant>", "")
            assert not LATIN.search(plain), line[:60]

    def no_arabic_digits():
        for line in train:
            assert not ARABIC_DIGIT.search(line), line[:60]

    def answers_are_sentences():
        for line in train + val:
            answer = line.split("<assistant>", 1)[1].strip()
            assert answer, line[:60]
            assert answer[-1] in "।?!", (line[:80], answer[-1:])
            assert "॥" not in answer, line[:60]        # no verse marker

    def no_duplicates():
        assert len(train) == len(set(train))
        assert not (set(train) & set(val))

    def eval_prompts_held_out():
        held = {e["prompt"].strip() for e in evals}
        for line in train + val:
            for turn in turns(line):
                assert turn not in held, turn

    def every_prompt_has_answer():
        for line in train:
            assert "<user>" in line and "<assistant>" in line, line[:60]

    def keyword_matches_inflection():
        from eval_instruct import keyword_in
        assert keyword_in("तत् पदं षष्ठ्याम् विभक्तौ।", ["षष्ठी"])
        assert keyword_in("तत् पदं सप्तम्याम् विभक्तौ।", ["सप्तमी"])
        assert keyword_in("मिलित्वा त्रीणि भवन्ति।", ["त्रीणि"])
        assert not keyword_in("मिलित्वा चत्वारि भवन्ति।", ["त्रीणि"])
        assert not keyword_in("अहं पद्भ्याम् आगतः अस्मि।", ["षष्ठी"])

    def mix_keeps_corpus():
        mix = ROOT / "data" / "instruct_mix"
        if not (mix / "mix_stats.json").exists():
            return
        m = json.loads((mix / "mix_stats.json").read_text(encoding="utf-8"))
        share = m["verse_share_of_training_tokens"]
        assert 0.10 <= share <= 0.40, share
        assert m["instruction_pairs_after_balancing"] > m["instruction_pairs_authored"]

    check("dataset size (>= 2000 pairs from >= 700 facts)", size)
    check("no Latin characters", devanagari_only)
    check("no Arabic digits", no_arabic_digits)
    check("answers are complete, punctuated sentences", answers_are_sentences)
    check("no duplicate pairs, no train/val overlap", no_duplicates)
    check("evaluation prompts are held out", eval_prompts_held_out)
    check("every pair has a prompt and an answer", every_prompt_has_answer)
    check("keyword matching handles inflection", keyword_matches_inflection)
    check("training mix replays the corpus", mix_keeps_corpus)

    print("\nall instruction checks passed" if not failures else f"\n{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
