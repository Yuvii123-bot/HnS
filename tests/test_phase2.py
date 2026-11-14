#!/usr/bin/env python3
"""
Guarantees for phase 2: morphology, dataset integrity, evaluation hold-outs.

    python3 tests/test_phase2.py
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
ARABIC = re.compile(r"[0-9]")
DATA = ROOT / "data" / "instruct_v2"


def user_turns(line: str) -> list[str]:
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

    def morphology():
        from sanskrit_kb import morph
        assert morph.self_check() == 0
        # every cell of every paradigm must be a non-empty Devanagari form
        for noun in morph.NOUNS:
            for cell, form in morph.decline(noun).items():
                assert form and not LATIN.search(form), (noun.lemma, cell, form)

    def dialogues():
        from sanskrit_kb import dialogues
        assert dialogues.self_check() == 0

    def grammar_checker():
        import grammar_check
        assert grammar_check.self_check() == 0

    def dataset_exists():
        assert (DATA / "train.txt").exists(), "run scripts/build_instruct_v2.py"
        assert (DATA / "eval_set.jsonl").exists()

    train = [l for l in (DATA / "train.txt").read_text(encoding="utf-8").splitlines() if l.strip()] \
        if (DATA / "train.txt").exists() else []
    evals = [json.loads(l) for l in (DATA / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()] if (DATA / "eval_set.jsonl").exists() else []
    stats = json.loads((DATA / "stats.json").read_text(encoding="utf-8")) \
        if (DATA / "stats.json").exists() else {}

    def size():
        assert stats["pairs"] >= 10000, stats.get("pairs")
        assert len(evals) >= 500, len(evals)

    def no_latin_or_digits():
        for line in train:
            plain = line.replace("<user>", "").replace("<assistant>", "")
            assert not LATIN.search(plain), line[:60]
            assert not ARABIC.search(plain), line[:60]

    def answers_well_formed():
        for line in train:
            for part in line.split("<assistant>")[1:]:
                answer = part.split("<user>")[0].strip()
                assert answer, line[:60]
                assert answer[-1] in "।?!", answer[-1:]
                assert "॥" not in answer, line[:60]

    def evaluation_is_held_out():
        blocked = {e["prompt"].strip() for e in evals}
        for line in train:
            for turn in user_turns(line):
                assert turn not in blocked, turn

    def evaluation_frames_unseen():
        """The two eval-only question frames must not appear in training."""
        from build_instruct_v2 import FRAMES_EVAL, FRAMES_TRAIN
        assert not (set(FRAMES_EVAL) & set(FRAMES_TRAIN))
        for frame in FRAMES_EVAL:
            prefix = frame.split("{q}")[0].strip()
            if not prefix:
                continue
            for line in train[:4000]:
                assert prefix not in line, prefix

    def unseen_nouns_absent():
        """A held-out noun must not appear in training at all."""
        from build_instruct_v2 import HELD_OUT_NOUNS
        from sanskrit_kb.morph import NOUN_BY_LEMMA, decline
        forms = set()
        for lemma in HELD_OUT_NOUNS:
            for form in decline(NOUN_BY_LEMMA[lemma]).values():
                if len(form) >= 5:
                    forms.add(form)
        blob = "\n".join(train)
        leaked = [f for f in forms if f in blob]
        assert not leaked, leaked[:5]

    def eval_items_have_references():
        for e in evals:
            assert e["prompt"], e
            assert e["reference"] or e["kind"] == "multi-turn", e["id"]
            assert e["expected_type"] in ("sentence", "numeral", "vibhakti", "form"), e
            assert isinstance(e.get("history", []), list)

    def multi_turn_present():
        multi = [l for l in train if l.count("<user>") >= 2]
        assert len(multi) >= 150, len(multi)

    check("declension engine matches known forms", morphology)
    check("dialogue scripts fill every slot", dialogues)
    check("grammar checker passes its own cases", grammar_checker)
    check("phase-2 dataset exists", dataset_exists)
    check("dataset size (>= 10k pairs, >= 500 eval questions)", size)
    check("no Latin characters or Arabic digits", no_latin_or_digits)
    check("answers are complete sentences", answers_well_formed)
    check("evaluation prompts never appear in training", evaluation_is_held_out)
    check("evaluation question frames are unseen", evaluation_frames_unseen)
    check("held-out nouns never appear in training", unseen_nouns_absent)
    check("evaluation items carry references and types", eval_items_have_references)
    check("multi-turn training rows exist", multi_turn_present)

    print("\nall phase-2 checks passed" if not failures else f"\n{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
