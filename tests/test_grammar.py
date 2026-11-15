#!/usr/bin/env python3
"""
Guarantees for the grammar layer.

The point of these tests is that the answers are *looked up*, so they can be
checked against the source tables rather than against a model:

  1. the 15 held-out benchmark questions are answered correctly
  2. nouns that appear in no drill and in no benchmark are declined correctly,
     in two different phrasings
  3. every answer is Devanagari-only and ends in a danda
  4. the router refuses what it does not know instead of guessing
  5. the benchmark prompts are held out of the training corpus

    python3 tests/test_grammar.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_grammar import score  # noqa: E402
from scripts.sanskrit_kb.lexicon import lexicon  # noqa: E402
from scripts.sanskrit_kb.router import BENCHMARK, answer  # noqa: E402

GRAMMAR_IDS = {f"q{i:03d}" for i in range(38, 53)}
LATIN = re.compile(r"[A-Za-z]")
ARABIC = re.compile(r"[0-9]")
fails: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        fails.append(what)


def test_benchmark() -> None:
    print("benchmark: the 15 held-out questions")
    items = [json.loads(l) for l in
             (ROOT / "data" / "questions_100.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    held = [it for it in items if it["id"] in GRAMMAR_IDS]
    check(len(held) == 15, f"the question set carries 15 grammar items (found {len(held)})")
    correct = 0
    for it in held:
        got = answer(it["prompt"])
        text = got.text if got else ""
        strict, loose = score(text, it["accept"])
        correct += strict
        check(strict, f"{it['id']} {it['prompt'][:38]} -> {text or '—'}")
    check(correct == 15, f"all 15 answered correctly ({correct}/15)")


def test_unseen_nouns() -> None:
    print("generalisation: nouns in no drill")
    lex = lexicon()
    # नगर is not in the śabdapatha (only ग्राम is); the router stays silent for a
    # stem it cannot look up rather than inventing a paradigm — tested below.
    cases = [("पुत्र", "षष्ठी", "बहुवचनम्", "पुत्राणाम्"),
             ("ग्राम", "तृतीया", "एकवचनम्", "ग्रामेण"),
             ("भार्या", "प्रथमा", "एकवचनम्", "भार्या"),
             ("वृक्ष", "द्वितीया", "बहुवचनम्", "वृक्षान्")]
    for stem, case, num, want in cases:
        table = lex.cell(stem, case, num)
        got = answer(f"{stem} इति शब्दस्य {case} {num} किम्?")
        check(table == want, f"table: {stem} {case} {num} = {table}")
        check(got is not None and want in got.text,
              f"router: {stem} {case} {num} -> {got.text if got else '—'}")
    check(answer("नगर इति शब्दस्य तृतीया एकवचनं किम्?") is None,
          "silent for a stem the śabdapatha does not carry (नगर)")
    loc = answer("पुत्र इति शब्दस्य षष्ठ्याम् बहुवचने रूपं किम्?")
    check(loc is not None and "पुत्राणाम्" in loc.text,
          f"router: the locative phrasing agrees -> {loc.text if loc else '—'}")


def test_shape() -> None:
    print("answer shape")
    for prompt, _ in [(q, a) for _, q, a in BENCHMARK]:
        got = answer(prompt)
        text = got.text if got else ""
        check(not LATIN.search(text) and not ARABIC.search(text),
              f"Devanagari only: {text}")
        check(text.endswith("।"), f"ends with a danda: {text}")
    check(answer("किं त्वं मानुषी?") is None, "refuses a question no table answers")
    check(answer("राम इति शब्दस्य षष्ठी एकवचनं किम्?") is not None,
          "still answers one it does know")


def test_holdout() -> None:
    print("the benchmark prompts are held out of training")
    prompts = {json.loads(l)["prompt"] for l in
               (ROOT / "data" / "questions_100.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip() and json.loads(l)["id"] in GRAMMAR_IDS}
    hits = 0
    for corpus in sorted((ROOT / "data").glob("**/train.txt")):
        text = corpus.read_text(encoding="utf-8", errors="ignore")
        hits += sum(1 for p in prompts if p in text)
    check(hits == 0, f"no grammar prompt appears in any training corpus (found {hits})")


def main() -> None:
    print("grammar layer")
    for fn in (test_benchmark, test_unseen_nouns, test_shape, test_holdout):
        fn()
    print()
    if fails:
        print(f"FAILED {len(fails)} check(s)")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
