# Feeding the model a book: Nārāyaṇa's Hitopadeśa

## The book

**Hitopadeśa**, Nārāyaṇa, M. R. Kale's edition (1896/1912). The traditional first
reader of Sanskrit: short prose stories with dialogue in simple sentences, with
moral verses between them. Public domain — Nārāyaṇa is mediaeval, Kale died in
1931, both editions are long out of copyright.

Text: `data/raw/hitopadesha/hitokale_deva_F.txt`, the Devanagari transcription
from `github.com/funderburkjim/hitopadesha-kale` (digitised from the scanned Kale
edition). **The transcription itself carries no explicit licence** — the
underlying text is public domain, the transcription is unlicensed. If this ever
ships as a dataset rather than as weights, ask the transcriber first.

Devanagari matters: the model's tokenizer is Devanagari-only and the API rejects
Latin input, so an IAST edition would have been useless without transliteration.

## Pipeline

```bash
python3 scripts/build_book_data.py --curate 150   # clean, split, derive instructions
python3 scripts/build_book_mix.py --stage2-book-repeat 30 --stage2-greet-repeat 20
python3 scripts/prepare_data.py --data data/book_mix/stage2 --out data/bin_book2
python3 scripts/train_model.py --bin-dir data/bin_book2 --resume <ckpt> --out artifacts/book5 ...
python3 scripts/eval_book.py --checkpoint artifacts/book5/latest.pt --label book5
```

`build_book_data.py` cleans 3,153 lines (drops 105 page numbers, keeps the verse
marker so verses can be told from prose), holds out the last 260 lines entirely,
and derives instruction rows **from the book itself** rather than inventing them:
recitation (`नीतिवचनं वद` → a real verse), narration (`कथां वद` → a real prose
line), completion (`पूर्णं कुरु — <half>` → the other half), and four facts about
the book.

`eval_book.py` scores 34 items in five kinds and reports them separately:
recall of trained lines, recall of held-out lines, recitation, narration, facts.

## Results

| checkpoint | trained on | book probe | greet A | 100 exchanges |
|---|---|---|---|---|
| deployed (`greet10`) | — | **0/25** | 43/44 | 72.0 % |
| book2 (2-stage, 450+450) | whole book, ×6 | 4/34 | 37/44 | — |
| book3 (book-heavy, 1300) | whole book, ×10, all lines as pairs | 4/34 | 37/44 | 57.0 % |
| book4 (curated, 800) | 300 curated lines, 27 % share | 6/34 | 40/44 | — |
| **book5 (fit, 800)** | 300 curated lines, **69 % share** | **22/34** | 40/44 | 54.0 % |
| **book6 (consolidated, 400)** | book 16 % + greet 71 % | **22/34** | **41/44** | 57.0 % |

book6 per-kind: **recall_trained 12/12, recite 4/4, narrate 2/2, facts 4/4,
recall_heldout 0/12.**

So after the right recipe the model does what "feed it the book" should mean: it
finishes lines it was taught, recites real maxims on request, tells a story in the
book's own prose, and states what the book is. What it cannot do is continue a
line it has *not* seen (0/12) — that is memorisation, and held-out lines are by
definition not memorised.

## Why the first three attempts failed

Not data quality — capacity and proportions, and the numbers say which:

* Teacher-forced loss on greeting rows: **NLL 0.053 (ppl 1.05)** — fully fitted.
* Teacher-forced loss on book rows at that time: **NLL 1.47 (ppl 4.36)** — never
  fitted, after 4 epochs over 2,541 lines.

A 9.9 M parameter model holds roughly 400 short prompt→answer mappings
(the greeting inventory is 437 rows and fits solidly). It does not hold 2,541
long lines. At 69 % of the mix the curated 300-line set fits in 800 steps
(loss 3.04 → 0.31); at 27 % of the mix it does not reach that in the same budget.
Teaching formats the evaluation then uses also mattered: the first probe asked for
`पूर्णं कुरु — …` completions that had never appeared in training, so 0/25 was
partly my measurement error, not only the model's.

## The cost, stated plainly

Book knowledge is bought with general conversation: 72 → 54 on the 100-exchange
set when the book dominates, recovering only to 57 after consolidation. Greetings
stay near 41/44. **This is displacement, not forgetting**: 9.9 M parameters cannot
hold both the greeting inventory and a book.

## Recommendation

Do not ship book6 as the production model — it trades away 15 points of everyday
conversation for recitation. For a 10 M parameter model the right way to "know a
book" is a small curated table served next to the model (a quote table and an
intent router), which is exact, costs no capacity, and leaves the deployed
checkpoint untouched at 72/100 and 43/44. That is roughly an hour of work and it
is what I would do next.

If the goal is instead for the *weights* to hold the book, the honest options are
a larger model or far more curated data with a longer schedule — not more epochs
of this one.
