# Phase 1 — Steps 2–4: corpus, data quality, training strategy

## 1. The 100 M token target: what is achievable here

The brief asks for **100 million akṣhara tokens**. Here is the arithmetic that
constrains it, measured in this environment rather than assumed:

| quantity | measured value |
|---|---|
| Sanskrit text on disk (`data/raw/gretil` + `data/raw/dcs`) | **163,027,543 chars** in 974 files |
| of which Devanagari | **0** — GRETIL ships IAST romanisation |
| usable Devanagari after transliteration + quality filters | **74,564,467 chars** |
| akṣhara tokens | **35,769,932** (measured with the real tokenizer) |
| previous corpus | 2,930,084 tokens (6.0 M chars) |
| increase | **12.2 ×** |

**100 M tokens cannot be reached from this source.** The entire source is 163 M
chars; even with a 100 % acceptance rate that is ~80 M tokens, and the filters
reject 34 % of lines (page numbers, editorial notation, Latin residue,
duplicate lines). 36 M tokens is 36 % of the target and a 12× improvement over
what the model was trained on.

Two ways to get more were considered and rejected, honestly:

* **Download more scripture.** The sandbox allows outbound access only to
  github.com, codeload.github.com, api.github.com, registry.npmjs.org, pypi.org
  and files.pythonhosted.org. The `fetch_page` tool can read individual pages but
  returns chunked markdown — moving 400 MB of Devanagari through it is not
  viable. I did not attempt to smuggle a corpus through a CDN or a package
  registry.
* **Synthesise the difference.** Generating 60 M more tokens from templates would
  be exactly the "100 million tokens of repetitive text" the brief forbids, and
  would inflate the token count without teaching anything.

I am reporting this as a **shortfall against the brief**, not as success.

## 2. Corpus composition and how it is allocated to the Step-1 deficits

Step 1 (`reports/phase1_step1_eval.md`) found: dhāturūpa 0 %, lakāra 0 %,
liṅga 0 %, puruṣa 0 %, samāsa 0 %, correction 0 %, relevance 0 %, abstention 0 %,
vibhakti 14 %, śabdarūpa 40 %, vacana 33 %, comprehension 0–3 %.

Scripture cannot fix any of that — it is not question-answer shaped. So the mix
is deliberately two-sided:

| component | rows | tokens | purpose |
|---|---|---|---|
| GRETIL scripture (transliterated) | 1,115,923 | ~35.8 M | fluency: sandhi, compounds, vocabulary, verse |
| morphology drills (generated) | 4,453 unique | ~1.1 M after repeat | vibhakti, śabdarūpa, liṅga, vacana |
| verb drills (hand-verified tables) | 351 unique | — | dhāturūpa, lakāra, puruṣa |
| comprehension passages | 1,500 | — | answer must quote the passage |
| grammatical correction | 800 | — | inject one agreement error, output the fix |
| abstention | 8 | — | "एतत् अहं न जानामि" |
| phase-2 replay (`instruct_mix2`) | 13,861 | — | protects the 1,012-item benchmark and verse recitation |
| **total** | **1,174,744** | **37,646,258** | |

Targeted rows are repeated ×6 and interleaved into the scripture stream so that
every context window sees both, rather than the drills being one contiguous
block the model forgets by the next step.

## 3. Data quality: what was removed and what was generated

**Removed or rewritten**

* 355 malformed `<statement>`-as-question rows and 15 false-persona rows
  (`अहं युधिष्ठिरः अस्मि।`) are **not** in this mix; the replay is the phase-2
  file, which those revisions already preceded. The persona topic is handled by
  explicit correction pairs instead.
* **9 of the 100 Step-1 evaluation items were found to exist in the phase-2
  replay set** (glossary and greeting rows). Those were rewritten with words that
  are provably absent from all training text, and the whole set now passes a
  leakage guard (exact match and ≥85 % word overlap): **0 leaks**.
* Generated comprehension sentences were first built from *any* noun as subject,
  producing grammatical nonsense ("the village leads"). Subjects are now
  restricted to a hand-written agent table per verb (`AGENTS`), so each sentence
  is meaningful as well as grammatical.

**Generated, and why that is not fabrication**

Every generated pair is derived from one of two verified tables, never invented:

* **Declension**: `scripts/sanskrit_kb/morph.py` — 61 nouns, 939 forms, and its
  own `self_check()` passes (24/24 reference paradigms, 0 issues). The drills ask
  the engine's own answers.
* **Conjugation**: 13 roots hand-checked for लट् (present) and लङ् (imperfect),
  parasmaipada. I wrote these tables myself and checked them one by one — पच् →
  पचामि/पचसि/पचति…, स्था → तिष्ठामि/तिष्ठसि/तिष्ठति…, नी → नयामि…, चिन्त् →
  चिन्तयामि…, and the corresponding imperfects (अपचम्…, अतिष्ठम्…).

**Held out on purpose.** The verb roots the Step-1 evaluation asks about (पठ्,
गम्, कृ, अस्, भू, दा, पा, श्रु, लिख्, वद्, इष्) are **excluded** from the verb
drills; training uses other roots (पच्, हस्, रक्ष्, त्यज्, वस्, नी, जि, स्था,
चिन्त्, कथ्, गण्, नृत्, क्षिप्). If the eval score moves, it is generalisation of
the *question type*, not recall of the answer.

## 4. Weight transfer from the 5.77 M model (brief §4)

The brief asks how weights are transferred from the original model and how new
parameters are initialised. I must report this plainly:

**No direct weight transfer from the 5.77 M checkpoint is possible.** The 10 M
model changed `d_model` (256 → 288), layer count (6 → 8) and `d_ff` (688 → 880).
Every tensor that touches the residual stream changes shape, and the tied
embedding matrix becomes 4000 × 288 instead of 4000 × 256. A state-dict load
cannot succeed, and silently reshaping a subset of tensors would not preserve
the function the weights encode.

How the 10 M model was therefore produced:

1. **Trained from scratch** on the same data and token budget as the 5.77 M model
   (1,900 pretrain steps + 300 chat + 900 instruct), documented in
   `reports/scaling_10m.md`. All new parameters are standard PyTorch
   initialisation; there is no transfer.
2. **This phase continues from that 10 M checkpoint** (`artifacts/x1_final`), so
   the Sanskrit it already knows is preserved rather than recomputed. That is the
   practical equivalent of the brief's "reinforce existing knowledge".
3. The original 5.77 M checkpoint is untouched: `artifacts/model.pt` and
   `serve/model.npz` remain the production bundle.

If literal transfer is required, the correct technique is function-preserving
widening (net2net): pad the width matrices with zeros and insert identity blocks.
I have **not** done that, because it needs its own validation and the brief also
requires the original checkpoint to stay intact. I flag it as the remaining
option rather than claiming transfer happened.

## 5. Training schedule

| setting | value |
|---|---|
| starting checkpoint | `artifacts/x1_final/best.pt` (9,896,544 params) |
| steps | 4,000 (16.4 M tokens, 43 % of one epoch) |
| batch × ctx | 16 × 256 |
| lr / warmup | 3e-4, 200 steps, cosine decay |
| eval / checkpoint | every 500 steps (8 intermediate checkpoints) |
| throughput | ~2,000 tok/s on 2 CPU threads → ~2.5 h |

Intermediate checkpoints are evaluated rather than waiting for the last step, as
the brief requires. Evaluation at each checkpoint: `data/phase1_eval.jsonl`
(100 items), `scripts/eval_semantic.py` (1,012 items), `eval_continuation.py`
(verse), and the 40-item conversational set.
