# Phase 1 — status and what the measurements say so far

Last updated after 2,500-step run #2 reached step 250. **Training is still
running**; numbers below are intermediate and will be superseded.

## Where this phase stands

| step | status | artefact |
|---|---|---|
| 1. Evaluate existing knowledge | **done** | `reports/phase1_step1_eval.md`, `data/phase1_eval.jsonl` |
| 2. Build targeted corpus | **done, 36 % of target** | `reports/phase1_step2_corpus.md`, `data/clean_full` |
| 3. Data quality | **done** | leakage guard 0, generated data verified, 9 contaminated eval items replaced |
| 4. Training | **in progress** | `artifacts/ph1b` (run #2), `artifacts/ph1` (run #1, abandoned) |
| 5. Independent 3-way evaluation | **not yet** | needs the finished checkpoint |

## Step 1 in one line

The model is fluent but not grounded: `grammar_ok` 98–100 % and `boundary_ok`
96–100 %, yet strict accuracy is **10.0 % (5.77 M)** and **4.0 % (9.9 M)** on 100
capability questions, with 0 % on dhāturūpa, lakāra, liṅga, puruṣa, samāsa,
correction, relevance and abstention. It completes the *shape* of the requested
answer without binding it to the word that was asked about.

## Corpus: 35.77 M tokens, not 100 M

| | |
|---|---|
| source on disk | 163,027,543 chars (GRETIL + DCS), **all IAST** |
| after transliteration + filters | 74,564,467 Devanagari chars |
| tokens (real tokenizer) | **35,769,932** |
| vs previous corpus | 12.2 × |
| vs brief's target | **36 %** |

The target is unreachable from this source: 163 M chars caps out near 80 M tokens
at a 100 % acceptance rate, and the filters reject 34 %. Sandbox network access
covers only GitHub/PyPI/npm hosts, so no further scripture can be fetched;
synthesising the remainder would be the repetitive padding the brief forbids. I
am reporting this as a shortfall.

## Training run #1 — failed, and the failure is informative

Configuration: 88 % scripture / 3.8 % drills, lr 3e-4, from `x1_final`.

| step | strict | F1 | grammar | vocab | śabdarūpa | vibhakti |
|---|---|---|---|---|---|---|
| start (`x1_final`) | 4.0 % | 7.1 % | 100 % | 46 % | 0 % | 0 % |
| 500 | 0.0 % | 4.5 % | 77 % | — | — | — |
| 1000 | 0.0 % | 4.9 % | 98 % | **0 %** | **0 %** | 10 % |

Flooding the model with scripture **destroyed its instruction behaviour**:
vocabulary fell 42 % → 0 % and the answers show cross-family bleed
(`हस्ती इति पदस्य अर्थः कः?` → `हस् धातोः अन्धकारः।`, `सूर्य + उदय` → `ॐ गण`).
Roughly 3.6 M scripture tokens was enough. Run stopped at step 1000.

This is the same lesson as `reports/scaling_10m.md` from the other direction: at
this data scale the model has a fixed budget of behaviour, and adding one regime
pays for it out of another.

## Training run #2 — rebalanced, in progress

Scripture capped at 15 M chars, drills ×12, replay ×4 → 11.96 M tokens per epoch,
**36 % instruction data**, lr 1.2e-4 (one third of run #1), 2,500 steps,
checkpoint and eval every 250.

At step 250:

| area | start (`x1_final`) | run #2 @250 |
|---|---|---|
| vibhakti | 0 % | **14 %** |
| correction | 0 % | **8 %** |
| greeting | 26 % | **33 %** |
| comprehension | 7 % | 8 % |
| **vocab** | 46 % | **4 %** |
| dhaturupa | 0 % | 0 % |

Overall strict 1.0 %, F1 5.9 %, grammar 98 %, boundary 95 %.

**Diagnosed cause of the vocabulary collapse — a flaw in my own data design.**
The morphology drills ask `X इति पदं कस्यां विभक्तौ?` while the replay's glossary
rows ask `X इति पदस्य पर्यायः कः?`. Those are near-identical carrier templates
with different answer types, so the model slides one response between them:
vocabulary 46 % → 4 % at almost exactly the rate vibhakti rises 0 % → 14 %. The
net is that the model has not learned two tasks, it has moved a response.

The fix is to make the drill template structurally distinct from the glossary
template (e.g. `व्याकरणप्रश्नः — …` prefix, or a different interrogative frame)
rather than merely different in wording, and to raise replay so both mappings are
supervised densely enough to stay separable. That is the next experiment; I am
not going to report the 0 → 14 % vibhakti gain as progress while vocabulary is
being destroyed.

## What is already trustworthy

* The 100-item benchmark, its leakage guard, and the per-area breakdown.
* The corpus figures (measured with the real tokenizer, not estimated from chars).
* The forgetting result of run #1 — reproduced at two checkpoints.
* The negative result on ordering: more scripture first is *worse* than a
  balanced mix at this model size and data volume.

## Not done, and not going to be claimed

* 100 M tokens.
* "Fluent with very few errors". Step 1 shows the ceiling is set by data scale
  and by the tokenizer, not by training duration; 5–10 hours of CPU training on
  a 10 M model does not change that, and I will not declare fluency without the
  Step 5 measurement to back it.
* Sanskrit↔English translation (impossible with a Devanagari-only tokenizer).

## Deployment status (2026-11-14)

On instruction, production was moved from the 5.77M checkpoint to the undamaged
10M checkpoint `artifacts/x1_final/best.pt` — not to either phase-1 run
(`artifacts/ph1`, `artifacts/ph1b`), both of which are damaged by the mechanisms
above and must not be served.

Production now reports, from `/api/info`:

* 9,896,544 parameters, `d_model` 288, `n_layers` 8, `d_ff` 880
* 12,697,600 training tokens seen over 3,100 steps
* its own `known_regressions` list, including the 13.2-point composite drop

This does not complete Phase 1: the 35.8M-token corpus has still never been
trained on successfully, and no capability in the Step 1 benchmark has been
improved except verse modelling and grammar well-formedness. Detail:
`reports/deploy_10m.md`.
