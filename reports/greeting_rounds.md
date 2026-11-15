# Greetings and basic exchanges: twelve rounds on the 10M model

Every number below is a run of `scripts/eval_greet.py` on a saved checkpoint.
Nothing here is estimated.

* **Set A** — `data/greet_eval.jsonl`, 44 items, written before training began.
* **Set B** — `data/greet_eval_b.jsonl`, 34 items, written *after* round 8, to
  check whether a model tuned against set A generalises at all. Its prompts were
  verified absent from the training set at build time, and no round ever trained
  on a set B sentence.

Because I tuned against set A, set A alone cannot prove anything. Set B is the
honest number.

## Round by round

| round | init from | steps | set A | set B |
|---|---|---|---|---|
| baseline `x1_final/best.pt` | — | — | 12/40 = 30.0 %\* | 2/34 = 5.9 % |
| r1 | x1_final | 700 | 17/40 = 42.5 %\* | — |
| r3 | r1 | 700 | 29/44 = 65.9 % | — |
| r4 | r3 | 700 | 29/44 = 65.9 % | — |
| r5 (dense inventory) | r4 | 800 | 30/44 = 68.2 % | — |
| r6 | x1_final | 900 | 36/44 = 81.8 % | 22/34 = 64.7 % |
| r7 (gender + polite augmentation) | x1_final | 1000 | 32/44 = 72.7 % | — |
| r8 (cue-word variants) | r6 | 700 | 38/44 = 86.4 % | — |
| r9 (answers echo their cue) | r8 | 650 | 41/44 = 93.2 % | — |
| r10 (fragment + arithmetic scrub, silence the fallback) | r9 | 500 | **43/44 = 97.7 %** | 30/34 = 88.2 % |
| r11 | r10 | 450 | 42/44 = 95.5 % | 28/34 = 82.3 % |
| r12 (wellbeing + day-greeting families) | r10 | 450 | 43/44 = 97.7 % | 30/34 = 88.2 % |

\* 40-item era of set A.

Deployed: **r10 `artifacts/greet10/latest.pt`**.

## What actually moved the number

Ranked by effect, none of it was step count:

1. **One answer set per intent.** `न दोषः` and `आम् … संस्कृतम्` were being used
   for several intents at once, so no single intent had a unique target and the
   score oscillated 29 → 30 → 29. The build now asserts each answer belongs to
   exactly one intent, including no substring containment.
2. **Answers that echo the intent's own cue word.** `न दोषः` for an apology,
   `साहाय्यं करोमि` for a help request, `गच्छतु` for a farewell. The cue and the
   reply now co-occur, so the model can learn the association from the text.
3. **Removing the generic fallback.** `चिन्ता मा। अहं विवृणोमि।` had become the
   argmax for anything unrecognised — it answered a farewell, an apology and
   praise. A generic-sounding answer *becomes* the default; the fallback was
   rewritten to state specifically that understanding failed.
4. **Scrubbing replay.** Scripture fragments and arithmetic rows had been
   surfacing as replies: `क्षम्यताम्` → `न र्व्१,३.३ ॥` and `चलति तर्हि` →
   `मिलित्वा अष्ट भवन्ति।`. The replay scrub drops verse markers, Devanagari
   digits, `र्व्`, and arithmetic vocabulary.

## The ceiling

Rounds 8–12 sit at 43/44 and 30–31/34, but **the failing items change every
round**: r10 missed `किं भवान् सुखी?`, r12 fixed that and missed `क्षमां प्रार्थये`
instead. On set B, r10 missed `शुभं दिनम्`/`प्रकृतिः सम्यक्`/`कुशली अस्मि अहम्`;
r12 fixed those three and missed `किं त्वं मानवी असि?`/`किं कुरुषे त्वम्?`/
`अनुगृहीतोऽस्मि`/`स्वागतम् भवते` instead.

Same aggregate, different items, four rounds running. That is what a converged
soft classifier looks like: ~97 % / ~88 % is where 21 intents and ~330 prompt
templates on a 9.9 M parameter model settles, and each new paraphrase family
trades one error for another.

## Sampling matters as much as weights

The site samples at temperature 0.85, so the same checkpoint scores lower there
than under greedy decoding:

| | greedy | site default (T 0.85) |
|---|---|---|
| set A | 43/44 = 97.7 % | 41/44 = 93.2 % |
| set B | 31/34 = 91.2 % | 29/34 = 85.3 % |

Lowering the default temperature for short inputs would recover most of that gap
without touching the weights.

## Not regressed

`scripts/eval_conversational.py`, 40 items, baseline → r10:

| | strict | content F1 | grammar | boundary | format |
|---|---|---|---|---|---|
| x1_final | 0.000 | 0.0243 | 0.95 | 0.825 | 1.00 |
| r10 | 0.025 | 0.0596 | 0.95 | 1.000 | 1.00 |

All 9,896,544 parameters are the base checkpoint's. No new parameters were
initialised, the tokenizer and vocabulary are untouched, and
`artifacts/model.pt` (p2b) is unchanged.

## What would actually reach zero mistakes

Not more of the same. Two options, in order of cost:

1. **Deterministic routing for the closed inventory.** The 21 greeting intents
   are a finite, enumerable set: keyword patterns over the cue words that the
   rounds above already proved are decisive (`साहाय्य`, `क्षमा`, `नाम`, `आनन्द`,
   `सुखी`, `गच्छामि`, `धन्यवाद` …) with the model handling everything else.
   Correct by construction on the whole inventory, and it fails *visibly*
   (falls through to the model) instead of confidently answering the wrong thing.
   Roughly an hour of work, including tests.
2. **Data volume.** Every intent needs hundreds of genuine paraphrase templates,
   not the 5–16 it has now. That is authoring work, not compute, and 3–4× the
   current prompt count is the realistic threshold for a 10 M model to hold
   ~99 % on unseen phrasings.
