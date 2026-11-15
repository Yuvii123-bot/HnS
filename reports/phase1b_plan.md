# How to train it next — plan for Phase 1b

Written after the 10M deployment, using the Step-1 measurements rather than
intuition. The short version: **more training will not make this model answer
correctly, because most of what it gets wrong is not a language problem.**

## 1. What the measurements actually say

| observation | number | what it rules out |
| --- | --- | --- |
| 71 % more parameters, same data | val 3.0770 → 3.0632 (0.45 %), composite 0.8161 → **0.6842** | "it needs more parameters" |
| grammar well-formed while content is wrong | 98–100 % grammar, 4–10 % strict | "it needs more grammar training" |
| factual recall on unseen facts | 0–6 % | "it needs more facts in the data" |
| paradigm production (generate a form) | 14 % (recognition 33 %) | "it knows the paradigm, just needs prompting" |
| multi-turn questions answerable only from context | 12 % | "it uses the history if you send it" |
| training loss 0.46 vs validation 0.87 | gap | it is memorising rows, not learning a rule |
| vocabulary 46 % → 4 % when drill share rose to 23 % | collapse | the drill *frames* collide, not the drill *content* |

Two separate failure modes are being confused under "it doesn't understand":

* **Language failure** — surface completion without binding, template collision,
  context ignored. Real, and fixable by data design.
* **Storage failure** — asked for one of ~200,000 verb forms, or a fact that
  appears in no training row. At 10M parameters with a 4,000-token vocabulary
  this is *not* fixable by training. It is an architecture problem.

Conflating them is why two training runs were spent and both made the model
worse. The plan below separates them.

## 2. Decide first: is the model allowed to use tools?

A 10M model cannot hold 2,229 roots × 10 lakāras × 3 prayogas × 3 puruṣas × 3
vacanas ≈ **600,000 verb forms**, plus noun paradigms, plus facts. No amount of
interleaved training changes that arithmetic.

So the honest choice:

**Option A — hybrid (recommended).** The LM does language: understanding the
question, choosing the reply shape, conversation, comprehension, phrasing. A
deterministic engine answers anything that has one provably correct answer
(paradigms, arithmetic, known facts), and the LM *renders* it into a sentence.
Randomised router decides; the LM learns to recognise the request type.

**Option B — model-only.** Keep asking the LM to produce forms from its weights.
Then accept a realistic ceiling: frequent roots correct, rare roots wrong, and
no way to tell in advance which — exactly today's behaviour.

Option A is the only one where "answers correctly" is a property we can
demonstrate rather than hope for. It also spends the model's small capacity on
the thing only a model can do.

## 3. The verified engine is already available

Checked in this sandbox, not assumed:

* `vidyut` 0.4.0 (PyPI) ships Ambuda's Aṣṭādhyāyī prakriyā engine — 2,000+ rules.
  Data files (`dhatupatha.tsv`, `sutrapatha.tsv`, …) are MIT-licensed and live in
  the `ambuda-org/vidyut` repo.
* `Data("path").load_dhatu_entries()` → **2,229 roots**.
* `Vyakarana().derive(Pada.Tinanta(dhatu, Prayoga.Kartari, Lakara.Lat, Purusha.Prathama, Vacana.Eka))`
  → √paṭh `पठति` / `पठामि`, √bhū `भवति` / `भवामि` — correct.
* Feeding a bare root string instead of the pāṭha entry gives **wrong** forms
  (√pac → `pati`, not `pacati`); the pāṭha encoding and its it-markers matter.
  Any generated dataset must go through `load_dhatu_entries()`, never hand-typed
  roots, and must force `dhatu_pada` where the root is ubhayapada (√kṛ defaulted
  to ātmanepada `कुरुते`, which is valid but not the expected parasmaipada form).
* Noun paradigms (`Pada.Subanta`) come from the same engine, which also lets us
  replace the 61 hand-entered nouns in `scripts/sanskrit_kb/morph.py`.

This is a *verified* source: a form is either derived by a rule with a
documented prakriyā, or it is not in the dataset. That is a stronger correctness
guarantee than anything a transformer of this size can provide.

## 4. What to train on

Target ≈ 12M tokens per run (roughly what fit in the last two runs), with the
mix shaped by the deficits. Percentages are of tokens.

| block | share | source | why |
| --- | --- | --- | --- |
| attested scripture (interleaved, never staged) | 45 % | `data/clean_full` (35.8M tokens available) | sandhi, compounds, natural phrasing. Run #1 proved 88 % scripture destroys skills; interleave instead of staging |
| instruction + verse replay | 20 % | `data/instruct_mix2` + verse replay | catastrophic-forgetting guard; run #2 showed 0 % replay → vocab collapse |
| comprehension (passage + held-out question) | 10 % | passages from the corpus, questions generated over spans, *not* the passage in the answer | teaches binding: the answer must come from the text, and the text is new |
| morphology drills — **structurally distinct frames** | 10 % | vidyut, root-disjoint from eval | fixes the run-#2 collision: vibhakti drill and glossary replay must not share a carrier sentence |
| multi-turn context pairs | 6 % | authored, contrastive | same final question, different history → different answer. Without the contrast the model learns to ignore history (12 %) |
| abstention / "I do not know" | 5 % | authored, 3 phrasings × many unanswerable topics | currently **0 rows**. The model answers confidently when it should not; this is a data hole, not a capability limit |
| correction / agreement errors | 4 % | generated, one injected error per pair | already built; keep |

Rules for the mix, each one paid for by a failed run:

1. **No carrier reuse across tasks.** If two question types share a sentence
   skeleton, one answer distribution interferes with the other. Enforce it in
   the builder and assert it in a test.
2. **Root-disjoint and stem-disjoint splits.** Train on √paṭh, √bhū, √nī; the
   eval asks √pac, √kṛ, √as. Improvement must be generalisation, not recall —
   this is already how `data/phase1_eval.jsonl` was built.
3. **Every contributed token carries provenance** (source URL, license, rows).
   GRETIL/DCS text, MIT pāṭha data, authored rows, generated-by-rule rows.
4. **Vocabulary coverage filter.** Generated forms whose akṣaras are outside the
   4,000-token vocabulary do not ship to training; they would teach the model
   nothing and waste slots.
5. **Leakage guard stays at 0 exact / 0 at ≥ 85 % overlap** before any run.

## 5. How to train it

* **Base: the 5.77M p2b checkpoint, widened to 10M by net2net.** p2b is the
  best-measured model (composite 0.8161). The deployed 10M was trained from
  scratch and is strictly worse, so there is nothing to inherit from it. Net2net
  keeps its function: pad `d_model`, duplicate layers (interleaved, halved
  residual output), pad `d_ff`, zero-init the new projections. Verify by checking
  that logits match p2b before any steps are taken — a widening that changes the
  output is a bug.
* **One interleaved stage, not scripture-then-skills.** Both failures came from
  staging.
* **lr 1.5e-4, cosine, 60-step warmup, 1,200–1,800 steps**, batch 16 × ctx 256.
* **Select checkpoints by capability composite with per-area floors**, never by
  validation loss. Concretely: evaluate every 150 steps on the 100-item set; keep
  a checkpoint only if no area drops more than 5 points. This single change would
  have rejected both phase-1 runs automatically.
* **Stop rule**: if two consecutive evals regress on vocabulary, stop and fix the
  data. Do not add steps.

## 6. What "understands and answers correctly" means here

Per area, with the plan above, what I expect and what I will not claim:

| area | now | realistic after | how |
| --- | --- | --- | --- |
| dhāturūpa / lakāra / puruṣa | 0 % | **~95 %** | engine answers, LM phrases; verified by construction |
| śabdarūpa / vibhakti / liṅga / vacana | 0–14 % | **~90 %** | vidyut subanta + distinct frames |
| abstention | 0 % | **~80 %** | 5 % abstention rows; the model must learn to say it does not know |
| multi-turn | 12 % | **40–60 %** | contrastive history pairs; still a genuine model skill |
| comprehension (span from new passage) | 7 % | **30–50 %** | held-out passages, span answers |
| unseen facts | 0–6 % | **~0 % without retrieval**, high with it | weights cannot store them at this size |
| open conversation, fluency | weak | **modestly better, not fluent** | — |

The last two rows are the honest part. A 10M-parameter model trained on ~12M
tokens will not become fluent, and I will not report it as fluent. What it can
become is a system that is *correct* on everything checkable, abstains where it
should, and keeps context across a short conversation — which is a defensible
reading of the original goal, and measurable.

## 7. Order of work

1. **Engine + router behind the API** (no training): paradigm requests answered
   by vidyut, LM only phrases. Immediately makes the 0 % areas correct.
2. **Data rebuild** with the table in §4, the frame-collision test, and the
   abstention/multi-turn blocks that do not exist yet.
3. **Net2net widening** of p2b → 10M with a logit-parity check.
4. **One interleaved run**, checkpoint selection by capability composite with
   per-area floors.
5. **Step-5 style independent eval** of p2b vs widened vs trained, on sets that
   were never touched, and report the failures too.

Steps 1–2 do not require a training run and cannot make the model worse; they
are the cheapest way to move the numbers.
