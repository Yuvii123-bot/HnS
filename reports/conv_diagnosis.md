# Conversational failure: diagnosis and two corrective experiments

**Input under investigation** — `मम अद्य अध्ययनं कर्तुं मनः नास्ति। किं करणीयम्?`
("I have no mind to study today. What should be done?")

**Production model** — `artifacts/model.pt` (p2b), 5,773,056 parameters.
**Date** — 2026-11-14. **Branch** — `arena/31b2a7fb-hns`.

## Verdict

The failure is **not a bug**. Tokenisation, turn boundaries and decoding are
correct; the model reproduces the reported failure class deterministically. The
cause is a **data-coverage gap**: the model has never been trained on a
conversational-advice register, so it emits the highest-probability sentence in
its distribution instead of an answer. Two data-only corrective experiments
removed the specific bad behaviour but **neither raised strict accuracy above
0%**, and both cost main-benchmark accuracy. **Nothing was deployed**;
production remains p2b.

---

## 1. Reproduction

```bash
# the reported failure, greedy (verified: prints the text shown below)
python3 - <<'PY'
import torch
from sanskrit_llm.model import load_checkpoint, stream_tokens
from sanskrit_llm.data import token_feature_table

model, tok, _ = load_checkpoint("artifacts/model.pt")
feat = token_feature_table(tok)
prompt = "मम अद्य अध्ययनं कर्तुं मनः नास्ति। किं करणीयम्?"
ids = torch.tensor([tok.encode(f"<user> {prompt} <assistant>")], dtype=torch.long)
out = []
for tid in stream_tokens(model, feat, ids, max_new_tokens=40, temperature=0.0, eos_id=None):
    s = tok.decode([tid])
    if s.startswith("<"):
        out.append(s); break
    out.append(s)
print("".join(out))
PY

# the benchmark used throughout this report (40 held-out items)
python3 scripts/eval_conversational.py --checkpoint artifacts/model.pt --label before
```

Greedy output (`max_new_tokens=40`):

```
 आम्, तत्र उपासे।  तस्य पूजा कदा क्रियते?  तस्य पूजा सायंकाले क्रियते।
```

The model answers "Yes, I sit there." and then, instead of stopping, fabricates a
new question-answer turn of its own — it overshoots the turn boundary even
greedily, because the input carries it far outside the distribution it was
trained on. At app sampling settings (T=0.85, top-k 40, top-p 0.95, rep 1.15)
the content varies across seeds (`आम्, तत्र अयोध्यायाः राजा। <user> …`,
`आम्, सत्यम्।`, …) but the failure class is identical every time. The user's
exact string did not reproduce; it is one sample from that distribution.

Note `<eos>` is deliberately disabled here (`eos_id=None`) — `stream_tokens`
otherwise swallows the EOS token, which makes boundary measurement impossible.

**Benchmark** — `data/conversational_eval.jsonl`, 40 items over 20 topics
(advice, emotion, meta, unknown-fact). Held out from all training. A leakage
guard (`check_leakage` in `scripts/build_conversational_fix.py`) rejects any
training question sharing ≥85% of a benchmark question's words; it currently
reports **zero** leakage. It caught three items (c34, c36, c37) that were in the
original mix, which is why the baseline was re-measured on the cleaned set —
the contaminated baseline overstated content F1 (6.8% → 4.8%).

---

## 2. Hypotheses, tested

| # | Hypothesis | Measurement | Result |
|---|---|---|---|
| 1 | No advice-register training data | `grep` on `data/instruct_mix2/train.txt` (13,861 rows) for `करणीयम्`, `किं करोमि`, `कुर्याम्`, `करवाणि`, `उपाय` | **0 rows** — confirmed primary cause |
| 2 | Answer space dominated by one template | 355 rows (2.6%) are `किं सत्यम्, <statement>` → `आम्, सत्यम्।`; generated at `scripts/build_instruct_data.py:757-761` (`CONFIRM_SOURCES`) | **Confirmed** — `आम्, सत्यम्।` was the model's most frequent reply (6/40) |
| 3 | No abstention behaviour | 5 unknown-fact items ("कति महासागराः?", "श्वः वर्षा भविष्यति किम्?") | **Confirmed** — confident wrong answers, 0% |
| 4 | Turn boundary / EOS handling broken | 200 held-out single-turn items, greedy, 60-token budget, no stop hacks | **Refuted** — 199/200 stopped on `<eos>` (99.5%), 1 on `<assistant>`, 0 hit the budget |
| 5 | Tokenizer / normalisation drift | `normalize_devanagari(IN) == IN`, decode round-trip exact, 0 `<unk>`; prompt tokens identical to the same prefix inside a training row; prompt ends on `<assistant>` | **Refuted** — byte-identical |
| 6 | Serving runtime differs from training runtime | `serve/runtime/sanskrit_llm/numpy_lm.NumpyLM` vs torch, 2 prompts | **Refuted** — first 8 tokens identical, same strings, same `<eos>` |
| 7 | Pure lookup, no generalisation | exact training prompts vs *held-out paraphrases* of the same facts | **Refuted** — 24/40 (60%) verbatim, **37/40 (92%)** content-correct on paraphrases |

Two further data defects surfaced while measuring:

* **Persona contamination (15 rows).** The mix teaches a human identity:
  `भवतः जन्म कुत्र अभवत्?` → `मम जन्म काश्याम् अभवत्।` and
  `त्वं कः असि?` → `अहं युधिष्ठिरः अस्मि।` A model that believes it is
  Yudhiṣṭhira born in Kāśī has no stable frame for "who are you" questions.
* **531 rows (3.8%)** have a user turn containing no question word at all.

Boundary failure is therefore a *symptom*: the model is out of distribution, so
it produces a wrong answer and then keeps generating. It is not a decoder fault.

---

## 3. Baseline (production p2b)

`reports/conv_before.json` — 40 held-out items, greedy:

| metric | before |
|---|---|
| strict (all reference content words present) | **0.0 %** |
| content F1 | **4.8 %** |
| grammar_ok | 92.5 % |
| boundary_ok | 97.5 % |
| format_ok (Devanagari only) | 100 % |
| answers in imperative/advice register | **0/40** |

Representative answers:

| prompt | reference | before |
|---|---|---|
| मम अद्य अध्ययनं कर्तुं मनः नास्ति। किं करणीयम्? | क्षणं विश्राम्य पुनः अध्ययनं करोतु। | आम्, तत्र उपासे। |
| अहं पठितुं न शक्नोमि। किं कुर्याम्? | अल्पं पठतु, नियमेन पठतु। | आम्, सत्यम्। |
| मम माता दुःखिनी अस्ति। किं करवाणि? | तां सान्त्वयतु, तस्याः सेवां करोतु। | यतः सत्यं ज्ञानम् यथार्थबोधः। |
| भवतः आवासः कुत्र वर्तते? | अहं यन्त्रे वसामि। | मम प्रियं भोजनं कदलीफलम्। |
| किं भवान् मानुषः? | न, अहं मानुषः न अस्मि। अहं भाषामयः। | आम्, अहं कुशली अस्मि। |

---

## 4. Experiment 1 — drop the malformed rows, add hand-written pairs

**Change** (`scripts/build_conversational_fix.py`, output `data/instruct_fix/`):
removed the 355 `किं सत्यम्,` rows; added 127 authored pairs (47 advice,
30 meta/fallback, 10 indirect, 40 phrase-framed copies). New data = **0.9 %** of
the mix. Trained 350 steps from p2b, lr 3e-4, batch 16, ctx 256
(`artifacts/fix/best.pt`, val loss 0.6501, params unchanged).

| metric | before | after 1 | Δ |
|---|---|---|---|
| strict | 0.0 % | 0.0 % | 0 |
| content F1 | 4.8 % | **3.3 %** | −1.5 |
| grammar_ok | 92.5 % | 95.0 % | +2.5 |
| boundary_ok | 97.5 % | 97.5 % | 0 |
| imperative register | 0/40 | 3/40 | +3 |
| **main benchmark composite** (`reports/sem_fix1.json`) | **0.8161** | **0.7875** | **−2.9** |
| main benchmark strict | 0.6285 | 0.5771 | −5.1 |

**Result: rejected.** The `आम्, सत्यम्।` template disappeared — but it was
replaced by *other* unrelated training sentences (`कृषकः क्षेत्रे बीजानि वपति।`,
"the farmer sows seeds", 4/40). 127 rows in 13,633 cannot install a response
mode, and none of them taught the behaviour relevance actually needs.

## 5. Experiment 2 — teach content echo, upweighted to 18 %

**Change** (`scripts/build_conversational_fix2.py`, output `data/instruct_fix2/`):
same row deletion, plus a synthesised family built on one mechanism — *echo the
thing the user mentioned, then say what to do about it*:

```
question: गृहं न सम्यक् वर्तते। किं करोमि?      (mentions गृह)
answer:   गृहं प्रति ध्यानं ददातु।               (echoes गृह)
```

40 case-annotated nouns × 5 question frames × 2 answer frames = 400 pairs,
repeated ×6; hand pairs repeated ×8. New data = **18 %** of characters. The noun
pool is disjoint from the benchmark's topics, so any gain is generalisation.
Trained 450 steps from p2b, lr 2e-4 (`artifacts/fix2/best.pt`, val 0.6549).

| metric | before | after 1 | after 2 |
|---|---|---|---|
| strict | 0.0 % | 0.0 % | **0.0 %** |
| content F1 | 4.8 % | 3.3 % | **8.8 %** |
| grammar_ok | 92.5 % | 95.0 % | 97.5 % |
| boundary_ok | 97.5 % | 97.5 % | **90.0 %** |
| imperative register | 0/40 | 3/40 | **20/40** |
| answer contains a *trained* pool noun | 5/40 | 4/40 | 9/40 |
| main benchmark composite | 0.8161 | 0.7875 | **0.7761** |
| main benchmark strict | 0.6285 | 0.5771 | 0.5514 |
| verse val loss | 4.0436 | — | **3.6587** (ppl 57.0 → 38.8) |

**Result: rejected.** This is the informative one. The model *did* learn the
register — half the answers are now imperatives, grammar rose to 97.5 %, and
verse modelling improved (extra steps on the replay pool). But it learned the
**shape of an answer without the ability to select its content**:

| prompt | reference | after 2 |
|---|---|---|
| मम अद्य अध्ययनं कर्तुं मनः नास्ति। किं करणीयम्? | क्षणं विश्राम्य पुनः अध्ययनं करोतु। | अन्नम् प्रति ध्यानं ददातु। (advice about *food*) |
| मम समयः न अस्ति। किं कुर्याम्? | व्यर्थं कार्यं त्यक्त्वा समयं रक्षतु। | अल्पम् आरभ्य शनैः करोतु। (generic, 3×) |
| अहं क्रुद्धः अस्मि। किं करोमि? | क्षणं मौनं तिष्ठतु, क्रोधं त्यजतु। | अहं शिशुम् उपासे। |

It echoes a noun **from its training pool** rather than the noun it was asked
about, and it newly picked up a question-asking frame from the mix
(`प्रश्नं पृच्छामि — भवान् श्वः आगमिष्यति किम्?`), which is why boundary
accuracy *fell* to 90 %.

---

## 6. Why it still fails

1. **Register is learnable cheaply; content selection is not.** 400 synthetic
   pairs were enough to move 20/40 answers into the imperative register, and
   not enough to move one answer onto the right topic. Choosing *which* of
   40 possible remedies applies to "I don't feel like studying" needs
   topic-level semantics the current data never provides a gradient for.
2. **The failure is not one template.** Deleting `आम्, सत्यम्।` (355 rows)
   moved the collapse to the next most likely sentence. Any patch that removes
   one high-probability reply exposes another; the defect is the absence of
   conditioned behaviour, not the presence of one bad string.
3. **The main benchmark is fragile under continued training.** Two different
   450-step runs both cost 3–4 composite points (−5 to −8 strict). The 13.6k-row
   instruction mix is close to the capacity limit of 5.77 M parameters, so every
   new behaviour is paid for out of the existing ones.
4. **Data defects compound it**: 355 malformed confirmation rows, 531
   question-less user turns, and 15 persona rows teaching a false human identity.

## 7. What would actually be needed

The corrective path is the one already scoped for this phase, not a patch:

* **Data at scale.** 10,000–20,000 genuine Q&A pairs with a real advice
  register, from varied topics — the experiments above show 127 pairs and 400
  synthetic pairs are each an order of magnitude too small.
* **Delete the defects at source**: rewrite `CONFIRM_SOURCES`
  (`scripts/build_instruct_data.py:757`) to emit real yes/no questions with both
  answers, drop the 15 persona rows, and ensure every user turn is a question.
* **Rehearse the old benchmark during any such training** (mix in the phase-2
  evaluation distribution) to stop the −4 composite slide seen twice here.
* **Capacity.** If 10–20k pairs still buy only register and not content, the
  limit is the 5.77 M parameters, and the answer is a larger model — which this
  phase explicitly forbids.

## 8. Deployment status

**Not deployed.** `serve/` and `artifacts/model.pt` are untouched; production
remains p2b. Neither experiment shows a genuine improvement: strict accuracy is
0 % in both, and both lose main-benchmark accuracy. The bundles are left in
`artifacts/fix/` and `artifacts/fix2/` for inspection only.

## 9. Artefacts

| path | contents |
|---|---|
| `data/conversational_eval.jsonl` | 40 held-out items, 20 topics |
| `scripts/eval_conversational.py` | scorer: stop token, strict, content F1, grammar, boundary, format |
| `scripts/build_conversational_fix.py` | experiment 1 data + leakage guard |
| `scripts/build_conversational_fix2.py` | experiment 2 data |
| `reports/conv_before.json`, `conv_after.json`, `conv_after2.json` | per-row results |
| `reports/sem_fix1.json`, `sem_fix2.json` | main benchmark, 1,012 items |
| `reports/cont_fix2.json` | continuation / verse loss |
| `artifacts/fix/`, `artifacts/fix2/` | experiment checkpoints (not deployed) |

## Remaining weaknesses (unchanged in production)

Advice-seeking questions, self/meta questions, and abstention on unknown facts
all still fail in p2b. The model answers confidently with unrelated content
(`मम प्रियं भोजनं कदलीफलम्।` to "where do you live") and cannot say "I don't
know". Conversation quality for *trained* phrasings is good (92 % content on
held-out paraphrases); the gap is entirely in untrained registers.
