# Phase 1 — Step 1: what the model actually knows

**Models compared**

| tag | checkpoint | params |
|---|---|---|
| `p2b` | `artifacts/model.pt` — current production | 5,773,056 |
| `x1` | `artifacts/x1_final/best.pt` — the 10M expansion, before this phase's foundation training | 9,896,544 |

**Benchmark.** `data/phase1_eval.jsonl` — 100 items I wrote for this phase, across
17 areas the brief names. Held out from every training set in the repo; greedy
decoding, 40 tokens, answers must be Devanagari-only. Scored on: `strict`
(every content word of the reference present), `content F1`, `grammar_ok`,
`boundary_ok` (stopped at `<eos>` / turn end), `format_ok` (no Latin).

Reproduce:

```bash
python3 scripts/eval_conversational.py --checkpoint artifacts/model.pt \
    --label p1_p2b --eval-set data/phase1_eval.jsonl
python3 scripts/eval_conversational.py --checkpoint artifacts/x1_final/best.pt \
    --label p1_x1 --eval-set data/phase1_eval.jsonl
```

## Headline

| | p2b (5.77M) | x1 (9.9M) |
|---|---|---|
| strict | **10.0 %** | 4.0 % |
| content F1 | **10.3 %** | 7.1 % |
| grammar_ok | 98.0 % | **100 %** |
| boundary_ok | **100 %** | 96 % |
| format_ok | 100 % | 100 % |

The 10M expansion is **worse than production on knowledge**, exactly as
`reports/scaling_10m.md` predicted for a data-limited regime. Grammar and turn
discipline are fine in both; *content* is the failure.

## Per-area results

| area | n | p2b strict | p2b F1 | x1 strict | x1 F1 | verdict |
|---|---|---|---|---|---|---|
| vocab / word meaning | 8 | **50 %** | 42 % | **50 %** | 46 % | knows reliably |
| śabdarūpa | 5 | **40 %** | 18 % | 0 % | 0 % | partial in p2b |
| vacana | 3 | **33 %** | 17 % | 0 % | 0 % | partial in p2b |
| vibhakti | 7 | **14 %** | 17 % | 0 % | 0 % | inconsistent |
| greeting | 6 | 17 % | 40 % | 0 % | 26 % | conversational register only |
| sandhi | 10 | 10 % | 3 % | 0 % | 0 % | does not understand |
| multi-turn | 3 | 0 % | 28 % | 0 % | 26 % | context leaks, no answer |
| comprehension | 10 | 0 % | 3 % | 0 % | 7 % | does not understand |
| question answering | 5 | 0 % | 12 % | 0 % | 7 % | wrong content |
| dhāturūpa | 13 | **0 %** | 0 % | **0 %** | 0 % | **absent** |
| lakāra | 5 | **0 %** | 0 % | 0 % | 0 % | **absent** |
| liṅga | 3 | **0 %** | 0 % | 0 % | 0 % | **absent** |
| puruṣa | 3 | **0 %** | 0 % | 0 % | 0 % | **absent** |
| samāsa | 6 | **0 %** | 0 % | 0 % | 0 % | **absent** |
| grammatical correction | 6 | **0 %** | 0 % | 0 % | 0 % | **absent** |
| relevance / advice | 4 | **0 %** | 0 % | 0 % | 0 % | **absent** |
| abstention | 3 | **0 %** | 0 % | 0 % | 0 % | **absent** |

## The four questions the brief asks

**1. Known reliably.** Concrete vocabulary with a fixed synonym
(`हस्ती → गजः`, `जलम् → उदकम्`, `सूर्यः → रविः`): 50 % strict, and the failures
are near-misses that answer about a *neighbouring* word. Also reliable: verse
recitation/continuation, Sanskrit sentence *shape* (98–100 % grammatical output),
and turn discipline (100 % `<eos>` in p2b).

**2. Known inconsistently.** Case and number naming, and greetings. The model
knows the *words* (`प्रथमायाम् विभक्तौ`, `द्विवचनम्`) and produces them fluently,
but attaches them to the wrong input:

```
रामः गच्छति — अत्र रामः कस्यां विभक्तौ?
  ref : प्रथमायां विभक्तौ।          (nominative)
  p2b : तत् पदं द्वितीयायाम् विभक्तौ।  (says accusative — wrong, but fluent)
```

**3. Not understood at all.** Zero across dhāturūpa, lakāra, liṅga, puruṣa,
samāsa, correction and relevance. These are not "shaky" — the model has no
mapping from the question to the answer *type*:

```
पठ् धातोः लटि प्रथमपुरुषैकवचनं किम्?     (expected: पठति)
  p2b : आम्, सत्यम्।

अशुद्धं शोधय — बालकः गच्छन्ति।            (expected: बालकाः गच्छन्ति)
  p2b : आम्, सत्यम्।
```

**4. Where responses become incoherent or ungrammatical.** Real comprehension is
where it breaks down — the sentence is grammatical but unrelated to the input:

```
रामः प्रातः उत्थाय पठति। रामः कदा पठति?
  p2b : रात्रौ चन्द्रः तारकाः च दृश्यन्ते।      (correct Sanskrit, unrelated)

सूर्य + उदय = ?
  p2b : सूर्य उदरम् इति स्थाने रामो गच्छति इति भवति।

हस्ती इति पदस्य अर्थः कः?
  p2b : हर्षः इति पदस्य अपरः पर्यायः प्रमोदः।  (answers about हर्ष, not हस्ती)

अहं क्षुधितः अस्मि। किं करोमि?
  p2b : अहं अन्नं खादामि।                    (describes, does not advise)
```

The dominant failure mode across all 17 areas is **surface-pattern completion
without semantic binding**: the response imitates the *form* of the requested
answer (a vibhakti sentence, a synonym sentence) while ignoring *which* word was
asked about. `grammar_ok` stays at 98–100 % throughout, which is precisely why
training loss and grammar checks cannot be used to judge this model.

## Translation: not testable with this tokenizer

The brief asks for Sanskrit→English and English→Sanskrit translation. This model
cannot do either, by construction:

* the vocabulary is 4,000 akṣhara tokens with **no Latin characters**, so English
  cannot be emitted;
* the API **refuses Latin input** by design (`tests/test_api.py::test_latin_is_refused`),
  so English cannot be submitted;
* the brief also requires preserving the akṣhara tokenizer and vocabulary.

I therefore evaluated *meaning explanation in Sanskrit* (`अर्थबोधः`, the vocab
and comprehension areas) instead of literal translation. Real translation needs a
tokenizer change, which is out of scope for this phase — this is a scoping
decision I am flagging rather than silently dropping.

## What this implies for the corpus (Step 2)

Ranked by measured deficit, the corpus must be dominated by:

1. **Morphology drills with verified answers** (dhāturūpa, śabdarūpa, vibhakti,
   liṅga, vacana, puruṣa, lakāra) — 37 of 100 items, currently 0–40 %. These can
   be generated from the repo's own declension engine
   (`scripts/sanskrit_kb/morph.py`, which has `self_check()`) instead of being
   hand-written, so every pair is verifiable.
2. **Comprehension and answer-relevance pairs** — passage + question, where the
   answer must quote the passage's actor/time/place. 21 items at 0–3 %.
3. **Grammatical correction** — inject a known error, output the fix. 6 items at 0 %.
4. **Abstention** — 3 items at 0 %; the model never says "I don't know".
5. **Scripture text** for general fluency — but see §the corpus reality below: it
   is a *fluency* resource, not a fix for the above deficits, because scripture
   is not question-answer shaped.
