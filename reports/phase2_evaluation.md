# Phase 2 — semantic accuracy and grammatical correctness

Measured on held-out questions only: the question frames, the facts and
the noun paradigms used here never appear in training
(checked by `tests/test_phase2.py`).

## Headline

| metric | production (v3) | v2 dataset (a) | v2 dataset (b), shipped | v2 dataset (c), rejected |
|---|---|---|---|---|
| strict ok | 37.0% | 63.4% | 62.8% | 52.7% |
| content f1 | 53.5% | 78.2% | 79.0% | 69.3% |
| type ok | 98.3% | 99.7% | 98.6% | 95.2% |
| grammar ok | 98.7% | 99.5% | 99.2% | 97.5% |
| format ok | 100.0% | 100.0% | 99.9% | 100.0% |
| composite | **67.3%** | **82.0%** | **81.6%** | **75.2%** |

n = 1012 questions per model.

`strict` asks whether every distinctive term of the reference answer is
present; `content f1` is the softer stemmed overlap. Both are surface
measures — no model judges meaning here, so a wrong answer built from the
right words still counts. The composite weights are stated in
`scripts/eval_semantic.py`: strict 0.40, content 0.15, type 0.15,
grammar 0.20, format 0.10.

## Where the difference is

| question kind | production (v3) | v2 dataset (a) | v2 dataset (b), shipped | v2 dataset (c), rejected |
|---|---|---|---|---|
| arithmetic | 0% (n=5) | 0% (n=5) | 0% (n=5) | 20% (n=5) |
| grammar-production | 0% (n=42) | 7% (n=42) | 14% (n=42) | 14% (n=42) |
| grammar-unseen-noun | 20% (n=168) | 36% (n=168) | 33% (n=168) | 11% (n=168) |
| multi-turn | 0% (n=24) | 12% (n=24) | 12% (n=24) | 8% (n=24) |
| number-word | 0% (n=10) | 0% (n=10) | 0% (n=10) | 0% (n=10) |
| paraphrase | 46% (n=520) | 85% (n=520) | 84% (n=520) | 74% (n=520) |
| paraphrase-2 | 41% (n=160) | 78% (n=160) | 75% (n=160) | 69% (n=160) |
| unseen-fact | 41% (n=83) | 14% (n=83) | 18% (n=83) | 12% (n=83) |

Cell = strict accuracy on that kind of question.

## Unseen facts, split by provenance

Some held-out items are facts the production model did train on in
phase 1; the rest are new facts that neither model ever saw. Mixing
them would flatter one model or the other.

| provenance | n | production (v3) | v2 dataset (a) | v2 dataset (b), shipped | v2 dataset (c), rejected |
|---|---|---|---|---|---|
| baseline trained on this fact | 48 | 71% | 21% | 27% | 19% |
| neither model saw this fact | 35 | 0% | 6% | 6% | 3% |

## Grammatical correctness

| | production (v3) | v2 dataset (a) | v2 dataset (b), shipped | v2 dataset (c), rejected |
|---|---|---|---|---|
| answers passing all rule checks | 98.7% | 99.5% | 99.2% | 97.5% |
| failures: numeral | 11 | 4 | 6 | 4 |
| failures: person | 1 | 0 | 0 | 0 |
| failures: termination | 0 | 0 | 1 | 0 |
| failures: vibhakti | 1 | 1 | 1 | 21 |

The checker (`scripts/grammar_check.py`) verifies script, termination,
pronoun/copula person, adjective-noun gender where both are known,
case labels and numeral agreement. It is a rule set, not a parser:
passing it is necessary, not sufficient.

## Continuation (the verse ability)

| | production (v3) | v2 dataset (a) | v2 dataset (b), shipped | v2 dataset (c), rejected |
|---|---|---|---|---|
| verse val loss | 4.9608 | 4.0599 | 4.0436 | 3.9312 |
| mean continuation chars | 34.0 | 21.0 | 23.7 | 24.5 |

## Answers — production vs v2 dataset (b), shipped

**dharma / unseen-fact** — अध्यापकः पृच्छति — मोक्षः कः?
- reference: मोक्षः जन्ममरणप्रबन्धात् मुक्तिः।
- production: आरामः इत्यस्य पर्यायः उद्यानम्।
- v2 dataset (b), shipped: प्रसादः देवाय निवेदितम् अन्नम्।

**dharma / unseen-fact** — शिष्यः पृच्छति — वेदाः के?
- reference: वेदाः चत्वारः — ऋग्वेदः, यजुर्वेदः, सामवेदः, अथर्ववेदः।
- production: नि, गौः इत्यस्य अर्थः कारुः।
- v2 dataset (b), shipped: दशवादनम्।

**dharma / unseen-fact** — अध्यापकः पृच्छति — संसारः कः?
- reference: संसारः जन्ममरणस्य चक्रम्।
- production: अहं ध्यानं यत् रोगः।
- v2 dataset (b), shipped: संसारः जन्ममरणस्य चक्रम्।

**dharma / unseen-fact** — शिष्यः पृच्छति — अज्ञानम् किम्?
- reference: अज्ञानम् अविद्या।
- production: अज्ञानम् अविद्या।
- v2 dataset (b), shipped: अविद्या अज्ञानम्।

**dharma / unseen-fact** — अध्यापकः पृच्छति — ध्यानम् किम्?
- reference: ध्यानं मनसः एकाग्रता।
- production: ध्यानं मनसः एकाग्रता।
- v2 dataset (b), shipped: ध्यानं मनसः एकाग्रता।

**glossary / unseen-fact** — शिष्यः पृच्छति — वायुः इत्यस्य पर्यायः कः?
- reference: वायुः इत्यस्य पर्यायः पवनः।
- production: वायुः इत्यस्य पर्यायः पवनः।
- v2 dataset (b), shipped: वायुः इत्यस्य पर्यायः पवनः।

**glossary / unseen-fact** — अध्यापकः पृच्छति — राजा इति पदस्य अपरः पर्यायः कः?
- reference: राजा इति पदस्य अपरः पर्यायः भूपतिः।
- production: राजा इति पदस्य अपरः पर्यायः भूपतिः।
- v2 dataset (b), shipped: राजा इति पदस्य अपरः पर्यायः भूपतिः।

**glossary / unseen-fact** — शिष्यः पृच्छति — पुत्रः इति पदस्य अपरः पर्यायः कः?
- reference: पुत्रः इति पदस्य अपरः पर्यायः तनयः।
- production: पुत्रः इति पदस्य अपरः पर्यायः तनयः।
- v2 dataset (b), shipped: पुत्रः इति पदस्य अपरः पर्यायः नरः।
