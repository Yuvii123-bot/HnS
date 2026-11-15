# संस्कृतम् · Saṃskṛta-LLM

A Sanskrit-native language model and chat interface, built from scratch.

Not "a Latin-script LLM with Devanagari glued on": the tokenizer's alphabet is
the **akṣara** (the syllable the Devanagari script writes as one unit), the
model is told each token's **phonological features** (virama, matra, anusvāra,
visarga…), the corpus is transliterated into Devanagari from real Sanskrit
e-texts, and the UI renders everything back as Sanskrit.

```
धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।
  →  धर्म | क्षे | त्रे |   | कु | रु | क्षे | त्रे | …
```

> **संस्कृतम् एव · Sanskrit only.**
> The model speaks Sanskrit and nothing else. It does **not** translate Sanskrit
> into English, and it does **not** accept English (or any romanised) input — the
> API refuses Latin characters rather than silently converting them, and the
> interface is written in Devanagari. Transliteration code exists in this repo
> *only* to build the corpus from romanised scholarly archives (GRETIL, DCS)
> before training; it is never part of the chat path.

---

## What is actually in here

| component | what it does |
|---|---|
| `sanskrit_llm/transliterate.py` | bidirectional **Devanagari ⇄ IAST** transliteration (virama/matra aware, GRETIL & DCS conventions, avagraha, daṇḍas, `oṃ`→ॐ, SLP1 helpers) |
| `sanskrit_llm/akshara.py` | **akṣara segmentation** + 10-dimensional phonological feature vectors |
| `sanskrit_llm/tokenizer.py` | **akṣara-BPE**: merges whole syllables only, never splits a conjunct, never crosses a daṇḍa, exact round-trip |
| `sanskrit_llm/model.py` | decoder-only transformer (5.8M or 9.9M config): RoPE, RMSNorm, SwiGLU, tied embeddings, KV-cache streaming |
| `sanskrit_llm/trainer.py` | CPU-friendly training loop (AdamW, cosine schedule, eval, checkpoints) |
| `api/main.py` | FastAPI backend: streaming chat, tokenizer inspector, transliteration |
| `public/` | chat interface, entirely in Devanagari, with a live akṣara-token visualiser (no build step) |
| `serve/` | deployment bundle: NumPy weights + tokenizer + staged torch-free runtime |
| `api/index.py` | Vercel serverless entrypoint (NumPy inference, no PyTorch) |
| `scripts/` | corpus build → tokenizer → data prep → training → sampling |

### Why the tokenizer is the interesting part

English tokenizers are trained on bytes; applied to Devanagari they cut
syllables in half (`क` + `्` + `ष` + `े`), producing pieces no Sanskrit
speaker recognises. This tokenizer:

1. segments text into **akṣaras** first — a merge can never break a conjunct
   or separate a consonant from its matra;
2. builds its alphabet from the corpus's **akṣara inventory**, not raw code
   points;
3. treats `।` and `॥` (daṇḍa, double daṇḍa) as hard boundaries, so verse
   structure survives tokenisation;
4. decodes exactly — `decode(encode(x)) == x`;
5. achieves **2.07 characters/token** on Devanagari at a 4,000-token vocabulary
   (where each character is ~3 UTF-8 bytes).

The 1,801 learned merges are themselves meaningful Sanskrit: `धर्म`, `क्षे`,
`त्रे`, `श्रीबृहन्नारदीय`, `महाभारत`…

### Why the model knows about akṣaras

Each token carries a feature vector (has virama? has matra? anusvāra? visarga?
candrabindu? vowel-initial? punctuation? digit? spans a word boundary? Latin?).
A linear projection of those features is added to the token embedding, so the
network starts from Sanskrit phonology instead of rediscovering orthography
from scratch.

---

## Numbers (this build)

| | |
|---|---|
| corpus | 6.0M Devanagari characters · 122,642 verses/lines · 974 source e-texts |
| sources | GRETIL (Veda, epic, purāṇa, poetry, śāstra, later) + Digital Corpus of Sanskrit |
| tokenizer | 4,000 tokens · 1,801 merges · 2.07 chars/token |
| model (production) | **9,896,544 parameters** · 8 layers · d=288 · d_ff 880 · 8 heads · ctx 256 |
| also in the repo | 5,773,056 parameters · 6 layers · d=256 · d_ff 688 — `artifacts/model.pt` |
| pretraining | 1,900 steps × 16 × 256 ≈ 7.8M tokens, 2 CPU cores, ~65 min |
| chat adaptation | 300 steps on the chat pairs, ~11 min |
| instruction tuning | 900 steps on `data/instruct_mix2`, ~33 min |
| quality | val loss **3.0632** · perplexity **21.4** (per akṣara-token) |

Everything above is reproducible from this repo on a laptop CPU.

**What is deployed is the 10M model, and it is not the better one.** Measured on
the same held-out sets, the 5.77M checkpoint scores composite **0.8161** against
the 10M model's **0.6842**; the 10M model wins only on verse modelling (val
loss 3.5469 vs 4.0436) and grammar well-formedness (0.9941 vs 0.9921). The swap
was made on explicit instruction, so the trade-off is recorded in the deployment
bundle (`serve/meta.json` → `known_regressions`) and in
[`reports/deploy_10m.md`](reports/deploy_10m.md), which also documents the
rollback. Do not read the numbers above as a quality claim for the live site.

---

## Quickstart

```bash
pip install -r requirements-dev.txt    # torch, fastapi (local training + dev server)
# requirements.txt holds only numpy — the deployed function's runtime

# 1. corpus  (clones GRETIL + DCS, transliterates to Devanagari)
git clone --depth 1 https://github.com/tokushige-koyasan/gretil-corpus data/raw/gretil
git clone --depth 1 https://github.com/cltk/sanskrit_text_dcs data/raw/dcs
python scripts/build_corpus.py --max-chars 6000000 --out data/clean

# 2. tokenizer
python scripts/train_tokenizer.py --vocab-size 4096 --sample 2500000

# 3. data + training
python scripts/prepare_data.py --data data/clean --out data/bin
python scripts/train_model.py --steps 1900 --batch 16 --ctx 256

# 4. chat
python api/main.py          # http://localhost:8000
```

Optional: chat-format adaptation (teaches the `<user>/<assistant>` template):

```bash
python scripts/build_chat_data.py --corpus data/clean/train.txt \
    --val-corpus data/clean/val.txt
python scripts/prepare_data.py --data data/chat --out data/bin_chat
python scripts/train_model.py --resume artifacts/model.pt --bin-dir data/bin_chat \
    --steps 300 --lr 4e-4 --warmup 20 --reset-steps
```

Sample from the command line:

```bash
python scripts/sample.py "धर्मक्षेत्रे कुरुक्षेत्रे" --n 3 --mode continuation
```

---

## API

| endpoint | purpose |
|---|---|
| `GET /api/info` | model config, parameter count, tokenizer stats, training metrics |
| `GET /api/health` | liveness + readiness |
| `POST /api/chat` | streaming generation → newline-delimited JSON events |
| `POST /api/tokenize` | akṣara/token breakdown with features |
| `POST /api/reload` | pick up a new checkpoint without restarting (local dev) |

```bash
curl -s localhost:8000/api/tokenize \
  -H 'content-type: application/json' \
  -d '{"text":"dharmakṣetre kurukṣetre"}'
```

---

## Honest limitations

This is a **research-scale prototype**, not a competitor to a frontier model:

* **5.8M parameters** trained on **~8M tokens** (plus ~1.2M tokens of format
  adaptation). It learns Sanskrit's
  orthography, sandhi-shaped letter sequences, verse rhythm and common
  formulas — it does not reason, translate, or reliably answer questions.
* The chat template was taught with **synthetic pairs** (verse *i* as prompt,
  verse *i+1* as answer). It learns the *format* of a dialogue turn, not
  question answering — there is no large Sanskrit instruction dataset to train
  on, because none exists publicly.
* `continuation` mode is the model doing what it was actually trained to do:
  continuing Sanskrit text in verse style.
* The corpus is editorial e-text (GRETIL, DCS) transliterated by rule. Vedic
  accent marks are stripped, Śrautasūtra-style `word.word.word` notation and
  commentary colons are filtered out, and editorial hyphens inside compounds are
  joined (`दिवे-दिवे` → `दिवेदिवे`).
* Evaluation is next-token loss/perplexity. There is no Sanskrit benchmark for
  chat quality here, and inventing one from these same verses would be circular.

What *is* production-quality here: the tokenizer, the transliterator, the
akṣara feature pipeline, the streaming API and the UI. Those are reusable for a
much larger model — swap `ModelConfig` and point the trainer at more data.

## Repository layout

```
sanskrit_llm/     the library (transliterate · akshara · tokenizer · model · trainer · inference)
api/              FastAPI server
web/              chat interface (vanilla HTML/CSS/JS)
scripts/          reproducible pipeline steps
tests/            13 tests: transliteration, segmentation, tokenizer invariants, model/KV-cache
artifacts/        tokenizer.json, model.pt, metrics
data/             corpus (generated; see scripts)
```

## Data & credit

Sanskrit source texts come from the **GRETIL** archive (Göttingen Register of
Electronic Texts in Indian Languages) via the `tokushige-koyasan/gretil-corpus`
mirror, and the **Digital Corpus of Sanskrit** (Oliver Hellwig) via
`cltk/sanskrit_text_dcs`. Both are scholarly resources with their own terms —
check them before redistributing the derived corpus. Code in this repo is
written for this project.

---

## Instruction tuning

The verse corpus teaches the model how Sanskrit *sounds*; it does not teach it
to answer. The first dialogue data paired consecutive verse lines, so the model
learned the `<user> … <assistant>` template and then produced unrelated verse
inside it. That is the failure this phase fixes.

```bash
python3 scripts/build_instruct_data.py     # curated pairs -> data/instruct/
python3 scripts/mix_instruct_data.py       # rebalance + verse replay -> data/instruct_mix/
python3 scripts/prepare_data.py --data data/instruct_mix --out data/bin_instruct_mix
python3 scripts/train_model.py --bin-dir data/bin_instruct_mix --resume artifacts/model.pt \
        --steps 700 --batch 16 --ctx 256 --lr 6e-4 --warmup 60 --reset-steps \
        --eval-every 50 --ckpt-every 100 --out artifacts/instruct
python3 scripts/eval_instruct.py --checkpoint artifacts/instruct/best.pt --label after
python3 scripts/report_instruct.py         # -> reports/instruction_tuning.md
```

**The dataset** is 2,010 pairs derived from 751 hand-written Sanskrit facts —
greetings and politeness, everyday conversation, grammar (vibhakti, liṅga,
vacana, sandhi), Sanskrit-to-Sanskrit glosses (पर्याय / अर्थ), causal reasoning,
numbers, and Sanātana Dharma vocabulary — plus 420 short multi-turn dialogues.
"Translation" is implemented as Sanskrit→Sanskrit glossing, because the chat
path stays Devanagari-only. Each fact is rendered in up to three natural
phrasings (plain, polite, reported question) so the model learns that the same
question asked differently takes the same answer.

Quality control is in the builder: no Latin characters, no Arabic digits, no
non-Devanagari answers, no verse markers, no duplicates, and no evaluation
prompt may appear as a training turn.
### Measured result (38 held-out prompts, greedy decoding)

| metric | before | after |
|---|---|---|
| instruction val loss | 3.9643 (ppl 52.68) | 0.4077 (ppl 1.5) |
| expected word present | 0.0% | **78.9%** |
| prompt-answer overlap | 0.0% | 53.0% |
| answer ends with punctuation | 2.6% | **100.0%** |
| verse fragments leaking into answers | 5.3% | **0.0%** |
| distinct answers | 65.8% | 100.0% |
| verse/continuation val loss | 3.8366 | 5.0389 (worse — replay limits, not removes, forgetting) |

Keyword coverage is a *coverage* check: an expected content word, or an
inflected form of it, appears in the answer. It does not certify that the
answer is semantically right, and the model is small enough that some answers
are off-topic. Both failure modes are visible in `reports/instruction_tuning.md`
alongside the raw answers.

`reports/instruction_tuning.md` holds the full comparison across every
checkpoint tried; nothing in this repository asserts an improvement that the
harness did not measure.

---

## Phase 2 — semantic accuracy and grammar

Phase 1 taught the model the *shape* of an answer. Phase 2 attacks the content:
a larger curated dataset, grammar pairs computed from a real declension engine,
multi-turn dialogues, and a benchmark that scores meaning, grammar and
instruction-following instead of keyword hits.

```bash
python3 scripts/build_instruct_v2.py      # -> data/instruct_v2 (13,104 pairs + 1,012 eval questions)
python3 scripts/mix_instruct_v2.py        # balance categories + 22% verse replay
python3 scripts/prepare_data.py --data data/instruct_mix2 --out data/bin_instruct_v2
python3 scripts/train_model.py --bin-dir data/bin_instruct_v2 --resume artifacts/model.pt \
        --steps 700 --batch 16 --ctx 256 --lr 6e-4 --warmup 60 --reset-steps \
        --eval-every 50 --ckpt-every 100 --out artifacts/instruct_p2b
python3 scripts/eval_semantic.py     --checkpoint artifacts/model.pt --label p2b_final
python3 scripts/eval_continuation.py --checkpoint artifacts/model.pt --label p2b_final
python3 scripts/report_phase2.py     # -> reports/phase2_evaluation.md
```

### What is in the dataset

| source | pairs | how it is made |
|---|---|---|
| curated facts | 6,000+ | 1,292 facts (greetings, conversation, dharma, world knowledge, Sanskrit glosses, reasoning) grown from phase 1, each asked in five phrasings |
| grammar | 7,084 | **computed** from `scripts/sanskrit_kb/morph.py`: 8 cases × 3 numbers for 61 nouns, both directions, plus sentence cloze |
| dialogues | 213 multi-turn | 156 scripted conversations, each prefix of a dialogue is its own example |
| numbers | 952 | numerals 1–100 and arithmetic, in Devanagari throughout |

Nothing is machine-translated: facts are authored in Sanskrit and grammar
answers are generated from the paradigm, so a wrong answer in the data is a bug
in the engine, and the engine is checked against 24 hand-verified forms.

### Measured on 1,012 held-out questions

| metric | production (v3) | shipped (v2-b) |
|---|---|---|
| strict content match | 37.0% | **62.8%** |
| content F1 | 53.5% | **79.0%** |
| answer type correct | 98.3% | 98.6% |
| passes grammar rules | 98.7% | 99.2% |
| format compliance | 100% | 99.9% |
| **composite** | **67.3%** | **81.6%** |
| verse val loss (continuation) | 4.961 | **4.044** |

Held out means held out: the question frames, the facts and the seven noun
paradigms scored here appear in no training row, enforced by
`tests/test_phase2.py`.

### Where it still fails

* **Unseen facts**: on facts that appear in no training row, both models score
  ~0–6%. The model composes fluent Sanskrit about the wrong thing rather than
  saying it does not know.
* **Paradigm production**: asked to *generate* a form ("गज इति पदस्य तृतीययां
  विभक्तौ किम् रूपम्?"), accuracy is 14%. Recognition of the same forms reaches
  33%; producing them is much harder at this size.
* **Multi-turn context**: 12% on questions answerable only from the preceding
  turns. The API and UI now send history, but the model rarely uses it.
* **Numerals**: digit→word is not reliably learned from ~950 examples.

A composite of 81.6% is a surface score: it rewards the right words in the right
shape, and no model judges meaning here. It is a large, reproducible improvement
over 67.3%, not a claim of fluency.

---

## Deploying to Vercel

`.vercelignore` keeps the function bundle small — Vercel rejected deploys at
676 MB while checkpoints were committed. Only `api/` and `serve/` ship; the
corpus, reports and training checkpoints stay in the repository history.

PyTorch cannot ship to a Vercel function (the wheel is several times the 250 MB
bundle limit), so the deployment does not use it. Instead:

```bash
python scripts/export_numpy.py   # artifacts/model.pt -> serve/model.npz (+ tokenizer, meta)
vercel deploy --prod             # or deploy from the Vercel dashboard
```

`serve/` contains the weights, vocabulary and a staged copy of the torch-free
runtime; `api/index.py` implements the same endpoints with NumPy (`RMSNorm`,
RoPE, SwiGLU, causal attention and a KV cache reimplemented in ~150 lines).
`tests/test_serverless.py` runs the real handler with **torch and fastapi
blocked from importing**, and checks that logits match the PyTorch model to
1.5e-5 — so the deployed model is the trained model, not an approximation.

Live: **https://sanskrit-llm.vercel.app** — the UI at `/`, the model at
`/api/selftest` (one deterministic generation) and `/api/chat`.

The live function serves the **10M** bundle (`serve/model.npz`, 44.2 MB, 60
tensors). `/api/info` reports 9,896,544 parameters, `d_model` 288, `n_layers` 8
and the full `known_regressions` list, so the deployed model states its own
weaknesses. Verified after promotion: `/api/info` params and config match the
exported checkpoint, and `/api/selftest` returns
`" धर्मः सत्याचरणं लोकरक्षणं"` (12 tokens) — byte-identical to running
`do_selftest()` against `serve/` locally, so what is live is what is in git.

Rollback to the 5.77M model is one export and one promote:

```bash
python3 scripts/export_numpy.py --checkpoint artifacts/model.pt   # rebuilds the 25 MB bundle
git commit -am "rollback: serve the 5.77M model" && git push
```

See [`reports/deploy_10m.md`](reports/deploy_10m.md) for the full record,
including the NumPy↔torch parity check and the measured cost of the swap.
