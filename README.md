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
| `sanskrit_llm/model.py` | 5.8M-parameter decoder-only transformer: RoPE, RMSNorm, SwiGLU, tied embeddings, KV-cache streaming |
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
| model | 5,773,056 parameters · 6 layers · d=256 · 8 heads · ctx 256 |
| pretraining | 1,900 steps × 16 × 256 ≈ 7.8M tokens, 2 CPU cores, ~42 min |
| chat adaptation | 300 steps on 60,079 `<user>/<assistant>` pairs, ~6 min |
| quality | val loss **3.077** · perplexity **21.7** (per akṣara-token; ≈1.49 bits/character) |

Everything above is reproducible from this repo on a laptop CPU.

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

## Deploying to Vercel

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

Live: see the deployment URL in the Vercel dashboard for this project.
