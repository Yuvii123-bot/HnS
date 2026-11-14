# Scaling to ~10M parameters: what it bought, and what it cost

**Question.** Expand the model to almost 10 million parameters, keeping the
architecture and the akṣara tokenizer.

**Answer.** The model was built and trained end to end at **9,896,544
parameters** (up from 5,773,056). At the current data scale it is **worse than
production on every conversational and semantic metric**, so it is **not
deployed**. The measurements show why: this is a **data-limited** regime, not a
capacity-limited one. Details and numbers below.

---

## 1. The model

| | production (p2b) | new (x1) |
|---|---|---|
| parameters | 5,773,056 | **9,896,544** |
| layers | 6 | 8 |
| d_model | 256 | 288 |
| heads / head_dim | 8 / 32 | 8 / 36 |
| d_ff | 688 | 880 |
| vocab (tokenizer) | 4,000 | 4,000 — **unchanged** |
| arch | RMSNorm + RoPE + SwiGLU, tied embeddings | identical |

Config: `--d-model 288 --layers 8 --heads 8 --d-ff 880`. `+71 %` parameters,
`+0.5 %` per-layer width in storage terms is irrelevant — the tokenizer,
`n_features=10`, `max_seq_len=512` and the tied-embedding scheme are all
untouched, so training data, eval harnesses and the tokenizer are shared.

## 2. Training (identical recipe, identical data, identical token budget)

| stage | production | new | tokens |
|---|---|---|---|
| pretrain on `data/clean` | 1,900 steps | 1,900 steps | 7,782,400 each |
| chat adaptation on `data/chat` | 300 steps | 300 steps | 1,228,800 each |
| instruction tuning on `data/instruct_mix2` | 600 steps | 900 steps | 2.46M / 3.69M |
| wall clock (2 CPU threads) | 41.7 + 5.7 + 11.8 min | 64.8 + 10.8 + 33.4 min | — |

Pretraining throughput: 2,035 → 2,030 tok/s (the extra layers cost time in the
instruction stage, where sequences are short: 0.9 min/100 steps → 3.7 min/100).

## 3. Results

### 3.1 Pretraining — a tie

| | val loss | perplexity |
|---|---|---|
| 5.77M @ 1,900 steps | 3.0770 | 21.7 |
| **9.90M @ 1,900 steps** | **3.0632** | **21.4** |

71 % more parameters bought **0.45 %** lower validation loss on the identical
token budget. The 5,000,000-parameter gap is not being used, because the corpus
is 2.93 M tokens — about **0.8 tokens per parameter** for the new model.

### 3.2 Instruction benchmark (`scripts/eval_semantic.py`, 1,012 held-out items)

| run | val loss | composite | strict | content F1 | grammar |
|---|---|---|---|---|---|
| **production p2b** | 0.7174 | **0.8161** | **0.6285** | **0.7899** | 0.9921 |
| x1 — single instruct stage, 600 steps | 0.8562 | 0.7106 | 0.4872 | 0.6082 | 0.9931 |
| x1drop — + dropout 0.1 | 0.8424 | 0.5966 | 0.2648 | 0.4475 | 0.9901 |
| x1final — chat stage + 900 steps | 0.8659 | 0.6842 | 0.4368 | 0.5667 | **0.9941** |

Per-kind detail for the best 9.9M run (`reports/sem_x1final.json`):

| kind | n | p2b composite | x1final composite |
|---|---|---|---|
| paraphrase | 520 | 88 % | 82 % |
| paraphrase-2 | 160 | 84 % | 80 % |
| grammar-unseen-noun | 168 | 66 % | **31 %** |
| unseen-fact | 83 | 54 % | 49 % |
| multi-turn | 24 | 57 % | 54 % |

### 3.3 Everything else

| metric | p2b | x1final | verdict |
|---|---|---|---|
| conversational content F1 (40 held out) | 4.8 % | 2.4 % | worse |
| conversational boundary_ok | 97.5 % | 82.5 % | worse |
| conversational strict | 0.0 % | 0.0 % | unchanged |
| verse val loss / ppl | 4.0436 / 57.0 | **3.5469 / 34.7** | **better** |
| grammar_ok | 99.21 % | **99.41 %** | better |
| serving latency (NumPy runtime) | 0.018 s/token | 0.024 s/token | +33 % |
| bundle size | 26 MB | 43 MB | +65 % |

## 4. Why more parameters made it worse

The evidence points at data volume, not at the architecture:

1. **Pretraining is saturated.** 71 % more parameters → 0.45 % better loss. The
   corpus cannot distinguish these models.
2. **The instruction benchmark is worse in the direction overfitting predicts.**
   On the best 9.9M run, training loss fell to **0.46** while validation sat at
   **0.87** (p2b: 0.54 / 0.7174). The larger model memorises the 13,610-row mix
   harder and generalises worse — which is exactly what
   `grammar-unseen-noun` collapsing from 66 % to 31 % looks like: the answer
   *type* (vibhakti) stops generalising and the model answers with the
   glossary template instead:
   `गजः इति पदं कस्यां विभक्तौ?` → `गजः इति पदस्य पर्यायः गणाः।` ("the synonym
   of गजः is गणाः") instead of `तत् पदं प्रथमायाम् विभक्तौ।`.
3. **Regularisation is not the fix.** dropout 0.1 lowered validation loss
   (0.8562 → 0.8424) but destroyed the strict metric (0.4872 → 0.2648), because
   the benchmark rewards faithful recall of trained answers. With 13.6 k rows
   the model is in a trade-off between memorising and generalising, and 10 M
   parameters sit on the wrong side of it.
4. **More steps do not close the gap.** A second 900-step stage on the final
   dataset moved validation from 0.8562 to a best of 0.8659 — flat, while
   training loss kept falling to 0.46.

## 5. Deployment status

**Not deployed.** Production remains p2b (5,773,056 parameters);
`serve/` and `artifacts/model.pt` are untouched. Standing rule for this phase is
to deploy only on measured improvement, and the best 9.9M run costs **−13.2
composite points**. The only genuine gains are verse modelling (ppl 57.0 → 34.7)
and grammar rules (99.21 % → 99.41 %) — both outweighed.

Checkpoints kept for inspection, not shipped: `artifacts/x1_pretrain/`,
`x1_chat/`, `x1_instruct/`, `x1_drop/`, `x1_final/`.

## 6. What would make 10M parameters actually help

1. **Pretraining data.** 2.93 M tokens is ~0.3 tokens/parameter for this model.
   The corpus needs to grow into the capacity before the parameters do any work.
2. **The 10,000–20,000 instruction pairs** already scoped for this phase. The
   current 13.6 k rows at 600–900 steps overfit; 10 M parameters need the
   larger set to show a gain rather than a regression.
3. **The missing warm start is not the explanation.** The working hypothesis
   after x1 was that the 9.9M model was handicapped by skipping the chat stage
   that p2b had (val 2.7707 for the new model vs 2.7871 for production on
   identical tokens, so the stage itself trained fine). Adding it plus 900
   instruct steps gave **0.6842**, statistically indistinguishable from the
   single-stage **0.7106**. Whatever separates the two models, it is not the
   curriculum; it is how much data each has to fit.

## 7. Code changes made along the way

* `scripts/train_model.py` — added `--dropout` (default 0.0, so existing runs are
  bit-identical). The model already had dropout wired up; it was not reachable
  from the CLI.
* `sanskrit_llm/model.py` — **bug fix**: `F.scaled_dot_product_attention(...,
  dropout_p=self.dropout ...)` makes PyTorch take the math path that materialises
  the full `(B, H, T, T)` attention matrix. With `--dropout 0.1` at batch 16 /
  ctx 256 on this 3 GB box that is an immediate OOM kill (reproduced twice,
  exit 137) and a 2× slowdown even when it survives. Regularisation is already
  applied on the residual branch in `Block` (GPT-2 style), so the inner dropout
  was redundant; it now always passes 0.0. **Inference is unaffected** — dropout
  is a no-op in eval mode, so `serve/` output is bit-identical.

## 8. Reproduction

```bash
python3 scripts/train_model.py --bin-dir data/bin_clean --out artifacts/x1_pretrain \
  --steps 1900 --batch 16 --ctx 256 --threads 2 \
  --d-model 288 --layers 8 --heads 8 --d-ff 880

python3 scripts/train_model.py --bin-dir data/bin_chat --out artifacts/x1_chat \
  --steps 300 --batch 16 --ctx 256 --lr 8e-4 --warmup 100 --threads 2 \
  --d-model 288 --layers 8 --heads 8 --d-ff 880 \
  --resume artifacts/x1_pretrain/best.pt --reset-steps

python3 scripts/train_model.py --bin-dir data/bin_instruct_v2 --out artifacts/x1_final \
  --steps 900 --batch 16 --ctx 256 --lr 6e-4 --warmup 100 --threads 2 \
  --d-model 288 --layers 8 --heads 8 --d-ff 880 \
  --resume artifacts/x1_chat/model.pt --reset-steps

python3 scripts/eval_semantic.py     --checkpoint artifacts/x1_final/best.pt --label x1final
python3 scripts/eval_conversational.py --checkpoint artifacts/x1_final/best.pt --label x1final
python3 scripts/eval_continuation.py --checkpoint artifacts/x1_final/best.pt --label x1final
```
