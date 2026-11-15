# Deploying the 10M model

Status: production bundle swapped from the 5.77M p2b checkpoint to the 10M
`x1_final` checkpoint on the explicit instruction "just deploy this 10M model".

This document records what was deployed, what was verified before deploying, and
what the deployment costs in measured quality. The measurements do **not**
support the swap; they are recorded here and in the live `/api/info` payload so
the regression is visible to anyone using the site.

## What is live

| | before | after |
| --- | --- | --- |
| checkpoint | `artifacts/model.pt` (p2b) | `artifacts/x1_final/best.pt` (x1) |
| parameters | 5,773,056 | **9,896,544** |
| layers / d_model / d_ff | 6 / 256 / 688 | 8 / 288 / 880 |
| tokenizer | akṣara BPE, vocab 4000 | unchanged |
| `serve/model.npz` | 25 MB, 46 tensors | 44.2 MB, 60 tensors |
| NumPy latency | 0.018 s/token | 0.024 s/token |

The tokenizer and vocabulary are unchanged, so `api/index.py`, the chat template
and `public/` needed no code changes.

## Verified before promoting to production

1. **Export.** `scripts/export_numpy.py --checkpoint artifacts/x1_final/best.pt`
   wrote 44.2 MB, 60 tensors, 9,896,544 params.
2. **NumPy == torch.** Greedy decode from the NumPy bundle and from
   `artifacts/x1_final/best.pt` was compared token-for-token on 5 prompts
   (2–6 words each, including a repeated-token prompt). With sampling
   parameters aligned and EOS disabled, all 5 matched **exactly**.
   - Two earlier comparison attempts reported mismatches. Both were harness
     errors, not runtime defects: `stream_tokens` defaults to
     `repetition_penalty=1.1` while `NumpyLM.sample` defaults to `1.0`, and
     `stream_tokens(..., eos_id=...)` swallows the EOS instead of emitting it.
     With both at 1.0 the outputs agree exactly.
3. **Test suites.** 5/5 green: `test_sanskrit` 13/13, `test_api` 7/7,
   `test_serverless` (real handler with torch and fastapi blocked from
   importing), `test_instruct`, `test_phase2`.
4. **Build.** The Vercel preview for commit `c15c62d` reached READY in 50 s
   (44 MB bundle inside `includeFiles: serve/**`, function memory 1024 MB,
   maxDuration 60 s).

## What the deployment costs

| measure | p2b (before) | x1 (after) | direction |
| --- | --- | --- | --- |
| semantic composite (1012 items) | 0.8161 | 0.6842 | **−13.2 pts** |
| strict | 0.6285 | 0.4368 | −19.2 pts |
| content F1 | 0.7899 | 0.5667 | −22.3 pts |
| conversational F1 (40 items) | 0.0476 | 0.0243 | −2.3 pts |
| conversational turn boundary | 0.975 | 0.825 | −15 pts |
| capability benchmark strict (100) | 0.10 | 0.04 | −6 pts |
| grammar well-formedness | 0.9921 | 0.9941 | +0.2 pts |
| verse validation loss | 4.0436 | 3.5469 | better (ppl 57.0 → 34.7) |

Zero-correctness areas in both models: dhāturūpa, lakāra, liṅga, puruṣa,
samāsa, correction, relevance, abstention. The 10M model does not fix any of
them; on the capability benchmark it is worse because it loses vocabulary recall
(46% → 4% after the phase-1 corpus runs, see `reports/phase1_status.md`).

The deployment-visible summary of this trade-off is embedded in the bundle:
`serve/meta.json` carries `known_regressions`, `known_gains` and a
`deployment_note` explaining that the swap was an instruction, not a measured
improvement.

## Rollback

`artifacts/model.pt` (p2b) is untouched. To restore the previous production
model:

```bash
python3 scripts/export_numpy.py --checkpoint artifacts/model.pt   # 25 MB bundle
git commit -am "rollback: serve p2b" && git push
```

That push builds a preview; promoting it to production returns the site to
composite 0.8161. The 5.77M bundle is regenerable in about 40 s from the
checkpoint, which is why it was not kept as a 25 MB duplicate in the repo.

## Phase 1 remains unfinished

This deployment is not the end of the Phase 1 work. The two phase-1 training
runs were stopped because both damaged the model (run #1 catastrophic
forgetting, run #2 template interference between the vibhakti drill and the
replay glossary — see `reports/phase1_status.md`). The 35.8M-token corpus built
for Phase 1 has not been trained on successfully. Deploying the 10M model
changes the parameter count on the site, not the capabilities being asked for.
