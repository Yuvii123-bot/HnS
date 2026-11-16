# Run 250 — live report

_updated 2026-11-16T01:25:05Z · written by `scripts/watchdog.py` every 15 min_

| field | value |
|---|---|
| step | **225 / 61,036** (0.4 %) |
| tokens | 921,600 / 250,000,000 |
| train loss | 2.8116 |
| val loss (recent) | — |
| speed | 2763 tok/s |
| ETA | 25.0 h |
| restarts | 0 |
| free disk | 8.15 GB |

Weights: `/home/user/HnS/artifacts/run250/latest.pt` (resumable, weights+Adam+step),
`/home/user/HnS/artifacts/run250/model.pt` (weights only), `/home/user/HnS/artifacts/run250/best.pt` (best val).
Health history: `reports/run250_health.jsonl`. Alerts: `reports/run250_alerts.log`.

Next automatic stage: `scripts/post_run_250.py` — 500 basic questions; if fewer
than 400 are correct the run extends to 350 M tokens (85,449 steps).
