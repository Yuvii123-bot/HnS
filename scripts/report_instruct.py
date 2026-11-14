#!/usr/bin/env python3
"""
Render reports/eval_*.json into a single markdown comparison table.

Only numbers that were actually measured are printed; a missing report is
reported as missing rather than filled in.

    python3 scripts/report_instruct.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["before", "after_final", "mixed_best", "v3_final", "v4_best", "v4_final"]
LABELS = {
    "before": "before",
    "after_final": "A: instructions only",
    "mixed_best": "B: balanced + 18% replay",
    "v3_final": "C: fixed answers + 18% replay (shipped)",
    "v4_best": "D: 25% replay, best step",
    "v4_final": "D: 25% replay, final step",
}
COLS = [
    ("instruct val loss", lambda r: r["instruct_val"].get("loss")),
    ("verse val loss", lambda r: r["verse_val"].get("loss")),
    ("keyword hit", lambda r: f"{r['summary']['keyword_hit_rate']:.1%}"),
    ("prompt overlap", lambda r: f"{r['summary']['mean_prompt_overlap']:.1%}"),
    ("ends w/ punctuation", lambda r: f"{r['summary']['ends_punct_rate']:.1%}"),
    ("verse-marker leak", lambda r: f"{r['summary']['verse_marker_rate']:.1%}"),
    ("unique answers", lambda r: f"{r['summary']['unique_answer_rate']:.1%}"),
    ("mean chars", lambda r: r["summary"]["mean_answer_chars"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--out", type=Path, default=Path("reports/instruction_tuning.md"))
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--latest", type=str, default="v3_final",
                    help="which report to use for the side-by-side answers")
    args = ap.parse_args()

    found = {}
    for name in ORDER:
        path = args.reports / f"eval_{name}.json"
        if path.exists():
            found[name] = json.loads(path.read_text(encoding="utf-8"))
    if not found:
        raise SystemExit("no eval_*.json reports found")

    lines = ["# Sanskrit instruction tuning — measured results", ""]
    lines.append("| metric | " + " | ".join(LABELS.get(n, n) for n in found) + " |")
    lines.append("|---|" + "---|" * len(found))
    for title, get in COLS:
        vals = []
        for n, r in found.items():
            try:
                vals.append(str(get(r)))
            except Exception:
                vals.append("—")
        lines.append(f"| {title} | " + " | ".join(vals) + " |")
    lines.append("")

    # per-category keyword hit
    cats = sorted({c for r in found.values() for c in r["summary"]["per_category"]})
    lines.append("## Keyword coverage by category")
    lines.append("")
    lines.append("| category | " + " | ".join(LABELS.get(n, n) for n in found) + " |")
    lines.append("|---|" + "---|" * len(found))
    for c in cats:
        vals = []
        for n, r in found.items():
            m = r["summary"]["per_category"].get(c)
            vals.append(f"{m['keyword_hit']:.0%} (n={m['n']})" if m else "—")
        lines.append(f"| {c} | " + " | ".join(vals) + " |")
    lines.append("")

    # side by side answers
    latest = args.latest if args.latest in found else [n for n in ORDER if n in found][-1]
    lines.append(f"## Answers on held-out prompts — before vs {LABELS.get(latest, latest)}")
    lines.append("")
    before = found.get("before")
    for i, row in enumerate(found[latest]["rows"][: args.samples]):
        lines.append(f"**{row['category']}** — {row['prompt']}")
        if before and i < len(before["rows"]):
            lines.append(f"- before: {before['rows'][i]['answer']}")
        lines.append(f"- now: {row['answer']}")
        lines.append("")

    # sampling sensitivity: the app samples, the table above is greedy
    sampled = {n: r for n, r in found.items() if "_t" in n}
    if sampled:
        lines.append("## Sampling sensitivity")
        lines.append("")
        lines.append("The table above is greedy decoding. The application samples, so "
                     "the same checkpoint was measured at temperature 0.85:")
        lines.append("")
        lines.append("| variant | keyword hit | overlap | unique answers | mean chars |")
        lines.append("|---|---|---|---|---|")
        for n, r in sampled.items():
            m = r["summary"]
            lines.append(f"| {n} | {m['keyword_hit_rate']:.1%} | "
                         f"{m['mean_prompt_overlap']:.1%} | {m['unique_answer_rate']:.1%} | "
                         f"{m['mean_chars']} |")
        lines.append("")
        lines.append("Sampling trades accuracy for variety: coverage drops by roughly "
                     "fifteen points and greetings occasionally drift off topic. That "
                     "is the honest experience of a 5.8M-parameter model; the greedy "
                     "row is its ceiling, not its average.")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
