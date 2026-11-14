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

ORDER = ["before", "after_best", "after_final", "mixed_best", "mixed_final"]
LABELS = {
    "before": "before (verse-trained)",
    "after_best": "instructions only, best step",
    "after_final": "instructions only, final step",
    "mixed_best": "balanced + verse replay, best step",
    "mixed_final": "balanced + verse replay, final step",
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
    latest = [n for n in ORDER if n in found][-1]
    lines.append(f"## Answers on held-out prompts — before vs {LABELS.get(latest, latest)}")
    lines.append("")
    before = found.get("before")
    for i, row in enumerate(found[latest]["rows"][: args.samples]):
        lines.append(f"**{row['category']}** — {row['prompt']}")
        if before and i < len(before["rows"]):
            lines.append(f"- before: {before['rows'][i]['answer']}")
        lines.append(f"- now: {row['answer']}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
