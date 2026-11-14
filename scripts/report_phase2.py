#!/usr/bin/env python3
"""
Render the phase-2 measurements into reports/phase2_evaluation.md.

Only numbers that exist on disk are printed. If a candidate has no report, it
is called missing rather than guessed.

    python3 scripts/report_phase2.py --candidates baseline_v3,p2_best,p2_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "baseline_v3": "production (v3)",
    "p2_best": "v2 dataset, best step",
    "p2_final": "v2 dataset (a)",
    "p2b_final": "v2 dataset (b), shipped",
    "p2c_final": "v2 dataset (c), rejected",
}
METRICS = ["strict_ok", "content_f1", "type_ok", "grammar_ok", "format_ok", "composite"]


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="baseline_v3,p2_final,p2b_final,p2c_final")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "phase2_evaluation.md")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--show", default="p2b_final", help="model to quote answers from")
    args = ap.parse_args()

    labels = [c.strip() for c in args.candidates.split(",") if c.strip()]
    sem = {l: load(ROOT / "reports" / f"sem_{l}.json") for l in labels}
    cont = {l: load(ROOT / "reports" / f"cont_{l}.json") for l in labels}
    present = [l for l in labels if sem[l]]

    lines: list[str] = []
    lines.append("# Phase 2 — semantic accuracy and grammatical correctness")
    lines.append("")
    lines.append("Measured on held-out questions only: the question frames, the facts and")
    lines.append("the noun paradigms used here never appear in training")
    lines.append("(checked by `tests/test_phase2.py`).")
    lines.append("")

    # ---- headline
    lines.append("## Headline")
    lines.append("")
    lines.append("| metric | " + " | ".join(LABELS.get(l, l) for l in present) + " |")
    lines.append("|---|" + "---|" * len(present))
    for metric in METRICS:
        cells = []
        for l in present:
            v = sem[l]["summary"][metric]
            cells.append(f"**{v:.1%}**" if metric == "composite" else f"{v:.1%}")
        lines.append(f"| {metric.replace('_', ' ')} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"n = {sem[present[0]]['n_items']} questions per model.")
    lines.append("")
    lines.append("`strict` asks whether every distinctive term of the reference answer is")
    lines.append("present; `content f1` is the softer stemmed overlap. Both are surface")
    lines.append("measures — no model judges meaning here, so a wrong answer built from the")
    lines.append("right words still counts. The composite weights are stated in")
    lines.append("`scripts/eval_semantic.py`: strict 0.40, content 0.15, type 0.15,")
    lines.append("grammar 0.20, format 0.10.")
    lines.append("")

    # ---- by kind
    lines.append("## Where the difference is")
    lines.append("")
    kinds = sorted({k for l in present for k in sem[l]["by_kind"]})
    header = " | ".join(LABELS.get(l, l) for l in present)
    lines.append(f"| question kind | {header} |")
    lines.append("|---|" + "---|" * len(present))
    for kind in kinds:
        cells = []
        for l in present:
            m = sem[l]["by_kind"].get(kind)
            cells.append(f"{m['strict_ok']:.0%} (n={m['n']})" if m else "—")
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cell = strict accuracy on that kind of question.")
    lines.append("")

    # ---- unseen facts, split by provenance
    phase1_file = ROOT / "data" / "instruct_mix" / "train.txt"
    if phase1_file.exists() and "baseline_v3" in present:
        seen = phase1_file.read_text(encoding="utf-8")
        unknown = [r for r in sem["baseline_v3"]["rows"] if r["kind"] == "unseen-fact"]
        split = {"baseline trained on this fact": [], "neither model saw this fact": []}
        for r in unknown:
            key = "baseline trained on this fact" if r["reference"] in seen \
                else "neither model saw this fact"
            split[key].append(r["id"])
        if any(split.values()):
            lines.append("## Unseen facts, split by provenance")
            lines.append("")
            lines.append("Some held-out items are facts the production model did train on in")
            lines.append("phase 1; the rest are new facts that neither model ever saw. Mixing")
            lines.append("them would flatter one model or the other.")
            lines.append("")
            lines.append("| provenance | n | " + " | ".join(LABELS.get(l, l) for l in present) + " |")
            lines.append("|---|---|" + "---|" * len(present))
            for key, ids in split.items():
                if not ids:
                    continue
                cells = []
                for l in present:
                    sub = [r for r in sem[l]["rows"] if r["id"] in set(ids)]
                    ok = sum(r["strict_ok"] for r in sub) / max(len(sub), 1)
                    cells.append(f"{ok:.0%}")
                lines.append(f"| {key} | {len(ids)} | " + " | ".join(cells) + " |")
            lines.append("")

    # ---- grammar
    lines.append("## Grammatical correctness")
    lines.append("")
    lines.append("| | " + " | ".join(LABELS.get(l, l) for l in present) + " |")
    lines.append("|---|" + "---|" * len(present))
    lines.append("| answers passing all rule checks | " +
                 " | ".join(f"{sem[l]['summary']['grammar_ok']:.1%}" for l in present) + " |")
    rules = sorted({r for l in present for r in sem[l]["grammar_rule_failures"]})
    for rule in rules:
        cells = [str(sem[l]["grammar_rule_failures"].get(rule, 0)) for l in present]
        lines.append(f"| failures: {rule} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("The checker (`scripts/grammar_check.py`) verifies script, termination,")
    lines.append("pronoun/copula person, adjective-noun gender where both are known,")
    lines.append("case labels and numeral agreement. It is a rule set, not a parser:")
    lines.append("passing it is necessary, not sufficient.")
    lines.append("")

    # ---- continuation
    if any(cont[l] for l in present):
        lines.append("## Continuation (the verse ability)")
        lines.append("")
        lines.append("| | " + " | ".join(LABELS.get(l, l) for l in present) + " |")
        lines.append("|---|" + "---|" * len(present))
        lines.append("| verse val loss | " + " | ".join(
            f"{cont[l]['verse_val_loss']}" if cont[l] else "—" for l in present) + " |")
        lines.append("| mean continuation chars | " + " | ".join(
            f"{cont[l]['mean_continuation_chars']}" if cont[l] else "—" for l in present) + " |")
        lines.append("")

    # ---- samples
    best = args.show if args.show in present else (present[-1] if present else None)
    if best and "baseline_v3" in present:
        lines.append(f"## Answers — production vs {LABELS.get(best, best)}")
        lines.append("")
        base_rows = sem["baseline_v3"]["rows"]
        cand_rows = sem[best]["rows"]
        by_id = {r["id"]: r for r in cand_rows}
        shown = 0
        for b in base_rows:
            c = by_id.get(b["id"])
            if not c or shown >= args.samples:
                continue
            lines.append(f"**{b['category']} / {b['kind']}** — {b['prompt']}")
            if b["reference"]:
                lines.append(f"- reference: {b['reference']}")
            lines.append(f"- production: {b['answer']}")
            lines.append(f"- {LABELS.get(best, best)}: {c['answer']}")
            lines.append("")
            shown += 1

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
