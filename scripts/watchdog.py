#!/usr/bin/env python3
"""Watchdog: health log + self-healing for the unattended run.

Runs detached, one sample every 15 minutes, and appends a JSON line to
``reports/run250_health.jsonl``. Beyond logging it can act:

* supervisor dead and the budget not reached  -> relaunch the supervisor
* free disk below 1.5 GB                      -> delete leftover text parts,
                                                quarantine corrupt checkpoints
* ``nan``/``inf`` loss in the recent log      -> quarantine ``latest.pt`` and
                                                resume from ``best.pt`` so the
                                                run continues from good weights
* no checkpoint progress for a long time      -> note it (does not kill: slow is
                                                not the same as stuck)

Every hour it rewrites ``reports/run250_REPORT.md`` — step, tokens, loss trend,
ETA, val history — so the run is readable without reading the log.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "run250"
REPORTS = ROOT / "reports"
HEALTH = REPORTS / "run250_health.jsonl"
ALERTS = REPORTS / "run250_alerts.log"
REPORT = REPORTS / "run250_REPORT.md"
LOG = REPORTS / "run250.log"
STEP_TARGET = 61036
TOKENS_PER_STEP = 4096
TOKENS_TARGET = 250_000_000
INTERVAL = 900


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def alert(msg: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    line = f"[{now()}] {msg}\n"
    with ALERTS.open("a") as fh:
        fh.write(line)
    print(line, end="", flush=True)


def alive(pattern: str) -> list[int]:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
    except OSError:
        return []
    return [int(p) for p in out.split() if p.strip()]


def free_gb() -> float:
    st = os.statvfs(ROOT)
    return st.f_bavail * st.f_frsize / 1e9


def mem_available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def tail_lines(n: int = 40) -> list[str]:
    if not LOG.exists():
        return []
    with LOG.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 200_000))
        return fh.read().decode("utf-8", "replace").splitlines()[-n:]


def parse_log(lines: list[str]) -> dict:
    """Last step/loss/tok-s and the whole val history seen in the tail."""
    info: dict = {}
    for line in reversed(lines):
        m = re.search(r"step\s+(\d+)/\s*(\d+)\s+loss\s+([\d.]+).*?([\d.]+) tok/s", line)
        if m and "step" not in info:
            info["step"] = int(m.group(1))
            info["loss"] = float(m.group(3))
            info["tok_per_s"] = float(m.group(4))
        v = re.search(r">> eval @ (\d+): val loss ([\d.]+)\s+ppl ([\d.]+)", line)
        if v:
            info.setdefault("val", []).append(
                {"step": int(v.group(1)), "val_loss": float(v.group(2)), "ppl": float(v.group(3))})
    for line in reversed(lines):
        if "Traceback" in line or "Error" in line:
            info["error_line"] = line.strip()[:300]
            break
    return info


def restart_supervisor(reason: str) -> None:
    alert(f"ACTION relaunching supervisor: {reason}")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "supervise_run.py"),
        "--steps", str(STEP_TARGET), "--budget-tokens", str(TOKENS_TARGET),
        "--out", str(OUT), "--bin-dir", str(ROOT / "data" / "bin_250m"),
        "--batch", "16", "--ctx", "256", "--d-model", "288", "--layers", "8",
        "--heads", "8", "--d-ff", "880", "--lr", "1e-3", "--warmup", "2000",
        "--threads", "2", "--eval-every", "4000", "--ckpt-every", "2000",
    ]
    with open(OUT / "watchdog_relaunch.out", "ab") as fh:
        subprocess.Popen(["setsid"] + cmd, cwd=str(ROOT), stdout=fh,
                         stderr=subprocess.STDOUT, start_new_session=True)


def guard_disk(gb: float) -> None:
    if gb > 1.5:
        return
    alert(f"ACTION free disk {gb:.2f} GB — trimming")
    for leftover in sorted((ROOT / "data" / "curriculum").glob("part_*.txt")):
        leftover.unlink(missing_ok=True)
    for corrupt in OUT.glob("*.corrupt"):
        corrupt.unlink(missing_ok=True)
    for big in OUT.glob("latest-*.pt"):
        big.unlink(missing_ok=True)


def guard_nan(lines: list[str]) -> None:
    recent = [l for l in lines[-25:] if "loss" in l]
    if not any(re.search(r"loss\s+(nan|inf)", l, re.I) for l in recent):
        return
    alert("ACTION nan/inf loss — quarantining latest.pt, resuming from best.pt")
    latest, best = OUT / "latest.pt", OUT / "best.pt"
    if latest.exists():
        latest.replace(OUT / f"latest-nan-{int(time.time())}.corrupt")
    if best.exists():
        shutil.copyfile(best, latest)
    for pid in alive("scripts/supervise_run.py"):
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def write_report(step: int, info: dict, restarts: int, gb: float) -> None:
    tokens = step * TOKENS_PER_STEP
    pct = 100.0 * step / STEP_TARGET
    tps = info.get("tok_per_s", 0.0)
    remain_h = (TOKENS_TARGET - tokens) / tps / 3600 if tps else float("nan")
    vals = info.get("val", [])
    val_line = " · ".join(f"{v['step']}:{v['val_loss']:.4f}" for v in vals[-4:]) or "—"
    REPORT.write_text(f"""# Run 250 — live report

_updated {now()} · written by `scripts/watchdog.py` every 15 min_

| field | value |
|---|---|
| step | **{step:,} / {STEP_TARGET:,}** ({pct:.1f} %) |
| tokens | {tokens:,} / {TOKENS_TARGET:,} |
| train loss | {info.get('loss', '—')} |
| val loss (recent) | {val_line} |
| speed | {tps:.0f} tok/s |
| ETA | {remain_h:.1f} h |
| restarts | {restarts} |
| free disk | {gb:.2f} GB |

Weights: `{OUT / 'latest.pt'}` (resumable, weights+Adam+step),
`{OUT / 'model.pt'}` (weights only), `{OUT / 'best.pt'}` (best val).
Health history: `reports/run250_health.jsonl`. Alerts: `reports/run250_alerts.log`.

Next automatic stage: `scripts/post_run_250.py` — 500 basic questions; if fewer
than 400 are correct the run extends to 350 M tokens (85,449 steps).
""", encoding="utf-8")


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    print(f"[watchdog] start pid {os.getpid()} interval {INTERVAL}s", flush=True)
    last_step_mtime = 0.0
    stalled_samples = 0
    while True:
        gb = free_gb()
        guard_disk(gb)
        lines = tail_lines()
        guard_nan(lines)
        info = parse_log(lines)
        status: dict = {}
        if (REPORTS / "run250_status.json").exists():
            try:
                status = json.loads((REPORTS / "run250_status.json").read_text())
            except json.JSONDecodeError:
                status = {}
        ckpt = OUT / "latest.pt"
        step = int(status.get("step") or info.get("step") or 0)
        restarts = int(status.get("restarts") or 0)
        budget_done = (OUT / "DONE").exists()
        sup = alive("scripts/supervise_run.py")
        trn = alive("scripts/train_model.py")

        if ckpt.exists():
            mt = ckpt.stat().st_mtime
            if mt != last_step_mtime:
                last_step_mtime = mt
                stalled_samples = 0
            else:
                stalled_samples += 1

        record = {
            "ts": now(), "step": step, "tokens": step * TOKENS_PER_STEP,
            "pct": round(100 * step / STEP_TARGET, 2),
            "tok_per_s": info.get("tok_per_s"), "loss": info.get("loss"),
            "val": info.get("val", [])[-2:], "restarts": restarts,
            "supervisor_alive": bool(sup), "trainer_alive": bool(trn),
            "budget_done": budget_done, "free_gb": round(gb, 2),
            "mem_avail_mb": mem_available_mb(),
            "stalled_samples": stalled_samples,
            "error": info.get("error_line"),
        }
        with HEALTH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        if not budget_done and not sup:
            restart_supervisor("supervisor not alive and budget not reached")
        if not budget_done and sup and not trn and step < STEP_TARGET:
            alert(f"note: supervisor up but no trainer yet (step {step})")
        if stalled_samples >= 3 and step < STEP_TARGET:
            alert(f"note: no checkpoint advance for {stalled_samples * INTERVAL // 60} min "
                  f"(step {step}) — watching, not killing")

        write_report(step, info, restarts, gb)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
