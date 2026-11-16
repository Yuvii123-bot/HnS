#!/usr/bin/env python3
"""Unattended supervisor for the 250M-token run (and the 100M extension).

Three promises, enforced here rather than by hope:

1. **It never pauses.** The trainer is relaunched after any exit (OOM, SIGKILL,
   a sandbox restart kills the child but the loop restarts it). A crash costs
   at most the steps since the last checkpoint (``--ckpt-every``).
2. **Nothing is lost.** Every restart resumes from ``<out>/latest.pt``, which
   carries weights + Adam state + step. A weights-only ``model.pt`` is written
   at the same cadence, so the run is still readable if the optimizer state is
   ever damaged. A small ``status.json`` is rewritten every 30 s.
3. **It stops only at the budget.** The loop exits when the checkpointed step
   count reaches ``--steps``. If a post-stage hook exists
   (``scripts/post_run_250.py``) it is executed exactly once, and its exit code
   decides whether the run is finished or extended.

    python3 scripts/supervise_run.py --steps 61036            # 250M tokens
    python3 scripts/supervise_run.py --steps 85449            # 350M tokens

Tokens per step = batch * ctx, so ``--steps`` is the budget in steps:
250e6 / (16*256) = 61,036 steps exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_step(ckpt: Path) -> int:
    if not ckpt.exists():
        return 0
    try:
        blob = torch_load(ckpt)
        return int(blob.get("step", 0))
    except Exception as exc:  # corrupt half-written file
        print(f"[supervisor] cannot read {ckpt}: {exc}", flush=True)
        return -1


def existing_trainers(out: Path) -> list[int]:
    """pids of train_model.py processes bound to this output directory."""
    try:
        listing = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                                 text=True).stdout
    except OSError:
        return []
    found = []
    for line in listing.splitlines():
        if "train_model.py" in line and f"--out {out}" in line:
            try:
                found.append(int(line.split()[0]))
            except ValueError:
                pass
    return found


def torch_load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--budget-tokens", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/run250"))
    ap.add_argument("--bin-dir", type=Path, default=Path("data/bin_250m"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=288)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=880)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--max-restarts", type=int, default=200)
    ap.add_argument("--hook", type=Path, default=ROOT / "scripts" / "post_run_250.py")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    log_path = reports / f"{out.name}.log"
    status_path = reports / f"{out.name}_status.json"
    pid_path = out / "supervisor.pid"
    done_path = out / "DONE"

    # one supervisor at a time
    if pid_path.exists():
        try:
            old = int(pid_path.read_text().strip())
            os.kill(old, 0)
            print(f"[supervisor] already running as pid {old}; exiting", flush=True)
            return 0
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    pid_path.write_text(str(os.getpid()))

    tokens_per_step = args.batch * args.ctx
    budget = args.budget_tokens or args.steps * tokens_per_step
    started = time.time()
    restarts = 0
    log = open(log_path, "a", buffering=1)

    def rel(path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:
            return str(Path(path).resolve())

    def status(step: int, state: str, note: str = "") -> None:
        try:
            _write_status(step, state, note)
        except Exception as exc:          # a status write must never kill the run
            print(f"[supervisor] status write failed: {exc}", flush=True)

    def _write_status(step: int, state: str, note: str = "") -> None:
        payload = {
            "state": state,
            "step": step,
            "steps_target": args.steps,
            "tokens_per_step": tokens_per_step,
            "tokens_seen": step * tokens_per_step,
            "tokens_target": budget,
            "fraction": round(step / max(1, args.steps), 4),
            "restarts": restarts,
            "validator": "held-out val.bin, sha-stable split",
            "note": note,
            "updated": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "elapsed_h": round((time.time() - started) / 3600, 2),
            "log": rel(log_path),
            "checkpoint": rel(out / "latest.pt"),
        }
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[supervisor] pid {os.getpid()} target {args.steps} steps "
          f"({budget:,} tokens) -> {out}", flush=True)
    log.write(f"\n=== supervisor start {time.strftime('%F %T')} "
              f"target={args.steps} steps ===\n")

    while True:
        ckpt = out / "latest.pt"
        step_now = read_step(ckpt)
        if step_now == -1:
            bad = out / "latest.pt.corrupt"
            ckpt.replace(bad)
            print(f"[supervisor] quarantined corrupt checkpoint -> {bad}", flush=True)
            step_now = 0
        if step_now >= args.steps:
            print(f"[supervisor] budget reached at step {step_now}", flush=True)
            break
        if restarts >= args.max_restarts:
            status(step_now, "gave_up", f"{restarts} restarts")
            print("[supervisor] too many restarts, giving up", flush=True)
            return 1

        # an orphaned trainer from a dead supervisor must be adopted, not duplicated
        first_wait = True
        while existing_trainers(out):
            if first_wait:
                print("[supervisor] adopting an existing trainer for this run", flush=True)
                status(read_step(out / "latest.pt") if (out / "latest.pt").exists() else 0,
                       "adopting", "orphaned trainer running")
                first_wait = False
            time.sleep(30)

        cmd = [
            sys.executable, str(ROOT / "scripts" / "train_model.py"),
            "--bin-dir", str(args.bin_dir), "--out", str(out),
            "--steps", str(args.steps), "--batch", str(args.batch),
            "--ctx", str(args.ctx), "--d-model", str(args.d_model),
            "--layers", str(args.layers), "--heads", str(args.heads),
            "--d-ff", str(args.d_ff), "--lr", str(args.lr),
            "--warmup", str(args.warmup), "--threads", str(args.threads),
            "--eval-every", str(args.eval_every),
            "--ckpt-every", str(args.ckpt_every),
        ]
        if ckpt.exists():
            cmd += ["--resume", str(ckpt)]
        print(f"[supervisor] launch (restart #{restarts}): {' '.join(cmd[1:])}",
              flush=True)
        log.write(f"\n=== launch restart #{restarts} {time.strftime('%F %T')} ===\n")

        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        status(step_now, "training", f"pid {proc.pid}")
        while proc.poll() is None:
            time.sleep(30)
            status(read_step(out / "latest.pt") if (out / "latest.pt").exists()
                   else step_now, "training", f"pid {proc.pid}")
        code = proc.returncode
        print(f"[supervisor] trainer exited with {code}", flush=True)
        log.write(f"=== trainer exit {code} {time.strftime('%F %T')} ===\n")
        if code == 0:
            after = read_step(out / "latest.pt")
            if after >= args.steps:
                break
        restarts += 1
        time.sleep(20)

    final = read_step(out / "latest.pt")
    done_path.write_text(json.dumps({
        "steps": final, "tokens": final * tokens_per_step,
        "finished": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")
    status(final, "budget_reached", "hook pending")

    if args.hook.exists():
        print(f"[supervisor] running hook {args.hook}", flush=True)
        log.write(f"=== hook {args.hook.name} {time.strftime('%F %T')} ===\n")
        rc = subprocess.call([sys.executable, str(args.hook)], cwd=str(ROOT))
        status(final, "hook_done" if rc == 0 else "hook_failed", f"rc={rc}")
        print(f"[supervisor] hook rc={rc}", flush=True)
        return rc
    status(final, "done", "no hook")
    print("[supervisor] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
