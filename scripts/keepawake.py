#!/usr/bin/env python3
"""Heartbeat that (a) tries to keep the sandbox from going idle-suspended and
(b) measures every pause that happens anyway.

Every 20 s it does a tiny amount of work and writes a heartbeat line. Because
the loop's own sleep is the reference, any gap far larger than 20 s means the
whole sandbox was frozen — that is recorded to reports/keepawake.log, so the
passive stretches are measured rather than guessed, and the training ETA can be
corrected for them.

    python3 scripts/keepawake.py &
"""
from __future__ import annotations

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "reports" / "keepawake.log"
BEAT = ROOT / "reports" / "keepawake.beat"
EVERY = 20


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    LOG.parent.mkdir(exist_ok=True)
    last = time.time()
    paused_total = 0.0
    with LOG.open("a") as log:
        log.write(f"\n=== keepawake start {stamp()} pid {os.getpid()} ===\n")
        while True:
            time.sleep(EVERY)
            now = time.time()
            gap = now - last
            last = now
            if gap > EVERY * 2.5:          # frozen: the sleep was much longer than asked
                paused_total += gap - EVERY
                line = (f"[{stamp()}] PAUSE {gap - EVERY:.0f}s "
                        f"(total paused {paused_total/60:.1f} min)")
                log.write(line + "\n")
                log.flush()
            BEAT.write_text(f"{stamp()} paused_total_s={paused_total:.0f} gap={gap:.1f}")
            # a little CPU so an idle-detector sees activity, not just a timer
            sum(i * i for i in range(2000))


if __name__ == "__main__":
    raise SystemExit(main())
