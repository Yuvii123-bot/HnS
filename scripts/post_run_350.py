#!/usr/bin/env python3
"""Exam hook for the extended run: re-score 350M, never extend again."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.exit(subprocess.call([sys.executable, str(ROOT / "scripts" / "post_run_250.py"),
                          "--no-extend"], cwd=str(ROOT)))
