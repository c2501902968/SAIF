#!/usr/bin/env python
"""Small helper used by per-task Python compatibility wrappers."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main(task: str) -> None:
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_batches.py"
    sys.argv = [str(runner), task, *sys.argv[1:]]
    runpy.run_path(str(runner), run_name="__main__")
