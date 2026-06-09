#!/usr/bin/env python
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runner = ROOT / "scripts" / "run_batches.py"
sys.argv = [str(runner), "run-finetune-ablation", *sys.argv[1:]]
runpy.run_path(str(runner), run_name="__main__")
