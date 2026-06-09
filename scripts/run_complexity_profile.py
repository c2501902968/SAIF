#!/usr/bin/env python
from __future__ import annotations

import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

runner = ROOT / "scripts" / "run_batches.py"
sys.argv = [str(runner), "run-complexity-profile", *sys.argv[1:]]
runpy.run_path(str(runner), run_name="__main__")
