# Scripts

This folder now keeps the shared Python utilities for experiment batches,
log summaries, comparisons, and case studies.

The old root-level `.sh` files are compatibility wrappers. Matching `.py`
wrappers are also available for environments without Bash. Both call the single
Python entry point:

```bash
python scripts/run_batches.py list
python scripts/run_batches.py run-anchor-table2
python scripts/run_batches.py run-complexity-profile
python run_wanzheng_anchor.py --dry-run
```

If your shell cannot find `python`, set the interpreter explicitly:

```bash
PYTHON=/path/to/python bash run_wanzheng_anchor.sh
```

See [SH_SCRIPTS.md](SH_SCRIPTS.md) for the purpose of every legacy `.sh`
wrapper, matching `.py` wrapper, and corresponding Python task.
