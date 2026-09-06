# Scripts

This folder keeps the shared Python utilities, experiment entrypoints, log
summaries, comparisons, and case-study scripts. The repository root is kept
small for the paper artifact.

Most convenience entrypoints call the single Python task runner:

```bash
python scripts/run_batches.py list
python scripts/run_batches.py run-anchor-table2
python scripts/run_batches.py run-complexity-profile
python scripts/run_wanzheng_anchor.py --dry-run
```

Use `python scripts/run_batches.py list` to see all available tasks.
