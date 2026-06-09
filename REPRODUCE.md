# Reproduce Experiments

This document describes how to reproduce the main experiments from a clean
GitHub clone.

Large files should not be committed directly to GitHub. Keep code in GitHub and
host datasets/checkpoints through GitHub Releases, Hugging Face, Zenodo, Google
Drive, or an institutional file server.

## 1. Clone

```bash
git clone <your-repo-url>
cd RevTrack-main
```

## 2. Create Environment

Python 3.10 is recommended:

```bash
conda create -n revtrack python=3.10
conda activate revtrack
pip install -r requirements.txt
```

Most batch commands already use offline Weights & Biases logging:

```bash
wandb.mode=offline
```

## 3. Download Data

Download the Elliptic / Elliptic2 preprocessing assets and place them here:

```text
data/elliptic/raw/data_df.pkl
data/elliptic/raw/node_idx_map.pt
data/elliptic/raw/raw_emb.pt
```

If you provide preprocessed files, place them under:

```text
data/elliptic/processed/
```

Add the real data URL to your GitHub README or Release page:

```text
Data download: <replace-with-your-data-url>
```

## 4. Download Checkpoints

To reproduce evaluation tables and plots, download checkpoints into this
layout:

```text
checkpoints/
|-- RevTrack/
|   |-- 0_tuned.ckpt
|   |-- 1_tuned.ckpt
|   `-- 2_tuned.ckpt
|-- AnchorRevFilter/
|   |-- tuned_seed0.ckpt
|   |-- tuned_seed1.ckpt
|   `-- tuned_seed2.ckpt
|-- MLP/
|   |-- 0.ckpt
|   |-- 1.ckpt
|   `-- 2.ckpt
|-- NGCF/
|   |-- 0.ckpt
|   |-- 1.ckpt
|   `-- 2.ckpt
`-- LightGCN/
    |-- 0.ckpt
    |-- 1.ckpt
    `-- 2.ckpt
```

Official RevTrack checkpoints can be linked from the original project README.
For AnchorRevFilter and ablation checkpoints generated in this repository, add
your own release/download URL:

```text
Checkpoint download: <replace-with-your-checkpoint-url>
```

## 5. Verify Setup

List all available tasks:

```bash
python scripts/run_batches.py list
```

Preview expensive commands before running them:

```bash
python scripts/run_batches.py run-anchor-table2 --dry-run
python scripts/run_anchor_topk.py --dry-run
```

The convenience Python entrypoints live in `scripts/`; the shared task runner is
`scripts/run_batches.py`.

## 6. Reproduce Main Table

Run official RevFilter and AnchorRevFilter evaluations:

```bash
python scripts/run_batches.py run-official-table2
python scripts/run_batches.py run-anchor-table2
python scripts/run_batches.py compare-anchor-official-table2
```

Expected outputs:

```text
logs-official-revfilter-table2/
logs-anchor-revfilter-table2/
logs-anchor-revfilter-table2/anchor_vs_official_delta.md
```

Run MLP, NGCF, and LightGCN baselines:

```bash
python scripts/run_batches.py run-baselines-table2
```

## 7. Reproduce Top-k Experiments

```bash
python scripts/run_batches.py run-official-topk
python scripts/run_batches.py run-anchor-topk
python scripts/run_batches.py compare-anchor-official-topk
python scripts/draw_topk-hr.py
python scripts/draw_topk-ncdg.py
```

Expected outputs:

```text
logs-official-topk/
logs-anchor-topk/
figures/topk_hr_curve.png
figures/topk_ndcg_curve.png
```

## 8. Reproduce Sparsity Experiments

```bash
python scripts/run_batches.py run-official-sparsity
python scripts/run_batches.py run-anchor-sparsity
python scripts/run_batches.py compare-anchor-official-sparsity
python scripts/draw_sparsity.py
python scripts/draw_sparsity-h.py
```

Expected outputs:

```text
logs-official-sparsity/
logs-anchor-sparsity/
figures/sparsity_ndcg_curve.png
figures/sparsity_hr_curve.png
```

## 9. Optional Experiments

Anchor-only evaluation:

```bash
python scripts/anchor-only.py
python scripts/evaluate_anchor_only.py
```

Ablation studies:

```bash
python scripts/run_xiaorong.py
python scripts/run_xiaorong_bu.py
python scripts/run_targeted_ablation.py
python scripts/run_order_robustness.py
```

See `ABLATION_EXPERIMENTS.md` for the targeted reviewer-control ablations,
including w/o fine-tuning, region-size features, removed sparsity proxy, removed
balance features, shuffled candidate order, and the optional no-LayerNorm run.

Cost and complexity evaluation:

```bash
python scripts/run_official_cost.py
python scripts/run_anchor_cost.py
python scripts/jiexi_cost.py
python scripts/run_complexity_profile.py
```

`run_complexity_profile.py` runs Official RevFilter and SAIF over three
checkpoints on `10+1000@100` and `10+10000@100`, then writes
`logs-complexity-profile/complexity_profile_summary.md` with runtime,
parameter count, scored-region count, and search-workload mean+/-std.

Case study:

```bash
python scripts/run_case_study.py
python scripts/draw_case_study_43.py
```

## 10. Expected Output Files

Most evaluation folders contain:

```text
revfilter_raw.csv
revfilter_summary.csv
revfilter_summary.md
```

Comparison tasks generate markdown tables such as:

```text
anchor_vs_official_delta.md
anchor_vs_official_topk_delta.md
anchor_vs_official_sparsity_delta.md
```

## 11. Reproducibility Notes

- Evaluation tasks usually use `seed=0` for test-time sampling.
- Multi-checkpoint summaries generally average over checkpoints/seeds `0`,
  `1`, and `2`.
- GPU is recommended for full reproduction. CPU runs can be slow.
- `logs-*`, `outputs/`, `checkpoints/`, and `data/` are ignored by
  `.gitignore` because they are large or generated.
- Always use `--dry-run` before expensive runs to inspect exact commands.

## 12. Troubleshooting

If `python` is not found, use the full interpreter path:

```bash
/path/to/python scripts/run_batches.py list
```

If a checkpoint is missing, verify that the file path matches the layout in
section 4.

If data loading fails, verify that required files are under `data/elliptic/`.

If Weights & Biases prompts for login, keep `wandb.mode=offline` or run:

```bash
wandb offline
```
