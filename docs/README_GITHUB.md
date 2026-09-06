# RevTrack Experiment Toolkit

This repository contains code, checkpoints, scripts, and experiment utilities
for reproducing and extending RevTrack-style money-laundering subgraph
recommendation experiments on the Elliptic dataset.

The project includes:

- RevFilter / AnchorRevFilter evaluation workflows
- Baseline comparisons with MLP, NGCF, and LightGCN
- Sparsity, top-k, cost, ablation, and case-study experiments
- Unified Python wrappers for the legacy shell scripts
- Log parsers and summary-table generators

## Project Structure

```text
.
├── algorithms/                  # Model and filtering algorithm code
├── configurations/              # Hydra configuration files
├── datasets/                    # Elliptic dataset loaders
├── experiments/                 # Experiment definitions
├── scripts/                     # Shared Python utilities and batch runner
├── checkpoints/                 # Model checkpoints
├── logs-*/                      # Experiment logs and generated summaries
├── figures/                     # Generated plots
├── main.py                      # Main Hydra entry point
├── requirements.txt             # Python dependencies
└── README_GITHUB.md             # This GitHub-oriented README
```

## Environment Setup

Create a Python environment and install dependencies:

```bash
conda create -n revtrack python=3.10
conda activate revtrack
pip install -r requirements.txt
```

The original project uses Weights & Biases for logging. For offline runs, most
scripts already pass:

```bash
wandb.mode=offline
```

## Data

The repository expects Elliptic data under:

```text
data/elliptic/
```

The original RevTrack README describes how to obtain the preprocessed Elliptic2
data and node embeddings. At minimum, the project expects files such as:

```text
data/elliptic/raw/data_df.pkl
data/elliptic/raw/node_idx_map.pt
```

Some workflows may also require processed embeddings or generated edge-index
files under:

```text
data/elliptic/processed/
```

## Unified Experiment Runner

Legacy `.sh` scripts have been deduplicated. Their shared logic now lives in:

```bash
python scripts/run_batches.py list
```

Each old shell script has a matching Python wrapper, so on Windows you can run:

```bash
python run_wanzheng_anchor.py --dry-run
python run_anchor_topk.py --dry-run
python draw_sparsity.py
```

Or call the unified task runner directly:

```bash
python scripts/run_batches.py run-anchor-table2
python scripts/run_batches.py run-official-topk
python scripts/run_batches.py compare-anchor-official-table2
```

Use `--dry-run` to preview commands without launching expensive experiments:

```bash
python scripts/run_batches.py run-anchor-topk --dry-run
```

## Common Tasks

List all available experiment tasks:

```bash
python scripts/run_batches.py list
```

Evaluate AnchorRevFilter on Table 2 settings:

```bash
python scripts/run_batches.py run-anchor-table2
```

Evaluate official RevFilter on Table 2 settings:

```bash
python scripts/run_batches.py run-official-table2
```

Run top-k experiments:

```bash
python scripts/run_batches.py run-official-topk
python scripts/run_batches.py run-anchor-topk
python scripts/run_batches.py compare-anchor-official-topk
```

Run sparsity experiments:

```bash
python scripts/run_batches.py run-official-sparsity
python scripts/run_batches.py run-anchor-sparsity
python scripts/run_batches.py compare-anchor-official-sparsity
```

Run the three-checkpoint complexity profile:

```bash
python run_complexity_profile.py
```

Generate plots:

```bash
python draw_topk-hr.py
python draw_topk-ncdg.py
python draw_sparsity.py
python draw_sparsity-h.py
```

## Script Mapping

The purpose of every legacy `.sh` script and its matching `.py` wrapper is
documented here:

```text
scripts/SH_SCRIPTS.md
```

## Outputs

Experiment logs are written into folders such as:

```text
logs-anchor-revfilter-table2/
logs-official-revfilter-table2/
logs-anchor-topk/
logs-official-topk/
logs-anchor-sparsity/
logs-official-sparsity/
```

Summary utilities usually generate:

```text
revfilter_raw.csv
revfilter_summary.csv
revfilter_summary.md
```

Figures are written to:

```text
figures/
```

## Reproducibility Notes

- Most evaluation tasks use seed `0` for test-time sampling.
- Multi-checkpoint summaries typically evaluate seeds/checkpoints `0`, `1`,
  and `2`.
- Checkpoint paths are expected under `checkpoints/`.
- Expensive tasks should be tested first with `--dry-run`.

## Citation

If you use this repository as part of RevTrack-related research, cite the
original paper:

```bibtex
@inproceedings{song2024revtrack,
  title={Identifying Money Laundering Subgraphs on the Blockchain},
  author={Kiwhan Song and Mohamed Ali Dhraief and Muhua Xu and Locke Cai and Xuhao Chen and Arvind and Jie Chen},
  booktitle={Proceedings of the Fifth ACM International Conference on AI in Finance},
  year={2024}
}
```

## License

See `LICENSE` for licensing details.
