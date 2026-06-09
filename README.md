# SAIF

SAIF is an anchor-aware extension of RevTrack-style subgraph recommendation for
money-laundering detection on blockchain transaction graphs.

This repository contains the code, experiment wrappers, and reproducibility
notes for running SAIF / AnchorRevFilter experiments on the Elliptic dataset.
It is built on top of the original RevTrack research codebase and keeps the
original implementation structure for compatibility.

## What Is Included

- SAIF / AnchorRevFilter evaluation workflows
- RevFilter baseline evaluation workflows
- Baseline comparisons with MLP, NGCF, and LightGCN
- Top-k, sparsity, cost, ablation, and case-study experiments
- Python experiment entrypoints collected under `scripts/`
- Log parsers, summary generators, and plotting utilities

## Repository Structure

```text
.
|-- algorithms/                  # Models and filtering algorithms
|-- configurations/              # Hydra configuration files
|-- datasets/                    # Elliptic dataset loaders
|-- experiments/                 # Experiment definitions
|-- scripts/                     # Batch runner and analysis utilities
|-- utils/                       # Shared helper utilities
|-- main.py                      # Main Hydra entry point
|-- requirements.txt             # Python dependencies
|-- REPRODUCE.md                 # End-to-end reproduction instructions
|-- README_ORIGINAL_REVTRACK.md  # Original upstream RevTrack README
`-- README_GITHUB.md             # Alternative detailed GitHub README draft
```

Large files such as datasets, checkpoints, logs, and Hydra outputs are ignored
by `.gitignore`. See `REPRODUCE.md` for where to download/place them.

## Setup

Python 3.10 is recommended.

```bash
conda create -n saif python=3.10
conda activate saif
pip install -r requirements.txt
```

Most batch commands use offline Weights & Biases logging by default:

```bash
wandb.mode=offline
```

## Data And Checkpoints

Place Elliptic data under:

```text
data/elliptic/
```

Place model checkpoints under:

```text
checkpoints/
```

For the exact expected file layout, see:

```text
REPRODUCE.md
```

## Run Experiments

List all available tasks:

```bash
python scripts/run_batches.py list
```

Preview commands without running expensive jobs:

```bash
python scripts/run_batches.py run-anchor-table2 --dry-run
python scripts/run_anchor_topk.py --dry-run
```

Run SAIF / AnchorRevFilter Table 2 style evaluation:

```bash
python scripts/run_batches.py run-anchor-table2
```

Run official RevFilter comparison:

```bash
python scripts/run_batches.py run-official-table2
python scripts/run_batches.py compare-anchor-official-table2
```

Run top-k experiments:

```bash
python scripts/run_batches.py run-official-topk
python scripts/run_batches.py run-anchor-topk
python scripts/run_batches.py compare-anchor-official-topk
python scripts/draw_topk-hr.py
python scripts/draw_topk-ncdg.py
```

Run sparsity experiments:

```bash
python scripts/run_batches.py run-official-sparsity
python scripts/run_batches.py run-anchor-sparsity
python scripts/run_batches.py compare-anchor-official-sparsity
python scripts/draw_sparsity.py
python scripts/draw_sparsity-h.py
```

Run the three-checkpoint complexity profile:

```bash
python scripts/run_complexity_profile.py
```

## Script Entrypoints

Experiment entrypoint scripts are kept in `scripts/` so the repository root
stays focused on the paper artifact. Most wrappers call the shared runner:

```bash
python scripts/run_batches.py <task>
```

## Reproduction

Use `REPRODUCE.md` for a full clean-clone workflow, including:

- environment setup
- required data paths
- required checkpoint paths
- main table reproduction
- top-k and sparsity reproduction
- optional ablation, cost, and case-study runs

## Citation And Upstream Credit

This repository is based on the RevTrack codebase:

```bibtex
@inproceedings{song2024revtrack,
  title={Identifying Money Laundering Subgraphs on the Blockchain},
  author={Kiwhan Song and Mohamed Ali Dhraief and Muhua Xu and Locke Cai and Xuhao Chen and Arvind and Jie Chen},
  booktitle={Proceedings of the Fifth ACM International Conference on AI in Finance},
  year={2024}
}
```

The original RevTrack README has been preserved as
`README_ORIGINAL_REVTRACK.md`.

This repo is forked from Boyuan Chen's research template repo
`https://github.com/buoyancy99/research-template`. By directly reading the
template repo's README, you can learn how this repo is structured and how to
use it.

## License

See `LICENSE` for details.
