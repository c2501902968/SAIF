# Updates since the initial paper-code release

This inventory compares the initial public `main` branch at commit
`58e2182d7b1d01284e493583f43b0b2623fac5e0` with the final reproducibility
release. It lists scientific and engineering changes; generated experiment
artifacts are intentionally excluded.

## Model and search implementation

- `algorithms/subgraph/models/anchor_double_deep_sets.py`
  - adds the final six-dimensional anchor/state representation;
  - supports `full`, `size_only`, `no_balance`, and `no_density` feature groups;
  - adds LayerNorm and deterministic `normal`, `zero`, `shuffle`, and `random`
    feature controls;
  - validates feature dimensions and control modes.
- `algorithms/subgraph/models/double_deep_sets.py`
  - adds representation controls needed by the final ablation and pooling
    workflows.
- `algorithms/subgraph/iterative_filtering_algo.py`
  - adds deterministic `original`/`shuffle` candidate order;
  - independently shuffles sender and receiver order;
  - records candidate membership, positive-pair, sender-order, and
    receiver-order hashes;
  - records search-loop time, scored regions, encoded nodes, initial candidate
    pairs, and iteration rounds.

## Dataset and evaluation correctness

- `datasets/elliptic/data.py` and `datasets/elliptic/dataset.py`
  - make the TRN/VAL/TST mask provenance explicit;
  - add official, symmetric, and receiver-balanced evaluation pools;
  - separate ground-truth positive pairs from endpoint membership;
  - expose deterministic candidate-order controls.
- `algorithms/subgraph/utils/edge_recommendation_evaluator.py`
  - evaluates distinct positive pairs correctly;
  - exports sample-level HR, NDCG, ranks, candidate counts, and audit hashes.

## Configuration and experiment plumbing

- `configurations/algorithm/deepsets.yaml` and
  `configurations/algorithm/iterative_filtering.yaml` expose the final anchor,
  normalization, candidate-order, and profiling fields.
- `configurations/algorithm/lightgcn.yaml` and the LightGCN tuning config fix
  final baseline composition/provenance.
- New VAL-only sweep configs cover Shared DeepSets, MLP, and NGCF.
- `main.py`, `utils/exp_utils.py`, and
  `experiments/exp_subgraph_classification.py` propagate resolved configs and
  validation metrics needed by audited selection.

## Final experiment launchers

- `scripts/run_deepsets_val_only_gpu.py` runs the 36-job GPU VAL-only sweep and
  rejects TEST-based selection.
- `scripts/run_mlp_val_only.py`, `run_mlp_val_selected_table1.py`,
  `run_ngcf_val_only.py`, and `run_ngcf_val_selected_table1.py` separate
  validation selection from formal TEST evaluation for baselines.
- `scripts/run_final_revfilter_saif_main.py` implements restartable two-stage
  training for two methods and three seeds, followed by the matched eight-setting
  evaluation under the frozen `128/1/0.3/max/ELU` backbone.
- `scripts/run_final_efficiency_receiver_balanced.py` implements the two-setting
  efficiency profile and receiver-balanced control.
- `scripts/run_final_ablation_pooling.py` implements the two-factor ablation and
  max/mean/sum pooling boundary experiment.
- `scripts/run_final_candidate_order_robustness.py` implements deterministic
  candidate-order perturbations without changing candidate membership.

## Statistics, provenance, and integrity

- `scripts/paired_instance_wilcoxon_bh.py` adds matched-run assertions,
  Wilcoxon signed-rank tests, tie/zero handling, effect sizes, and global or
  per-metric BH correction.
- `tests/test_paired_instance_wilcoxon_bh.py` regression-tests the statistical
  implementation.
- New provenance audits cover Shared DeepSets, MLP, NGCF, LightGCN, and split
  masks.
- Final workflows record resolved commands/configs, checkpoint SHA-256 hashes,
  environment metadata, dependency snapshots, raw instance records, and
  machine-readable `integrity.json` assertions.
- `reproducibility/*.yaml` freezes training, evaluation, receiver-balanced, and
  candidate-order protocols independently of prose documentation.

## Documentation

- `README.md` now distinguishes legacy workflows from the frozen final protocol.
- `REPRODUCE.md` provides the complete VAL-selection -> retraining -> evaluation
  -> paired-statistics -> supporting-controls sequence.
- Dedicated GPU instructions document efficiency/receiver-balanced, ablation/
  pooling, and candidate-order experiments.
