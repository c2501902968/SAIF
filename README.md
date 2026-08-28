# SAIF

SAIF is a state-aware, anchor-conditioned extension of RevFilter-style
subgraph recommendation for anti-money-laundering analysis on blockchain
transaction graphs.

This branch is the final reproducibility-oriented code release. It retains the
initial paper implementation and adds the frozen model-selection, training,
evaluation, statistical-testing, efficiency, ablation, pooling, and robustness
workflows used by the final experiments. See
[`UPDATES_SINCE_INITIAL_RELEASE.md`](UPDATES_SINCE_INITIAL_RELEASE.md) for the
file-level change inventory.

## Final frozen protocol

- Dataset split masks: `0 = TRN`, `1 = VAL`, `2 = TST`.
- Shared DeepSets selection metric: validation PR-AUC only; TEST is excluded
  from hyperparameter selection.
- Frozen backbone: input 43, hidden 128, one layer, dropout 0.3, max pooling,
  ELU activation.
- Search parameters: keep multiplier 1.5, augmentation gamma 0.4, merge range
  `[1, 20]`.
- Methods: RevFilter and full SAIF, independently trained with seeds 0, 1, and
  2 in two stages.
- Main evaluation: eight settings, 256 matched instances per setting, HR and
  NDCG, with candidate-membership hash assertions.
- Statistical analysis: paired instance-level Wilcoxon signed-rank tests after
  averaging the three independently trained runs per `(setting, sample_id)`,
  followed by Benjamini-Hochberg correction.

Machine-readable specifications are under [`reproducibility/`](reproducibility/).

## What changed after the initial paper code

1. **Anchor-conditioned representation.** The SAIF scorer now supports six
   anchor/state features, feature-group variants, LayerNorm, and deterministic
   zero/shuffle/random controls.
2. **Correct matched evaluation.** The evaluator distinguishes positive pairs
   from positive endpoints, exports per-instance HR/NDCG, and supports official,
   symmetric, and receiver-balanced candidate pools.
3. **VAL-only selection and provenance.** Dedicated Shared DeepSets, MLP, NGCF,
   and LightGCN audit workflows prevent TEST-derived selection and record the
   selected configuration.
4. **Frozen final retraining.** A restartable launcher trains RevFilter and SAIF
   for three seeds and evaluates the same eight settings with integrity checks.
5. **Statistical inference.** The repository includes paired Wilcoxon tests,
   BH correction, effect sizes, tie handling, and unit tests.
6. **Efficiency accounting.** Search-loop time, total process time, parameter
   count, initial candidate pairs, scored regions, encoded nodes, and iteration
   rounds are recorded.
7. **Boundary and defense experiments.** Receiver-balanced control, two-factor
   ablation, max/mean/sum pooling sensitivity, and deterministic candidate-order
   robustness are implemented as audited workflows.
8. **Reproducibility records.** Final scripts write resolved configurations,
   checkpoint hashes, command manifests, environment metadata, `pip freeze`,
   instance records, and `integrity.json` assertions.

## Repository layout

```text
algorithms/          model and iterative-search implementations
configurations/      Hydra configs and VAL-only sweeps
datasets/            Elliptic data loading and candidate-pool construction
experiments/         training/evaluation experiment definitions
reproducibility/     machine-readable frozen protocols
scripts/             launchers, audits, statistics, and legacy utilities
tests/               statistical regression tests
main.py              Hydra entry point
REPRODUCE.md          end-to-end final workflow
```

Generated checkpoints, Hydra outputs, detailed logs, and final result bundles
are intentionally not added by this update. Keep them under `checkpoints/`,
`outputs/`, and `results/`; the launchers record their provenance locally.

## Environment

The formal GPU runs used Python 3.12.3, PyTorch 2.5.1+cu124, CUDA 12.4, and an
NVIDIA RTX 4090. Create an isolated environment and install the project
dependencies:

```bash
conda create -n saif python=3.12
conda activate saif
pip install -r requirements.txt
```

Place the Elliptic assets under `data/elliptic/`. See [`REPRODUCE.md`](REPRODUCE.md)
for the required paths and checkpoint layout.

## Main final workflow

First audit the 36-run Shared DeepSets VAL-only sweep:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_deepsets_val_only_gpu.py --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/run_deepsets_val_only_gpu.py
```

After freezing the selected configuration, run final two-stage retraining and
the matched eight-setting evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_revfilter_saif_main.py
```

The final launcher writes to:

```text
checkpoints/final_revfilter_saif_128_l1_d0p3/
outputs/final_revfilter_saif_128_l1_d0p3/
results/final_revfilter_saif_128_l1_d0p3/
```

Run the paired main-table analysis with the two method-specific instance files:

```bash
python scripts/paired_instance_wilcoxon_bh.py \
  --a <revfilter-instance-csv> \
  --b <saif-instance-csv> \
  --a-name RevFilter --b-name SAIF \
  --out results/paired_main_wilcoxon_bh.csv
```

## Supporting experiments

All supporting launchers offer a non-computing preflight:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py --phase all --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py --phase all --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_candidate_order_robustness.py --dry-run
```

Formal execution:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py --phase all
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py --phase all
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_candidate_order_robustness.py
```

These workflows are restartable: valid completed stages are reused, while
configuration and checkpoint hashes are checked before collection. Detailed
GPU instructions are available in:

- [`SUPPORTING_GPU_RUN_INSTRUCTIONS.md`](SUPPORTING_GPU_RUN_INSTRUCTIONS.md)
- [`ABLATION_POOLING_GPU_RUN_INSTRUCTIONS.md`](ABLATION_POOLING_GPU_RUN_INSTRUCTIONS.md)
- [`CANDIDATE_ORDER_GPU_RUN_INSTRUCTIONS.md`](CANDIDATE_ORDER_GPU_RUN_INSTRUCTIONS.md)
- [`FINAL_REPRODUCIBILITY_FILES.md`](FINAL_REPRODUCIBILITY_FILES.md)

## Legacy workflows

The initial top-k, sparsity, complexity, case-study, and batch wrappers remain
available. List them with:

```bash
python scripts/run_batches.py list
```

They are retained for compatibility; the `run_final_*` launchers above define
the frozen final protocol.

## License

See [`LICENSE`](LICENSE).
