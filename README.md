# SAIF

SAIF is a lightweight candidate-region search-state conditioning mechanism for
RevFilter-style coarse-to-fine sender–receiver candidate retrieval in
blockchain anti-money laundering. It conditions each candidate-region score on
a structural summary of that region's current state within the iterative
search.

This repository provides the model implementation and reproducibility workflows
used for the paper, including frozen model selection, training, evaluation,
statistical testing, efficiency analysis, ablation, pooling sensitivity, and
candidate-order sensitivity experiments.

## Frozen paper protocol

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

## Implemented components

1. **Candidate-region search-state conditioning.** The SAIF scorer combines the
   candidate-region representation with a six-dimensional structural summary
   of its current search state. Feature-group variants, LayerNorm, and
   deterministic zero/shuffle/random controls support the component analyses.
2. **Matched evaluation.** The evaluator distinguishes positive pairs
   from positive endpoints, exports per-instance HR/NDCG, and supports official,
   symmetric, and receiver-balanced candidate pools.
3. **VAL-only selection and provenance.** Dedicated Shared DeepSets, MLP, NGCF,
   and LightGCN audit workflows prevent TEST-derived selection and record the
   selected configuration.
4. **Frozen retraining.** A restartable launcher trains RevFilter and SAIF
   for three seeds and evaluates the same eight settings with integrity checks.
5. **Statistical inference.** The repository includes paired Wilcoxon tests,
   BH correction, effect sizes, tie handling, and unit tests.
6. **Efficiency accounting.** Search-loop time, total process time, parameter
   count, initial candidate pairs, scored regions, encoded nodes, and iteration
   rounds are recorded.
7. **Boundary and sensitivity experiments.** Receiver-balanced control, two-factor
   ablation, max/mean/sum pooling sensitivity, and deterministic candidate-order
   sensitivity are implemented as audited workflows.
8. **Reproducibility records.** Experiment launchers write resolved configurations,
   checkpoint hashes, command manifests, environment metadata, `pip freeze`,
   instance records, and `integrity.json` assertions.

## Repository layout

```text
algorithms/          model and iterative-search implementations
configurations/      Hydra configs and VAL-only sweeps
datasets/            Elliptic data loading and candidate-pool construction
experiments/         training/evaluation experiment definitions
reproducibility/     machine-readable frozen protocols
scripts/             launchers, audits, statistics, and additional utilities
tests/               statistical regression tests
main.py              Hydra entry point
REPRODUCE.md          end-to-end paper workflow
```

Generated checkpoints, Hydra outputs, detailed logs, and result bundles are not
versioned. Keep them under `checkpoints/`,
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

## Main paper workflow

First audit the 36-run Shared DeepSets VAL-only sweep:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_deepsets_val_only_gpu.py --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/run_deepsets_val_only_gpu.py
```

Once the selected configuration is frozen, run two-stage retraining and
the matched eight-setting evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_revfilter_saif_main.py
```

The main launcher writes to:

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
commands, expected record counts, integrity gates, output paths, and artifact
retention requirements are consolidated in [`REPRODUCE.md`](REPRODUCE.md).

## Additional workflows

Additional top-k, sparsity, complexity, case-study, and batch wrappers are
available. List them with:

```bash
python scripts/run_batches.py list
```

The `run_final_*` launchers above define the frozen paper protocol.

## License

See [`LICENSE`](LICENSE).
