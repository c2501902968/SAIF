# Final d0p3 Ablation and Pooling Sensitivity

This package adds one restartable launcher:

```text
scripts/run_final_ablation_pooling.py
```

Do not use the older `scripts/run_finetune_ablation.py` or
`scripts/run_pooling_sensitivity.sh` for the final paper results.  Those launchers
do not enforce the final `128/1/0.3` provenance and do not provide the complete
matched-instance integrity/statistical audit used here.

## Frozen design

- Backbone: `input=43, hidden=128, layers=1, dropout=0.3, activation=ELU`.
- Search: `pool=max` for ablation; `pool in {max, mean, sum}` for pooling.
- Other settings: `alpha=1.5`, `gamma=0.4`, `n_merge=[1,20]`.
- Training seeds: `0, 1, 2`.
- Evaluation seed: `0`; 256 matched candidate instances per setting.
- Evaluation settings: `10+1000@100`, `10+10000@100`.
- TEST is disabled during every training stage.
- No checkpoint, pooling operator, seed, or result is selected using TEST.

## Experiment 1: component ablation

The ablation is a 2x2 factorial design:

| Label | Anchor features | Stage-2 fine-tuning | Checkpoint |
|---|---|---|---|
| RevFilter-S1 | off | off | final RevFilter Stage-1 |
| RevFilter-S2 | off | on | final RevFilter Stage-2 |
| SAIF-S1 | on | off | final SAIF Stage-1 |
| Full-SAIF | on | on | final SAIF Stage-2 |

It reuses all frozen final checkpoints and therefore needs no new training.  It
runs 24 evaluations: 4 variants x 3 independently trained checkpoints x 2
settings.  The four planned contrasts are:

1. anchor effect after Stage-1;
2. anchor effect after Stage-2;
3. Stage-2 effect on embedding-only RevFilter;
4. Stage-2 effect on SAIF.

Each contrast is tested for HR and NDCG in both settings (16 paired tests, one BH
family).  Four additional difference-in-differences tests estimate the
anchor-by-Stage-2 interaction (a separate BH family).

Feature-group ablations are intentionally excluded from this confirmatory run.
They require new training for every feature group and would create a post-hoc
multiple-testing family.  They can remain exploratory if already available.

### Preflight

Run inside the repository containing the completed final d0p3 experiment:

```bash
cd /root/SAIF-main-gpu-val

test -f checkpoints/final_revfilter_saif_128_l1_d0p3/RevFilter/seed0_stage1.ckpt
test -f checkpoints/final_revfilter_saif_128_l1_d0p3/SAIF/seed2_stage2.ckpt
test -f results/final_revfilter_saif_128_l1_d0p3/checkpoint_provenance.json
test -f results/final_revfilter_saif_128_l1_d0p3/integrity.json

CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase ablation --dry-run
```

The final line must be:

```text
DRY_RUN = PASS
```

### Run

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase ablation \
  2>&1 | tee final_ablation_128_l1_d0p3_launcher.log
```

Expected time on the same RTX 4090: approximately 25-40 minutes.  Re-running the
same command resumes completed jobs.

### Audit and inspect

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase ablation --audit-only

cat results/final_ablation_128_l1_d0p3/integrity.json
cat results/final_ablation_128_l1_d0p3/ablation_summary.csv
cat results/final_ablation_128_l1_d0p3/ablation_planned_pairwise_wilcoxon_bh_all16.md
cat results/final_ablation_128_l1_d0p3/ablation_interaction_wilcoxon_bh_all4.md
```

Required counts:

- evaluations: `24/24`;
- instance records: `6144/6144`;
- candidate hash assertion: `PASS`;
- planned tests: `16`;
- interaction tests: `4`.

## Experiment 2: pooling sensitivity

`max` uses the already frozen final d0p3 checkpoints.  `mean` and `sum` are each
trained from scratch with the same two-stage procedure and all other settings
frozen.  This requires 24 new training stages:

```text
2 new pooling operators x 2 methods x 3 seeds x 2 stages = 24
```

All max/mean/sum models are evaluated again, giving 36 evaluation jobs.  The
analysis includes:

- SAIF minus RevFilter within every pool, setting, and metric: 12 paired tests
  with BH over all 12;
- `(non-max SAIF effect) - (max SAIF effect)`: 8 paired interaction tests with
  BH over all 8.

The interaction tests, rather than a visual comparison of separate p-values,
are the direct evidence for representation dependence.

### Preflight

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase pooling --dry-run
```

### Run

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase pooling \
  2>&1 | tee final_pooling_sensitivity_128_l1_d0p3_launcher.log
```

Expected time on the same RTX 4090: approximately 3-5 hours.  Re-running the
same command safely resumes completed training/evaluation jobs.

### Monitor

```bash
tail -f final_pooling_sensitivity_128_l1_d0p3_launcher.log
```

### Audit and inspect

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase pooling --audit-only

cat results/final_pooling_sensitivity_128_l1_d0p3/integrity.json
cat results/final_pooling_sensitivity_128_l1_d0p3/pooling_summary.csv
cat results/final_pooling_sensitivity_128_l1_d0p3/pooling_saif_vs_revfilter_wilcoxon_bh_all12.md
cat results/final_pooling_sensitivity_128_l1_d0p3/pooling_interaction_vs_max_wilcoxon_bh_all8.md
```

Required counts:

- new training stages: `24/24`;
- evaluations: `36/36`;
- instance records: `9216/9216`;
- candidate hash assertion: `PASS`;
- within-pooling tests: `12`;
- pooling-interaction tests: `8`.

## Files to download

Download both complete result directories and both launcher logs:

```text
results/final_ablation_128_l1_d0p3/
results/final_pooling_sensitivity_128_l1_d0p3/
final_ablation_128_l1_d0p3_launcher.log
final_pooling_sensitivity_128_l1_d0p3_launcher.log
```

Also preserve the newly trained pooling checkpoints and their markers:

```text
checkpoints/final_pooling_sensitivity_128_l1_d0p3/
```

Do not delete the per-instance CSV files.  They are the source data for every
paired test and interaction test.
