# Reproduce the SAIF experiments

SAIF conditions each candidate-region score on a structural summary of the
region's current state within iterative filtering. This is the terminology used
throughout the paper and the reproduction workflow below.

This document describes the frozen `128/1/0.3/max/ELU` workflow. It separates
validation-only model selection, frozen retraining, TEST evaluation, and
supporting controls so that TEST results cannot influence model choice.

## 1. Clone and install

```bash
git clone https://github.com/c2501902968/SAIF.git
cd SAIF
conda create -n saif python=3.12
conda activate saif
pip install -r requirements.txt
```

The recorded formal environment was Python 3.12.3, PyTorch 2.5.1+cu124, CUDA
12.4, and an NVIDIA RTX 4090. Every paper-experiment launcher writes its own
environment record, so a compatible CUDA environment may be used without
silently claiming the reference environment.

## 2. Data layout and split semantics

Required Elliptic assets:

```text
data/elliptic/raw/data_df.pkl
data/elliptic/raw/node_idx_map.pt
data/elliptic/processed/data.pt
data/elliptic/processed/edge_index.pt
data/elliptic/processed/emb.pt
data/elliptic/processed/mask.pt
```

The frozen split contract is:

```text
mask == 0  training
mask == 1  validation
mask == 2  test
```

Run the provenance audit before expensive jobs:

```bash
python scripts/audit_split_mask_provenance.py
```

## 3. VAL-only Shared DeepSets selection

The validation sweep contains 12 configurations and three seeds per
configuration. It ranks configurations only by mean `validation/prauc`; its
source scan rejects TEST-derived selection logic.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_deepsets_val_only_gpu.py --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/run_deepsets_val_only_gpu.py
CUDA_VISIBLE_DEVICES=0 python scripts/run_deepsets_val_only_gpu.py --audit-only
```

Expected selection:

```text
hidden=128, layers=1, dropout=0.3, pool=max, activation=ELU
```

Outputs are written under `results/deepsets_val_only_gpu/` and include the raw
36-run table, ranking, environment, dependency snapshot, and selection audit.

MLP and NGCF use the same selection/evaluation separation:

```bash
python scripts/run_mlp_val_only.py
python scripts/run_mlp_val_selected_table1.py
python scripts/run_ngcf_val_only.py
python scripts/run_ngcf_val_selected_table1.py
```

Relevant audit entrypoints are `audit_deepsets_provenance.py`,
`audit_mlp_provenance.py`, `audit_ngcf_provenance.py`, and
`audit_lightgcn_val_table1.py`.

## 4. Frozen RevFilter and SAIF retraining

The main launcher freezes all model/search hyperparameters, trains RevFilter
and full SAIF independently for seeds 0, 1, and 2 in two stages, and only then
evaluates TEST. It is restartable and validates existing stage markers and
checkpoint hashes before reusing them.

Stage 1 is trained for at most 150 epochs with validation-based checkpoint
selection. Stage 2 resumes from the selected Stage-1 checkpoint and runs for
300 epochs without early stopping, with the best validation checkpoint
retained. Main matched test instances use evaluation seed 0.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_revfilter_saif_main.py \
  2>&1 | tee final_revfilter_saif_128_l1_d0p3_launcher.log
```

Expected artifact roots:

```text
checkpoints/final_revfilter_saif_128_l1_d0p3/
outputs/final_revfilter_saif_128_l1_d0p3/
results/final_revfilter_saif_128_l1_d0p3/
```

The integrity audit must report:

```text
48 aggregate records
12,288 instance records
6 stage-2 models
candidate hash assertion = PASS
TEST exclusion during training = PASS
GPU assertion for 60 jobs = PASS
```

Inspect:

```bash
cat results/final_revfilter_saif_128_l1_d0p3/main_table_summary.csv
cat results/final_revfilter_saif_128_l1_d0p3/integrity.json
cat results/final_revfilter_saif_128_l1_d0p3/checkpoint_provenance.json
```

## 5. Paired statistics

`scripts/paired_instance_wilcoxon_bh.py` performs two-sided paired Wilcoxon
signed-rank tests. It first verifies matched settings, sample identifiers, and
three independently trained runs; then it averages those runs per instance,
tests HR and NDCG, and applies BH correction.

```bash
python scripts/paired_instance_wilcoxon_bh.py \
  --a <revfilter-instance-csv> \
  --b <saif-instance-csv> \
  --a-name RevFilter --b-name SAIF \
  --expected-runs 3 --expected-samples 256 --expected-settings 8 \
  --bh-scope all \
  --out results/paired_main_wilcoxon_bh.csv
```

Run the statistical regression tests with:

```bash
python -m unittest tests.test_paired_instance_wilcoxon_bh -v
```

## 6. Efficiency and receiver-balanced control

These evaluations reuse the six frozen paper-protocol stage-2 checkpoints and
only run `10+1000@100` and `10+10000@100`.

Verify the retained main-experiment artifacts first:

```bash
test -f results/final_revfilter_saif_128_l1_d0p3/checkpoint_provenance.json
test -f results/final_revfilter_saif_128_l1_d0p3/raw_48_run_records.csv
test -f results/final_revfilter_saif_128_l1_d0p3/main_table_summary.csv
for method in RevFilter SAIF; do
  for seed in 0 1 2; do
    test -f "checkpoints/final_revfilter_saif_128_l1_d0p3/${method}/seed${seed}_stage2.ckpt"
  done
done
```

Run the configuration-only preflight:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py --phase all --dry-run
```

It must report two settings, three training seeds, and 12 planned evaluations
for each phase.

### 6.1 Efficiency

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py \
  --phase efficiency 2>&1 | tee final_efficiency_128_l1_d0p3_launcher.log

python scripts/run_final_efficiency_receiver_balanced.py \
  --phase efficiency --audit-only
```

The efficiency audit requires 12/12 evaluations, exact checkpoint hashes,
GPU/config assertions, and main-table HR/NDCG replication. It records:

- exhaustive initial sender-receiver pairs;
- split/filter and forward rounds;
- scored regions and scored-region ratio;
- encoded node tokens and represented edge volume;
- maximum live regions and model parameters;
- CUDA-synchronized search-loop time;
- whole-process time including startup, loading, candidate construction, and
  evaluation.

The principal outputs are:

```text
results/final_efficiency_128_l1_d0p3/RESULTS.md
results/final_efficiency_128_l1_d0p3/logs/complexity_profile_raw.csv
results/final_efficiency_128_l1_d0p3/logs/complexity_profile_summary.csv
results/final_efficiency_128_l1_d0p3/efficiency_comparison.csv
results/final_efficiency_128_l1_d0p3/integrity.json
```

For interpretation, let `N = |S||R|`, `K` be the top-k budget, `alpha` the keep
multiplier, and `T = ceil(log_4 N)` the maximum split depth. Exhaustive scoring
is `O(N)`. After the first split, iterative search scores at most
`O(alpha K T)` newly generated regions; the DeepSets work is proportional to
the sum of sender and receiver nodes in those regions. SAIF adds a
constant-dimensional state vector and therefore has the same asymptotic search
order as RevFilter, with a constant scoring overhead.

### 6.2 Receiver-balanced control

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py \
  --phase receiver-balanced 2>&1 | tee final_receiver_balanced_128_l1_d0p3_launcher.log

python scripts/run_final_efficiency_receiver_balanced.py \
  --phase receiver-balanced --audit-only
```

The audit requires 12/12 evaluations, 3,072/3,072 instance records, identical
candidate hashes across methods and checkpoints, identical receiver density,
and four paired Wilcoxon tests with BH correction over the four-test family.

Inspect:

```bash
cat results/final_receiver_balanced_128_l1_d0p3/receiver_balanced_summary.csv
cat results/final_receiver_balanced_128_l1_d0p3/official_vs_receiver_balanced_deltas.csv
cat results/final_receiver_balanced_128_l1_d0p3/paired_receiver_balanced_wilcoxon_bh_all4.md
cat results/final_receiver_balanced_128_l1_d0p3/integrity.json
```

## 7. Ablation and pooling sensitivity

Both phases use the two sparse settings, training seeds 0/1/2, evaluation seed
0, and 256 matched instances per setting. TEST is disabled during training and
is never used to select a checkpoint, pooling operator, seed, or result.

Do not use `run_finetune_ablation.py` or `run_pooling_sensitivity.sh` for
reported paper results; they do not enforce the frozen d0p3 provenance and
complete matched-instance audit.

### 7.1 Two-factor component ablation

The confirmatory design crosses state conditioning with Stage-2 fine-tuning:

| Variant | Search-state conditioning | Stage-2 fine-tuning | Checkpoint |
|---|---:|---:|---|
| RevFilter-S1 | off | off | frozen RevFilter Stage-1 |
| RevFilter-S2 | off | on | frozen RevFilter Stage-2 |
| SAIF-S1 | on | off | frozen SAIF Stage-1 |
| Full-SAIF | on | on | frozen SAIF Stage-2 |

It reuses the frozen checkpoints and runs 24 evaluations. Four planned
contrasts are tested for HR and NDCG in both settings, giving 16 paired tests in
one BH family. Four additional difference-in-differences tests estimate the
state-by-Stage-2 interaction in a separate BH family.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase ablation --dry-run

set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase ablation 2>&1 | tee final_ablation_128_l1_d0p3_launcher.log

CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase ablation --audit-only
```

Required integrity counts are 24 evaluations, 6,144 instance records, 16
planned paired tests, four interaction tests, and a passing candidate-hash
assertion. Inspect:

```text
results/final_ablation_128_l1_d0p3/ablation_summary.csv
results/final_ablation_128_l1_d0p3/ablation_planned_pairwise_wilcoxon_bh_all16.md
results/final_ablation_128_l1_d0p3/ablation_interaction_wilcoxon_bh_all4.md
results/final_ablation_128_l1_d0p3/integrity.json
```

Feature-group variants (`size_only`, `no_density`, `no_balance`) remain
exploratory because they require new training and introduce a post-hoc
multiple-testing family. The legacy internal identifier `no_density` disables
the inverse-candidate-volume search-granularity proxy; it does not refer to
suspicious or positive-pair density.

### 7.2 Pooling sensitivity

Max pooling reuses the frozen paper checkpoints. Mean and sum pooling are
trained from scratch under the same two-stage protocol, producing 24 new
training stages:

```text
2 new pooling operators x 2 methods x 3 seeds x 2 stages = 24
```

All max/mean/sum models are evaluated again in 36 jobs. The paper reports the
12 within-pooling SAIF-versus-RevFilter comparisons. Eight interaction outputs
of the form `(non-max SAIF effect) - (max SAIF effect)` are retained as
diagnostic artifacts, but they are not used to support a stronger "7 of 8
interactions significant" claim. The reported conclusion is a
representation-dependent conditioning pattern.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase pooling --dry-run

set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase pooling 2>&1 | tee final_pooling_sensitivity_128_l1_d0p3_launcher.log

CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py \
  --phase pooling --audit-only
```

Required integrity counts are 24 new training stages, 36 evaluations, 9,216
instance records, 12 reported within-pooling comparisons, eight diagnostic
interaction outputs, and a passing candidate-hash assertion. Inspect:

```text
results/final_pooling_sensitivity_128_l1_d0p3/pooling_summary.csv
results/final_pooling_sensitivity_128_l1_d0p3/pooling_saif_vs_revfilter_wilcoxon_bh_all12.md
results/final_pooling_sensitivity_128_l1_d0p3/pooling_interaction_vs_max_wilcoxon_bh_all8.md
results/final_pooling_sensitivity_128_l1_d0p3/integrity.json
checkpoints/final_pooling_sensitivity_128_l1_d0p3/
```

## 8. Candidate-order sensitivity

This sensitivity experiment reuses the frozen checkpoints and matched main-table
instances. It does not train or select a model. Candidate membership stays
fixed while sender and receiver order are independently shuffled using seeds 1,
2, and 3, alongside the original order.

The full design contains 48 evaluations and 12,288 instance records:

```text
2 methods x 3 checkpoints x 2 settings x 4 orders = 48
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_candidate_order_robustness.py --dry-run

set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_candidate_order_robustness.py \
  2>&1 | tee final_candidate_order_robustness_128_l1_d0p3_launcher.log

CUDA_VISIBLE_DEVICES=0 python scripts/run_final_candidate_order_robustness.py --audit-only
```

The audit requires membership/positive-pair hashes to remain invariant and
order hashes to change for shuffled conditions. It also checks matched order
across methods and training seeds, main-table replication under original order,
GPU/config assertions, checkpoint provenance, and absence of TEST-derived
selection.

This experiment evaluates sensitivity under the specified perturbations; it
does not establish order invariance. Report the stability of the principal NDCG
pattern together with any observed HR order sensitivity.

Inspect:

```text
results/final_candidate_order_robustness_128_l1_d0p3/candidate_order_summary.csv
results/final_candidate_order_robustness_128_l1_d0p3/candidate_order_delta_summary.csv
results/final_candidate_order_robustness_128_l1_d0p3/candidate_order_saif_vs_revfilter_wilcoxon_bh_all16.md
results/final_candidate_order_robustness_128_l1_d0p3/candidate_order_interaction_wilcoxon_bh_all12.md
results/final_candidate_order_robustness_128_l1_d0p3/order_hash_audit.csv
results/final_candidate_order_robustness_128_l1_d0p3/integrity.json
```

Re-running any experiment launcher resumes valid completed jobs. Do not retrain,
tune, select an order, or discard a seed based on these results.

## 9. Additional and exploratory workflows

The following additional scripts are available, but they are not substitutes
for the audited `run_final_*` workflows:

- `run_targeted_ablation.py`: exploratory `full`, `no_finetune`, `size_only`,
  `no_density`, and `no_balance` comparison;
- `run_finetune_ablation.py`: compact tuned-versus-no-fine-tuning comparison;
- `train_no_layernorm.py` and `run_no_layernorm_ablation.py`: optional
  normalization control requiring new checkpoints;
- `run_order_robustness.py`: a historical internal filename for a less strictly
  audited candidate-order sensitivity experiment;
- `run_complexity_profile.py` and `summarize_complexity_profile.py`: additional
  three-checkpoint complexity workflow.

Additional tasks and outputs can be listed with:

```bash
python scripts/run_batches.py list
```

## 10. Runtime provenance and artifact retention

Files under `reproducibility/` are frozen protocol manifests, not
runtime-resolved Hydra configurations. Preserve the actual `.hydra/config.yaml`
and log from each cloud job as authoritative execution provenance.

Training:

```text
outputs/final_revfilter_saif_128_l1_d0p3/training/{RevFilter|SAIF}/seed{0|1|2}/{stage1|stage2}/.hydra/config.yaml
outputs/final_revfilter_saif_128_l1_d0p3/training/{RevFilter|SAIF}/seed{0|1|2}/{stage1|stage2}/training.log
```

Main evaluation:

```text
outputs/final_revfilter_saif_128_l1_d0p3/evaluation/{RevFilter|SAIF}/seed{0|1|2}/{setting_tag}/.hydra/config.yaml
outputs/final_revfilter_saif_128_l1_d0p3/evaluation/{RevFilter|SAIF}/seed{0|1|2}/{setting_tag}/evaluation.log
```

Receiver-balanced and candidate-order controls:

```text
outputs/final_receiver_balanced_128_l1_d0p3/{RevFilter|SAIF}/seed{0|1|2}/{setting_tag}/.hydra/config.yaml
outputs/final_candidate_order_robustness_128_l1_d0p3/{RevFilter|SAIF}/seed{0|1|2}/{original|shuffle_1|shuffle_2|shuffle_3}/{setting_tag}/.hydra/config.yaml
```

Source code, configs, tests, and protocol manifests are versioned. Generated
checkpoints, full Hydra outputs, launcher logs, instance-level result bundles,
and environment snapshots are kept outside the source commit. Archive those
artifacts together with their `integrity.json` and checkpoint SHA-256 records in
a release or long-term research repository.

For every reported experiment retain the complete `results/` subdirectory,
launcher log, resolved configs, per-instance CSV/JSONL, `environment.json`,
`pip_freeze.txt`, checkpoint provenance, and `integrity.json`. Never synthesize
a launcher log or resolved configuration after the run.
