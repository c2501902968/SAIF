# Reproduce the final SAIF experiments

This document describes the frozen `128/1/0.3/max/ELU` workflow. It separates
validation-only model selection, final retraining, TEST evaluation, and
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
12.4, and an NVIDIA RTX 4090. Every final launcher writes its own environment
record, so a compatible CUDA environment may be used without silently claiming
the reference environment.

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

The final sweep contains 12 configurations and three seeds per configuration.
It ranks configurations only by mean `validation/prauc`; its source scan rejects
TEST-derived selection logic.

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

## 4. Final RevFilter and SAIF retraining

The final launcher freezes all model/search hyperparameters, trains RevFilter
and full SAIF independently for seeds 0, 1, and 2 in two stages, and only then
evaluates TEST. It is restartable and validates existing stage markers and
checkpoint hashes before reusing them.

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

The final integrity audit must report:

```text
48 aggregate records
12,288 instance records
6 final stage-2 models
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

These evaluations reuse the six frozen final stage-2 checkpoints and only run
`10+1000@100` and `10+10000@100`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py --phase all --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py --phase all
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py --phase all --audit-only
```

Efficiency compares actual scored regions and encoded nodes, not merely wall
clock time. Receiver-balanced evaluation asserts identical receiver density
across methods and writes direct official-versus-balanced delta comparisons.

## 7. Ablation and pooling sensitivity

The ablation is a two-factor design over state conditioning and Stage-2
fine-tuning. Pooling sensitivity compares max, mean, and sum while holding the
rest of the architecture fixed.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py --phase all --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py --phase ablation
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py --phase pooling
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_ablation_pooling.py --phase all --audit-only
```

The collector writes both within-pooling SAIF-versus-RevFilter tests and direct
interaction tests against max pooling. See
[`ABLATION_POOLING_GPU_RUN_INSTRUCTIONS.md`](ABLATION_POOLING_GPU_RUN_INSTRUCTIONS.md).

## 8. Candidate-order robustness

This defense experiment reuses the frozen checkpoints and matched main-table
instances. Candidate membership stays fixed while sender and receiver order are
independently shuffled with deterministic seeds.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_candidate_order_robustness.py --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_candidate_order_robustness.py
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_candidate_order_robustness.py --audit-only
```

The audit requires membership/positive-pair hashes to remain invariant and
order hashes to change for shuffled conditions. See
[`CANDIDATE_ORDER_GPU_RUN_INSTRUCTIONS.md`](CANDIDATE_ORDER_GPU_RUN_INSTRUCTIONS.md).

## 9. What is and is not versioned

Source code, configs, tests, and protocol manifests are versioned. Generated
checkpoints, full Hydra outputs, launcher logs, instance-level result bundles,
and environment snapshots are kept outside the source commit. Archive those
artifacts together with their `integrity.json` and checkpoint SHA-256 records in
a release or long-term research repository.

The exact handoff inventory is documented in
[`FINAL_REPRODUCIBILITY_FILES.md`](FINAL_REPRODUCIBILITY_FILES.md).
