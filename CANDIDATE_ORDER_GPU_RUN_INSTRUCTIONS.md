# Final Candidate-Order Robustness — GPU Instructions

Formal evaluation must run on the cloud GPU server.  The local package only
contains the launcher and audit instrumentation; it does not contain or retrain
models.

## Frozen protocol

- Checkpoints: final d0p3 RevFilter/SAIF Stage-2, training seeds 0/1/2.
- Backbone: `43/128/1/0.3/max/ELU`.
- Search: `alpha=1.5`; all other final settings unchanged.
- Evaluation settings: `10+1000@100`, `10+10000@100`.
- Evaluation seed: 0; 256 matched instances per setting.
- Orders: `original`, `shuffle_1`, `shuffle_2`, `shuffle_3`.
- Shuffle seeds: 1, 2, 3.
- Training, tuning, checkpoint selection, and order selection are prohibited.

The patch to `algorithms/subgraph/iterative_filtering_algo.py` only adds audit
hashes to the existing instance JSONL output.  It does not change model scores,
candidate membership, partitioning, retention, terminal handling, or metrics.

## Job counts

```text
2 methods x 3 checkpoints x 2 settings x 4 orders = 48 evaluations
48 x 256 = 12,288 instance records
```

Expected time on the same RTX 4090: approximately 55-80 minutes.

## Update and verify

Run from the cloud GPU checkout:

```bash
cd /root/SAIF-main-gpu-val
git pull --ff-only
git status --short

python -m py_compile \
  scripts/run_final_candidate_order_robustness.py \
  scripts/paired_instance_wilcoxon_bh.py \
  algorithms/subgraph/iterative_filtering_algo.py
```

## Preflight

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_final_candidate_order_robustness.py \
  --dry-run
```

Required final line:

```text
DRY_RUN = PASS
```

The preflight verifies CUDA, the six final checkpoint hashes, the final d0p3
provenance, and prints all 48 commands.  It does not run evaluation.

## Formal run

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0 \
python scripts/run_final_candidate_order_robustness.py \
  2>&1 | tee final_candidate_order_robustness_128_l1_d0p3_launcher.log
```

Monitor with:

```bash
tail -f final_candidate_order_robustness_128_l1_d0p3_launcher.log
```

If the connection or process stops, run the same formal command again.  Jobs
with a complete 256-row audit, matching checkpoint hash, GPU log, and resolved
configuration are skipped.

## Final audit

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_final_candidate_order_robustness.py \
  --audit-only
```

Required final line:

```text
AUDIT_ONLY = PASS
```

Required integrity values:

```text
aggregate_record_count = 48
instance_record_count = 12288
same_candidate_membership = PASS
same_ground_truth = PASS
same_candidate_counts = PASS
matched_order_across_methods = PASS
matched_order_across_training_seeds = PASS
original_main_table_replication = PASS
GPU_ASSERTION = PASS
CONFIG_ASSERTION = PASS
CHECKPOINT_PROVENANCE = PASS
ORDER_HASH_ASSERTION = PASS
test_derived_model_selection = false
```

## Inspect results

```bash
cat results/final_candidate_order_robustness_128_l1_d0p3/integrity.json

cat results/final_candidate_order_robustness_128_l1_d0p3/candidate_order_summary.csv

cat results/final_candidate_order_robustness_128_l1_d0p3/candidate_order_delta_summary.csv

cat results/final_candidate_order_robustness_128_l1_d0p3/candidate_order_saif_vs_revfilter_wilcoxon_bh_all16.md

cat results/final_candidate_order_robustness_128_l1_d0p3/candidate_order_interaction_wilcoxon_bh_all12.md
```

## Files to download

Download the complete directory and launcher log:

```text
results/final_candidate_order_robustness_128_l1_d0p3/
final_candidate_order_robustness_128_l1_d0p3_launcher.log
```

The most important audit source files are:

```text
candidate_order_instance_metrics.csv
order_hash_audit.csv
candidate_order_raw_48.csv
integrity.json
```

Stop after this experiment.  Do not retrain, tune, select an order, or start a
new experiment.
