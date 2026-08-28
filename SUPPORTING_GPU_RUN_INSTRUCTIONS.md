# Final efficiency and receiver-balanced controls

Run all commands from `/root/SAIF-main-gpu-val`.

## 1. Update and verify the source tree

```bash
git pull --ff-only
git status --short
```

Run from a clean checkout of the repository revision recorded for the paper.
If files were transferred as an offline package instead, verify that package
against its separately archived SHA-256 manifest before extraction.

## 2. Verify retained final artifacts

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

## 3. Syntax and configuration-only dry run

```bash
python -m py_compile \
  scripts/run_final_efficiency_receiver_balanced.py \
  scripts/summarize_complexity_profile.py \
  scripts/paired_instance_wilcoxon_bh.py \
  scripts/edge_recommendation_analysis_utils.py
```

```bash
python scripts/run_final_efficiency_receiver_balanced.py --dry-run --phase all
```

Expected output includes `DRY_RUN_ONLY`, two settings, three training seeds, and
12 planned evaluations for each phase.

## 4. GPU preflight

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("CUDA PREFLIGHT = PASS")
PY
```

## 5. Run efficiency first

This evaluates the official candidate pool with profiling enabled. It performs
12 evaluations: two methods by three frozen checkpoints by two settings.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py \
  --phase efficiency 2>&1 | tee final_efficiency_128_l1_d0p3_launcher.log
```

Completion must include:

```text
FINAL_CHECKPOINT_PROVENANCE = PASS
EFFICIENCY RESULTS READY
AUTHORIZED SUPPORTING EXPERIMENTS COMPLETE
```

Run the independent audit without evaluation:

```bash
python scripts/run_final_efficiency_receiver_balanced.py \
  --phase efficiency --audit-only
```

Expected audit fields include 12/12 runs, GPU/config assertions, exact final
checkpoint hashes, and Main-Table HR/NDCG replication. The primary efficiency
quantities are `scored_regions_per_sample`, `region_score_ratio`, and the
CUDA-synchronized `search_elapsed_sec`. Whole-process `elapsed_sec` is reported
separately and includes startup/data construction.

## 6. Run receiver-balanced control

This performs 12 evaluations on newly constructed receiver-balanced candidate
instances. The evaluation seed is 0, and every method/checkpoint sees the same
256 instances within each setting.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/run_final_efficiency_receiver_balanced.py \
  --phase receiver-balanced 2>&1 | tee final_receiver_balanced_128_l1_d0p3_launcher.log
```

Completion must include:

```text
FINAL_CHECKPOINT_PROVENANCE = PASS
RECEIVER-BALANCED RESULTS READY
AUTHORIZED SUPPORTING EXPERIMENTS COMPLETE
```

Run the independent audit without evaluation:

```bash
python scripts/run_final_efficiency_receiver_balanced.py \
  --phase receiver-balanced --audit-only
```

The receiver-balanced audit requires 12/12 evaluations, 3,072/3,072 instance
records, candidate-hash matching across both methods and all three checkpoints,
and four two-sided paired Wilcoxon tests with BH correction over the four-test
family.

## 7. Inspect results

```bash
cat results/final_efficiency_128_l1_d0p3/RESULTS.md
cat results/final_receiver_balanced_128_l1_d0p3/receiver_balanced_summary.csv
cat results/final_receiver_balanced_128_l1_d0p3/official_vs_receiver_balanced_deltas.csv
cat results/final_receiver_balanced_128_l1_d0p3/paired_receiver_balanced_wilcoxon_bh_all4.md
```

Do not retrain models and do not run any other supporting experiments in this
round.
