# SAIF Ablation Experiments

This note lists the ablation runs that address the most likely reviewer
questions about SAIF.

## 1. Targeted Ablations

Run:

```bash
python scripts/run_targeted_ablation.py
```

This evaluates the two sparse settings:

- `10+1000@100`
- `10+10000@100`

and the following variants over three checkpoints:

- `full`: Full SAIF with tuned checkpoints.
- `no_finetune`: Full SAIF pretrain checkpoints without the candidate-region
  fine-tuning stage.
- `size_only`: RevFilter plus region size features
  (`log |S|`, `log |R|`, `log |S||R|`).
- `no_density`: SAIF without the sparsity proxy.
- `no_balance`: SAIF without sender-receiver balance features.

Outputs:

- `logs-targeted-ablation/anchor_ablation_raw.csv`
- `logs-targeted-ablation/anchor_ablation_summary.md`

Use this table to rule out simpler explanations: that SAIF only benefits from
two-stage tuning, only uses region size, or only learns generic balance cues.

## 2. Fine-Tuning Only

Run:

```bash
python scripts/run_finetune_ablation.py
```

This smaller table compares only:

- `full_tuned`
- `no_finetune`

Outputs:

- `logs-finetune-ablation/anchor_ablation_raw.csv`
- `logs-finetune-ablation/anchor_ablation_summary.md`

Use this if you want a compact table dedicated to the contribution of the
candidate-region fine-tuning stage.

## 3. Candidate Order Robustness

Run:

```bash
python scripts/run_order_robustness.py
```

This evaluates Full SAIF under the original candidate order and three shuffled
sender/receiver orders. The model is unchanged; only the contiguous split order
inside iterative filtering changes.

Outputs:

- `logs-order-robustness/order_robustness_raw.csv`
- `logs-order-robustness/order_robustness_summary.md`

Use this table to show that SAIF is not relying on one favorable deterministic
candidate ordering.

## 4. Optional No-LayerNorm Ablation

This ablation requires new checkpoints because the architecture changes.

Train:

```bash
python scripts/train_no_layernorm.py
```

Evaluate:

```bash
python scripts/run_no_layernorm_ablation.py
```

Outputs:

- `logs-no-layernorm-ablation/anchor_ablation_raw.csv`
- `logs-no-layernorm-ablation/anchor_ablation_summary.md`

This is lower priority than fine-tuning and order robustness. It is mainly an
engineering-control ablation for the anchor feature normalization layer.

## Recommended Paper Tables

Main appendix ablation table:

```text
Variant | 10+1000@100 HR | 10+1000@100 NDCG | 10+10000@100 HR | 10+10000@100 NDCG
Full SAIF
SAIF w/o fine-tuning
RevFilter + size features
SAIF w/o sparsity proxy
SAIF w/o balance features
```

Robustness table:

```text
Condition | n | 10+1000@100 HR/NDCG | 10+10000@100 HR/NDCG
Original order
Shuffle seed 0
Shuffle seed 1
Shuffle seed 2
All shuffled
```

Report all values as mean+/-std.
