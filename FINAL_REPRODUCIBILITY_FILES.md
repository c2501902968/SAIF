# Final d0p3 Reproducibility Files

## Important provenance distinction

Files under `reproducibility/` are frozen reproduction-protocol manifests. They
are not claimed to be the runtime-resolved Hydra files emitted by the cloud GPU
runs. The latter must be copied from each cloud output directory's
`.hydra/config.yaml` and retained with its corresponding log.

`scripts/run_final_revfilter_saif_main.py` has been aligned with the final d0p3
protocol (`128/1/0.3/max/ELU`). Runtime-resolved Hydra configurations and the
launcher log emitted by each GPU run remain the authoritative execution
provenance and must be retained with the corresponding results.

## Files available locally now

```text
reproducibility/final_training_protocol_d0p3.yaml
reproducibility/final_evaluation_protocol_d0p3.yaml
reproducibility/receiver_balanced_control_protocol_d0p3.yaml
reproducibility/candidate_order_robustness_protocol_d0p3.yaml
scripts/run_final_efficiency_receiver_balanced.py
scripts/run_final_ablation_pooling.py
scripts/run_final_candidate_order_robustness.py
scripts/run_final_revfilter_saif_main.py
algorithms/subgraph/iterative_filtering_algo.py
```

## Exact cloud runtime configs

### Training

Every method/seed/stage has its own resolved file:

```text
outputs/final_revfilter_saif_128_l1_d0p3/training/{RevFilter|SAIF}/seed{0|1|2}/{stage1|stage2}/.hydra/config.yaml
```

The corresponding training logs are:

```text
outputs/final_revfilter_saif_128_l1_d0p3/training/{RevFilter|SAIF}/seed{0|1|2}/{stage1|stage2}/training.log
```

### Main evaluation

```text
outputs/final_revfilter_saif_128_l1_d0p3/evaluation/{RevFilter|SAIF}/seed{0|1|2}/{setting_tag}/.hydra/config.yaml
outputs/final_revfilter_saif_128_l1_d0p3/evaluation/{RevFilter|SAIF}/seed{0|1|2}/{setting_tag}/evaluation.log
```

### Receiver-balanced control

```text
outputs/final_receiver_balanced_128_l1_d0p3/{RevFilter|SAIF}/seed{0|1|2}/{setting_tag}/.hydra/config.yaml
outputs/final_receiver_balanced_128_l1_d0p3/{RevFilter|SAIF}/seed{0|1|2}/{setting_tag}/evaluation.log
final_receiver_balanced_128_l1_d0p3_launcher.log
```

### Candidate-order robustness

```text
outputs/final_candidate_order_robustness_128_l1_d0p3/{RevFilter|SAIF}/seed{0|1|2}/{original|shuffle_1|shuffle_2|shuffle_3}/{setting_tag}/.hydra/config.yaml
outputs/final_candidate_order_robustness_128_l1_d0p3/{RevFilter|SAIF}/seed{0|1|2}/{original|shuffle_1|shuffle_2|shuffle_3}/{setting_tag}/evaluation.log
final_candidate_order_robustness_128_l1_d0p3_launcher.log
```

The candidate-order launcher log exists only after that formal cloud experiment
has been run. It must not be synthesized locally.
