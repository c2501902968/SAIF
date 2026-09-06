# Shell Script Inventory

所有 `.sh` 文件现在都只是兼容 wrapper，实际逻辑集中在
`scripts/run_batches.py`。每个 `.sh` 也有一个同名 `.py` wrapper，更适合
Windows 或没有 Bash 的环境。直接运行旧脚本仍然可用，也可以运行：

```bash
python scripts/run_batches.py <task>
python <同名脚本>.py
```

| `.sh` 文件 | `.py` 文件 | Python task | 作用 |
|---|---|---|
| `anchor-only.sh` | `anchor-only.py` | `anchor-only` | 训练 SAIF anchor-only 版本的 seed 0/1/2 预训练和微调 checkpoint。 |
| `check_checkpoints.sh` | `check_checkpoints.py` | `check-checkpoints` | 检查几个关键 checkpoint 中 anchor 相关权重和预测 MLP 权重是否存在及其形状。 |
| `draw_case_study_43.sh` | `draw_case_study_43.py` | `draw-case-study-43` | 绘制固定 case 43 的 RevFilter 与 AnchorRevFilter 命中排名对比图。 |
| `draw_sparsity.sh` | `draw_sparsity.py` | `draw-sparsity-ndcg` | 从各方法 sparsity summary CSV 绘制候选密度与 NDCG 曲线。 |
| `draw_sparsity-h.sh` | `draw_sparsity-h.py` | `draw-sparsity-hr` | 从各方法 sparsity summary CSV 绘制候选密度与 HR 曲线。 |
| `draw_topk-hr.sh` | `draw_topk-hr.py` | `draw-topk-hr` | 从 top-k summary CSV 绘制 top-k budget 与 HR 曲线。 |
| `draw_topk-ncdg.sh` | `draw_topk-ncdg.py` | `draw-topk-ndcg` | 从 top-k summary CSV 绘制 top-k budget 与 NDCG 曲线；保留原文件名里的拼写。 |
| `evaluate_anchor_only.sh` | `evaluate_anchor_only.py` | `evaluate-anchor-only` | 用 SAIF anchor-only 的 tuned checkpoint 跑 Table 2 设置并汇总日志。 |
| `jiexi_cost.sh` | `jiexi_cost.py` | `summarize-cost` | 解析 `logs-cost/*.log`，输出时间、内存、HR、NDCG 的成本汇总表。 |
| `jiexisymetric.sh` | `jiexisymetric.py` | `summarize-balanced-receiver-control` | 解析 balanced receiver control 日志并生成 raw CSV 与 markdown summary。 |
| `keep_multiplier.sh` | `keep_multiplier.py` | `keep-multiplier` | 跑 keep multiplier 敏感性实验，并生成 SAIF 相对 RevFilter 的 delta 表。 |
| `receiver_balance_control.sh` | `receiver_balance_control.py` | `receiver-balance-control` | 在 `balanced_receivers` evaluation pool 下运行 geometry control 评估。 |
| `run_3models_topk.sh` | `run_3models_topk.py` | `run-3models-topk` | 对 MLP、NGCF、LightGCN 三个 baseline 跑 top-k 设置并汇总。 |
| `run_3model-sparsity.sh` | `run_3model-sparsity.py` | `run-3models-sparsity` | 对 MLP、NGCF、LightGCN 三个 baseline 跑 sparsity 设置并汇总。 |
| `run_anchor.sh` | `run_anchor.py` | `run-anchor` | 用 AnchorRevFilter seed0 跑四个主设置，写入 `logs-anchor-revfilter`。 |
| `run_anchor_bu34.sh` | `run_anchor_bu34.py` | `train-anchor-seeds-3-4` | 训练 AnchorRevFilter seed 3 和 seed 4 的预训练/微调 checkpoint。 |
| `run_anchor_cost.sh` | `run_anchor_cost.py` | `run-anchor-cost` | 对 AnchorRevFilter 在两个大设置上做计时和内存成本评估。 |
| `run_anchor_result.sh` | `run_anchor_result.py` | `run-anchor-result` | 用 AnchorRevFilter tuned_seed0/1/2 跑四个主设置并汇总。 |
| `run_anchor_sparsity.sh` | `run_anchor_sparsity.py` | `run-anchor-sparsity` | 用 AnchorRevFilter tuned_seed0/1/2 跑 sparsity 设置并生成 summary。 |
| `run_anchor_topk.sh` | `run_anchor_topk.py` | `run-anchor-topk` | 用 AnchorRevFilter tuned_seed0/1/2 跑 top-k 设置并生成 summary。 |
| `run_case_study.sh` | `run_case_study.py` | `run-case-study` | 调用 `scripts/case_study_anchor_revfilter.py` 生成 case-study CSV/Markdown。 |
| `run_check_checkpoints.sh` | `run_check_checkpoints.py` | `paired-official-anchor` | 对 official 与 anchor 的 Table 2 raw logs 做 paired 统计检验。 |
| `run_compare.sh` | `run_compare.py` | `compare-anchor-official-table2` | 比较 official 与 anchor Table 2 summary，输出 HR/NDCG delta 表。 |
| `run_delta.sh` | `run_delta.py` | `compare-anchor-official-sparsity` | 比较 official 与 anchor sparsity summary，输出 delta 表。 |
| `run_edge_index_shengcheng.sh` | `run_edge_index_shengcheng.py` | `build-edge-index` | 通过数据集 loader 生成 `data/elliptic/processed/edge_index.pt`。 |
| `run_offical_result.sh` | `run_offical_result.py` | `run-official-result` | 用 official RevFilter checkpoint 跑四个主设置并汇总；保留原文件名拼写。 |
| `run_official_cost.sh` | `run_official_cost.py` | `run-official-cost` | 对 official RevFilter 在两个大设置上做计时和内存成本评估。 |
| `run_official_sparsity.sh` | `run_official_sparsity.py` | `run-official-sparsity` | 用 official RevFilter checkpoint 跑 sparsity 设置并生成 summary。 |
| `run_official_topk.sh` | `run_official_topk.py` | `run-official-topk` | 用 official RevFilter checkpoint 跑 top-k 设置并生成 summary。 |
| `run_seed1.sh` | `run_seed1.py` | `train-anchor-seed1` | 训练 AnchorRevFilter seed 1 的预训练/微调 checkpoint。 |
| `run_seed2.sh` | `run_seed2.py` | `train-anchor-seed2` | 训练 AnchorRevFilter seed 2 的预训练/微调 checkpoint。 |
| `run_topk-delta.sh` | `run_topk-delta.py` | `compare-anchor-official-topk` | 比较 official 与 anchor top-k summary，输出 delta 表。 |
| `run_wanzheng_anchor.sh` | `run_wanzheng_anchor.py` | `run-anchor-table2` | 用 AnchorRevFilter tuned_seed0/1/2 跑完整 Table 2 设置并汇总。 |
| `run_wanzheng_offical.sh` | `run_wanzheng_offical.py` | `run-official-table2` | 用 official RevFilter tuned checkpoint 跑完整 Table 2 设置并汇总；保留原文件名拼写。 |
| `run_xiaorong.sh` | `run_xiaorong.py` | `run-ablation-quick` | 训练 seed0 的 anchor ablation 变体并跑 quick ablation 评估。 |
| `run_xiaorong_bu.sh` | `run_xiaorong_bu.py` | `run-ablation-3seeds` | 补训练 seed1/2 的 ablation 变体，并汇总三 seed ablation 结果。 |
| `run_zhubiao_bu3model.sh` | `run_zhubiao_bu3model.py` | `run-baselines-table2` | 对 MLP、NGCF、LightGCN 跑完整 Table 2 设置并汇总。 |
| `scripts/dummy_script.sh` | `scripts/dummy_script.py` | `dummy` | 保留模板脚本行为，输出 `hello world`。 |
| `symmetric-com.sh` | `symmetric-com.py` | `run-symmetric-control` | 在 `symmetric` evaluation pool 下比较 official、SAIF 和 geometry control。 |
| `tianjiasymmetricindatasets.sh` | `tianjiasymmetricindatasets.py` | `patch-eval-pool-mode` | 给 dataset/config 增加 `eval_pool_mode` 支持；该补丁是幂等的。 |

整理后的重复逻辑主要包括：统一 `python -m main` 调用、setting 到日志 tag 的转换、
`tee` 式日志写入、最新 checkpoint 复制、RevFilter summary 生成、delta 表生成、
绘图和 source patch。
