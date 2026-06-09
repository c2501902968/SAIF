# Complexity Profiling Experiment

This experiment supports the complexity discussion for the sparse sender-receiver
search procedure used by RevFilter and SAIF.

## What The Code Records

Enable profiling with:

```bash
algorithm.profile_search=true
```

During `IterativeFilteringAlgo.test_step`, the code records:

- `initial_pairs_per_sample`: exhaustive sender-receiver pair count, `|S| * |R|`.
- `search_rounds_per_sample`: number of split/filter rounds until top-k one-to-one pairs remain.
- `forward_rounds_per_sample`: number of rounds requiring model scoring.
- `scored_regions_per_sample`: number of candidate regions passed through the scorer.
- `region_score_ratio`: `scored_regions_per_sample / initial_pairs_per_sample`.
- `scored_node_tokens_per_sample`: total sender/receiver node tokens encoded by DeepSets.
- `scored_edge_volume_per_sample`: total `|S_q| * |R_q|` represented by scored regions.
- `max_live_regions_per_sample`: largest live candidate-region pool during search.
- `model_parameters`: number of trainable/evaluated model parameters.
- `search_elapsed_sec`: wall-clock time inside the iterative filtering loop.

The outer `elapsed_sec` written by `scripts/run_batches.py` includes process
startup, data loading, candidate construction, and testing.

## How To Run

Run the three-checkpoint profile on the two highly sparse settings:

```bash
python run_complexity_profile.py
```

This evaluates:

- Official RevFilter: `checkpoints/RevTrack/{0,1,2}_tuned.ckpt`
- SAIF: `checkpoints/AnchorRevFilter/tuned_seed{0,1,2}.ckpt`
- Settings: `10+1000@100`, `10+10000@100`

To summarize existing logs without rerunning:

```bash
python summarize_complexity_profile.py
```

Outputs:

- `logs-complexity-profile/complexity_profile_raw.csv`
- `logs-complexity-profile/complexity_profile_summary.csv`
- `logs-complexity-profile/complexity_profile_summary.md`

All summary values are reported as `mean+/-std` over three checkpoints.

## Paper Reporting

Report two complementary timings:

1. End-to-end evaluation time (`elapsed_sec`): includes data loading and candidate construction.
2. Search-only time (`search_elapsed_sec`): isolates the iterative region scoring loop.

A concise complexity statement:

> Let `N = |S||R|` be the exhaustive sender-receiver pair count, `K` the top-k
> budget, `alpha` the keep multiplier, and `T = ceil(log_4 N)` the maximum
> number of recursive split rounds. Exhaustive pair scoring evaluates `O(N)`
> candidate pairs. The iterative search scores only newly generated regions;
> already selected one-to-one pairs are carried forward. Therefore the number
> of scored regions is bounded by `O(alpha K T)` after the first split. The
> DeepSets scorer costs `O(sum_q (|S_q| + |R_q|))` over scored regions. SAIF
> adds only a constant-dimensional structural anchor vector per region, so its
> asymptotic search cost matches RevFilter while adding a small constant
> scoring overhead.

In the complexity table, include at least:

- `scored_regions_per_sample`
- `region_score_ratio`
- `scored_node_tokens_per_sample`
- `model_parameters`
- `search_elapsed_sec`
- `elapsed_sec`
