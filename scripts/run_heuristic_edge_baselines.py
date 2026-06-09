#!/usr/bin/env python
"""Evaluate no-training Random and Degree edge-recommendation baselines."""

from __future__ import annotations

import argparse
import csv
import heapq
import random
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from algorithms.subgraph.utils.edge_recommendation_evaluator import (
    EdgeRecommendationEvaluator,
)
from datasets.elliptic.dataset import EllipticRecommendationDataset
from scripts.edge_recommendation_analysis_utils import (
    TABLE2_SETTINGS,
    compose_edge_cfg,
    dcg_at_k,
    mean_std,
    parse_setting,
    setting_sort_key,
)
from utils.exp_utils import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="+", default=TABLE2_SETTINGS)
    parser.add_argument("--out-dir", type=Path, default=Path("logs-heuristic-table2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Random", "Degree"],
        choices=["Random", "Degree"],
    )
    parser.add_argument(
        "--random-mode",
        default="expected",
        choices=["expected", "sampled"],
        help="Use analytic expected random-ranking metrics or sample one ranking.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides for dataset construction.",
    )
    return parser.parse_args()


def tensor_label(data) -> int:
    if isinstance(data.y, torch.Tensor):
        return int(data.y.flatten()[0].item())
    return int(data.y)


def build_degree_maps(train_dataset) -> tuple[Counter[int], Counter[int]]:
    sender_degree: Counter[int] = Counter()
    receiver_degree: Counter[int] = Counter()
    for data in train_dataset:
        if tensor_label(data) != 1:
            continue
        if data.senders.size(0) != 1 or data.receivers.size(0) != 1:
            continue
        sender = int(data.senders.item())
        receiver = int(data.receivers.item())
        sender_degree[sender] += 1
        receiver_degree[receiver] += 1
    return sender_degree, receiver_degree


def candidate_edges(senders: torch.Tensor, receivers: torch.Tensor):
    sender_ids = [int(x) for x in senders.tolist()]
    receiver_ids = [int(x) for x in receivers.tolist()]
    return [(s, r) for s in sender_ids for r in receiver_ids]


def expected_random_metrics(num_candidates: int, num_gt: int, top_k: int) -> dict[str, str]:
    k = min(top_k, num_candidates)
    hit_prob = k / num_candidates if num_candidates else 0.0
    ideal = dcg_at_k(min(num_gt, k))
    ndcg = 0.0
    if ideal > 0:
        ndcg = (num_gt / num_candidates) * dcg_at_k(k) / ideal
    return {
        "HR": f"{hit_prob:.10f}",
        "NDCG": f"{ndcg:.10f}",
        "hit_count": f"{num_gt * hit_prob:.10f}",
        "gt_count": str(num_gt),
        "hit_ranks": "",
    }


def sampled_random_metrics(
    edges: list[tuple[int, int]],
    gt_edges: torch.Tensor,
    top_k: int,
    rng: random.Random,
) -> dict[str, str]:
    pred = rng.sample(edges, min(top_k, len(edges)))
    metrics = EdgeRecommendationEvaluator.evaluate_instance(pred, gt_edges)
    return {
        "HR": f"{metrics['HR']:.10f}",
        "NDCG": f"{metrics['NDCG']:.10f}",
        "hit_count": str(metrics["hit_count"]),
        "gt_count": str(metrics["gt_count"]),
        "hit_ranks": ";".join(str(rank) for rank in metrics["hit_ranks"]),
    }


def stable_setting_offset(setting: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(setting))


def degree_metrics(
    edges: list[tuple[int, int]],
    gt_edges: torch.Tensor,
    top_k: int,
    sender_degree: Counter[int],
    receiver_degree: Counter[int],
) -> dict[str, str]:
    def score(edge: tuple[int, int]) -> tuple[int, int, int]:
        sender, receiver = edge
        degree_score = sender_degree[sender] + receiver_degree[receiver]
        return degree_score, -sender, -receiver

    pred = heapq.nlargest(min(top_k, len(edges)), edges, key=score)
    metrics = EdgeRecommendationEvaluator.evaluate_instance(pred, gt_edges)
    return {
        "HR": f"{metrics['HR']:.10f}",
        "NDCG": f"{metrics['NDCG']:.10f}",
        "hit_count": str(metrics["hit_count"]),
        "gt_count": str(metrics["gt_count"]),
        "hit_ranks": ";".join(str(rank) for rank in metrics["hit_ranks"]),
    }


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = []
    for setting in sorted({row["setting"] for row in rows}, key=setting_sort_key):
        for method in ["Random", "Degree"]:
            group = [row for row in rows if row["setting"] == setting and row["method"] == method]
            if not group:
                continue
            summary.append(
                {
                    "method": method,
                    "setting": setting,
                    "n": str(len(group)),
                    "HR": mean_std(float(row["HR"]) for row in group),
                    "NDCG": mean_std(float(row["NDCG"]) for row in group),
                }
            )
    return summary


def write_markdown(summary_rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| method | setting | n | HR | NDCG |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['method']} | {row['setting']} | {row['n']} | "
                f"{row['HR']} | {row['NDCG']} |\n"
            )


def main() -> None:
    args = parse_args()
    rows = []

    for setting in args.settings:
        num_illicits, num_licits, top_k = parse_setting(setting)
        cfg = compose_edge_cfg(
            setting=setting,
            algorithm="iterative_filtering",
            seed=args.seed,
            overrides=args.overrides,
        )
        set_deterministic_seed(args.seed)
        dataset = EllipticRecommendationDataset(cfg.dataset)
        train_dataset, _, test_dataset = dataset.split()
        sender_degree, receiver_degree = build_degree_maps(train_dataset)

        for sample_id, (senders, receivers, gt_edges) in enumerate(test_dataset.data_list):
            edges = candidate_edges(senders, receivers)
            density = num_illicits / max(1, len(edges))

            for method in args.methods:
                if method == "Random" and args.random_mode == "expected":
                    metrics = expected_random_metrics(len(edges), num_illicits, top_k)
                    ckpt = "expected"
                elif method == "Random":
                    rng = random.Random(
                        args.random_seed
                        + sample_id * 1_000_003
                        + stable_setting_offset(setting) % 1_000_003
                    )
                    metrics = sampled_random_metrics(edges, gt_edges, top_k, rng)
                    ckpt = f"seed{args.random_seed}"
                else:
                    metrics = degree_metrics(
                        edges,
                        gt_edges,
                        top_k,
                        sender_degree,
                        receiver_degree,
                    )
                    ckpt = "train_degree"

                rows.append(
                    {
                        "method": method,
                        "ckpt": ckpt,
                        "setting": setting,
                        "sample_id": str(sample_id),
                        "num_illicits": str(num_illicits),
                        "num_licits": str(num_licits),
                        "top_k": str(top_k),
                        "density": f"{density:.8f}",
                        **metrics,
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "heuristic_instance_metrics.csv"
    summary_path = args.out_dir / "heuristic_summary.csv"
    md_path = args.out_dir / "heuristic_summary.md"

    fieldnames = [
        "method",
        "ckpt",
        "setting",
        "sample_id",
        "num_illicits",
        "num_licits",
        "top_k",
        "density",
        "HR",
        "NDCG",
        "hit_count",
        "gt_count",
        "hit_ranks",
    ]
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "setting", "n", "HR", "NDCG"])
        writer.writeheader()
        writer.writerows(summary_rows)
    write_markdown(summary_rows, md_path)

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
