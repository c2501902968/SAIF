#!/usr/bin/env python
"""Export per-instance edge-recommendation metrics for trained checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch_geometric.data import Batch

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from algorithms.subgraph.utils.edge_recommendation_evaluator import (
    EdgeRecommendationEvaluator,
)
from datasets.elliptic.data import SenderToReceiverData
from datasets.elliptic.dataset import EllipticRecommendationDataset
from experiments.exp_edge_recommendation import EdgeRecommendationExperiment
from scripts.edge_recommendation_analysis_utils import (
    PROJECT_ROOT,
    TABLE2_SETTINGS,
    compose_edge_cfg,
    mean_std,
    parse_setting,
    setting_sort_key,
)
from utils.exp_utils import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="Display name written to CSV.")
    parser.add_argument("--algorithm", default="iterative_filtering")
    parser.add_argument("--ckpt", required=True, action="append", type=Path)
    parser.add_argument("--ckpt-id", action="append", default=None)
    parser.add_argument("--settings", nargs="+", default=TABLE2_SETTINGS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides, e.g. algorithm.use_anchor_features=true.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_algorithm(cfg, ckpt_path: Path, device: torch.device):
    algo_cls = EdgeRecommendationExperiment.compatible_algorithms[cfg.algorithm._name]
    algo = algo_cls(cfg.algorithm)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    try:
        algo.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        algo.load_state_dict(stripped, strict=True)
    algo.to(device)
    algo.eval()
    return algo


def batches(items: list, batch_size: int) -> Iterable[tuple[int, list]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


@torch.no_grad()
def predict_dot_product(algo, senders, receivers, device: torch.device):
    new_batch = Batch.from_data_list(
        [
            SenderToReceiverData.from_data(s, r, torch.tensor([1]))
            for s, r in zip(senders, receivers)
        ],
        follow_batch=["senders", "receivers"],
    ).to(device)
    sender_features, receiver_features = algo.model(new_batch)
    top_k_edges = []

    for idx in range(len(senders)):
        curr_senders = new_batch.senders[new_batch.senders_batch == idx]
        curr_receivers = new_batch.receivers[new_batch.receivers_batch == idx]
        curr_sender_features = sender_features[new_batch.senders_batch == idx]
        curr_receiver_features = receiver_features[new_batch.receivers_batch == idx]
        scores = curr_sender_features @ curr_receiver_features.t()
        top_k_indices = torch.topk(
            scores.flatten(),
            min(algo.top_k, scores.size(0) * scores.size(1)),
        ).indices
        top_k_senders = curr_senders[top_k_indices // scores.size(1)]
        top_k_receivers = curr_receivers[top_k_indices % scores.size(1)]
        top_k_edges.append(
            [(int(s.item()), int(r.item())) for s, r in zip(top_k_senders, top_k_receivers)]
        )

    return top_k_edges


@torch.no_grad()
def predict_iterative_filtering(algo, senders, receivers, batch_idx: int, device: torch.device):
    senders, receivers = algo._maybe_shuffle_candidate_order(senders, receivers, batch_idx)
    groups = [
        [SenderToReceiverData.from_data(s, r, torch.tensor([1]))]
        for s, r in zip(senders, receivers)
    ]

    initial_edges = max(
        1,
        groups[0][0].senders.size(0) * groups[0][0].receivers.size(0),
    )
    estimated_iters = max(1, math.ceil(math.log(initial_edges, 4)))
    keep_top_k = algo.keep_top_k
    decrease_k_by = 2 * (algo.keep_top_k - algo.top_k) / estimated_iters

    while not all(algo._is_done(groups)):
        is_done = algo._is_done(groups)
        groups = [
            group if done else algo._split_group(group)
            for group, done in zip(groups, is_done)
        ]
        should_forward = [len(group) > algo.top_k for group in groups]
        data_list = [
            data
            for group, forward in zip(groups, should_forward)
            if forward
            for data in group
        ]

        if data_list:
            model_batch = Batch.from_data_list(
                data_list, follow_batch=["senders", "receivers"]
            ).to(device)
            scores = algo.model(model_batch).flatten().detach().cpu().tolist()
            groups = algo._sort_by_scores(groups, scores, should_forward)

        groups = [
            group[:keep_top_k] if forward else group
            for group, forward in zip(groups, should_forward)
        ]
        groups = [
            (
                group[: algo.top_k]
                if forward and all(algo._is_data_1_1(data) for data in group)
                else group
            )
            for group, forward in zip(groups, should_forward)
        ]
        keep_top_k = int(max(algo.top_k, keep_top_k - decrease_k_by))

    return [
        [(int(data.senders[0].item()), int(data.receivers[0].item())) for data in group]
        for group in groups
    ]


def predict_batch(algo, algorithm_name: str, senders, receivers, batch_idx: int, device: torch.device):
    if algorithm_name in {"mlp", "ngcf", "lightgcn"}:
        return predict_dot_product(algo, senders, receivers, device)
    if algorithm_name == "iterative_filtering":
        return predict_iterative_filtering(algo, senders, receivers, batch_idx, device)
    raise ValueError(f"Unsupported algorithm={algorithm_name}")


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = []
    for setting in sorted({row["setting"] for row in rows}, key=setting_sort_key):
        group = [row for row in rows if row["setting"] == setting]
        ckpts = ",".join(sorted({row["ckpt"] for row in group}))
        summary.append(
            {
                "method": group[0]["method"],
                "ckpts": ckpts,
                "setting": setting,
                "n": str(len(group)),
                "HR": mean_std(float(row["HR"]) for row in group),
                "NDCG": mean_std(float(row["NDCG"]) for row in group),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.ckpt_id is not None and len(args.ckpt_id) != len(args.ckpt):
        raise ValueError("--ckpt-id must be supplied once per --ckpt")
    ckpt_ids = args.ckpt_id or [path.stem for path in args.ckpt]
    rows = []

    for setting in args.settings:
        num_illicits, num_licits, top_k = parse_setting(setting)
        cfg = compose_edge_cfg(
            setting=setting,
            algorithm=args.algorithm,
            seed=args.seed,
            overrides=args.overrides,
        )
        set_deterministic_seed(args.seed)
        dataset = EllipticRecommendationDataset(cfg.dataset)
        _, _, test_dataset = dataset.split()

        for ckpt_path, ckpt_id in zip(args.ckpt, ckpt_ids):
            algo = load_algorithm(cfg, PROJECT_ROOT / ckpt_path, device)

            for batch_idx, (start, batch) in enumerate(
                batches(test_dataset.data_list, args.batch_size)
            ):
                senders, receivers, gt_edges = zip(*batch)
                top_k_edges = predict_batch(
                    algo,
                    cfg.algorithm._name,
                    list(senders),
                    list(receivers),
                    batch_idx,
                    device,
                )
                for local_idx, (pred_edges, gt, s, r) in enumerate(
                    zip(top_k_edges, gt_edges, senders, receivers)
                ):
                    metrics = EdgeRecommendationEvaluator.evaluate_instance(pred_edges, gt)
                    sample_id = start + local_idx
                    density = num_illicits / (int(s.size(0)) * int(r.size(0)))
                    rows.append(
                        {
                            "method": args.method,
                            "ckpt": ckpt_id,
                            "setting": setting,
                            "sample_id": str(sample_id),
                            "num_illicits": str(num_illicits),
                            "num_licits": str(num_licits),
                            "top_k": str(top_k),
                            "density": f"{density:.8f}",
                            "HR": f"{metrics['HR']:.10f}",
                            "NDCG": f"{metrics['NDCG']:.10f}",
                            "hit_count": str(metrics["hit_count"]),
                            "gt_count": str(metrics["gt_count"]),
                            "hit_ranks": ";".join(str(rank) for rank in metrics["hit_ranks"]),
                        }
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
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
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    summary_rows = summarize(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "ckpts", "setting", "n", "HR", "NDCG"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote: {args.out}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
