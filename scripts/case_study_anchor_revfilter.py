#!/usr/bin/env python
"""Case study for RevFilter vs AnchorRevFilter.

This script compares two iterative filtering checkpoints on the same generated
edge-recommendation test samples. It reports the samples where AnchorRevFilter
most improves NDCG and records where the ground-truth illicit edges appear in
each ranked top-k list.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.subgraph.models import AnchorAwareDoubleDeepSets, DoubleDeepSets
from datasets.elliptic.data import SenderToReceiverData
from datasets.elliptic.dataset import EllipticRecommendationDataset


@dataclass
class SampleResult:
    sample_idx: int
    num_senders: int
    num_receivers: int
    candidate_edges: int
    density: float
    official_hr: float
    official_ndcg: float
    anchor_hr: float
    anchor_ndcg: float
    delta_hr: float
    delta_ndcg: float
    official_hit_ranks: list[int | str]
    anchor_hit_ranks: list[int | str]
    official_edges: list[tuple[int, int]]
    anchor_edges: list[tuple[int, int]]
    gt_edges: list[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="10+1000@100")
    parser.add_argument("--official-ckpt", required=True)
    parser.add_argument("--anchor-ckpt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-cases", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--keep-multiplier", type=float, default=1.5)
    parser.add_argument("--anchor-feature-mode", default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="logs-case-study")
    parser.add_argument("--edge-preview", type=int, default=20)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_setting(setting: str) -> tuple[int, int, int]:
    dataset_str, top_k = setting.split("@")
    num_illicits, num_licits = dataset_str.split("+")
    return int(num_illicits), int(num_licits), int(top_k)


def build_dataset_cfg(num_illicits: int, num_licits: int, num_samples: int):
    return OmegaConf.create(
        {
            "shot_size": -1,
            "num_illicits": num_illicits,
            "num_licits": num_licits,
            "num_samples": num_samples,
            "filter_1_1": False,
            "augment": {
                "enabled": False,
                "min": 1,
                "max": 20,
                "gamma": 0.4,
            },
            "use_edge_index": False,
        }
    )


def build_model_cfg(anchor_feature_mode: str):
    return OmegaConf.create(
        {
            "emb_path": "data/elliptic/processed/emb.pt",
            "num_classes": 2,
            "input_dim": 43,
            "num_layers": 2,
            "hidden_dim": 128,
            "dropout": 0.2,
            "activation": "ELU",
            "pool": "max",
            "anchor_feature_mode": anchor_feature_mode,
        }
    )


def load_model(model_cls, ckpt_path: str | Path, model_cfg, device: torch.device):
    model = model_cls(model_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model.") :]] = value
        elif not key.startswith(("criterion.", "train_", "val_", "test_")):
            model_state[key] = value

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    serious_missing = [key for key in missing if not key.startswith(("criterion",))]
    if serious_missing or unexpected:
        print(f"[WARN] {ckpt_path} missing={serious_missing} unexpected={unexpected}")

    model.to(device)
    model.eval()
    return model


def is_data_1_1(data: SenderToReceiverData) -> bool:
    return len(data.senders) == 1 and len(data.receivers) == 1


def split_data(data: SenderToReceiverData) -> list[SenderToReceiverData]:
    if len(data.senders) == 1 and len(data.receivers) == 1:
        return [data]

    def split_nodes(nodes: torch.Tensor):
        return torch.chunk(nodes, 2) if len(nodes) > 1 else (nodes,)

    return [
        SenderToReceiverData.from_data(senders, receivers, data.y)
        for senders in split_nodes(data.senders)
        for receivers in split_nodes(data.receivers)
    ]


def split_group(group: list[SenderToReceiverData]) -> list[SenderToReceiverData]:
    return [child for data in group for child in split_data(data)]


def sort_by_scores(
    groups: list[list[SenderToReceiverData]],
    scores: list[float],
    forwarded: list[bool],
) -> list[list[SenderToReceiverData]]:
    num_data = [len(group) if forward else 0 for group, forward in zip(groups, forwarded)]
    score_start_indices = [0] + list(np.cumsum(num_data))[:-1]
    sorted_groups = []

    for i, (group, forward) in enumerate(zip(groups, forwarded)):
        if not forward:
            sorted_groups.append(group)
            continue
        group_scores = scores[score_start_indices[i] : score_start_indices[i] + len(group)]
        sorted_groups.append(
            [
                data
                for _, data in sorted(
                    zip(group_scores, group),
                    reverse=True,
                    key=lambda item: item[0],
                )
            ]
        )
    return sorted_groups


@torch.no_grad()
def rank_edges(
    model: torch.nn.Module,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    top_k: int,
    keep_multiplier: float,
    device: torch.device,
) -> list[tuple[int, int]]:
    groups = [[SenderToReceiverData.from_data(senders, receivers, torch.tensor([1]))]]
    num_candidates = max(1, int(senders.size(0) * receivers.size(0)))
    estimated_iters = max(1, math.ceil(math.log(num_candidates, 4)))
    keep_top_k = int(top_k * keep_multiplier)
    decrease_k_by = 2 * (keep_top_k - top_k) / estimated_iters

    def is_done(group_list: list[list[SenderToReceiverData]]) -> list[bool]:
        return [
            all(is_data_1_1(data) for data in group) and len(group) <= top_k
            for group in group_list
        ]

    while not all(is_done(groups)):
        done = is_done(groups)
        groups = [group if curr_done else split_group(group) for group, curr_done in zip(groups, done)]
        should_forward = [len(group) > top_k for group in groups]
        data_list = [
            data
            for group, forward in zip(groups, should_forward)
            if forward
            for data in group
        ]

        if not data_list:
            continue

        batch = Batch.from_data_list(data_list, follow_batch=["senders", "receivers"])
        batch = batch.to(device)
        scores = model(batch).flatten().detach().cpu().tolist()
        groups = sort_by_scores(groups, scores, should_forward)
        groups = [
            group[:keep_top_k] if forward else group
            for group, forward in zip(groups, should_forward)
        ]
        groups = [
            group[:top_k] if forward and all(is_data_1_1(data) for data in group) else group
            for group, forward in zip(groups, should_forward)
        ]
        keep_top_k = int(max(top_k, keep_top_k - decrease_k_by))

    return [(data.senders[0].item(), data.receivers[0].item()) for data in groups[0]]


def evaluate_edges(
    ranked_edges: list[tuple[int, int]],
    gt_edges: Iterable[tuple[int, int]],
) -> tuple[float, float, list[int | str]]:
    gt_set = set(gt_edges)
    pred_set = set(ranked_edges)
    hits = pred_set.intersection(gt_set)
    hr = len(hits) / len(gt_set)

    dcg = 0.0
    rank_map = {edge: idx + 1 for idx, edge in enumerate(ranked_edges)}
    for edge in ranked_edges:
        if edge in gt_set:
            rank = rank_map[edge]
            dcg += 1.0 / math.log2(rank + 1)

    max_hit = min(len(gt_set), len(ranked_edges))
    perfect_dcg = sum(1.0 / math.log2(i + 2) for i in range(max_hit))
    ndcg = dcg / perfect_dcg if perfect_dcg > 0 else 0.0

    hit_ranks = [rank_map.get(edge, f">{len(ranked_edges)}") for edge in sorted(gt_set)]
    hit_ranks = sorted(hit_ranks, key=lambda x: x if isinstance(x, int) else 10**9)
    return hr, ndcg, hit_ranks


def tensor_edges_to_tuples(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    return [tuple(edge) for edge in edge_index.t().detach().cpu().tolist()]


def format_ranks(ranks: list[int | str]) -> str:
    return ", ".join(str(rank) for rank in ranks)


def write_summary_csv(results: list[SampleResult], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "num_senders",
                "num_receivers",
                "candidate_edges",
                "density",
                "official_hr",
                "official_ndcg",
                "anchor_hr",
                "anchor_ndcg",
                "delta_hr",
                "delta_ndcg",
                "official_hit_ranks",
                "anchor_hit_ranks",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "sample_idx": r.sample_idx,
                    "num_senders": r.num_senders,
                    "num_receivers": r.num_receivers,
                    "candidate_edges": r.candidate_edges,
                    "density": f"{r.density:.8f}",
                    "official_hr": f"{r.official_hr:.6f}",
                    "official_ndcg": f"{r.official_ndcg:.6f}",
                    "anchor_hr": f"{r.anchor_hr:.6f}",
                    "anchor_ndcg": f"{r.anchor_ndcg:.6f}",
                    "delta_hr": f"{r.delta_hr:+.6f}",
                    "delta_ndcg": f"{r.delta_ndcg:+.6f}",
                    "official_hit_ranks": format_ranks(r.official_hit_ranks),
                    "anchor_hit_ranks": format_ranks(r.anchor_hit_ranks),
                }
            )


def write_markdown(results: list[SampleResult], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("| case | senders | receivers | density | method | HR | NDCG | hit ranks |\n")
        f.write("|---:|---:|---:|---:|---|---:|---:|---|\n")
        for r in results:
            f.write(
                f"| {r.sample_idx} | {r.num_senders} | {r.num_receivers} | "
                f"{r.density:.6f} | RevFilter | {r.official_hr:.4f} | "
                f"{r.official_ndcg:.4f} | {format_ranks(r.official_hit_ranks)} |\n"
            )
            f.write(
                f"| {r.sample_idx} | {r.num_senders} | {r.num_receivers} | "
                f"{r.density:.6f} | AnchorRevFilter | {r.anchor_hr:.4f} | "
                f"{r.anchor_ndcg:.4f} | {format_ranks(r.anchor_hit_ranks)} |\n"
            )


def write_edge_preview(result: SampleResult, out_path: Path, edge_preview: int) -> None:
    rows = []
    gt_set = set(result.gt_edges)
    for method, edges in [
        ("RevFilter", result.official_edges),
        ("AnchorRevFilter", result.anchor_edges),
    ]:
        for rank, edge in enumerate(edges[:edge_preview], start=1):
            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "sender": edge[0],
                    "receiver": edge[1],
                    "is_illicit": int(edge in gt_set),
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "rank", "sender", "receiver", "is_illicit"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_illicits, num_licits, top_k = parse_setting(args.setting)
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset_cfg = build_dataset_cfg(num_illicits, num_licits, args.num_samples)
    dataset = EllipticRecommendationDataset(dataset_cfg)
    _, _, test_dataset = dataset.split()

    model_cfg = build_model_cfg(args.anchor_feature_mode)
    official_model = load_model(DoubleDeepSets, args.official_ckpt, model_cfg, device)
    anchor_model = load_model(AnchorAwareDoubleDeepSets, args.anchor_ckpt, model_cfg, device)

    results = []
    for sample_idx in range(len(test_dataset)):
        senders, receivers, illicit_edge_index = test_dataset[sample_idx]
        gt_edges = tensor_edges_to_tuples(illicit_edge_index)

        official_edges = rank_edges(
            official_model,
            senders,
            receivers,
            top_k,
            args.keep_multiplier,
            device,
        )
        anchor_edges = rank_edges(
            anchor_model,
            senders,
            receivers,
            top_k,
            args.keep_multiplier,
            device,
        )

        official_hr, official_ndcg, official_ranks = evaluate_edges(official_edges, gt_edges)
        anchor_hr, anchor_ndcg, anchor_ranks = evaluate_edges(anchor_edges, gt_edges)
        candidate_edges = int(senders.size(0) * receivers.size(0))
        density = len(gt_edges) / candidate_edges

        results.append(
            SampleResult(
                sample_idx=sample_idx,
                num_senders=int(senders.size(0)),
                num_receivers=int(receivers.size(0)),
                candidate_edges=candidate_edges,
                density=density,
                official_hr=official_hr,
                official_ndcg=official_ndcg,
                anchor_hr=anchor_hr,
                anchor_ndcg=anchor_ndcg,
                delta_hr=anchor_hr - official_hr,
                delta_ndcg=anchor_ndcg - official_ndcg,
                official_hit_ranks=official_ranks,
                anchor_hit_ranks=anchor_ranks,
                official_edges=official_edges,
                anchor_edges=anchor_edges,
                gt_edges=gt_edges,
            )
        )

    results.sort(key=lambda r: (r.delta_ndcg, r.delta_hr), reverse=True)
    top_results = results[: args.top_cases]

    write_summary_csv(results, out_dir / "case_study_all_samples.csv")
    write_summary_csv(top_results, out_dir / "case_study_top_cases.csv")
    write_markdown(top_results, out_dir / "case_study_top_cases.md")

    for result in top_results:
        write_edge_preview(
            result,
            out_dir / f"case_{result.sample_idx}_top_edges.csv",
            args.edge_preview,
        )

    print(f"Wrote: {out_dir / 'case_study_all_samples.csv'}")
    print(f"Wrote: {out_dir / 'case_study_top_cases.csv'}")
    print(f"Wrote: {out_dir / 'case_study_top_cases.md'}")


if __name__ == "__main__":
    main()
