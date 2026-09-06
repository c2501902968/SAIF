from typing import Sequence, Tuple
import math
import torch
import numpy as np


class EdgeRecommendationEvaluator:
    @staticmethod
    def __call__(
        top_k_edges: Sequence[Sequence[Tuple[int, int]]],
        gt_edges: Sequence[torch.Tensor],
    ):
        """
        Evaluate edge recommendation task
        """
        assert len(top_k_edges) == len(
            gt_edges
        ), "top_k_edges and gt_edges must have the same number of batches"

        hit_ratio_list = []
        ndcg_list = []

        for top_k_edges_batch, gt_edges_batch in zip(top_k_edges, gt_edges):
            hit_ratio, ndcg = EdgeRecommendationEvaluator._evaluate_batch(
                top_k_edges_batch, gt_edges_batch
            )
            hit_ratio_list.append(hit_ratio)
            ndcg_list.append(ndcg)

        return np.mean(hit_ratio_list), np.mean(ndcg_list)

    @staticmethod
    def _evaluate_batch(
        top_k_edges_batch: Sequence[Tuple[int, int]],
        gt_edges_batch: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Evaluate edge recommendation task for a batch
        """
        metrics = EdgeRecommendationEvaluator.evaluate_instance(
            top_k_edges_batch, gt_edges_batch
        )
        return metrics["HR"], metrics["NDCG"]

    @staticmethod
    def evaluate_instance(
        top_k_edges_batch: Sequence[Tuple[int, int]],
        gt_edges_batch: torch.Tensor,
    ) -> dict[str, float | int | list[int]]:
        """
        Evaluate one recommendation instance.
        """
        gt_edges_batch = gt_edges_batch.t().detach().cpu().tolist()
        # compute HR@K
        pred_edges = set(top_k_edges_batch)
        gt_edges = set(tuple(edge) for edge in gt_edges_batch)
        num_hits = len(pred_edges.intersection(gt_edges))
        hit_ratio = num_hits / len(gt_edges)

        # compute NDCG@K
        ndcg = 0
        hit_ranks = []
        for i, edge in enumerate(top_k_edges_batch):
            if edge in gt_edges:
                hit_ranks.append(i + 1)
                ndcg += 1 / math.log2(i + 2)
        max_hit = min(len(gt_edges), len(top_k_edges_batch))
        perfect_ndcg = sum(1 / math.log2(i + 2) for i in range(max_hit))
        ndcg /= perfect_ndcg

        return {
            "HR": hit_ratio,
            "NDCG": ndcg,
            "hit_count": num_hits,
            "gt_count": len(gt_edges),
            "hit_ranks": hit_ranks,
        }
