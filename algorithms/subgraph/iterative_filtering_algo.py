import math
import numpy as np
import time
from typing import List
from functools import partial
import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch
from algorithms.subgraph.utils.edge_recommendation_evaluator import (
    EdgeRecommendationEvaluator,
)
from datasets.elliptic.data import SenderToReceiverData
from .models import AnchorAwareDoubleDeepSets, DoubleDeepSets
from .subgraph_algo import SubgraphAlgo


class IterativeFilteringAlgo(SubgraphAlgo):
    def __init__(self, cfg: DictConfig):
        self.top_k = cfg.top_k  # number of top edges to recommend
        self.keep_top_k = int(
            self.top_k * cfg.keep_multiplier
        )  # keep at most this many edges after each iteration
        self.test_every_n_epoch = cfg.test_every_n_epoch
        self.profile_search = getattr(cfg, "profile_search", False)
        self.candidate_order = getattr(cfg, "candidate_order", "original")
        self.candidate_order_seed = int(getattr(cfg, "candidate_order_seed", 0))
        if self.candidate_order not in {"original", "shuffle"}:
            raise ValueError(
                f"Unknown candidate_order={self.candidate_order}; "
                "expected 'original' or 'shuffle'."
            )
        self.evaluator = EdgeRecommendationEvaluator()
        super().__init__(cfg)

    def _get_model_cls(self):
        if getattr(self.cfg, "use_anchor_features", False):
            return AnchorAwareDoubleDeepSets
        return DoubleDeepSets

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            super().validation_step(batch, batch_idx, dataloader_idx)
        elif self.current_epoch % self.test_every_n_epoch == 0:
            self.test_step(batch, batch_idx, namespace="test")

    def test_step(self, batch, batch_idx: int, namespace: str = "final_test"):
        senders, receivers, illicit_edge_indices = batch
        senders, receivers = self._maybe_shuffle_candidate_order(
            senders, receivers, batch_idx
        )
        batch_size = len(senders)

        groups = [
            [SenderToReceiverData.from_data(s, r, torch.tensor([1]))]
            for s, r in zip(senders, receivers)
        ]
        initial_candidate_edges = [
            int(s.size(0) * r.size(0)) for s, r in zip(senders, receivers)
        ]

        if self.profile_search:
            search_profile = self._init_search_profile(
                batch_size, initial_candidate_edges
            )
            search_start = self._start_search_timer()

        estimated_iters = math.ceil(
            math.log(groups[0][0].senders.size(0) * groups[0][0].receivers.size(0), 4)
        )
        keep_top_k = self.keep_top_k
        decrease_k_by = 2 * (self.keep_top_k - self.top_k) / estimated_iters

        while not all(self._is_done(groups)):
            is_done = self._is_done(groups)
            if self.profile_search:
                self._profile_search_round(search_profile, is_done)

            # done group remains the same
            # undone group split into 4
            groups = [
                group if done else self._split_group(group)
                for group, done in zip(groups, is_done)
            ]
            if self.profile_search:
                self._profile_live_regions(search_profile, groups)

            # choose groups to forward pass (compute scores)
            should_forward = [len(group) > self.top_k for group in groups]
            if self.profile_search:
                self._profile_forwarded_groups(
                    search_profile, groups, should_forward
                )

            # choose data to forward pass (all data in should_forward groups, by zipping with should_forward)
            data_list = [
                data
                for group, forward in zip(groups, should_forward)
                if forward
                for data in group
            ]

            if len(data_list) == 0:
                continue

            batch = Batch.from_data_list(
                data_list, follow_batch=["senders", "receivers"]
            )

            # sort each forwarded group by scores
            scores = self.model(batch).flatten().detach().cpu().tolist()
            groups = self._sort_by_scores(groups, scores, should_forward)

            # keep keep_top_k data in each forwarded group
            groups = [
                group[:keep_top_k] if forward else group
                for group, forward in zip(groups, should_forward)
            ]

            # for each forwarded group, if top-k edges are all 1-1, only keep the top-k edges (mark as done)
            groups = [
                (
                    group[: self.top_k]
                    if forward and all(self._is_data_1_1(data) for data in group)
                    else group
                )
                for group, forward in zip(groups, should_forward)
            ]

            keep_top_k = int(max(self.top_k, keep_top_k - decrease_k_by))

        if self.profile_search:
            search_elapsed_sec = self._stop_search_timer(search_start)

        top_k_edges = [
            [(data.senders[0].item(), data.receivers[0].item()) for data in group]
            for group in groups
        ]

        hit_ratio, ndcg = self.evaluator(top_k_edges, illicit_edge_indices)

        log_fn = partial(
            self.log,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        log_fn(f"{namespace}/HR", hit_ratio)
        log_fn(f"{namespace}/NDCG", ndcg)
        if self.profile_search:
            self._log_search_profile(
                namespace,
                search_profile,
                search_elapsed_sec,
                batch_size,
            )

    @staticmethod
    def _init_search_profile(
        batch_size: int, initial_candidate_edges: List[int]
    ) -> dict[str, List[float]]:
        return {
            "initial_candidate_edges": [float(x) for x in initial_candidate_edges],
            "search_rounds": [0.0] * batch_size,
            "forward_rounds": [0.0] * batch_size,
            "scored_regions": [0.0] * batch_size,
            "scored_sender_tokens": [0.0] * batch_size,
            "scored_receiver_tokens": [0.0] * batch_size,
            "scored_edge_volume": [0.0] * batch_size,
            "max_live_regions": [1.0] * batch_size,
        }

    @staticmethod
    def _profile_search_round(
        profile: dict[str, List[float]], is_done: List[bool]
    ) -> None:
        for idx, done in enumerate(is_done):
            if not done:
                profile["search_rounds"][idx] += 1.0

    @staticmethod
    def _profile_live_regions(
        profile: dict[str, List[float]], groups: List[List[SenderToReceiverData]]
    ) -> None:
        for idx, group in enumerate(groups):
            profile["max_live_regions"][idx] = max(
                profile["max_live_regions"][idx], float(len(group))
            )

    @staticmethod
    def _profile_forwarded_groups(
        profile: dict[str, List[float]],
        groups: List[List[SenderToReceiverData]],
        should_forward: List[bool],
    ) -> None:
        for idx, (group, forward) in enumerate(zip(groups, should_forward)):
            if not forward:
                continue
            profile["forward_rounds"][idx] += 1.0
            profile["scored_regions"][idx] += float(len(group))
            for data in group:
                num_senders = float(data.senders.size(0))
                num_receivers = float(data.receivers.size(0))
                profile["scored_sender_tokens"][idx] += num_senders
                profile["scored_receiver_tokens"][idx] += num_receivers
                profile["scored_edge_volume"][idx] += num_senders * num_receivers

    def _start_search_timer(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _stop_search_timer(self, start: float) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - start

    def _log_search_profile(
        self,
        namespace: str,
        profile: dict[str, List[float]],
        search_elapsed_sec: float,
        batch_size: int,
    ) -> None:
        def mean(values: List[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        initial_candidate_edges = profile["initial_candidate_edges"]
        total_scored_tokens = [
            s + r
            for s, r in zip(
                profile["scored_sender_tokens"],
                profile["scored_receiver_tokens"],
            )
        ]
        region_score_ratio = [
            scored / max(1.0, exhaustive)
            for scored, exhaustive in zip(
                profile["scored_regions"], initial_candidate_edges
            )
        ]

        values = {
            "initial_pairs_per_sample": mean(initial_candidate_edges),
            "search_rounds_per_sample": mean(profile["search_rounds"]),
            "forward_rounds_per_sample": mean(profile["forward_rounds"]),
            "scored_regions_per_sample": mean(profile["scored_regions"]),
            "region_score_ratio": mean(region_score_ratio),
            "scored_sender_tokens_per_sample": mean(
                profile["scored_sender_tokens"]
            ),
            "scored_receiver_tokens_per_sample": mean(
                profile["scored_receiver_tokens"]
            ),
            "scored_node_tokens_per_sample": mean(total_scored_tokens),
            "scored_edge_volume_per_sample": mean(
                profile["scored_edge_volume"]
            ),
            "max_live_regions_per_sample": mean(profile["max_live_regions"]),
            "model_parameters": float(
                sum(p.numel() for p in self.model.parameters())
            ),
        }

        log_kwargs = dict(
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        for key, value in values.items():
            self.log(
                f"{namespace}/profile_{key}",
                torch.tensor(value, device=self.device, dtype=torch.float),
                **log_kwargs,
            )

        self.log(
            f"{namespace}/profile_search_elapsed_sec",
            torch.tensor(search_elapsed_sec, device=self.device, dtype=torch.float),
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
            add_dataloader_idx=False,
            reduce_fx="sum",
        )

    def _maybe_shuffle_candidate_order(
        self,
        senders: List[torch.Tensor],
        receivers: List[torch.Tensor],
        batch_idx: int,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.candidate_order == "original":
            return senders, receivers

        shuffled_senders = []
        shuffled_receivers = []
        batch_seed = self.candidate_order_seed + int(batch_idx) * 1_000_003
        for sample_idx, (sample_senders, sample_receivers) in enumerate(
            zip(senders, receivers)
        ):
            seed = batch_seed + sample_idx * 2
            shuffled_senders.append(self._shuffle_nodes(sample_senders, seed))
            shuffled_receivers.append(self._shuffle_nodes(sample_receivers, seed + 1))
        return shuffled_senders, shuffled_receivers

    @staticmethod
    def _shuffle_nodes(nodes: torch.Tensor, seed: int) -> torch.Tensor:
        if nodes.size(0) <= 1:
            return nodes
        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(nodes.size(0), generator=generator).to(nodes.device)
        return nodes[perm]

    @staticmethod
    def _sort_by_scores(
        groups: List[List[SenderToReceiverData]],
        scores: List[float],
        forwarded: List[bool],
    ) -> List[List[SenderToReceiverData]]:
        """
        Sort each group by scores and return the sorted groups
        """
        num_data = [
            len(group) if forward else 0 for group, forward in zip(groups, forwarded)
        ]
        score_start_indices = [0] + list(np.cumsum(num_data))[:-1]
        return [
            (
                [
                    data
                    for _, data in sorted(
                        zip(
                            scores[
                                score_start_indices[i] : score_start_indices[i]
                                + len(group)
                            ],
                            group,
                        ),
                        reverse=True,
                        key=lambda x: x[0],
                    )
                ]
                if forward
                else group
            )
            for i, (group, forward) in enumerate(zip(groups, forwarded))
        ]

    def _is_done(self, groups: List[List[SenderToReceiverData]]) -> List[bool]:
        """
        Given a batch of SenderToReceiverData objects, return a list of booleans indicating whether each group is done recommending top-k edges
        """
        return [
            all(IterativeFilteringAlgo._is_data_1_1(data) for data in group)
            and len(group) <= self.top_k
            for group in groups
        ]

    @staticmethod
    def _split_data(
        data: SenderToReceiverData,
    ) -> List[SenderToReceiverData]:
        """
        Split a SenderToReceiverData object into 4 SenderToReceiverData objects
        by splitting the senders and receivers into 2 parts each.
        """
        if len(data.senders) == 1 and len(data.receivers) == 1:
            return [data]

        def split_nodes(nodes):
            return torch.chunk(nodes, 2) if len(nodes) > 1 else (nodes,)

        all_senders = split_nodes(data.senders)
        all_receivers = split_nodes(data.receivers)
        return [
            SenderToReceiverData.from_data(senders, receivers, data.y)
            for senders in all_senders
            for receivers in all_receivers
        ]

    @classmethod
    def _split_group(
        cls, group: List[SenderToReceiverData]
    ) -> List[SenderToReceiverData]:
        return [d for data in group for d in cls._split_data(data)]

    @staticmethod
    def _is_data_1_1(data: SenderToReceiverData) -> bool:
        return len(data.senders) == 1 and len(data.receivers) == 1
