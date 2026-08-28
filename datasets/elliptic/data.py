from typing import Union
import torch
from torch_geometric.data import Data


def unique_edge_pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Return distinct directed pairs from a 2-by-E edge-index tensor."""
    if edge_index.numel() == 0:
        return set()
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"Expected edge_index with shape [2, E], got {tuple(edge_index.shape)}"
        )
    return {
        tuple(int(value) for value in edge)
        for edge in edge_index.t().detach().cpu().tolist()
    }


def candidate_density(
    senders: torch.Tensor,
    receivers: torch.Tensor,
    positive_edge_index: torch.Tensor,
) -> float:
    """Compute unique-positive density over the sender-receiver Cartesian product."""
    candidate_count = int(senders.numel()) * int(receivers.numel())
    if candidate_count <= 0:
        raise ValueError("Candidate density is undefined for an empty candidate space")
    return len(unique_edge_pairs(positive_edge_index)) / candidate_count


class SenderToReceiverData(Data):
    """
    A data object representing a sender-to-receiver bipartite graph.
    node_idx: node indices (senders + receivers)
    senders: node indices of senders
    receivers: node indices of receivers
    edge_index: (relabeled) edge indices (senders -> receivers)
    num_nodes: number of nodes (senders + receivers)
    y: label
    """

    @staticmethod
    def from_data(senders: torch.Tensor, receivers: torch.Tensor, y: torch.Tensor):
        num_senders = senders.size(0)
        num_receivers = receivers.size(0)
        edge_index = torch.stack(
            torch.meshgrid(
                torch.arange(num_senders),
                torch.arange(num_senders, num_senders + num_receivers),
                indexing="ij",
            )
        ).reshape(2, -1)
        return SenderToReceiverData(
            node_idx=torch.cat([senders, receivers]),
            senders=senders,
            receivers=receivers,
            edge_index=edge_index,
            num_nodes=num_senders + num_receivers,
            y=y,
        )

    def __add__(self, other: Union["SenderToReceiverData", int]):
        if isinstance(other, int):
            return self
        senders = torch.cat([self.senders, other.senders]).unique()
        receivers = torch.cat([self.receivers, other.receivers]).unique()
        y = self.y or other.y
        return SenderToReceiverData.from_data(senders, receivers, y)

    def __radd__(self, other):
        return self + other
