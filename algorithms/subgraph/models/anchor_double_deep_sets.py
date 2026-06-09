import torch
import torch.nn as nn
from omegaconf import DictConfig

from .activation import activation_registry
from .deep_sets import DeepSets
from .feature_encoder import NodeIdFeatureEncoder


class AnchorAwareDoubleDeepSets(nn.Module):
    """
    DoubleDeepSets with lightweight structural anchor features.

    RevFilter scores candidate sender/receiver groups during iterative
    filtering. Besides the sender and receiver set embeddings, these candidates
    have rule-relevant group structure: branching size, merging size, candidate
    edge volume, and sender/receiver imbalance. The features below expose those
    anchors without running an expensive GNN over the background graph.
    """

    anchor_feature_dims = {
        "full": 6,
        "size_only": 3,
        "no_balance": 4,
        "no_density": 5,
    }

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.emb_path = cfg.emb_path
        self.num_classes = cfg.num_classes
        self.hidden_dim = cfg.hidden_dim
        self.activation_type = cfg.activation
        self.dropout = cfg.dropout
        self.cfg = cfg
        self.anchor_feature_mode = getattr(cfg, "anchor_feature_mode", "full")
        self.anchor_normalization = getattr(cfg, "anchor_normalization", "layernorm")
        if self.anchor_feature_mode not in self.anchor_feature_dims:
            raise ValueError(
                f"Unknown anchor_feature_mode={self.anchor_feature_mode}; "
                f"expected one of {sorted(self.anchor_feature_dims)}"
            )
        if self.anchor_normalization not in {"layernorm", "none"}:
            raise ValueError(
                f"Unknown anchor_normalization={self.anchor_normalization}; "
                "expected 'layernorm' or 'none'."
            )
        self.anchor_feature_dim = self.anchor_feature_dims[self.anchor_feature_mode]
        self._build_model()

    def _build_model(self):
        self.feature_encoder = NodeIdFeatureEncoder(self.emb_path)
        self.sender_deep_sets = DeepSets(self.cfg)
        self.receiver_deep_sets = DeepSets(self.cfg)
        self.anchor_norm = (
            nn.LayerNorm(self.anchor_feature_dim)
            if self.anchor_normalization == "layernorm"
            else nn.Identity()
        )
        self.pred_mlp = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2 + self.anchor_feature_dim, self.hidden_dim),
            activation_registry[self.activation_type](),
            nn.Linear(self.hidden_dim, self.num_classes if self.num_classes > 2 else 1),
        )

    def _anchor_features(
        self,
        senders_batch: torch.Tensor,
        receivers_batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        device = senders_batch.device
        sender_count = torch.bincount(senders_batch, minlength=batch_size).to(
            device=device,
            dtype=torch.float,
        )
        receiver_count = torch.bincount(receivers_batch, minlength=batch_size).to(
            device=device,
            dtype=torch.float,
        )
        candidate_edges = sender_count * receiver_count
        log_sender = torch.log1p(sender_count)
        log_receiver = torch.log1p(receiver_count)
        log_edges = torch.log1p(candidate_edges)
        log_balance = log_sender - log_receiver
        abs_log_balance = torch.abs(log_balance)
        inverse_density_proxy = 1.0 / candidate_edges.clamp_min(1.0)
        size_features = [log_sender, log_receiver, log_edges]

        match self.anchor_feature_mode:
            case "full":
                features = size_features + [
                    log_balance,
                    abs_log_balance,
                    inverse_density_proxy,
                ]
            case "size_only":
                features = size_features
            case "no_balance":
                features = size_features + [inverse_density_proxy]
            case "no_density":
                features = size_features + [log_balance, abs_log_balance]
            case _:
                raise RuntimeError(f"Unhandled anchor_feature_mode={self.anchor_feature_mode}")

        return torch.stack(features, dim=-1)

    def forward(self, batched_data):
        senders, receivers, senders_batch, receivers_batch = (
            batched_data.senders,
            batched_data.receivers,
            batched_data.senders_batch,
            batched_data.receivers_batch,
        )
        batch_size = int(batched_data.num_graphs)
        senders = self.feature_encoder(senders)
        receivers = self.feature_encoder(receivers)
        senders = self.sender_deep_sets(senders, senders_batch)
        receivers = self.receiver_deep_sets(receivers, receivers_batch)
        anchors = self.anchor_norm(
            self._anchor_features(senders_batch, receivers_batch, batch_size)
        )
        return self.pred_mlp(torch.cat([senders, receivers, anchors], dim=-1))
