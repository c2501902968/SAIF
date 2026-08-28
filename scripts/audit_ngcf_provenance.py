"""Audit NGCF configuration, interaction graph, sweeps, and checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "ngcf_val_only" / "provenance_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=ROOT,
        help="Root containing any released legacy NGCF checkpoints.",
    )
    parser.add_argument("--out", type=Path, default=RESULT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_record(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", {})
    conv_indices = sorted(
        {
            int(key.split(".convs.", 1)[1].split(".", 1)[0])
            for key in state
            if ".convs." in key and ".lin_1.weight" in key
        }
    )
    shapes = {
        key: list(value.shape)
        for key, value in state.items()
        if any(
            token in key
            for token in (
                "feature_encoder.embedding.weight",
                "post_mlp.weight",
                "convs.0.lin_1.weight",
                "convs.0.lin_2.weight",
                "all_edge_index",
            )
        )
    }
    callbacks = []
    for key, state_value in payload.get("callbacks", {}).items():
        callbacks.append(
            {
                "callback": str(key),
                "monitor": state_value.get("monitor"),
                "best_model_score": (
                    None
                    if state_value.get("best_model_score") is None
                    else float(state_value["best_model_score"])
                ),
                "best_model_path": state_value.get("best_model_path"),
            }
        )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "hyper_parameters": payload.get("hyper_parameters"),
        "callbacks": callbacks,
        "selected_state_shapes": shapes,
        "architecture_inference": {
            "input_embedding_dim": shapes.get("model.feature_encoder.embedding.weight", [None, None])[-1],
            "num_propagation_layers": len(conv_indices),
            "post_projection_dim": shapes.get("model.post_mlp.weight", [None, None])[0],
            "ngcf_bilinear_message_parameters_present": any(".lin_2.weight" in key for key in state),
        },
        "not_recoverable_from_state_dict": ["dropout", "normalize", "selection_split"],
    }


def main() -> None:
    args = parse_args()
    legacy_root = args.legacy_root.resolve()
    default_path = ROOT / "configurations" / "algorithm" / "ngcf.yaml"
    tuning_path = ROOT / "configurations" / "sweep" / "subgraph_recommendation" / "tuning" / "NGCF.yaml"
    current = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    tuning = yaml.safe_load(tuning_path.read_text(encoding="utf-8"))
    edge_index = torch.load(ROOT / "data" / "elliptic" / "processed" / "edge_index.pt", weights_only=False)
    edges = [tuple(pair) for pair in edge_index.t().tolist()]

    released = sorted((legacy_root / "checkpoints" / "NGCF").glob("*.ckpt"))
    formal = sorted(
        (ROOT / "outputs" / "ngcf_val_only_formal").glob(
            "seed*/job/run/checkpoints/*.ckpt"
        )
    )
    report = {
        "current_default": {
            "path": str(default_path.relative_to(ROOT)),
            "sha256": sha256(default_path),
            "model": current["model"],
            "train_with_1_1": current["train_with_1_1"],
            "use_edge_index": current["use_edge_index"],
        },
        "historical_tuning": {
            "path": str(tuning_path.relative_to(ROOT)),
            "sha256": sha256(tuning_path),
            "method": tuning["method"],
            "metric": tuning["metric"],
            "search_space": {
                "num_layers": tuning["parameters"]["algorithm.model.num_layers"]["values"],
                "dropout": tuning["parameters"]["algorithm.model.dropout"]["values"],
                "normalize": tuning["parameters"]["algorithm.model.normalize"]["values"],
            },
            "seed": tuning["parameters"]["seed"]["value"],
            "shortcut": tuning["parameters"]["+shortcut"]["value"],
            "test_during_training": tuning["parameters"]["experiment.validation.test_during_training"]["value"],
            "resolved_tasks_from_experiment_default": ["training", "test"],
            "final_test_after_training": True,
            "local_sweep_winner_mapping_found": False,
            "certification": "The current NGCF configuration cannot be certified as independent of test-based hyperparameter selection.",
        },
        "interaction_graph": {
            "source": "TRN suspicious (y=1) one-to-one records only",
            "licit_edges_included": False,
            "validation_or_test_edges_included": False,
            "directed_input_records": True,
            "stored_graph": "undirected, coalesced by torch_geometric.utils.to_undirected",
            "stored_edge_columns": int(edge_index.shape[1]),
            "unique_stored_edge_columns": len(set(edges)),
            "is_symmetric": set(edges) == {(b, a) for a, b in edges},
            "observed_source_records": 926,
            "observed_unique_directed_pairs_before_undirecting": 454,
        },
        "released_checkpoints": [checkpoint_record(path) for path in released],
        "new_val_only_formal_checkpoints": [checkpoint_record(path) for path in formal],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
