"""Record MLP configuration, historical sweep, and checkpoint provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "mlp_val_only" / "provenance_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=ROOT,
        help="Root containing any released legacy MLP checkpoints.",
    )
    parser.add_argument("--out", type=Path, default=RESULT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def callback_metadata(callbacks: dict[Any, Any]) -> list[dict[str, Any]]:
    records = []
    for key, state in callbacks.items():
        records.append(
            {
                "callback": str(key),
                "monitor": state.get("monitor"),
                "mode": state.get("mode"),
                "best_model_score": (
                    None if state.get("best_model_score") is None else float(state["best_model_score"])
                ),
                "best_model_path": state.get("best_model_path"),
            }
        )
    return records


def checkpoint_record(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", {})
    tower_shapes = {
        key: list(value.shape)
        for key, value in state.items()
        if "sender_mlp.layers" in key or "receiver_mlp.layers" in key
    }
    linear_weights = {
        key: shape for key, shape in tower_shapes.items() if key.endswith("weight") and len(shape) == 2
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "hyper_parameters": payload.get("hyper_parameters"),
        "callbacks": callback_metadata(payload.get("callbacks", {})),
        "tower_linear_weight_shapes": linear_weights,
        "architecture_inference": {
            "sender_receiver_separate_towers": any("sender_mlp" in key for key in linear_weights)
            and any("receiver_mlp" in key for key in linear_weights),
            "input_dim": 43 if any(shape[-1] == 43 for shape in linear_weights.values()) else None,
            "hidden_or_output_dim": 128 if any(shape[0] == 128 for shape in linear_weights.values()) else None,
            "linear_layer_count_per_tower": sum("sender_mlp" in key for key in linear_weights),
        },
        "not_encoded_by_state_dict": ["dropout_probability", "activation_name", "selection_split"],
    }


def main() -> None:
    args = parse_args()
    legacy_root = args.legacy_root.resolve()
    current_config_path = ROOT / "configurations" / "algorithm" / "mlp.yaml"
    historical_sweep_path = ROOT / "configurations" / "sweep" / "subgraph_recommendation" / "tuning" / "MLP.yaml"
    current = yaml.safe_load(current_config_path.read_text(encoding="utf-8"))
    historical = yaml.safe_load(historical_sweep_path.read_text(encoding="utf-8"))

    released_paths = sorted((legacy_root / "checkpoints" / "MLP").glob("*.ckpt"))
    existing_formal_paths = sorted((ROOT / "outputs" / "mlp_final").glob("seed*/checkpoints/*.ckpt"))
    new_formal_paths = sorted(
        (ROOT / "outputs" / "mlp_val_only_formal").glob("seed*/run/checkpoints/*.ckpt")
    )
    report = {
        "current_formal_config": {
            "path": str(current_config_path.relative_to(ROOT)),
            "sha256": sha256(current_config_path),
            "model": current["model"],
            "train_with_1_1": current["train_with_1_1"],
            "scoring": "separate sender/receiver towers followed by dot product",
        },
        "historical_sweep": {
            "path": str(historical_sweep_path.relative_to(ROOT)),
            "sha256": sha256(historical_sweep_path),
            "method": historical["method"],
            "metric": historical["metric"],
            "candidates": {
                "num_layers": historical["parameters"]["algorithm.model.num_layers"]["values"],
                "hidden_dim": historical["parameters"]["algorithm.model.hidden_dim"]["values"],
                "dropout": historical["parameters"]["algorithm.model.dropout"]["values"],
            },
            "seed": historical["parameters"]["seed"]["value"],
            "shortcut": historical["parameters"]["+shortcut"]["value"],
            "experiment_tasks_override_present": "experiment.tasks" in historical["parameters"],
            "test_during_training_override_present": "experiment.validation.test_during_training" in historical["parameters"],
            "resolved_from_defaults": {
                "experiment_tasks": ["training", "test"],
                "test_during_training": True,
                "checkpoint_monitor": "validation/f1",
            },
            "local_wandb_run_or_winner_manifest_found": False,
            "certification": "The current MLP configuration cannot be certified as independent of test-based hyperparameter selection.",
        },
        "released_checkpoints": [checkpoint_record(path) for path in released_paths],
        "preexisting_val_selected_formal_checkpoints": [
            checkpoint_record(path) for path in existing_formal_paths
        ],
        "new_three_seed_val_only_formal_checkpoints": [
            checkpoint_record(path) for path in new_formal_paths
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
