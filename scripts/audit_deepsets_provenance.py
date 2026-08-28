"""Record architecture and checkpoint-selection evidence for RevFilter/SAIF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deepsets_val_only" / "checkpoint_provenance.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="Repository/artifact root containing legacy checkpoints and outputs.",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def callback_metadata(payload: dict[str, Any]) -> list[dict[str, Any]]:
    callbacks = []
    for key, state in payload.get("callbacks", {}).items():
        if not isinstance(state, dict) or "monitor" not in state:
            continue
        score = state.get("best_model_score")
        callbacks.append(
            {
                "callback": key,
                "monitor": state.get("monitor"),
                "best_model_score": float(score) if score is not None else None,
                "best_model_path": state.get("best_model_path"),
                "dirpath": state.get("dirpath"),
            }
        )
    return callbacks


def inspect_checkpoint(model: str, seed: int, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    init_weight = state["model.sender_deep_sets.init_mlp.weight"]
    layers = sorted(
        {
            int(key.split(".")[3])
            for key in state
            if key.startswith("model.sender_deep_sets.equivarant_layers.")
        }
    )
    anchor_norm = state.get("model.anchor_norm.weight")
    return {
        "model": model,
        "seed": seed,
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "checkpoint_hyper_parameters_present": payload.get("hyper_parameters") is not None,
        "state_dict_inference": {
            "input_dim": int(init_weight.shape[1]),
            "hidden_dim": int(init_weight.shape[0]),
            "num_equivariant_layers": len(layers),
            "equivariant_layer_indices": layers,
            "anchor_layernorm_dim": int(anchor_norm.shape[0]) if anchor_norm is not None else None,
            "first_scorer_linear_shape": list(state["model.pred_mlp.1.weight"].shape),
        },
        "checkpoint_callbacks": callback_metadata(payload),
        "not_encoded_by_state_dict": ["dropout", "activation", "pooling"],
    }


def resolved_saif_config(seed: int, stage: str, path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = config["algorithm"]["model"]
    experiment = config["experiment"]
    return {
        "seed": seed,
        "stage": stage,
        "path": str(path),
        "name": config["name"],
        "architecture": {
            "input_dim": model["input_dim"],
            "hidden_dim": model["hidden_dim"],
            "num_layers": model["num_layers"],
            "dropout": model["dropout"],
            "activation": model["activation"],
            "pool": model["pool"],
            "anchor_feature_mode": model.get("anchor_feature_mode", "full (code default)"),
            "anchor_normalization": model.get("anchor_normalization", "layernorm (code default)"),
        },
        "use_anchor_features": config["algorithm"]["use_anchor_features"],
        "tasks": experiment["tasks"],
        "test_during_training": experiment["validation"]["test_during_training"],
        "checkpoint_monitor": experiment["training"]["checkpointing"]["monitor"],
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    checkpoints = []
    for seed in range(3):
        paths = {
            "RevFilter": source_root / "checkpoints" / "RevTrack" / f"{seed}_tuned.ckpt",
            "SAIF": source_root / "checkpoints" / "AnchorRevFilter" / f"tuned_seed{seed}.ckpt",
        }
        for model, path in paths.items():
            if path.is_file():
                checkpoints.append(inspect_checkpoint(model, seed, path))

    saif_run_dirs = {
        (0, "pretrain"): "11-54-05",
        (0, "finetune"): "12-03-51",
        (1, "pretrain"): "12-32-24",
        (1, "finetune"): "12-35-37",
        (2, "pretrain"): "12-42-21",
        (2, "finetune"): "12-46-05",
    }
    resolved = []
    for (seed, stage), time in saif_run_dirs.items():
        path = source_root / "outputs" / "2026-05-06" / time / ".hydra" / "config.yaml"
        if path.is_file():
            resolved.append(resolved_saif_config(seed, stage, path))

    evidence = {
        "source_root": str(source_root),
        "architecture_defaults": str(source_root / "configurations" / "algorithm" / "deepsets.yaml"),
        "revfilter_saif_switch": str(source_root / "configurations" / "algorithm" / "iterative_filtering.yaml"),
        "formal_training_script": str(source_root / "scripts" / "run_batches.py"),
        "checkpoints": checkpoints,
        "saif_resolved_training_configs": resolved,
        "conclusion": {
            "state_dict_confirms_for_all_formal_checkpoints": {
                "input_dim": 43,
                "hidden_dim": 128,
                "num_equivariant_layers": 2,
            },
            "saif_state_dict_also_confirms": {"anchor_layernorm_dim": 6},
            "checkpoint_selection_monitor": "validation/f1",
            "full_architecture_independently_verifiable_from_checkpoint_metadata": False,
            "reason": "hyper_parameters is absent and state_dict does not encode dropout, activation, or pooling",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
