"""Run and audit MLP validation-only architecture selection.

The twelve candidates exactly match the historical MLP grid.  Each candidate
is trained with seeds 0, 1, and 2.  Selection uses only validation/f1; no test
dataloader or test task is run during the sweep.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "mlp_val_only"
RESULT_ROOT = ROOT / "results" / "mlp_val_only"
HIDDEN_DIMS = (64, 128)
LAYERS = (1, 2)
DROPOUTS = (0.1, 0.2, 0.3)
SEEDS = (0, 1, 2)
SELECTION_METRIC = "validation/f1"


def tag(hidden_dim: int, layers: int, dropout: float, seed: int) -> str:
    return f"h{hidden_dim}_l{layers}_d{str(dropout).replace('.', 'p')}_s{seed}"


def checkpoint_score(path: Path) -> tuple[float, int, int, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, state in payload.get("callbacks", {}).items():
        if "ModelCheckPoint" in key and state.get("monitor") == SELECTION_METRIC:
            score = state.get("best_model_score")
            if score is None:
                break
            return float(score), int(payload["epoch"]), int(payload["global_step"]), str(key)
    raise RuntimeError(f"No {SELECTION_METRIC} ModelCheckpoint score in {path}")


def run_one(python: Path, hidden_dim: int, layers: int, dropout: float, seed: int) -> dict[str, Any]:
    run_tag = tag(hidden_dim, layers, dropout, seed)
    run_dir = OUT_ROOT / run_tag
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            "main.py",
            f"+name=MLP_VAL_ONLY_{run_tag}",
            "experiment=exp_edge_recommendation",
            "dataset=elliptic_recommendation",
            "algorithm=mlp",
            "experiment.tasks=[training]",
            "experiment.validation.test_during_training=false",
            "experiment.training.checkpointing.monitor=validation/f1",
            "experiment.training.batch_size=1024",
            "experiment.training.data.num_workers=0",
            "experiment.validation.data.num_workers=0",
            f"algorithm.model.hidden_dim={hidden_dim}",
            f"algorithm.model.num_layers={layers}",
            f"algorithm.model.dropout={dropout}",
            f"seed={seed}",
            "wandb.mode=disabled",
            f"hydra.run.dir={run_dir.as_posix()}",
        ]
        env = os.environ.copy()
        env.setdefault("WANDB_SILENT", "true")
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (run_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(
                f"Run {run_tag} failed with exit code {completed.returncode}; "
                f"see {run_dir / 'run.log'}"
            )
        ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if len(ckpts) != 1:
        raise RuntimeError(f"Expected one checkpoint for {run_tag}, found {len(ckpts)}")

    config_path = run_dir / ".hydra" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["experiment"]["tasks"] == ["training"]
    assert config["experiment"]["validation"]["test_during_training"] is False
    assert config["experiment"]["training"]["checkpointing"]["monitor"] == SELECTION_METRIC
    log_path = run_dir / "run.log"
    log_text = log_path.read_text(encoding="utf-8")
    forbidden = ("final_test/", "test/f1", "test/HR", "test/NDCG")
    if any(metric in log_text for metric in forbidden):
        raise RuntimeError(f"Forbidden test metric found in {log_path}")

    score, epoch, global_step, callback = checkpoint_score(ckpts[0])
    return {
        "hidden_dim": hidden_dim,
        "layers": layers,
        "dropout": dropout,
        "activation": config["algorithm"]["model"]["activation"],
        "scoring": "sender_receiver_dot_product",
        "seed": seed,
        "validation_f1": score,
        "best_epoch": epoch,
        "global_step": global_step,
        "checkpoint": str(ckpts[0].relative_to(ROOT)),
        "resolved_config": str(config_path.relative_to(ROOT)),
        "checkpoint_callback": callback,
        "test_metric_scan": "PASS",
    }


def write_results(rows: list[dict[str, Any]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_path = RESULT_ROOT / "mlp_val_only_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: list[dict[str, Any]] = []
    for hidden_dim, layers, dropout in itertools.product(HIDDEN_DIMS, LAYERS, DROPOUTS):
        subset = [
            row for row in rows
            if row["hidden_dim"] == hidden_dim
            and row["layers"] == layers
            and math.isclose(row["dropout"], dropout)
        ]
        subset.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in subset] != list(SEEDS):
            raise RuntimeError(f"Incomplete seeds for hidden={hidden_dim}, layers={layers}, dropout={dropout}")
        scores = [row["validation_f1"] for row in subset]
        grouped.append(
            {
                "hidden_dim": hidden_dim,
                "layers": layers,
                "dropout": dropout,
                "activation": "ELU",
                "scoring": "sender_receiver_dot_product",
                "seed0_validation_f1": scores[0],
                "seed1_validation_f1": scores[1],
                "seed2_validation_f1": scores[2],
                "mean_validation_f1": statistics.fmean(scores),
                "sd_validation_f1": statistics.stdev(scores),
            }
        )

    grouped.sort(
        key=lambda row: (
            -row["mean_validation_f1"],
            row["sd_validation_f1"],
            row["layers"],
            row["hidden_dim"],
            row["dropout"],
        )
    )
    for rank, row in enumerate(grouped, 1):
        row["rank"] = rank

    summary_path = RESULT_ROOT / "mlp_val_only_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)

    sweep_path = ROOT / "configurations" / "sweep" / "subgraph_recommendation" / "tuning" / "MLP_val_only.yaml"
    selected = grouped[0]
    audit = {
        "protocol": {
            "training_split": "TRN",
            "selection_split": "VAL",
            "test_used_during_selection": False,
            "tasks": ["training"],
            "test_during_training": False,
            "selection_metric": SELECTION_METRIC,
            "seeds": list(SEEDS),
            "aggregation": "arithmetic mean across seeds 0, 1, and 2",
            "ranking": "descending mean validation/f1; exact ties: lower SD, fewer layers, smaller hidden dimension, lower dropout",
        },
        "search_space": {
            "hidden_dim": list(HIDDEN_DIMS),
            "num_layers": list(LAYERS),
            "dropout": list(DROPOUTS),
            "activation": ["ELU"],
            "scoring": ["sender_receiver_dot_product"],
            "input_dim": [43],
        },
        "training_protocol": {
            "training_examples": "TRN one-to-one samples after filter_1_1",
            "positive_examples": "suspicious label (y=1)",
            "negative_examples": "licit label (y=0)",
            "loss": "unweighted BCEWithLogitsLoss",
            "random_corruption": False,
            "in_batch_negative_sampling": False,
            "hard_negative_mining": False,
            "observed_filtered_counts": {
                "TRN_positive": 926,
                "TRN_negative": 45307,
                "VAL_positive": 127,
                "VAL_negative": 5713,
            },
        },
        "selected": selected,
        "completeness": {
            "expected_runs": 36,
            "observed_runs": len(rows),
            "all_test_metric_scans_pass": all(row["test_metric_scan"] == "PASS" for row in rows),
        },
        "sweep_config": str(sweep_path.relative_to(ROOT)),
        "sweep_config_sha256": hashlib.sha256(sweep_path.read_bytes()).hexdigest(),
    }
    (RESULT_ROOT / "selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# MLP VAL-only selection",
        "",
        "Selection metric: `validation/f1`; score is the arithmetic mean across seeds 0, 1, and 2.",
        "",
        "| Rank | Hidden | Layers | Dropout | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            "| {rank} | {hidden_dim} | {layers} | {dropout:.1f} | "
            "{seed0_validation_f1:.6f} | {seed1_validation_f1:.6f} | "
            "{seed2_validation_f1:.6f} | {mean_validation_f1:.6f} | "
            "{sd_validation_f1:.6f} |".format(**row)
        )
    lines.extend([
        "",
        f"Selected: layers={selected['layers']}, hidden={selected['hidden_dim']}, "
        f"dropout={selected['dropout']}, activation=ELU, scoring=sender/receiver dot product.",
    ])
    (RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    specs = list(itertools.product(HIDDEN_DIMS, LAYERS, DROPOUTS, SEEDS))
    if args.audit_only:
        for hidden_dim, layers, dropout, seed in specs:
            run_dir = OUT_ROOT / tag(hidden_dim, layers, dropout, seed)
            if not list((run_dir / "checkpoints").glob("*.ckpt")):
                raise FileNotFoundError(f"Missing completed run: {run_dir}")
        for index, spec in enumerate(specs, 1):
            rows.append(run_one(args.python, *spec))
            print(f"[{index:02d}/36] audit {tag(*spec)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, args.python, *spec): spec for spec in specs}
            for index, future in enumerate(as_completed(futures), 1):
                spec = futures[future]
                rows.append(future.result())
                print(f"[{index:02d}/36] complete {tag(*spec)}", flush=True)

    rows.sort(key=lambda row: (row["hidden_dim"], row["layers"], row["dropout"], row["seed"]))
    write_results(rows)
    selected = json.loads((RESULT_ROOT / "selection_audit.json").read_text(encoding="utf-8"))["selected"]
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
