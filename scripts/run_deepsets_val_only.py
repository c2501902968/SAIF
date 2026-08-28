"""Run and audit the shared DeepSets validation-only architecture selection.

The architecture candidate values match the historical DeepSets sweep, except
that pooling is fixed to max.  Twelve configurations are each repeated for
seeds 0, 1, and 2.  No test dataloader or final test task is run.
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
OUT_ROOT = ROOT / "outputs" / "deepsets_val_only"
RESULT_ROOT = ROOT / "results" / "deepsets_val_only"
HIDDEN_DIMS = (64, 128)
LAYERS = (1, 2)
DROPOUTS = (0.1, 0.2, 0.3)
SEEDS = (0, 1, 2)
POOL = "max"
SELECTION_METRIC = "validation/prauc"


def tag(hidden_dim: int, layers: int, dropout: float, seed: int) -> str:
    return f"h{hidden_dim}_l{layers}_d{str(dropout).replace('.', 'p')}_s{seed}"


def checkpoint_score(path: Path) -> tuple[float, int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, state in payload.get("callbacks", {}).items():
        if "ModelCheckPoint" in key and state.get("monitor") == SELECTION_METRIC:
            score = state.get("best_model_score")
            if score is None:
                break
            return float(score), int(payload["epoch"]), int(payload["global_step"])
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
            f"+name=DeepSets_VAL_ONLY_{run_tag}",
            "experiment=exp_subgraph_classification",
            "dataset=elliptic",
            "algorithm=deepsets",
            "experiment.tasks=[training]",
            "experiment.validation.test_during_training=false",
            "experiment.training.checkpointing.monitor=validation/prauc",
            "experiment.training.data.num_workers=0",
            "experiment.validation.data.num_workers=0",
            f"algorithm.model.hidden_dim={hidden_dim}",
            f"algorithm.model.num_layers={layers}",
            f"algorithm.model.dropout={dropout}",
            "algorithm.model.pool=max",
            f"seed={seed}",
            "wandb.mode=disabled",
            f"hydra.run.dir={run_dir.as_posix()}",
        ]
        env = os.environ.copy()
        env.setdefault("WANDB_SILENT", "true")
        # Small per-batch matrix operations are faster and more predictable on
        # this CPU-only Windows audit host without large BLAS thread pools.
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
    assert config["algorithm"]["model"]["pool"] == POOL
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    forbidden = ("final_test/", "test/prauc", "test/f1", "test/HR", "test/NDCG")
    if any(metric in log_text for metric in forbidden):
        raise RuntimeError(f"Forbidden test metric found in {run_dir / 'run.log'}")

    score, epoch, global_step = checkpoint_score(ckpts[0])
    return {
        "hidden_dim": hidden_dim,
        "layers": layers,
        "dropout": dropout,
        "pool": POOL,
        "seed": seed,
        "validation_prauc": score,
        "best_epoch": epoch,
        "global_step": global_step,
        "checkpoint": str(ckpts[0].relative_to(ROOT)),
        "resolved_config": str(config_path.relative_to(ROOT)),
        "test_metric_scan": "PASS",
    }


def write_results(rows: list[dict[str, Any]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_path = RESULT_ROOT / "deepsets_val_only_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: list[dict[str, Any]] = []
    for hidden_dim, layers, dropout in itertools.product(HIDDEN_DIMS, LAYERS, DROPOUTS):
        subset = [
            row
            for row in rows
            if row["hidden_dim"] == hidden_dim
            and row["layers"] == layers
            and math.isclose(row["dropout"], dropout)
        ]
        subset.sort(key=lambda row: row["seed"])
        scores = [row["validation_prauc"] for row in subset]
        grouped.append(
            {
                "hidden_dim": hidden_dim,
                "layers": layers,
                "dropout": dropout,
                "pool": POOL,
                "seed0_validation_prauc": scores[0],
                "seed1_validation_prauc": scores[1],
                "seed2_validation_prauc": scores[2],
                "mean_validation_prauc": statistics.fmean(scores),
                "sd_validation_prauc": statistics.stdev(scores),
            }
        )

    # Predeclared selection rule: descending mean validation PR-AUC. Exact
    # numeric ties prefer lower SD, then fewer layers, smaller hidden width,
    # and lower dropout.
    grouped.sort(
        key=lambda row: (
            -row["mean_validation_prauc"],
            row["sd_validation_prauc"],
            row["layers"],
            row["hidden_dim"],
            row["dropout"],
        )
    )
    for rank, row in enumerate(grouped, 1):
        row["rank"] = rank

    summary_path = RESULT_ROOT / "deepsets_val_only_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)

    sweep_path = ROOT / "configurations" / "sweep" / "subgraph_classification" / "tuning" / "DS_val_only.yaml"
    sweep_sha256 = hashlib.sha256(sweep_path.read_bytes()).hexdigest()
    selected = grouped[0]
    audit = {
        "protocol": {
            "training_split": "training",
            "selection_split": "validation",
            "test_used_during_selection": False,
            "tasks": ["training"],
            "test_during_training": False,
            "selection_metric": SELECTION_METRIC,
            "pooling": POOL,
            "seeds": list(SEEDS),
            "aggregation": "arithmetic mean across seeds 0, 1, and 2",
            "ranking": "descending mean validation/prauc; exact ties: lower SD, fewer layers, smaller hidden dimension, lower dropout",
        },
        "search_space": {
            "hidden_dim": list(HIDDEN_DIMS),
            "num_layers": list(LAYERS),
            "dropout": list(DROPOUTS),
            "pool": [POOL],
            "activation": ["ELU"],
            "input_dim": [43],
        },
        "selected": selected,
        "completeness": {
            "expected_runs": len(HIDDEN_DIMS) * len(LAYERS) * len(DROPOUTS) * len(SEEDS),
            "observed_runs": len(rows),
            "all_test_metric_scans_pass": all(row["test_metric_scan"] == "PASS" for row in rows),
        },
        "sweep_config": str(sweep_path.relative_to(ROOT)),
        "sweep_config_sha256": sweep_sha256,
    }
    (RESULT_ROOT / "selection_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    lines = [
        "# Shared DeepSets VAL-only selection",
        "",
        "Selection metric: `validation/prauc`; pooling fixed to `max`; score is mean across seeds 0, 1, and 2.",
        "",
        "| Rank | Hidden | Layers | Dropout | Seed 0 | Seed 1 | Seed 2 | Mean | SD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            "| {rank} | {hidden_dim} | {layers} | {dropout:.1f} | "
            "{seed0_validation_prauc:.6f} | {seed1_validation_prauc:.6f} | "
            "{seed2_validation_prauc:.6f} | {mean_validation_prauc:.6f} | "
            "{sd_validation_prauc:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Selected: hidden={selected['hidden_dim']}, layers={selected['layers']}, "
            f"dropout={selected['dropout']}, pool={POOL}.",
        ]
    )
    (RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable with the project dependencies installed.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Do not launch missing runs; only audit completed output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Maximum number of independent CPU training processes.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    specs = list(itertools.product(HIDDEN_DIMS, LAYERS, DROPOUTS, SEEDS))
    for hidden_dim, layers, dropout, seed in specs:
        run_dir = OUT_ROOT / tag(hidden_dim, layers, dropout, seed)
        if args.audit_only and not list((run_dir / "checkpoints").glob("*.ckpt")):
            raise FileNotFoundError(f"Missing completed run: {run_dir}")
    if args.audit_only:
        for index, (hidden_dim, layers, dropout, seed) in enumerate(specs, 1):
            print(f"[{index:02d}/36] audit {tag(hidden_dim, layers, dropout, seed)}", flush=True)
            rows.append(run_one(args.python, hidden_dim, layers, dropout, seed))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_one, args.python, hidden_dim, layers, dropout, seed):
                (hidden_dim, layers, dropout, seed)
                for hidden_dim, layers, dropout, seed in specs
            }
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
