"""Run the strict three-seed validation-only NGCF tuning grid."""

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
OUT_ROOT = ROOT / "outputs" / "ngcf_val_only"
RESULT_ROOT = ROOT / "results" / "ngcf_val_only"
LAYERS = (1, 2, 3, 4)
DROPOUTS = (0.1, 0.2, 0.3)
NORMALIZE = (False, True)
SEEDS = (0, 1, 2)
SELECTION_METRIC = "validation/f1"


def tag(layers: int, dropout: float, normalize: bool, seed: int) -> str:
    return f"l{layers}_d{str(dropout).replace('.', 'p')}_n{int(normalize)}_s{seed}"


def checkpoint_score(path: Path) -> tuple[float, int, int, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, state in payload.get("callbacks", {}).items():
        if "ModelCheckPoint" in key and state.get("monitor") == SELECTION_METRIC:
            score = state.get("best_model_score")
            if score is not None:
                return float(score), int(payload["epoch"]), int(payload["global_step"]), str(key)
    raise RuntimeError(f"No {SELECTION_METRIC} score in {path}")


def run_one(python: Path, layers: int, dropout: float, normalize: bool, seed: int) -> dict[str, Any]:
    run_tag = tag(layers, dropout, normalize, seed)
    # Two extra path levels keep main.py's latest-run link private per job.
    run_dir = OUT_ROOT / run_tag / "attempt2" / "job" / "run"
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    log_path = run_dir / "run.log"
    completed = (
        log_path.exists()
        and "AUDIT_RETURN_CODE=0" in log_path.read_text(encoding="utf-8", errors="replace")
    )
    if not ckpts or not completed:
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(python), "main.py", f"+name=NGCF_VAL_ONLY_{run_tag}",
            "experiment=exp_edge_recommendation",
            "dataset=elliptic_recommendation",
            "algorithm=ngcf",
            "experiment.tasks=[training]",
            "experiment.validation.test_during_training=false",
            "experiment.training.checkpointing.monitor=validation/f1",
            "experiment.training.batch_size=1024",
            "experiment.training.data.num_workers=0",
            "experiment.validation.data.num_workers=0",
            f"algorithm.model.num_layers={layers}",
            f"algorithm.model.dropout={dropout}",
            f"algorithm.model.normalize={'true' if normalize else 'false'}",
            f"seed={seed}",
            "wandb.mode=offline",
            f"hydra.run.dir={run_dir.as_posix()}",
        ]
        env = os.environ.copy()
        env["WANDB_SILENT"] = "true"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        wandb_root = OUT_ROOT / "_wandb"
        for key, subdir in (
            ("WANDB_DATA_DIR", "data"),
            ("WANDB_CACHE_DIR", "cache"),
            ("WANDB_CONFIG_DIR", "config"),
        ):
            path = wandb_root / subdir
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        proc = subprocess.run(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
        (run_dir / "run.log").write_text(
            proc.stdout + f"\nAUDIT_RETURN_CODE={proc.returncode}\n", encoding="utf-8"
        )
        if proc.returncode:
            raise RuntimeError(f"Run {run_tag} failed; see {run_dir / 'run.log'}")
        ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if len(ckpts) != 1:
        raise RuntimeError(f"Expected one checkpoint for {run_tag}, found {len(ckpts)}")

    config_path = run_dir / ".hydra" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = config["algorithm"]["model"]
    assert config["experiment"]["tasks"] == ["training"]
    assert config["experiment"]["validation"]["test_during_training"] is False
    assert config["experiment"]["training"]["checkpointing"]["monitor"] == SELECTION_METRIC
    assert model["num_layers"] == layers
    assert math.isclose(model["dropout"], dropout)
    assert model["normalize"] is normalize
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if any(token in text for token in ("final_test/", "test/HR", "test/NDCG", "test/f1")):
        raise RuntimeError(f"Forbidden TEST metric found in {log_path}")
    score, epoch, step, callback = checkpoint_score(ckpts[0])
    return {
        "layers": layers,
        "dropout": dropout,
        "normalize": normalize,
        "seed": seed,
        "validation_f1": score,
        "best_epoch": epoch,
        "global_step": step,
        "checkpoint": str(ckpts[0].relative_to(ROOT)),
        "resolved_config": str(config_path.relative_to(ROOT)),
        "checkpoint_callback": callback,
        "test_metric_scan": "PASS",
    }


def write_results(rows: list[dict[str, Any]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "ngcf_val_only_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = []
    for layers, dropout, normalize in itertools.product(LAYERS, DROPOUTS, NORMALIZE):
        subset = sorted(
            (row for row in rows if row["layers"] == layers
             and math.isclose(row["dropout"], dropout)
             and row["normalize"] is normalize),
            key=lambda row: row["seed"],
        )
        if [row["seed"] for row in subset] != list(SEEDS):
            raise RuntimeError(f"Incomplete seeds for {layers=}, {dropout=}, {normalize=}")
        scores = [row["validation_f1"] for row in subset]
        grouped.append({
            "layers": layers,
            "dropout": dropout,
            "normalize": normalize,
            "seed0_validation_f1": scores[0],
            "seed1_validation_f1": scores[1],
            "seed2_validation_f1": scores[2],
            "mean_validation_f1": statistics.fmean(scores),
            "sd_validation_f1": statistics.stdev(scores),
        })

    # Exact ties: lower SD, fewer layers, lower dropout, then normalize=False.
    grouped.sort(key=lambda row: (
        -row["mean_validation_f1"], row["sd_validation_f1"],
        row["layers"], row["dropout"], row["normalize"],
    ))
    for rank, row in enumerate(grouped, 1):
        row["rank"] = rank
    with (RESULT_ROOT / "ngcf_val_only_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)

    sweep_path = ROOT / "configurations" / "sweep" / "subgraph_recommendation" / "tuning" / "NGCF_val_only.yaml"
    audit = {
        "protocol": {
            "training_split": "TRN", "selection_split": "VAL",
            "test_used_during_selection": False, "tasks": ["training"],
            "test_during_training": False, "selection_metric": SELECTION_METRIC,
            "seeds": list(SEEDS),
            "aggregation": "arithmetic mean across seeds 0, 1, and 2",
            "ranking": "descending mean validation/f1; exact ties: lower SD, fewer layers, lower dropout, normalize=False",
        },
        "search_space": {
            "num_layers": list(LAYERS), "dropout": list(DROPOUTS),
            "normalize": [True, False],
        },
        "fixed_model": {
            "input_embedding_dim": 43, "conv": "NGCF", "aggregation": "concat",
            "activation": "LeakyReLU", "scoring": "sender_receiver_dot_product",
            "objective": "unweighted BCEWithLogitsLoss",
            "optimizer": "Adam", "learning_rate": 0.001,
        },
        "interaction_graph": {
            "source": "TRN suspicious one-to-one pairs only",
            "stored_as": "undirected coalesced edge_index", "stored_edges": 885,
            "licit_edges": False, "VAL_or_TST_edges": False,
        },
        "selected": grouped[0],
        "completeness": {
            "expected_runs": 72, "observed_runs": len(rows),
            "all_test_metric_scans_pass": all(row["test_metric_scan"] == "PASS" for row in rows),
        },
        "sweep_config": str(sweep_path.relative_to(ROOT)),
        "sweep_config_sha256": hashlib.sha256(sweep_path.read_bytes()).hexdigest(),
    }
    (RESULT_ROOT / "selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# NGCF VAL-only selection", "",
        "Selection metric: `validation/f1`; arithmetic mean across seeds 0, 1, and 2.", "",
        "| Rank | Layers | Dropout | Normalize | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            "| {rank} | {layers} | {dropout:.1f} | {normalize} | "
            "{seed0_validation_f1:.6f} | {seed1_validation_f1:.6f} | "
            "{seed2_validation_f1:.6f} | {mean_validation_f1:.6f} | "
            "{sd_validation_f1:.6f} |".format(**row)
        )
    winner = grouped[0]
    lines.extend(["", f"Selected: layers={winner['layers']}, dropout={winner['dropout']}, normalize={winner['normalize']}."])
    (RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    specs = list(itertools.product(LAYERS, DROPOUTS, NORMALIZE, SEEDS))
    if args.audit_only:
        rows = []
        for index, spec in enumerate(specs, 1):
            rows.append(run_one(args.python, *spec))
            print(f"[{index:02d}/72] audit {tag(*spec)}", flush=True)
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, args.python, *spec): spec for spec in specs}
            for index, future in enumerate(as_completed(futures), 1):
                spec = futures[future]
                rows.append(future.result())
                print(f"[{index:02d}/72] complete {tag(*spec)}", flush=True)
    rows.sort(key=lambda row: (row["layers"], row["dropout"], row["normalize"], row["seed"]))
    write_results(rows)
    print(json.dumps(json.loads((RESULT_ROOT / "selection_audit.json").read_text(encoding="utf-8"))["selected"], indent=2))


if __name__ == "__main__":
    main()
