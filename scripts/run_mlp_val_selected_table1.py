"""Retrain the VAL-only selected MLP and evaluate all eight main settings."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "results" / "mlp_val_only" / "selection_audit.json"
FORMAL_ROOT = ROOT / "outputs" / "mlp_val_only_formal"
EVAL_ROOT = ROOT / "outputs" / "mlp_val_only_formal_eval"
RESULT_ROOT = ROOT / "results" / "mlp_val_only" / "formal_test"
SEEDS = (0, 1, 2)
SETTINGS = (
    "1+5@1",
    "1+10@1",
    "1+10@3",
    "1+100@3",
    "3+100@10",
    "3+1000@10",
    "10+1000@100",
    "10+10000@100",
)


def env() -> dict[str, str]:
    result = os.environ.copy()
    result["WANDB_SILENT"] = "true"
    result["OMP_NUM_THREADS"] = "1"
    result["MKL_NUM_THREADS"] = "1"
    wandb_root = EVAL_ROOT / "_wandb"
    paths = {
        "WANDB_DATA_DIR": wandb_root / "data",
        "WANDB_CACHE_DIR": wandb_root / "cache",
        "WANDB_CONFIG_DIR": wandb_root / "config",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        result[key] = str(path)
    return result


def selected() -> dict[str, Any]:
    return json.loads(SELECTION.read_text(encoding="utf-8"))["selected"]


def checkpoint_score(path: Path) -> tuple[float, int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for _, state in payload.get("callbacks", {}).items():
        if state.get("monitor") == "validation/f1" and state.get("best_model_score") is not None:
            return float(state["best_model_score"]), int(payload["epoch"]), int(payload["global_step"])
    raise RuntimeError(f"No validation/f1 checkpoint score in {path}")


def train_formal(python: Path, seed: int, choice: dict[str, Any]) -> dict[str, Any]:
    # The extra per-job parent prevents concurrent Hydra jobs from racing on
    # the repository-wide ``outputs/latest-run`` convenience link.
    run_dir = FORMAL_ROOT / f"seed{seed}" / "run"
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            "main.py",
            f"+name=MLP_VAL_ONLY_FORMAL_seed{seed}",
            "experiment=exp_edge_recommendation",
            "dataset=elliptic_recommendation",
            "algorithm=mlp",
            "experiment.tasks=[training]",
            "experiment.validation.test_during_training=false",
            "experiment.training.checkpointing.monitor=validation/f1",
            "experiment.training.batch_size=1024",
            "experiment.training.data.num_workers=0",
            "experiment.validation.data.num_workers=0",
            f"algorithm.model.num_layers={choice['layers']}",
            f"algorithm.model.hidden_dim={choice['hidden_dim']}",
            f"algorithm.model.dropout={choice['dropout']}",
            f"seed={seed}",
            "wandb.mode=disabled",
            f"hydra.run.dir={run_dir.as_posix()}",
        ]
        proc = subprocess.run(
            command, cwd=ROOT, env=env(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
        (run_dir / "run.log").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"Formal training seed {seed} failed; see {run_dir / 'run.log'}")
        ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if len(ckpts) != 1:
        raise RuntimeError(f"Expected one formal checkpoint for seed {seed}, found {len(ckpts)}")

    config_path = run_dir / ".hydra" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = config["algorithm"]["model"]
    assert config["seed"] == seed
    assert config["experiment"]["tasks"] == ["training"]
    assert config["experiment"]["validation"]["test_during_training"] is False
    assert config["experiment"]["training"]["checkpointing"]["monitor"] == "validation/f1"
    assert model["num_layers"] == choice["layers"]
    assert model["hidden_dim"] == choice["hidden_dim"]
    assert model["dropout"] == choice["dropout"]
    log_text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    if any(token in log_text for token in ("final_test/", "test/HR", "test/NDCG", "test/f1")):
        raise RuntimeError(f"Test metric leaked into formal training seed {seed}")
    score, epoch, step = checkpoint_score(ckpts[0])
    return {
        "training_seed": seed,
        "checkpoint": str(ckpts[0].relative_to(ROOT)),
        "resolved_config": str(config_path.relative_to(ROOT)),
        "validation_f1": score,
        "best_epoch": epoch,
        "global_step": step,
        "test_metric_scan": "PASS",
    }


def parse_setting(setting: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\+(\d+)@(\d+)", setting)
    if not match:
        raise ValueError(setting)
    return tuple(map(int, match.groups()))


def last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None


def evaluate_one(
    python: Path,
    formal: dict[str, Any],
    setting: str,
    choice: dict[str, Any],
) -> dict[str, Any]:
    train_seed = int(formal["training_seed"])
    tag = setting.replace("+", "p").replace("@", "at")
    run_dir = EVAL_ROOT / f"seed{train_seed}_{tag}" / "job" / "run"
    log_path = run_dir / "run.log"
    num_illicits, num_licits, top_k = parse_setting(setting)
    if not log_path.exists() or "AUDIT_RETURN_CODE=0" not in log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = ROOT / str(formal["checkpoint"])
        command = [
            str(python),
            "main.py",
            f"+name=MLP_VAL_ONLY_FORMAL_seed{train_seed}_{tag}",
            "experiment=exp_edge_recommendation",
            "dataset=elliptic_recommendation",
            "algorithm=mlp",
            "experiment.tasks=[test]",
            "experiment.validation.test_during_training=false",
            "experiment.test.batch_size=16",
            "experiment.test.data.num_workers=0",
            "dataset.num_samples=256",
            f"dataset.num_illicits={num_illicits}",
            f"dataset.num_licits={num_licits}",
            f"algorithm.top_k={top_k}",
            f"algorithm.model.num_layers={choice['layers']}",
            f"algorithm.model.hidden_dim={choice['hidden_dim']}",
            f"algorithm.model.dropout={choice['dropout']}",
            # Fixed evaluation seed materializes the same 256 matched instances
            # for all three independently trained checkpoints.
            "seed=0",
            # The legacy on_test_end hook logs a confusion-matrix table through
            # the W&B logger API. Offline mode keeps the run local while
            # providing that API; disabled mode falls back to CSVLogger and
            # crashes only after HR/NDCG have been computed.
            "wandb.mode=offline",
            f"load='{checkpoint.as_posix()}'",
            f"+shortcut={setting}",
            f"hydra.run.dir={run_dir.as_posix()}",
        ]
        proc = subprocess.run(
            command, cwd=ROOT, env=env(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
        text = proc.stdout + f"\nAUDIT_RETURN_CODE={proc.returncode}\n"
        log_path.write_text(text, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"Evaluation failed; inspect {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    hr = last_float(r"final_test/HR\D+([0-9]+(?:\.[0-9]+)?)", text)
    ndcg = last_float(r"final_test/NDCG\D+([0-9]+(?:\.[0-9]+)?)", text)
    density = last_float(r"Avg density:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if hr is None or ndcg is None:
        raise ValueError(f"Missing final metrics in {log_path}")
    config_path = run_dir / ".hydra" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["seed"] == 0
    assert config["dataset"]["num_samples"] == 256
    assert config["dataset"]["num_illicits"] == num_illicits
    assert config["dataset"]["num_licits"] == num_licits
    assert config["algorithm"]["top_k"] == top_k
    return {
        "training_seed": train_seed,
        "evaluation_seed": 0,
        "num_matched_test_instances": 256,
        "setting": setting,
        "checkpoint": formal["checkpoint"],
        "HR": hr,
        "NDCG": ndcg,
        "density": density,
        "resolved_config": str(config_path.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
    }


def write_results(formals: list[dict[str, Any]], rows: list[dict[str, Any]], choice: dict[str, Any]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "formal_checkpoint_provenance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(formals[0]))
        writer.writeheader()
        writer.writerows(formals)
    with (RESULT_ROOT / "mlp_val_selected_raw_24.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for setting in SETTINGS:
        group = sorted((row for row in rows if row["setting"] == setting), key=lambda row: row["training_seed"])
        hr = [float(row["HR"]) for row in group]
        ndcg = [float(row["NDCG"]) for row in group]
        summaries.append({
            "setting": setting,
            "seed0_HR": hr[0], "seed1_HR": hr[1], "seed2_HR": hr[2],
            "mean_HR": statistics.fmean(hr), "sd_HR": statistics.stdev(hr),
            "seed0_NDCG": ndcg[0], "seed1_NDCG": ndcg[1], "seed2_NDCG": ndcg[2],
            "mean_NDCG": statistics.fmean(ndcg), "sd_NDCG": statistics.stdev(ndcg),
        })
    with (RESULT_ROOT / "mlp_val_selected_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# MLP VAL-selected formal test",
        "",
        f"Architecture: layers={choice['layers']}, hidden={choice['hidden_dim']}, dropout={choice['dropout']}, activation=ELU, sender/receiver dot-product scoring.",
        "Each cell is arithmetic mean +/- sample SD across independently trained seeds 0, 1, and 2; evaluation seed is fixed to 0 for the same 256 instances.",
        "",
        "| Setting | HR | NDCG |",
        "|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['setting']} | {row['mean_HR']:.6f} +/- {row['sd_HR']:.6f} | "
            f"{row['mean_NDCG']:.6f} +/- {row['sd_NDCG']:.6f} |"
        )
    (RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--train-workers", type=int, default=3)
    parser.add_argument("--eval-workers", type=int, default=6)
    args = parser.parse_args()
    choice = selected()

    with ThreadPoolExecutor(max_workers=args.train_workers) as executor:
        futures = {executor.submit(train_formal, args.python, seed, choice): seed for seed in SEEDS}
        formals = []
        for future in as_completed(futures):
            formals.append(future.result())
            print(f"formal seed {futures[future]} complete", flush=True)
    formals.sort(key=lambda row: row["training_seed"])

    specs = [(formal, setting) for formal in formals for setting in SETTINGS]
    with ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
        futures = {
            executor.submit(evaluate_one, args.python, formal, setting, choice):
            (formal["training_seed"], setting)
            for formal, setting in specs
        }
        rows = []
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            seed, setting = futures[future]
            print(f"[{index:02d}/24] seed {seed} {setting} complete", flush=True)
    rows.sort(key=lambda row: (SETTINGS.index(row["setting"]), row["training_seed"]))
    write_results(formals, rows, choice)
    print((RESULT_ROOT / "RESULTS.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
