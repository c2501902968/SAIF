#!/usr/bin/env python3
"""GPU-only Shared DeepSets validation selection launcher.

This file orchestrates the existing project training entrypoint. It does not
implement or modify model training, dataset splitting, or evaluation metrics.
The formal mode requires exactly one visible CUDA device and never runs TEST.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "deepsets_val_only_gpu"
RESULT_ROOT = ROOT / "results" / "deepsets_val_only_gpu"
DRY_RUN_ROOT = ROOT / "results" / "deepsets_val_only_gpu_dry_run"
SWEEP_PATH = (
    ROOT
    / "configurations"
    / "sweep"
    / "subgraph_classification"
    / "tuning"
    / "DS_val_only.yaml"
)

HIDDEN_DIMS = (64, 128)
LAYERS = (1, 2)
DROPOUTS = (0.1, 0.2, 0.3)
SEEDS = (0, 1, 2)
INPUT_DIM = 43
POOLING = "max"
ACTIVATION = "ELU"
SELECTION_METRIC = "validation/prauc"
EXPECTED_CONFIGS = 12
EXPECTED_RUNS = 36

RAW_FIELDS = [
    "hidden",
    "layers",
    "dropout",
    "pooling",
    "activation",
    "seed",
    "best_validation_prauc",
    "best_epoch",
    "global_step",
    "checkpoint_path",
    "status",
    "checkpoint_sha256",
    "resolved_config_path",
]

MANIFEST_FIELDS = [
    "run_index",
    "run_tag",
    "hidden",
    "layers",
    "dropout",
    "pooling",
    "activation",
    "seed",
    "output_dir",
    "command",
    "status",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "return_code",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_tag(hidden: int, layers: int, dropout: float, seed: int) -> str:
    return f"h{hidden}_l{layers}_d{str(dropout).replace('.', 'p')}_s{seed}"


def all_specs() -> list[tuple[int, int, float, int]]:
    return list(itertools.product(HIDDEN_DIMS, LAYERS, DROPOUTS, SEEDS))


def command_overrides(
    hidden: int, layers: int, dropout: float, seed: int, output_dir: Path
) -> list[str]:
    tag = run_tag(hidden, layers, dropout, seed)
    return [
        f"+name=DeepSets_GPU_VAL_ONLY_{tag}",
        "experiment=exp_subgraph_classification",
        "dataset=elliptic",
        "algorithm=deepsets",
        "experiment.tasks=[training]",
        "experiment.validation.test_during_training=false",
        f"experiment.training.checkpointing.monitor={SELECTION_METRIC}",
        "experiment.training.checkpointing.mode=max",
        "experiment.training.data.num_workers=0",
        "experiment.validation.data.num_workers=0",
        f"dataset.num_features={INPUT_DIM}",
        f"algorithm.model.hidden_dim={hidden}",
        f"algorithm.model.num_layers={layers}",
        f"algorithm.model.dropout={dropout}",
        f"algorithm.model.pool={POOLING}",
        f"algorithm.model.activation={ACTIVATION}",
        f"seed={seed}",
        "wandb.mode=disabled",
        f"hydra.run.dir={output_dir.as_posix()}",
    ]


def build_command(
    python: Path, hidden: int, layers: int, dropout: float, seed: int
) -> list[str]:
    output_dir = OUT_ROOT / run_tag(hidden, layers, dropout, seed)
    return [str(python), "main.py", *command_overrides(hidden, layers, dropout, seed, output_dir)]


def validate_sweep_yaml() -> None:
    sweep = yaml.safe_load(SWEEP_PATH.read_text(encoding="utf-8"))
    params = sweep["parameters"]
    assert sweep["metric"] == {"goal": "maximize", "name": SELECTION_METRIC}
    assert set(params["algorithm.model.hidden_dim"]["values"]) == set(HIDDEN_DIMS)
    assert set(params["algorithm.model.num_layers"]["values"]) == set(LAYERS)
    assert set(params["algorithm.model.dropout"]["values"]) == set(DROPOUTS)
    assert params["algorithm.model.pool"]["value"] == POOLING
    assert params["algorithm.model.activation"]["value"] == ACTIVATION
    assert set(params["seed"]["values"]) == set(SEEDS)
    assert params["dataset"]["value"] == "elliptic"
    assert params["algorithm"]["value"] == "deepsets"
    assert params["experiment"]["value"] == "exp_subgraph_classification"
    assert params["experiment.tasks"]["value"] == ["training"]
    assert params["experiment.validation.test_during_training"]["value"] is False
    assert params["experiment.training.checkpointing.monitor"]["value"] == SELECTION_METRIC


def validate_data_dependencies() -> None:
    required = [
        ROOT / "data" / "elliptic" / "processed" / "emb.pt",
        ROOT / "data" / "elliptic" / "processed" / "mask.pt",
        ROOT / "data" / "elliptic" / "processed" / "data.pt",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(
            "Missing processed Elliptic files. Upload these files before running:\n"
            + formatted
        )


def compose_and_validate_all_configs() -> None:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "configurations")
    ):
        for hidden, layers, dropout, seed in all_specs():
            output_dir = OUT_ROOT / run_tag(hidden, layers, dropout, seed)
            cfg = compose(
                config_name="config",
                overrides=command_overrides(
                    hidden, layers, dropout, seed, output_dir
                ),
            )
            assert list(cfg.experiment.tasks) == ["training"]
            assert cfg.experiment.validation.test_during_training is False
            assert cfg.experiment.training.checkpointing.monitor == SELECTION_METRIC
            assert cfg.experiment.training.checkpointing.mode == "max"
            assert cfg.dataset.num_features == INPUT_DIM
            assert cfg.algorithm.model.hidden_dim == hidden
            assert cfg.algorithm.model.num_layers == layers
            assert math.isclose(float(cfg.algorithm.model.dropout), dropout)
            assert cfg.algorithm.model.pool == POOLING
            assert cfg.algorithm.model.activation == ACTIVATION
            assert cfg.seed == seed


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_state() -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"])
    status = command_output(["git", "status", "--porcelain"])
    return {
        "commit": commit or "unavailable-no-git-metadata",
        "dirty": None if status is None else bool(status),
        "status": "unavailable-no-git-metadata" if status is None else status,
    }


def environment_snapshot(cuda_required: str) -> dict[str, Any]:
    try:
        import torch_geometric

        pyg_version = torch_geometric.__version__
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        pyg_version = f"unavailable: {exc}"

    device_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    return {
        "generated_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "pytorch_geometric": pyg_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": device_count,
        "gpu_models": gpu_names,
        "current_cuda_device": torch.cuda.current_device() if device_count else None,
        "git": git_state(),
        "CUDA_REQUIRED": cuda_required,
    }


def cuda_preflight() -> dict[str, Any]:
    print(f"Python version: {sys.version}", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA runtime version: {torch.version.cuda}", flush=True)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Exactly one CUDA device must be visible. Launch with "
            "CUDA_VISIBLE_DEVICES=<gpu-index>."
        )
    torch.cuda.set_device(0)
    probe = torch.ones(1, device="cuda")
    if probe.device.type != "cuda":  # pragma: no cover - defensive assertion
        raise RuntimeError("CUDA allocation probe did not run on CUDA.")
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"current CUDA device: {torch.cuda.current_device()}", flush=True)
    print("CUDA_REQUIRED = PASS", flush=True)
    return environment_snapshot("PASS")


def write_environment(snapshot: dict[str, Any]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "environment.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (RESULT_ROOT / "pip_freeze.txt").write_text(
        completed.stdout, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError("pip freeze failed; see results/deepsets_val_only_gpu/pip_freeze.txt")


def write_manifest(rows: list[dict[str, Any]], root: Path = RESULT_ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def resolved_config_check(
    config_path: Path, hidden: int, layers: int, dropout: float, seed: int
) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg["experiment"]["tasks"] == ["training"]
    assert cfg["experiment"]["validation"]["test_during_training"] is False
    assert cfg["experiment"]["training"]["checkpointing"]["monitor"] == SELECTION_METRIC
    assert cfg["experiment"]["training"]["checkpointing"]["mode"] == "max"
    assert cfg["dataset"]["num_features"] == INPUT_DIM
    assert cfg["algorithm"]["model"]["hidden_dim"] == hidden
    assert cfg["algorithm"]["model"]["num_layers"] == layers
    assert math.isclose(float(cfg["algorithm"]["model"]["dropout"]), dropout)
    assert cfg["algorithm"]["model"]["pool"] == POOLING
    assert cfg["algorithm"]["model"]["activation"] == ACTIVATION
    assert cfg["seed"] == seed


TEST_EXECUTION_PATTERNS = [
    re.compile(r"Executing task:\s*test\b", re.IGNORECASE),
    re.compile(r"\bTesting DataLoader\b", re.IGNORECASE),
    re.compile(r"\bfinal_test/", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z_])test/(?:prauc|f1|loss|accuracy|HR|NDCG)\b", re.IGNORECASE),
]
GPU_USED_PATTERN = re.compile(
    r"GPU available:\s*True(?:\s*\(cuda\))?,\s*used:\s*True", re.IGNORECASE
)


def scan_run_log(log_path: Path) -> tuple[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    hits = [pattern.pattern for pattern in TEST_EXECUTION_PATTERNS if pattern.search(text)]
    if hits:
        raise RuntimeError(f"TEST execution/metric evidence found in {log_path}: {hits}")
    if not GPU_USED_PATTERN.search(text):
        raise RuntimeError(
            f"Lightning did not confirm GPU usage in {log_path}; expected "
            "'GPU available: True (cuda), used: True'."
        )
    return "PASS", "PASS"


def checkpoint_score(path: Path) -> tuple[float, int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, state in payload.get("callbacks", {}).items():
        if "ModelCheckPoint" in key and state.get("monitor") == SELECTION_METRIC:
            score = state.get("best_model_score")
            if score is not None:
                return float(score), int(payload["epoch"]), int(payload["global_step"])
    raise RuntimeError(f"No {SELECTION_METRIC} checkpoint score in {path}")


def collect_run(
    hidden: int, layers: int, dropout: float, seed: int
) -> dict[str, Any]:
    tag = run_tag(hidden, layers, dropout, seed)
    run_dir = OUT_ROOT / tag
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected exactly one checkpoint for {tag}, found {len(checkpoints)}")
    config_path = run_dir / ".hydra" / "config.yaml"
    log_path = run_dir / "run.log"
    if not config_path.is_file() or not log_path.is_file():
        raise FileNotFoundError(f"Missing resolved config or run log for {tag}")
    resolved_config_check(config_path, hidden, layers, dropout, seed)
    test_scan, gpu_scan = scan_run_log(log_path)
    if test_scan != "PASS" or gpu_scan != "PASS":  # pragma: no cover
        raise RuntimeError(f"Integrity scan failed for {tag}")
    score, epoch, global_step = checkpoint_score(checkpoints[0])
    return {
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "pooling": POOLING,
        "activation": ACTIVATION,
        "seed": seed,
        "best_validation_prauc": score,
        "best_epoch": epoch,
        "global_step": global_step,
        "checkpoint_path": str(checkpoints[0].relative_to(ROOT)).replace("\\", "/"),
        "status": "complete",
        "checkpoint_sha256": sha256(checkpoints[0]),
        "resolved_config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
    }


def launch_run(command: list[str], run_dir: Path) -> tuple[str, float, int]:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if len(checkpoints) == 1:
        return "reused", 0.0, 0
    if len(checkpoints) > 1:
        raise RuntimeError(f"Multiple checkpoints found in {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(
        subprocess.list2cmdline(command) + "\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env.update(
        {
            "HYDRA_FULL_ERROR": "1",
            "PYTHONUNBUFFERED": "1",
            "WANDB_SILENT": "true",
        }
    )
    started = time.perf_counter()
    with (run_dir / "run.log").open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Training failed with code {completed.returncode}; see {run_dir / 'run.log'}"
        )
    return "completed", elapsed, completed.returncode


def write_results(rows: list[dict[str, Any]], environment: dict[str, Any]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if len(rows) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} raw rows, observed {len(rows)}")

    configs = {(r["hidden"], r["layers"], r["dropout"]) for r in rows}
    if len(configs) != EXPECTED_CONFIGS:
        raise RuntimeError(f"Expected {EXPECTED_CONFIGS} configs, observed {len(configs)}")
    for config in configs:
        seeds = {
            row["seed"]
            for row in rows
            if (row["hidden"], row["layers"], row["dropout"]) == config
        }
        if seeds != set(SEEDS):
            raise RuntimeError(f"Invalid seed set for {config}: {seeds}")

    rows.sort(key=lambda r: (r["hidden"], r["layers"], r["dropout"], r["seed"]))
    raw_path = RESULT_ROOT / "deepsets_val_only_gpu_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, Any]] = []
    for hidden, layers, dropout in sorted(configs):
        subset = [
            row
            for row in rows
            if row["hidden"] == hidden
            and row["layers"] == layers
            and math.isclose(float(row["dropout"]), dropout)
        ]
        subset.sort(key=lambda row: row["seed"])
        scores = [float(row["best_validation_prauc"]) for row in subset]
        summary.append(
            {
                "rank": 0,
                "hidden": hidden,
                "layers": layers,
                "dropout": dropout,
                "seed_0": scores[0],
                "seed_1": scores[1],
                "seed_2": scores[2],
                "mean_validation_prauc": statistics.fmean(scores),
                "sample_sd": statistics.stdev(scores),
            }
        )

    summary.sort(
        key=lambda row: (
            -row["mean_validation_prauc"],
            row["sample_sd"],
            row["layers"],
            row["hidden"],
            row["dropout"],
        )
    )
    for rank, row in enumerate(summary, 1):
        row["rank"] = rank

    summary_fields = [
        "rank",
        "hidden",
        "layers",
        "dropout",
        "seed_0",
        "seed_1",
        "seed_2",
        "mean_validation_prauc",
        "sample_sd",
    ]
    with (RESULT_ROOT / "deepsets_val_only_gpu_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    winner = summary[0]
    audit = {
        "generated_at": utc_now(),
        "protocol": {
            "training_split": "TRN",
            "selection_split": "VAL",
            "tasks": ["training"],
            "test_during_training": False,
            "selection_metric": SELECTION_METRIC,
            "input_dim": INPUT_DIM,
            "pooling": POOLING,
            "activation": ACTIVATION,
            "ranking": (
                "maximum mean validation/prauc; exact ties: smaller sample SD, "
                "fewer layers, smaller hidden dimension, smaller dropout"
            ),
            "lightning_gpu_enforcement": (
                "exactly one visible CUDA device plus mandatory post-run "
                "Lightning 'GPU available ... used: True' evidence"
            ),
        },
        "search_space": {
            "hidden": list(HIDDEN_DIMS),
            "layers": list(LAYERS),
            "dropout": list(DROPOUTS),
            "seeds": list(SEEDS),
        },
        "integrity": {
            "expected_runs": EXPECTED_RUNS,
            "observed_runs": len(rows),
            "expected_configs": EXPECTED_CONFIGS,
            "observed_configs": len(configs),
            "seed_set_each_config": list(SEEDS),
            "resolved_configs_checked": EXPECTED_RUNS,
            "TEST_SELECTION_SCAN": "PASS",
            "GPU_ASSERTION": "PASS",
            "CUDA_REQUIRED": environment["CUDA_REQUIRED"],
        },
        "winner": winner,
        "sweep_config": str(SWEEP_PATH.relative_to(ROOT)).replace("\\", "/"),
        "sweep_config_sha256": sha256(SWEEP_PATH),
    }
    (RESULT_ROOT / "selection_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    lines = [
        "# GPU Shared DeepSets VAL-only selection",
        "",
        "- Selection metric: `validation/prauc`",
        "- TEST selection scan: **PASS**",
        "- GPU assertion: **PASS**",
        f"- Runs: **{len(rows)}/{EXPECTED_RUNS}**",
        f"- Configurations: **{len(configs)}/{EXPECTED_CONFIGS}**",
        "- Seeds per configuration: **0, 1, 2**",
        "- Pooling: **max**; activation: **ELU**",
        "",
        "| Rank | Hidden | Layers | Dropout | Seed 0 | Seed 1 | Seed 2 | Mean VAL PR-AUC | Sample SD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {rank} | {hidden} | {layers} | {dropout:.1f} | {seed_0:.6f} | "
            "{seed_1:.6f} | {seed_2:.6f} | {mean_validation_prauc:.6f} | "
            "{sample_sd:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Winner",
            "",
            f"`hidden={winner['hidden']}, layers={winner['layers']}, "
            f"dropout={winner['dropout']}, pooling={POOLING}, activation={ACTIVATION}`",
            "",
            "Selection stopped after VAL ranking. No TEST, RevFilter/SAIF, HR, or NDCG experiment was run.",
        ]
    )
    (RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(python: Path, skip_config_resolution: bool) -> None:
    print("DRY_RUN_ONLY", flush=True)
    validate_sweep_yaml()
    validate_data_dependencies()
    if not skip_config_resolution:
        compose_and_validate_all_configs()
    manifests: list[dict[str, Any]] = []
    for index, (hidden, layers, dropout, seed) in enumerate(all_specs(), 1):
        command = build_command(python, hidden, layers, dropout, seed)
        tag = run_tag(hidden, layers, dropout, seed)
        manifests.append(
            {
                "run_index": index,
                "run_tag": tag,
                "hidden": hidden,
                "layers": layers,
                "dropout": dropout,
                "pooling": POOLING,
                "activation": ACTIVATION,
                "seed": seed,
                "output_dir": str((OUT_ROOT / tag).relative_to(ROOT)).replace("\\", "/"),
                "command": subprocess.list2cmdline(command),
                "status": "DRY_RUN_ONLY",
                "started_at": "",
                "finished_at": "",
                "elapsed_seconds": "",
                "return_code": "",
            }
        )
    write_manifest(manifests, DRY_RUN_ROOT)
    audit = {
        "mode": "DRY_RUN_ONLY",
        "formal_gpu_results_generated": False,
        "run_count": len(manifests),
        "configuration_count": len(
            {(r["hidden"], r["layers"], r["dropout"]) for r in manifests}
        ),
        "seeds": list(SEEDS),
        "config_resolution": "SKIPPED" if skip_config_resolution else "PASS",
        "sweep_validation": "PASS",
        "data_dependency_validation": "PASS",
    }
    (DRY_RUN_ROOT / "DRY_RUN_ONLY.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"36 planned runs written to {DRY_RUN_ROOT / 'run_manifest.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable containing the project's CUDA dependencies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and audit all 36 commands without CUDA or training.",
    )
    parser.add_argument(
        "--skip-config-resolution",
        action="store_true",
        help="With --dry-run only, skip Hydra composition (static checks still run).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit already-completed GPU outputs without launching training.",
    )
    args = parser.parse_args()
    if args.dry_run and args.audit_only:
        parser.error("--dry-run and --audit-only are mutually exclusive")

    validate_sweep_yaml()
    validate_data_dependencies()
    if args.dry_run:
        dry_run(args.python, args.skip_config_resolution)
        return

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        environment_path = RESULT_ROOT / "environment.json"
        if not environment_path.is_file():
            raise FileNotFoundError("Missing formal environment.json")
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        if environment.get("CUDA_REQUIRED") != "PASS":
            raise RuntimeError("Formal environment did not pass CUDA preflight")
    else:
        environment = cuda_preflight()
        write_environment(environment)

    manifests: list[dict[str, Any]] = []
    for index, (hidden, layers, dropout, seed) in enumerate(all_specs(), 1):
        tag = run_tag(hidden, layers, dropout, seed)
        command = build_command(args.python, hidden, layers, dropout, seed)
        manifests.append(
            {
                "run_index": index,
                "run_tag": tag,
                "hidden": hidden,
                "layers": layers,
                "dropout": dropout,
                "pooling": POOLING,
                "activation": ACTIVATION,
                "seed": seed,
                "output_dir": str((OUT_ROOT / tag).relative_to(ROOT)).replace("\\", "/"),
                "command": subprocess.list2cmdline(command),
                "status": "planned",
                "started_at": "",
                "finished_at": "",
                "elapsed_seconds": "",
                "return_code": "",
            }
        )
    write_manifest(manifests)

    rows: list[dict[str, Any]] = []
    for manifest, (hidden, layers, dropout, seed) in zip(manifests, all_specs()):
        tag = manifest["run_tag"]
        run_dir = OUT_ROOT / tag
        if args.audit_only:
            manifest["status"] = "auditing"
            write_manifest(manifests)
            mode = "audited"
            elapsed = 0.0
            return_code = 0
        else:
            print(f"[{manifest['run_index']:02d}/{EXPECTED_RUNS}] START {tag}", flush=True)
            manifest["status"] = "running"
            manifest["started_at"] = utc_now()
            write_manifest(manifests)
            try:
                mode, elapsed, return_code = launch_run(
                    build_command(args.python, hidden, layers, dropout, seed), run_dir
                )
            except Exception:
                manifest["status"] = "failed"
                manifest["finished_at"] = utc_now()
                manifest["return_code"] = 1
                write_manifest(manifests)
                raise
        row = collect_run(hidden, layers, dropout, seed)
        rows.append(row)
        manifest["status"] = mode
        manifest["finished_at"] = utc_now()
        manifest["elapsed_seconds"] = f"{elapsed:.3f}"
        manifest["return_code"] = return_code
        write_manifest(manifests)
        print(f"[{manifest['run_index']:02d}/{EXPECTED_RUNS}] PASS {tag} ({mode})", flush=True)

    write_results(rows, environment)
    print("36/36 runs", flush=True)
    print("12 unique configs", flush=True)
    print("TEST_SELECTION_SCAN = PASS", flush=True)
    print("GPU_ASSERTION = PASS", flush=True)
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
