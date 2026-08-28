#!/usr/bin/env python3
"""Frozen d0p3 component ablation and pooling-sensitivity experiments.

The launcher is restartable and has two independent phases:

* ablation: a 2x2 design over anchor features (off/on) and Stage-2
  fine-tuning (off/on), using the already-trained final max-pooling
  checkpoints from both stages;
* pooling: max/mean/sum sensitivity.  Max reuses the final checkpoints;
  mean and sum are independently trained with the same two-stage protocol.

Only the two predeclared large evaluation settings are used.  Model training
never runs TEST, and no result is selected or discarded using TEST metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAIN_CHECKPOINT_ROOT = ROOT / "checkpoints" / "final_revfilter_saif_128_l1_d0p3"
MAIN_RESULT_ROOT = ROOT / "results" / "final_revfilter_saif_128_l1_d0p3"

ABLATION_OUTPUT_ROOT = ROOT / "outputs" / "final_ablation_128_l1_d0p3"
ABLATION_RESULT_ROOT = ROOT / "results" / "final_ablation_128_l1_d0p3"

POOL_OUTPUT_ROOT = ROOT / "outputs" / "final_pooling_sensitivity_128_l1_d0p3"
POOL_RESULT_ROOT = ROOT / "results" / "final_pooling_sensitivity_128_l1_d0p3"
POOL_CHECKPOINT_ROOT = ROOT / "checkpoints" / "final_pooling_sensitivity_128_l1_d0p3"

SETTINGS = ["10+1000@100", "10+10000@100"]
SEEDS = [0, 1, 2]
POOLS = ["max", "mean", "sum"]
TRAINED_POOLS = ["mean", "sum"]

METHODS = {
    "RevFilter": ["algorithm.use_anchor_features=false"],
    "SAIF": [
        "algorithm.use_anchor_features=true",
        "algorithm.model.anchor_feature_mode=full",
        "+algorithm.model.anchor_fusion_mode=full",
        "algorithm.model.anchor_input_dim=6",
        "algorithm.model.anchor_normalization=layernorm",
        "algorithm.model.anchor_control_mode=normal",
    ],
}

FROZEN_COMMON = [
    "algorithm.model.hidden_dim=128",
    "algorithm.model.num_layers=1",
    "algorithm.model.dropout=0.3",
    "algorithm.model.activation=ELU",
    "algorithm.keep_multiplier=1.5",
    "dataset.augment.gamma=0.4",
    "dataset.augment.min=1",
    "dataset.augment.max=20",
    "algorithm.train_with_1_1=false",
    "algorithm.candidate_order=original",
    "algorithm.candidate_order_seed=0",
]

ABLATION_VARIANTS = {
    "RevFilter-S1": ("RevFilter", "stage1"),
    "RevFilter-S2": ("RevFilter", "stage2"),
    "SAIF-S1": ("SAIF", "stage1"),
    "Full-SAIF": ("SAIF", "stage2"),
}

ABLATION_CONTRASTS = [
    ("Anchor effect after Stage-1", "RevFilter-S1", "SAIF-S1"),
    ("Anchor effect after Stage-2", "RevFilter-S2", "Full-SAIF"),
    ("Stage-2 effect on RevFilter", "RevFilter-S1", "RevFilter-S2"),
    ("Stage-2 effect on SAIF", "SAIF-S1", "Full-SAIF"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tag(value: str) -> str:
    return (
        value.replace("+", "p")
        .replace("@", "at")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha256() -> dict[str, str]:
    paths = [
        "scripts/run_final_ablation_pooling.py",
        "scripts/paired_instance_wilcoxon_bh.py",
        "algorithms/subgraph/iterative_filtering_algo.py",
        "algorithms/subgraph/models/anchor_double_deep_sets.py",
        "algorithms/subgraph/models/double_deep_sets.py",
        "algorithms/subgraph/models/deep_sets.py",
        "algorithms/subgraph/models/pool.py",
        "datasets/elliptic/dataset.py",
        "configurations/algorithm/iterative_filtering.yaml",
        "configurations/algorithm/deepsets.yaml",
        "configurations/dataset/elliptic_recommendation.yaml",
        "configurations/experiment/exp_edge_recommendation.yaml",
    ]
    return {relative: sha256(ROOT / relative) for relative in paths}


def cuda_preflight() -> dict:
    import torch

    info = {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_device_count": torch.cuda.device_count(),
        "hostname": platform.node(),
        "platform": platform.platform(),
    }
    print(f"Python version: {sys.version}", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA runtime version: {torch.version.cuda}", flush=True)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for these formal experiments")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU with CUDA_VISIBLE_DEVICES; "
            f"found {torch.cuda.device_count()} visible devices"
        )
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["current_cuda_device"] = torch.cuda.current_device()
    print(f"GPU name: {info['gpu_name']}", flush=True)
    print("CUDA_REQUIRED = PASS", flush=True)
    return info


def write_environment(root: Path, cuda: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "environment.json").write_text(
        json.dumps(cuda, indent=2) + "\n", encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (root / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")


def main_checkpoint(method: str, seed: int, stage: str) -> Path:
    return MAIN_CHECKPOINT_ROOT / method / f"seed{seed}_{stage}.ckpt"


def load_main_provenance() -> dict[tuple[str, int], dict]:
    path = MAIN_RESULT_ROOT / "checkpoint_provenance.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing final checkpoint provenance: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    indexed = {(str(row["method"]), int(row["seed"])): row for row in rows}
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(indexed) != expected:
        raise RuntimeError(
            f"Final provenance grid mismatch: missing={expected-set(indexed)}, "
            f"extra={set(indexed)-expected}"
        )
    return indexed


def assert_main_checkpoints() -> dict[tuple[str, int], dict]:
    provenance = load_main_provenance()
    integrity_path = MAIN_RESULT_ROOT / "integrity.json"
    if not integrity_path.is_file():
        raise FileNotFoundError(f"Missing final Main-Table integrity: {integrity_path}")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    frozen = integrity.get("frozen", {})
    if (
        frozen.get("hidden") != 128
        or frozen.get("layers") != 1
        or float(frozen.get("dropout", -1)) != 0.3
        or frozen.get("pool") != "max"
        or frozen.get("activation") != "ELU"
        or integrity.get("test_derived_model_selection") is not False
    ):
        raise RuntimeError(f"Unexpected final Main-Table freeze: {integrity}")

    for method in METHODS:
        for seed in SEEDS:
            row = provenance[(method, seed)]
            if row.get("backbone") != "128/1/0.3/max":
                raise RuntimeError(
                    f"Unexpected backbone for {method} seed={seed}: {row.get('backbone')}"
                )
            for stage in ("stage1", "stage2"):
                path = main_checkpoint(method, seed, stage)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing frozen checkpoint: {path}")
                expected_hash = row["stages"][stage]["sha256"]
                observed_hash = sha256(path)
                if observed_hash != expected_hash:
                    raise RuntimeError(
                        f"Checkpoint hash mismatch for {method} seed={seed} {stage}: "
                        f"expected={expected_hash}, observed={observed_hash}"
                    )
                if row["stages"][stage].get("test_metric_during_training") is True:
                    raise RuntimeError(
                        f"TEST metric found during training for {method} seed={seed} {stage}"
                    )
    print("FINAL_CHECKPOINT_PROVENANCE = PASS", flush=True)
    print("TEST_EXCLUSION_DURING_TRAINING = PASS", flush=True)
    return provenance


def base_main(name: str) -> list[str]:
    return [sys.executable, "-m", "main", f"+name={name}"]


def run_logged(cmd: list[str], log_path: Path, *, dry_run: bool) -> float:
    if dry_run:
        print("DRY_RUN " + subprocess.list2cmdline(cmd), flush=True)
        return 0.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("COMMAND=" + subprocess.list2cmdline(cmd) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "WANDB_SILENT": "true", "HYDRA_FULL_ERROR": "1"},
        )
        rc = proc.wait()
        elapsed = time.perf_counter() - started
        handle.write(f"\nRETURN_CODE={rc}\nelapsed_sec={elapsed:.3f}\n")
    if rc != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        )
        raise RuntimeError(f"Command failed ({rc}): {log_path}\n{tail}")
    return elapsed


def find_checkpoint(output: Path) -> Path:
    paths = sorted((output / "checkpoints").glob("*.ckpt"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one checkpoint in {output}, found {paths}")
    return paths[0]


def pool_checkpoint(pool: str, method: str, seed: int, stage: str) -> Path:
    if pool == "max":
        return main_checkpoint(method, seed, stage)
    return POOL_CHECKPOINT_ROOT / pool / method / f"seed{seed}_{stage}.ckpt"


def pool_training_paths(
    pool: str, method: str, seed: int, stage: str
) -> tuple[Path, Path, Path]:
    output = POOL_OUTPUT_ROOT / "training" / pool / method / f"seed{seed}" / stage
    copied = pool_checkpoint(pool, method, seed, stage)
    marker = copied.with_suffix(".complete.json")
    return output, copied, marker


def assert_gpu_log(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "GPU available: True (cuda), used: True" not in text:
        raise RuntimeError(f"GPU-use assertion failed: {path}")


def assert_resolved_config(
    path: Path,
    *,
    method: str,
    pool: str,
    training_seed: int,
    task: str,
    stage: str | None,
) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    assert int(cfg.algorithm.model.input_dim) == 43
    assert int(cfg.algorithm.model.hidden_dim) == 128
    assert int(cfg.algorithm.model.num_layers) == 1
    assert float(cfg.algorithm.model.dropout) == 0.3
    assert cfg.algorithm.model.pool == pool
    assert cfg.algorithm.model.activation == "ELU"
    assert float(cfg.algorithm.keep_multiplier) == 1.5
    assert float(cfg.dataset.augment.gamma) == 0.4
    assert int(cfg.dataset.augment.min) == 1
    assert int(cfg.dataset.augment.max) == 20
    assert bool(cfg.algorithm.train_with_1_1) is False
    assert cfg.algorithm.candidate_order == "original"
    assert int(cfg.algorithm.candidate_order_seed) == 0
    assert bool(cfg.algorithm.use_anchor_features) is (method == "SAIF")
    if method == "SAIF":
        assert cfg.algorithm.model.anchor_feature_mode == "full"
        assert int(cfg.algorithm.model.anchor_input_dim) == 6
        assert cfg.algorithm.model.anchor_normalization == "layernorm"
        assert cfg.algorithm.model.anchor_control_mode == "normal"
    assert bool(cfg.experiment.validation.test_during_training) is False
    assert list(cfg.experiment.tasks) == [task]
    if task == "training":
        assert int(cfg.seed) == training_seed
        assert bool(cfg.dataset.augment.enabled) is (stage == "stage2")
        assert int(cfg.experiment.training.max_epochs) == (150 if stage == "stage1" else 300)
        assert bool(cfg.experiment.training.early_stopping.enabled) is (stage == "stage1")
    else:
        assert int(cfg.seed) == 0
        assert int(cfg.dataset.num_samples) == 256
        assert cfg.dataset.eval_pool_mode == "official"
        assert bool(cfg.dataset.augment.enabled) is False


def train_pool_stage(
    pool: str,
    method: str,
    seed: int,
    stage: str,
    load: Path | None,
    *,
    dry_run: bool,
) -> Path:
    output, copied, marker = pool_training_paths(pool, method, seed, stage)
    if copied.is_file() and marker.is_file() and not dry_run:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("sha256") != sha256(copied):
            raise RuntimeError(f"Completed checkpoint hash mismatch: {copied}")
        assert_gpu_log(output / "training.log")
        assert_resolved_config(
            output / ".hydra" / "config.yaml",
            method=method,
            pool=pool,
            training_seed=seed,
            task="training",
            stage=stage,
        )
        print(f"SKIP completed pool={pool} {method} seed={seed} {stage}", flush=True)
        return copied

    output.mkdir(parents=True, exist_ok=True)
    copied.parent.mkdir(parents=True, exist_ok=True)
    cmd = base_main(f"FinalPool_{pool}_{method}_{stage}_seed{seed}") + [
        "dataset=elliptic_recommendation",
        "algorithm=iterative_filtering",
        *METHODS[method],
        *FROZEN_COMMON,
        f"algorithm.model.pool={pool}",
        "experiment=exp_edge_recommendation",
        "experiment.tasks=[training]",
        "experiment.validation.test_during_training=false",
        "experiment.training.checkpointing.monitor=validation/f1",
        "experiment.training.checkpointing.mode=max",
        "experiment.training.data.num_workers=0",
        "experiment.validation.data.num_workers=0",
        f"seed={seed}",
        "wandb.mode=offline",
        f"hydra.run.dir={output.as_posix()}",
    ]
    if stage == "stage1":
        cmd += [
            "dataset.augment.enabled=false",
            "experiment.training.max_epochs=150",
            "experiment.training.early_stopping.enabled=true",
            "experiment.training.early_stopping.patience=30",
        ]
    elif stage == "stage2":
        if load is None:
            raise ValueError("Stage-2 requires a Stage-1 checkpoint")
        cmd += [
            "dataset.augment.enabled=true",
            "experiment.training.max_epochs=300",
            "experiment.training.early_stopping.enabled=false",
            f"load={load.as_posix()}",
        ]
    else:
        raise ValueError(stage)

    print(f"START pool={pool} {method} seed={seed} {stage} at {now()}", flush=True)
    elapsed = run_logged(cmd, output / "training.log", dry_run=dry_run)
    if dry_run:
        return copied
    source = find_checkpoint(output)
    shutil.copy2(source, copied)
    assert_gpu_log(output / "training.log")
    assert_resolved_config(
        output / ".hydra" / "config.yaml",
        method=method,
        pool=pool,
        training_seed=seed,
        task="training",
        stage=stage,
    )
    log_text = (output / "training.log").read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?:^|\s)(?:final_test|test)/(?:HR|NDCG|f1|prauc)", log_text):
        raise RuntimeError(f"TEST metric appeared during training: {output}")
    payload = {
        "pool": pool,
        "method": method,
        "seed": seed,
        "stage": stage,
        "started_from": str(load) if load else None,
        "source_checkpoint": str(source),
        "copied_checkpoint": str(copied),
        "sha256": sha256(copied),
        "elapsed_sec": elapsed,
        "test_metric_during_training": False,
        "resolved_config": str(output / ".hydra" / "config.yaml"),
        "training_log": str(output / "training.log"),
        "completed_at": now(),
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"DONE  pool={pool} {method} seed={seed} {stage} "
        f"elapsed={elapsed/60:.1f} min",
        flush=True,
    )
    return copied


def evaluation_paths(
    phase: str, label: str, seed: int, setting: str
) -> tuple[Path, Path, Path]:
    if phase == "ablation":
        output_root, result_root = ABLATION_OUTPUT_ROOT, ABLATION_RESULT_ROOT
    elif phase == "pooling":
        output_root, result_root = POOL_OUTPUT_ROOT, POOL_RESULT_ROOT
    else:
        raise ValueError(phase)
    output = output_root / "evaluation" / label / f"seed{seed}" / tag(setting)
    audit = result_root / "instance_jsonl" / label / f"seed{seed}" / f"{tag(setting)}.jsonl"
    return output, audit, output / "complete.json"


def evaluate(
    *,
    phase: str,
    label: str,
    method: str,
    pool: str,
    seed: int,
    setting: str,
    checkpoint: Path,
    dry_run: bool,
) -> None:
    output, audit, marker = evaluation_paths(phase, label, seed, setting)
    if marker.is_file() and audit.is_file() and not dry_run:
        rows = [line for line in audit.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if len(rows) != 256:
            raise RuntimeError(f"Incomplete saved audit: {audit}")
        if payload.get("checkpoint_sha256") != sha256(checkpoint):
            raise RuntimeError(f"Evaluation checkpoint hash mismatch: {marker}")
        assert_gpu_log(output / "evaluation.log")
        assert_resolved_config(
            output / ".hydra" / "config.yaml",
            method=method,
            pool=pool,
            training_seed=seed,
            task="test",
            stage=None,
        )
        print(f"SKIP completed {phase} {label} seed={seed} {setting}", flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        audit.unlink(missing_ok=True)
    cmd = base_main(f"Final_{phase}_{tag(label)}_seed{seed}_{tag(setting)}") + [
        "dataset=elliptic_recommendation",
        "algorithm=iterative_filtering",
        *METHODS[method],
        *FROZEN_COMMON,
        f"algorithm.model.pool={pool}",
        "experiment=exp_edge_recommendation",
        "experiment.tasks=[test]",
        "experiment.validation.test_during_training=false",
        "experiment.test.batch_size=16",
        "experiment.test.data.num_workers=0",
        "dataset.num_samples=256",
        "dataset.eval_pool_mode=official",
        "dataset.augment.enabled=false",
        "seed=0",
        "wandb.mode=offline",
        f"load={checkpoint.as_posix()}",
        f"+shortcut={setting}",
        f"+algorithm.audit_output_path={audit.as_posix()}",
        f"+algorithm.audit_method={label}",
        f"+algorithm.audit_training_seed={seed}",
        "+algorithm.audit_eval_seed=0",
        f"+algorithm.audit_setting={tag(setting)}",
        f"hydra.run.dir={output.as_posix()}",
    ]
    print(f"START {phase} {label} seed={seed} setting={setting}", flush=True)
    elapsed = run_logged(cmd, output / "evaluation.log", dry_run=dry_run)
    if dry_run:
        return
    count = len([line for line in audit.read_text(encoding="utf-8").splitlines() if line.strip()])
    if count != 256:
        raise RuntimeError(f"Expected 256 instance records, found {count}: {audit}")
    assert_gpu_log(output / "evaluation.log")
    assert_resolved_config(
        output / ".hydra" / "config.yaml",
        method=method,
        pool=pool,
        training_seed=seed,
        task="test",
        stage=None,
    )
    marker.write_text(
        json.dumps(
            {
                "phase": phase,
                "label": label,
                "method": method,
                "pool": pool,
                "training_seed": seed,
                "setting": setting,
                "eval_seed": 0,
                "instances": count,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "elapsed_sec": elapsed,
                "instance_jsonl": str(audit),
                "completed_at": now(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"DONE  {phase} {label} seed={seed} {setting} elapsed={elapsed/60:.1f} min", flush=True)


def read_phase_records(
    *, phase: str, labels: list[str]
) -> list[dict]:
    result_root = ABLATION_RESULT_ROOT if phase == "ablation" else POOL_RESULT_ROOT
    records: list[dict] = []
    for label in labels:
        for seed in SEEDS:
            for setting in SETTINGS:
                path = result_root / "instance_jsonl" / label / f"seed{seed}" / f"{tag(setting)}.jsonl"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing instance audit: {path}")
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if len(rows) != 256:
                    raise RuntimeError(f"Expected 256 rows in {path}, found {len(rows)}")
                for row in rows:
                    row["label"] = label
                    row["setting"] = setting
                    row["ckpt"] = str(seed)
                records.extend(rows)
    return records


def assert_candidate_matching(records: list[dict], expected_unique: int) -> None:
    reference: dict[tuple[str, int], str] = {}
    for row in records:
        key = (str(row["setting"]), int(row["sample_index"]))
        candidate_hash = str(row["candidate_hash"])
        if key in reference and reference[key] != candidate_hash:
            raise RuntimeError(f"Candidate hash mismatch for {key}")
        reference[key] = candidate_hash
    if len(reference) != expected_unique:
        raise RuntimeError(
            f"Expected {expected_unique} unique candidate instances, found {len(reference)}"
        )


INSTANCE_FIELDS = [
    "label",
    "method",
    "training_seed",
    "ckpt",
    "setting",
    "eval_seed",
    "sample_index",
    "sample_id",
    "candidate_hash",
    "num_senders",
    "num_receivers",
    "gt_unique_count",
    "hit_count",
    "HR",
    "NDCG",
]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(str(row["label"]), int(row["training_seed"]), str(row["setting"]))].append(row)
    raw = []
    for (label, seed, setting), rows in sorted(grouped.items()):
        if len(rows) != 256:
            raise RuntimeError(f"Expected 256 rows for {(label, seed, setting)}")
        raw.append(
            {
                "Label": label,
                "Training seed": seed,
                "Setting": setting,
                "Eval seed": 0,
                "# instances": len(rows),
                "HR": statistics.mean(float(row["HR"]) for row in rows),
                "NDCG": statistics.mean(float(row["NDCG"]) for row in rows),
            }
        )
    summary = []
    for label in sorted({str(row["label"]) for row in records}):
        for setting in SETTINGS:
            rows = [r for r in raw if r["Label"] == label and r["Setting"] == setting]
            if len(rows) != 3:
                raise RuntimeError(f"Expected three runs for {(label, setting)}")
            summary.append(
                {
                    "Label": label,
                    "Setting": setting,
                    "n": 3,
                    "HR mean": statistics.mean(float(r["HR"]) for r in rows),
                    "HR SD": statistics.stdev(float(r["HR"]) for r in rows),
                    "NDCG mean": statistics.mean(float(r["NDCG"]) for r in rows),
                    "NDCG SD": statistics.stdev(float(r["NDCG"]) for r in rows),
                }
            )
    return raw, summary


def averaged_instance_values(records: list[dict]) -> dict[tuple[str, str, int, str], float]:
    grouped: dict[tuple[str, str, int, str], dict[str, float]] = defaultdict(dict)
    for row in records:
        for metric in ("HR", "NDCG"):
            key = (str(row["label"]), str(row["setting"]), int(row["sample_index"]), metric)
            run = str(row["ckpt"])
            if run in grouped[key]:
                raise RuntimeError(f"Duplicate run {run} for {key}")
            grouped[key][run] = float(row[metric])
    averaged = {}
    for key, values in grouped.items():
        if set(values) != {"0", "1", "2"}:
            raise RuntimeError(f"Run grid mismatch for {key}: {sorted(values)}")
        averaged[key] = statistics.mean(values.values())
    return averaged


def paired_test_row(
    *,
    comparison: str,
    setting: str,
    metric: str,
    a_name: str,
    b_name: str,
    a_values: list[float],
    b_values: list[float],
) -> dict:
    from scripts.paired_instance_wilcoxon_bh import wilcoxon_signed_rank

    if len(a_values) != 256 or len(b_values) != 256:
        raise RuntimeError(f"Expected 256 paired values for {comparison} {setting} {metric}")
    diffs = [b - a for a, b in zip(a_values, b_values)]
    test = wilcoxon_signed_rank(diffs, alternative="two-sided")
    return {
        "comparison": comparison,
        "setting": setting,
        "metric": metric,
        "n": len(diffs),
        "nonzero_n": test["nonzero_n"],
        "a_name": a_name,
        "b_name": b_name,
        "a_mean": f"{statistics.mean(a_values):.10f}",
        "b_mean": f"{statistics.mean(b_values):.10f}",
        "delta_mean": f"{statistics.mean(diffs):+.10f}",
        "positive": f"{sum(diff > 0 for diff in diffs)}/{len(diffs)}",
        "z": f"{test['z']:.6f}",
        "effect_r": f"{test['effect_r']:.6f}",
        "p": f"{test['p']:.10g}",
        "q_bh": "",
    }


def interaction_test_row(
    *, comparison: str, setting: str, metric: str, values: list[float]
) -> dict:
    from scripts.paired_instance_wilcoxon_bh import wilcoxon_signed_rank

    if len(values) != 256:
        raise RuntimeError(f"Expected 256 interaction values for {comparison}")
    test = wilcoxon_signed_rank(values, alternative="two-sided")
    return {
        "comparison": comparison,
        "setting": setting,
        "metric": metric,
        "n": len(values),
        "nonzero_n": test["nonzero_n"],
        "interaction_mean": f"{statistics.mean(values):+.10f}",
        "positive": f"{sum(value > 0 for value in values)}/{len(values)}",
        "z": f"{test['z']:.6f}",
        "effect_r": f"{test['effect_r']:.6f}",
        "p": f"{test['p']:.10g}",
        "q_bh": "",
    }


def apply_bh(rows: list[dict]) -> None:
    from scripts.paired_instance_wilcoxon_bh import bh_adjust

    for row, q in zip(rows, bh_adjust([float(row["p"]) for row in rows])):
        row["q_bh"] = f"{q:.10g}"


def write_stats_markdown(path: Path, title: str, rows: list[dict]) -> None:
    interaction = "interaction_mean" in rows[0]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(
            "Two-sided paired instance-level Wilcoxon signed-rank tests. "
            "The three independently trained runs are averaged per "
            "`(setting, sample_index)` before testing; BH correction covers "
            "every row in this file.\n\n"
        )
        if interaction:
            handle.write(
                "| comparison | setting | metric | n | nonzero n | "
                "interaction mean | positive | z | effect r | p | BH q |\n"
            )
            handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                handle.write(
                    f"| {row['comparison']} | {row['setting']} | {row['metric']} | "
                    f"{row['n']} | {row['nonzero_n']} | {row['interaction_mean']} | "
                    f"{row['positive']} | {row['z']} | {row['effect_r']} | "
                    f"{row['p']} | {row['q_bh']} |\n"
                )
        else:
            handle.write(
                "| comparison | setting | metric | n | nonzero n | A | B | "
                "A mean | B mean | delta B-A | positive | z | effect r | p | BH q |\n"
            )
            handle.write(
                "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            )
            for row in rows:
                handle.write(
                    f"| {row['comparison']} | {row['setting']} | {row['metric']} | "
                    f"{row['n']} | {row['nonzero_n']} | {row['a_name']} | "
                    f"{row['b_name']} | {row['a_mean']} | {row['b_mean']} | "
                    f"{row['delta_mean']} | {row['positive']} | {row['z']} | "
                    f"{row['effect_r']} | {row['p']} | {row['q_bh']} |\n"
                )


def ablation_stats(records: list[dict]) -> tuple[list[dict], list[dict]]:
    averaged = averaged_instance_values(records)
    pairwise = []
    for comparison, a_name, b_name in ABLATION_CONTRASTS:
        for setting in SETTINGS:
            for metric in ("HR", "NDCG"):
                a_values = [averaged[(a_name, setting, idx, metric)] for idx in range(256)]
                b_values = [averaged[(b_name, setting, idx, metric)] for idx in range(256)]
                pairwise.append(
                    paired_test_row(
                        comparison=comparison,
                        setting=setting,
                        metric=metric,
                        a_name=a_name,
                        b_name=b_name,
                        a_values=a_values,
                        b_values=b_values,
                    )
                )
    apply_bh(pairwise)

    interactions = []
    for setting in SETTINGS:
        for metric in ("HR", "NDCG"):
            values = [
                (
                    averaged[("Full-SAIF", setting, idx, metric)]
                    - averaged[("RevFilter-S2", setting, idx, metric)]
                )
                - (
                    averaged[("SAIF-S1", setting, idx, metric)]
                    - averaged[("RevFilter-S1", setting, idx, metric)]
                )
                for idx in range(256)
            ]
            interactions.append(
                interaction_test_row(
                    comparison="Anchor x Stage-2 interaction",
                    setting=setting,
                    metric=metric,
                    values=values,
                )
            )
    apply_bh(interactions)
    return pairwise, interactions


def pooling_stats(records: list[dict]) -> tuple[list[dict], list[dict]]:
    averaged = averaged_instance_values(records)
    pairwise = []
    for pool in POOLS:
        rev = f"{pool}/RevFilter"
        saif = f"{pool}/SAIF"
        for setting in SETTINGS:
            for metric in ("HR", "NDCG"):
                pairwise.append(
                    paired_test_row(
                        comparison=f"SAIF effect under {pool} pooling",
                        setting=setting,
                        metric=metric,
                        a_name=rev,
                        b_name=saif,
                        a_values=[averaged[(rev, setting, idx, metric)] for idx in range(256)],
                        b_values=[averaged[(saif, setting, idx, metric)] for idx in range(256)],
                    )
                )
    apply_bh(pairwise)

    interactions = []
    for pool in ("mean", "sum"):
        for setting in SETTINGS:
            for metric in ("HR", "NDCG"):
                values = [
                    (
                        averaged[(f"{pool}/SAIF", setting, idx, metric)]
                        - averaged[(f"{pool}/RevFilter", setting, idx, metric)]
                    )
                    - (
                        averaged[("max/SAIF", setting, idx, metric)]
                        - averaged[("max/RevFilter", setting, idx, metric)]
                    )
                    for idx in range(256)
                ]
                interactions.append(
                    interaction_test_row(
                        comparison=f"({pool} SAIF effect) - (max SAIF effect)",
                        setting=setting,
                        metric=metric,
                        values=values,
                    )
                )
    apply_bh(interactions)
    return pairwise, interactions


def collect_ablation(main_provenance: dict[tuple[str, int], dict]) -> None:
    labels = list(ABLATION_VARIANTS)
    records = read_phase_records(phase="ablation", labels=labels)
    expected_records = len(labels) * len(SEEDS) * len(SETTINGS) * 256
    if len(records) != expected_records:
        raise RuntimeError(f"Expected {expected_records} ablation rows, found {len(records)}")
    assert_candidate_matching(records, len(SETTINGS) * 256)
    ABLATION_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(ABLATION_RESULT_ROOT / "ablation_instance_metrics_6144.csv", records, INSTANCE_FIELDS)
    raw, summary = aggregate_records(records)
    write_csv(ABLATION_RESULT_ROOT / "ablation_raw_24.csv", raw)
    write_csv(ABLATION_RESULT_ROOT / "ablation_summary.csv", summary)
    pairwise, interactions = ablation_stats(records)
    pair_path = ABLATION_RESULT_ROOT / "ablation_planned_pairwise_wilcoxon_bh_all16.csv"
    interaction_path = ABLATION_RESULT_ROOT / "ablation_interaction_wilcoxon_bh_all4.csv"
    write_csv(pair_path, pairwise)
    write_csv(interaction_path, interactions)
    write_stats_markdown(pair_path.with_suffix(".md"), "Final Ablation Planned Contrasts", pairwise)
    write_stats_markdown(
        interaction_path.with_suffix(".md"), "Final Ablation Interaction Tests", interactions
    )
    provenance = []
    for label, (method, stage) in ABLATION_VARIANTS.items():
        for seed in SEEDS:
            path = main_checkpoint(method, seed, stage)
            provenance.append(
                {
                    "label": label,
                    "method": method,
                    "stage": stage,
                    "seed": seed,
                    "path": str(path),
                    "sha256": sha256(path),
                    "expected_sha256": main_provenance[(method, seed)]["stages"][stage]["sha256"],
                    "source": "frozen final d0p3 Main-Table training",
                }
            )
    (ABLATION_RESULT_ROOT / "checkpoint_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    integrity = {
        "phase": "ablation",
        "design": "2x2 anchor-features (off/on) x Stage-2 fine-tuning (off/on)",
        "frozen_backbone": "128/1/0.3/max/ELU",
        "settings": SETTINGS,
        "variants": labels,
        "training_seeds": SEEDS,
        "evaluation_seed": 0,
        "instances_per_setting": 256,
        "expected_evaluations": 24,
        "observed_evaluations": len(raw),
        "expected_instance_records": 6144,
        "observed_instance_records": len(records),
        "candidate_hash_assertion": "PASS",
        "same_candidate_instances_across_variants_and_runs": True,
        "planned_pairwise_tests": len(pairwise),
        "interaction_tests": len(interactions),
        "BH_scope_pairwise": "all 16 planned contrasts",
        "BH_scope_interaction": "all 4 interaction tests",
        "test_derived_model_selection": False,
        "GPU_ASSERTION": "PASS",
        "CONFIG_ASSERTION": "PASS",
        "FINAL_CHECKPOINT_PROVENANCE": "PASS",
        "code_sha256": code_sha256(),
        "generated_at": now(),
    }
    (ABLATION_RESULT_ROOT / "integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Final component ablation (d0p3)",
        "",
        "- Design: **2x2 anchor features x Stage-2 fine-tuning**",
        "- Evaluations: **24/24**",
        "- Instance records: **6144/6144**",
        "- Candidate matching: **PASS**",
        "- TEST-derived selection: **false**",
        "- Pairwise BH family: **16 planned tests**",
        "- Interaction BH family: **4 tests**",
        "",
        pair_path.with_suffix(".md").read_text(encoding="utf-8"),
        "",
        interaction_path.with_suffix(".md").read_text(encoding="utf-8"),
    ]
    (ABLATION_RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"ABLATION RESULTS READY: {ABLATION_RESULT_ROOT}", flush=True)


def collect_pooling(main_provenance: dict[tuple[str, int], dict]) -> None:
    labels = [f"{pool}/{method}" for pool in POOLS for method in METHODS]
    records = read_phase_records(phase="pooling", labels=labels)
    expected_records = len(labels) * len(SEEDS) * len(SETTINGS) * 256
    if len(records) != expected_records:
        raise RuntimeError(f"Expected {expected_records} pooling rows, found {len(records)}")
    assert_candidate_matching(records, len(SETTINGS) * 256)
    POOL_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(POOL_RESULT_ROOT / "pooling_instance_metrics_9216.csv", records, INSTANCE_FIELDS)
    raw, summary = aggregate_records(records)
    for row in raw:
        row["Pooling"], row["Method"] = str(row["Label"]).split("/", 1)
    for row in summary:
        row["Pooling"], row["Method"] = str(row["Label"]).split("/", 1)
    write_csv(POOL_RESULT_ROOT / "pooling_raw_36.csv", raw)
    write_csv(POOL_RESULT_ROOT / "pooling_summary.csv", summary)
    pairwise, interactions = pooling_stats(records)
    pair_path = POOL_RESULT_ROOT / "pooling_saif_vs_revfilter_wilcoxon_bh_all12.csv"
    interaction_path = POOL_RESULT_ROOT / "pooling_interaction_vs_max_wilcoxon_bh_all8.csv"
    write_csv(pair_path, pairwise)
    write_csv(interaction_path, interactions)
    write_stats_markdown(pair_path.with_suffix(".md"), "Final Pooling Sensitivity", pairwise)
    write_stats_markdown(
        interaction_path.with_suffix(".md"), "Pooling Dependence Interaction Tests", interactions
    )
    provenance = []
    for pool in POOLS:
        for method in METHODS:
            for seed in SEEDS:
                for stage in ("stage1", "stage2"):
                    path = pool_checkpoint(pool, method, seed, stage)
                    row = {
                        "pool": pool,
                        "method": method,
                        "seed": seed,
                        "stage": stage,
                        "path": str(path),
                        "sha256": sha256(path),
                    }
                    if pool == "max":
                        row["source"] = "frozen final d0p3 Main-Table training"
                        row["expected_sha256"] = main_provenance[(method, seed)]["stages"][stage]["sha256"]
                    else:
                        marker = path.with_suffix(".complete.json")
                        payload = json.loads(marker.read_text(encoding="utf-8"))
                        if payload["sha256"] != row["sha256"]:
                            raise RuntimeError(f"Pooling checkpoint marker mismatch: {path}")
                        row["source"] = "pooling-sensitivity two-stage retraining"
                        row["marker"] = payload
                    provenance.append(row)
    (POOL_RESULT_ROOT / "checkpoint_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    integrity = {
        "phase": "pooling-sensitivity",
        "frozen_except_pooling": "128/1/0.3/ELU; alpha=1.5; gamma=0.4; n_merge=[1,20]",
        "pools": POOLS,
        "newly_trained_pools": TRAINED_POOLS,
        "max_source": "frozen final d0p3 Main-Table checkpoints",
        "methods": list(METHODS),
        "settings": SETTINGS,
        "training_seeds": SEEDS,
        "evaluation_seed": 0,
        "instances_per_setting": 256,
        "expected_training_stages": 24,
        "expected_evaluations": 36,
        "observed_evaluations": len(raw),
        "expected_instance_records": 9216,
        "observed_instance_records": len(records),
        "candidate_hash_assertion": "PASS",
        "same_candidate_instances_across_pools_methods_and_runs": True,
        "saif_vs_revfilter_tests": len(pairwise),
        "pooling_interaction_tests": len(interactions),
        "BH_scope_pairwise": "all 12 pool-setting-metric tests",
        "BH_scope_interaction": "all 8 nonmax-vs-max interaction tests",
        "test_derived_model_selection": False,
        "GPU_ASSERTION": "PASS",
        "CONFIG_ASSERTION": "PASS",
        "FINAL_CHECKPOINT_PROVENANCE": "PASS",
        "code_sha256": code_sha256(),
        "generated_at": now(),
    }
    (POOL_RESULT_ROOT / "integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Final pooling sensitivity (d0p3)",
        "",
        "- Pools: **max / mean / sum**",
        "- Mean/sum training stages: **24/24**",
        "- Evaluations: **36/36**",
        "- Instance records: **9216/9216**",
        "- Candidate matching: **PASS**",
        "- TEST-derived selection: **false**",
        "- Pairwise BH family: **12 tests**",
        "- Pooling-interaction BH family: **8 tests**",
        "",
        pair_path.with_suffix(".md").read_text(encoding="utf-8"),
        "",
        interaction_path.with_suffix(".md").read_text(encoding="utf-8"),
    ]
    (POOL_RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"POOLING RESULTS READY: {POOL_RESULT_ROOT}", flush=True)


def run_ablation(*, dry_run: bool) -> None:
    for label, (method, stage) in ABLATION_VARIANTS.items():
        for seed in SEEDS:
            checkpoint = main_checkpoint(method, seed, stage)
            for setting in SETTINGS:
                evaluate(
                    phase="ablation",
                    label=label,
                    method=method,
                    pool="max",
                    seed=seed,
                    setting=setting,
                    checkpoint=checkpoint,
                    dry_run=dry_run,
                )


def run_pooling(*, dry_run: bool) -> None:
    for pool in TRAINED_POOLS:
        for method in METHODS:
            for seed in SEEDS:
                stage1 = train_pool_stage(pool, method, seed, "stage1", None, dry_run=dry_run)
                train_pool_stage(pool, method, seed, "stage2", stage1, dry_run=dry_run)
    if not dry_run:
        print("ALL POOLING TRAINING COMPLETE", flush=True)
    for pool in POOLS:
        for method in METHODS:
            for seed in SEEDS:
                checkpoint = pool_checkpoint(pool, method, seed, "stage2")
                for setting in SETTINGS:
                    evaluate(
                        phase="pooling",
                        label=f"{pool}/{method}",
                        method=method,
                        pool=pool,
                        seed=seed,
                        setting=setting,
                        checkpoint=checkpoint,
                        dry_run=dry_run,
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "ablation", "pooling"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if sum((args.dry_run, args.collect_only, args.audit_only)) > 1:
        parser.error("--dry-run, --collect-only, and --audit-only are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    provenance = assert_main_checkpoints()
    cuda = cuda_preflight()
    phases = ["ablation", "pooling"] if args.phase == "all" else [args.phase]
    for phase in phases:
        result_root = ABLATION_RESULT_ROOT if phase == "ablation" else POOL_RESULT_ROOT
        if not args.dry_run:
            write_environment(result_root, cuda)

    if args.audit_only:
        for phase in phases:
            if phase == "ablation":
                collect_ablation(provenance)
            else:
                collect_pooling(provenance)
        print("AUDIT_ONLY = PASS", flush=True)
        return

    if not args.collect_only:
        if "ablation" in phases:
            run_ablation(dry_run=args.dry_run)
        if "pooling" in phases:
            run_pooling(dry_run=args.dry_run)
    if args.dry_run:
        print("DRY_RUN = PASS", flush=True)
        return
    for phase in phases:
        if phase == "ablation":
            collect_ablation(provenance)
        else:
            collect_pooling(provenance)
    print("ALL REQUESTED EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
