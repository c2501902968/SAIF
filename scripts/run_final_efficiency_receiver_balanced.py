#!/usr/bin/env python3
"""Run the two authorized post-Main-Table supporting experiments.

This launcher evaluates the frozen d0p3 RevFilter/SAIF checkpoints only.  It
does not train, tune, or select a model.  The two restartable phases are:

1. efficiency: official-pool search profiling on the two largest settings;
2. receiver-balanced: matched receiver-balanced evaluation plus paired tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints" / "final_revfilter_saif_128_l1_d0p3"
MAIN_RESULT_ROOT = ROOT / "results" / "final_revfilter_saif_128_l1_d0p3"
EFF_OUTPUT_ROOT = ROOT / "outputs" / "final_efficiency_128_l1_d0p3"
EFF_RESULT_ROOT = ROOT / "results" / "final_efficiency_128_l1_d0p3"
RB_OUTPUT_ROOT = ROOT / "outputs" / "final_receiver_balanced_128_l1_d0p3"
RB_RESULT_ROOT = ROOT / "results" / "final_receiver_balanced_128_l1_d0p3"
SETTINGS = ["10+1000@100", "10+10000@100"]
METHODS = {
    "RevFilter": {
        "log_key": "official",
        "overrides": ["algorithm.use_anchor_features=false"],
    },
    "SAIF": {
        "log_key": "saif",
        "overrides": [
            "algorithm.use_anchor_features=true",
            "algorithm.model.anchor_feature_mode=full",
            "+algorithm.model.anchor_fusion_mode=full",
            "algorithm.model.anchor_input_dim=6",
            "algorithm.model.anchor_normalization=layernorm",
            "algorithm.model.anchor_control_mode=normal",
        ],
    },
}
FROZEN = [
    "algorithm.model.hidden_dim=128",
    "algorithm.model.num_layers=1",
    "algorithm.model.dropout=0.3",
    "algorithm.model.pool=max",
    "algorithm.model.activation=ELU",
    "algorithm.keep_multiplier=1.5",
    "dataset.augment.gamma=0.4",
    "dataset.augment.min=1",
    "dataset.augment.max=20",
    "algorithm.train_with_1_1=false",
    "algorithm.candidate_order=original",
    "algorithm.candidate_order_seed=0",
]
PROFILE_FIELDS = [
    "elapsed_sec",
    "search_elapsed_sec",
    "model_parameters",
    "initial_pairs_per_sample",
    "scored_regions_per_sample",
    "region_score_ratio",
    "scored_node_tokens_per_sample",
    "scored_edge_volume_per_sample",
    "search_rounds_per_sample",
    "forward_rounds_per_sample",
    "max_live_regions_per_sample",
    "HR",
    "NDCG",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tag(value: str) -> str:
    return value.replace("+", "p").replace("@", "at").replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha256() -> dict[str, str]:
    relative_paths = [
        "scripts/run_final_efficiency_receiver_balanced.py",
        "scripts/summarize_complexity_profile.py",
        "scripts/paired_instance_wilcoxon_bh.py",
        "scripts/edge_recommendation_analysis_utils.py",
        "algorithms/subgraph/iterative_filtering_algo.py",
        "datasets/elliptic/dataset.py",
        "configurations/algorithm/iterative_filtering.yaml",
        "configurations/dataset/elliptic_recommendation.yaml",
    ]
    return {
        relative: sha256(ROOT / relative)
        for relative in relative_paths
    }


def checkpoint(method: str, seed: int) -> Path:
    return CHECKPOINT_ROOT / method / f"seed{seed}_stage2.ckpt"


def assert_checkpoints() -> None:
    missing = [
        path
        for method in METHODS
        for seed in range(3)
        if not (path := checkpoint(method, seed)).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing frozen final checkpoints: {missing}")

    provenance_path = MAIN_RESULT_ROOT / "checkpoint_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Missing final-training checkpoint provenance: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    indexed = {
        (str(row["method"]), int(row["seed"])): row
        for row in provenance
    }
    expected_keys = {(method, seed) for method in METHODS for seed in range(3)}
    if set(indexed) != expected_keys:
        raise RuntimeError(
            "Final checkpoint provenance grid mismatch: "
            f"missing={expected_keys-set(indexed)}, extra={set(indexed)-expected_keys}"
        )
    for method, seed in sorted(expected_keys):
        row = indexed[(method, seed)]
        if row.get("backbone") != "128/1/0.3/max":
            raise RuntimeError(
                f"Unexpected frozen backbone for {method} seed={seed}: "
                f"{row.get('backbone')}"
            )
        expected_hash = row["stages"]["stage2"]["sha256"]
        observed_hash = sha256(checkpoint(method, seed))
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Final checkpoint hash mismatch for {method} seed={seed}: "
                f"expected={expected_hash}, observed={observed_hash}"
            )
    print("FINAL_CHECKPOINT_PROVENANCE = PASS", flush=True)


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
        raise RuntimeError("CUDA GPU is required for these formal experiments.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Exactly one visible CUDA device is required; set CUDA_VISIBLE_DEVICES."
        )
    info["gpu"] = torch.cuda.get_device_name(0)
    info["current_cuda_device"] = torch.cuda.current_device()
    print(f"GPU name: {info['gpu']}", flush=True)
    print("CUDA_REQUIRED = PASS", flush=True)
    return info


def git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_environment(root: Path, cuda: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {**cuda, "git": git_info(), "recorded_at": now()}
    (root / "environment.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (root / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")


def base_command(name: str) -> list[str]:
    return [sys.executable, "-m", "main", f"+name={name}"]


def eval_command(
    *,
    method: str,
    seed: int,
    setting: str,
    pool: str,
    output: Path,
    profile: bool,
    audit_path: Path | None,
) -> list[str]:
    cmd = base_command(
        f"Final_{'Efficiency' if profile else 'ReceiverBalanced'}_"
        f"{method}_seed{seed}_{tag(setting)}"
    ) + [
        "dataset=elliptic_recommendation",
        "algorithm=iterative_filtering",
        *METHODS[method]["overrides"],
        *FROZEN,
        "experiment=exp_edge_recommendation",
        "experiment.tasks=[test]",
        "experiment.validation.test_during_training=false",
        "experiment.test.batch_size=16",
        "experiment.test.data.num_workers=0",
        "dataset.num_samples=256",
        f"dataset.eval_pool_mode={pool}",
        "dataset.augment.enabled=false",
        f"algorithm.profile_search={'true' if profile else 'false'}",
        "seed=0",
        "wandb.mode=offline",
        f"load={checkpoint(method, seed).as_posix()}",
        f"+shortcut={setting}",
        f"hydra.run.dir={output.as_posix()}",
    ]
    if audit_path is not None:
        cmd += [
            f"+algorithm.audit_output_path={audit_path.as_posix()}",
            f"+algorithm.audit_method={method}",
            f"+algorithm.audit_training_seed={seed}",
            "+algorithm.audit_eval_seed=0",
            f"+algorithm.audit_setting={tag(setting)}",
        ]
    return cmd


def child_max_memory_kb() -> int:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except (ImportError, ValueError):
        return 0


def run_logged(cmd: list[str], log_path: Path) -> float:
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
        handle.write(f"\nRETURN_CODE={rc}\n")
        handle.write(f"elapsed_sec={elapsed:.3f}\n")
        handle.write(f"max_mem_kb={child_max_memory_kb()}\n")
    if rc != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        )
        raise RuntimeError(f"Command failed ({rc}): {log_path}\n{tail}")
    return elapsed


def assert_gpu_log(log_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "GPU available: True (cuda), used: True" not in text:
        raise RuntimeError(f"GPU-use assertion failed: {log_path}")


def assert_resolved_config(
    path: Path, *, method: str, pool: str, profile: bool
) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    assert cfg.algorithm.model.input_dim == 43
    assert cfg.algorithm.model.hidden_dim == 128
    assert cfg.algorithm.model.num_layers == 1
    assert float(cfg.algorithm.model.dropout) == 0.3
    assert cfg.algorithm.model.pool == "max"
    assert cfg.algorithm.model.activation == "ELU"
    assert float(cfg.algorithm.keep_multiplier) == 1.5
    assert float(cfg.dataset.augment.gamma) == 0.4
    assert int(cfg.dataset.augment.min) == 1
    assert int(cfg.dataset.augment.max) == 20
    assert bool(cfg.algorithm.train_with_1_1) is False
    assert cfg.algorithm.candidate_order == "original"
    assert int(cfg.algorithm.candidate_order_seed) == 0
    assert bool(cfg.algorithm.profile_search) is profile
    assert bool(cfg.algorithm.use_anchor_features) is (method == "SAIF")
    if method == "SAIF":
        assert cfg.algorithm.model.anchor_feature_mode == "full"
        assert int(cfg.algorithm.model.anchor_input_dim) == 6
        assert cfg.algorithm.model.anchor_normalization == "layernorm"
        assert cfg.algorithm.model.anchor_control_mode == "normal"
    assert list(cfg.experiment.tasks) == ["test"]
    assert bool(cfg.experiment.validation.test_during_training) is False
    assert int(cfg.dataset.num_samples) == 256
    assert cfg.dataset.eval_pool_mode == pool
    assert bool(cfg.dataset.augment.enabled) is False
    assert int(cfg.seed) == 0


def validate_completed_job(
    *,
    output: Path,
    log_path: Path,
    method: str,
    pool: str,
    profile: bool,
    audit_path: Path | None,
) -> None:
    marker = output / "complete.json"
    if not marker.is_file() or not log_path.is_file():
        raise RuntimeError(f"Incomplete job artifacts: {output}")
    if audit_path is not None:
        if not audit_path.is_file():
            raise RuntimeError(f"Missing instance audit: {audit_path}")
        count = sum(1 for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if count != 256:
            raise RuntimeError(f"Expected 256 instance rows, found {count}: {audit_path}")
    assert_gpu_log(log_path)
    assert_resolved_config(
        output / ".hydra" / "config.yaml",
        method=method,
        pool=pool,
        profile=profile,
    )


def run_job(
    *,
    method: str,
    seed: int,
    setting: str,
    pool: str,
    profile: bool,
    output: Path,
    log_path: Path,
    audit_path: Path | None,
) -> None:
    marker = output / "complete.json"
    if marker.is_file():
        validate_completed_job(
            output=output,
            log_path=log_path,
            method=method,
            pool=pool,
            profile=profile,
            audit_path=audit_path,
        )
        print(
            f"SKIP completed {method} seed={seed} setting={setting} pool={pool}",
            flush=True,
        )
        return

    output.mkdir(parents=True, exist_ok=True)
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.unlink(missing_ok=True)
    cmd = eval_command(
        method=method,
        seed=seed,
        setting=setting,
        pool=pool,
        output=output,
        profile=profile,
        audit_path=audit_path,
    )
    print(
        f"START {method} seed={seed} setting={setting} pool={pool} at {now()}",
        flush=True,
    )
    elapsed = run_logged(cmd, log_path)
    marker.write_text(
        json.dumps(
            {
                "method": method,
                "training_seed": seed,
                "setting": setting,
                "evaluation_seed": 0,
                "instances": 256,
                "pool": pool,
                "profile_search": profile,
                "checkpoint": str(checkpoint(method, seed)),
                "checkpoint_sha256": sha256(checkpoint(method, seed)),
                "elapsed_sec": elapsed,
                "log": str(log_path),
                "resolved_config": str(output / ".hydra" / "config.yaml"),
                "instance_jsonl": str(audit_path) if audit_path else None,
                "completed_at": now(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_completed_job(
        output=output,
        log_path=log_path,
        method=method,
        pool=pool,
        profile=profile,
        audit_path=audit_path,
    )
    print(
        f"DONE  {method} seed={seed} setting={setting} elapsed={elapsed/60:.2f} min",
        flush=True,
    )


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def collect_efficiency() -> None:
    for method in METHODS:
        for seed in range(3):
            for setting in SETTINGS:
                output = EFF_OUTPUT_ROOT / method / f"seed{seed}" / tag(setting)
                log = (
                    EFF_RESULT_ROOT
                    / "logs"
                    / f"{METHODS[method]['log_key']}_ckpt{seed}_{tag(setting)}.log"
                )
                validate_completed_job(
                    output=output,
                    log_path=log,
                    method=method,
                    pool="official",
                    profile=True,
                    audit_path=None,
                )
    log_root = EFF_RESULT_ROOT / "logs"
    summary_script = ROOT / "scripts" / "summarize_complexity_profile.py"
    subprocess.run([sys.executable, str(summary_script), str(log_root)], cwd=ROOT, check=True)
    raw_path = log_root / "complexity_profile_raw.csv"
    with raw_path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 12:
        raise RuntimeError(f"Expected 12 efficiency rows, found {len(raw)}")
    observed = {(r["method"], r["ckpt"], r["setting"]) for r in raw}
    expected = {
        (METHODS[method]["log_key"], str(seed), setting)
        for method in METHODS
        for seed in range(3)
        for setting in SETTINGS
    }
    if observed != expected:
        raise RuntimeError(f"Efficiency grid mismatch: missing={expected-observed}, extra={observed-expected}")
    for row in raw:
        for field in PROFILE_FIELDS:
            if row[field] == "":
                raise RuntimeError(f"Missing {field} in efficiency row: {row}")

    main_raw_path = MAIN_RESULT_ROOT / "raw_48_run_records.csv"
    if not main_raw_path.is_file():
        raise FileNotFoundError(f"Missing final Main Table raw records: {main_raw_path}")
    with main_raw_path.open(newline="", encoding="utf-8") as handle:
        main_raw = {
            (row["Method"], row["Training seed"], row["Setting"]): row
            for row in csv.DictReader(handle)
        }
    method_from_log_key = {
        config["log_key"]: method for method, config in METHODS.items()
    }
    for row in raw:
        key = (
            method_from_log_key[str(row["method"])],
            str(row["ckpt"]),
            str(row["setting"]),
        )
        if key not in main_raw:
            raise RuntimeError(f"Efficiency row missing from Main Table raw records: {key}")
        for metric in ("HR", "NDCG"):
            observed_metric = float(row[metric])
            expected_metric = float(main_raw[key][metric])
            if abs(observed_metric - expected_metric) > 5e-4:
                raise RuntimeError(
                    f"Profile run changed {metric} for {key}: "
                    f"profile={observed_metric}, main={expected_metric}"
                )

    comparison_rows = []
    for setting in SETTINGS:
        for field in PROFILE_FIELDS:
            rev = [
                float(r[field])
                for r in raw
                if r["method"] == "official" and r["setting"] == setting
            ]
            saif = [
                float(r[field])
                for r in raw
                if r["method"] == "saif" and r["setting"] == setting
            ]
            rev_mean, rev_sd = mean_sd(rev)
            saif_mean, saif_sd = mean_sd(saif)
            comparison_rows.append(
                {
                    "setting": setting,
                    "metric": field,
                    "RevFilter_mean": rev_mean,
                    "RevFilter_SD": rev_sd,
                    "SAIF_mean": saif_mean,
                    "SAIF_SD": saif_sd,
                    "SAIF_minus_RevFilter": saif_mean - rev_mean,
                    "SAIF_over_RevFilter": saif_mean / rev_mean if rev_mean != 0 else "",
                }
            )
    comparison_path = EFF_RESULT_ROOT / "efficiency_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    checkpoint_hashes = {
        f"{method}_seed{seed}": sha256(checkpoint(method, seed))
        for method in METHODS
        for seed in range(3)
    }
    integrity = {
        "phase": "efficiency",
        "frozen_backbone": "128/1/0.3/max/ELU",
        "settings": SETTINGS,
        "methods": list(METHODS),
        "training_seeds": [0, 1, 2],
        "evaluation_seed": 0,
        "instances_per_setting": 256,
        "expected_runs": 12,
        "observed_runs": len(raw),
        "pool": "official",
        "profile_search": True,
        "GPU_ASSERTION": "PASS",
        "CONFIG_ASSERTION": "PASS",
        "FINAL_CHECKPOINT_PROVENANCE": "PASS",
        "MAIN_METRIC_REPLICATION": "PASS",
        "checkpoint_sha256": checkpoint_hashes,
        "code_sha256": code_sha256(),
        "generated_at": now(),
    }
    (EFF_RESULT_ROOT / "integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Final efficiency profile (d0p3)",
        "",
        "- Runs: **12/12**",
        "- Settings: `10+1000@100`, `10+10000@100`",
        "- Checkpoints: final frozen RevFilter/SAIF seeds 0, 1, 2",
        "- GPU assertion: **PASS**",
        "- `search_elapsed_sec` is synchronized inside the iterative search loop.",
        "- `elapsed_sec` includes process startup, data loading, candidate construction, and evaluation.",
        "",
        (log_root / "complexity_profile_summary.md").read_text(encoding="utf-8"),
    ]
    (EFF_RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"EFFICIENCY RESULTS READY: {EFF_RESULT_ROOT}", flush=True)


def run_efficiency() -> None:
    log_root = EFF_RESULT_ROOT / "logs"
    for method in METHODS:
        for seed in range(3):
            for setting in SETTINGS:
                output = EFF_OUTPUT_ROOT / method / f"seed{seed}" / tag(setting)
                log = log_root / f"{METHODS[method]['log_key']}_ckpt{seed}_{tag(setting)}.log"
                run_job(
                    method=method,
                    seed=seed,
                    setting=setting,
                    pool="official",
                    profile=True,
                    output=output,
                    log_path=log,
                    audit_path=None,
                )
    collect_efficiency()


def collect_receiver_balanced() -> None:
    records = []
    reference_hashes: dict[tuple[str, int], str] = {}
    for method in METHODS:
        for seed in range(3):
            for setting in SETTINGS:
                output = RB_OUTPUT_ROOT / method / f"seed{seed}" / tag(setting)
                path = (
                    RB_RESULT_ROOT
                    / "instance_jsonl"
                    / method
                    / f"seed{seed}"
                    / f"{tag(setting)}.jsonl"
                )
                validate_completed_job(
                    output=output,
                    log_path=output / "evaluation.log",
                    method=method,
                    pool="balanced_receivers",
                    profile=False,
                    audit_path=path,
                )
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if len(rows) != 256:
                    raise RuntimeError(f"Expected 256 records in {path}, found {len(rows)}")
                for row in rows:
                    row["setting"] = setting
                    row["ckpt"] = str(seed)
                    denom = int(row["num_senders"]) * int(row["num_receivers"])
                    row["density"] = float(row["gt_unique_count"]) / denom
                    key = (setting, int(row["sample_index"]))
                    if key in reference_hashes and reference_hashes[key] != row["candidate_hash"]:
                        raise RuntimeError(f"Candidate hash mismatch for {key}")
                    reference_hashes[key] = row["candidate_hash"]
                    records.append(row)
    if len(records) != 3072:
        raise RuntimeError(f"Expected 3072 receiver-balanced records, found {len(records)}")
    if len(reference_hashes) != 512:
        raise RuntimeError(f"Expected 512 unique balanced candidates, found {len(reference_hashes)}")

    instance_path = RB_RESULT_ROOT / "receiver_balanced_instance_metrics_3072.csv"
    fields = [
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
        "density",
        "gt_unique_count",
        "hit_count",
        "HR",
        "NDCG",
    ]
    with instance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    raw_rows = []
    for method in METHODS:
        for seed in range(3):
            for setting in SETTINGS:
                group = [
                    row
                    for row in records
                    if row["method"] == method
                    and int(row["training_seed"]) == seed
                    and row["setting"] == setting
                ]
                raw_rows.append(
                    {
                        "Method": method,
                        "Training seed": seed,
                        "Setting": setting,
                        "Eval seed": 0,
                        "# instances": len(group),
                        "Density": statistics.mean(float(r["density"]) for r in group),
                        "HR": statistics.mean(float(r["HR"]) for r in group),
                        "NDCG": statistics.mean(float(r["NDCG"]) for r in group),
                    }
                )
    raw_path = RB_RESULT_ROOT / "receiver_balanced_raw_12.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows = []
    for setting in SETTINGS:
        row: dict[str, str | float] = {"Setting": setting}
        for method in METHODS:
            method_rows = [r for r in raw_rows if r["Method"] == method and r["Setting"] == setting]
            for metric in ("HR", "NDCG"):
                values = [float(r[metric]) for r in method_rows]
                mean_value, sd_value = mean_sd(values)
                row[f"{method} {metric} mean"] = mean_value
                row[f"{method} {metric} SD"] = sd_value
            row[f"{method} density"] = statistics.mean(float(r["Density"]) for r in method_rows)
        row["Delta HR"] = float(row["SAIF HR mean"]) - float(row["RevFilter HR mean"])
        row["Delta NDCG"] = float(row["SAIF NDCG mean"]) - float(row["RevFilter NDCG mean"])
        summary_rows.append(row)
    summary_path = RB_RESULT_ROOT / "receiver_balanced_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    stats_path = RB_RESULT_ROOT / "paired_receiver_balanced_wilcoxon_bh_all4.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "paired_instance_wilcoxon_bh.py"),
            "--a",
            str(instance_path),
            "--b",
            str(instance_path),
            "--a-name",
            "RevFilter",
            "--b-name",
            "SAIF",
            "--a-method",
            "RevFilter",
            "--b-method",
            "SAIF",
            "--out",
            str(stats_path),
            "--metrics",
            "HR",
            "NDCG",
            "--alternative",
            "two-sided",
            "--bh-scope",
            "all",
            "--expected-runs",
            "3",
            "--expected-samples",
            "256",
            "--expected-settings",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )
    with stats_path.open(newline="", encoding="utf-8") as handle:
        stats_rows = list(csv.DictReader(handle))
    if len(stats_rows) != 4:
        raise RuntimeError(f"Expected 4 receiver-balanced tests, found {len(stats_rows)}")

    official_path = (
        ROOT
        / "results"
        / "final_revfilter_saif_128_l1_d0p3"
        / "main_table_summary.csv"
    )
    if not official_path.is_file():
        raise FileNotFoundError(f"Missing official Main Table summary: {official_path}")
    with official_path.open(newline="", encoding="utf-8") as handle:
        official = {row["Setting"]: row for row in csv.DictReader(handle)}
    comparison = []
    for row in summary_rows:
        setting = str(row["Setting"])
        for metric in ("HR", "NDCG"):
            official_delta = float(official[setting][f"Delta {metric}"])
            balanced_delta = float(row[f"Delta {metric}"])
            comparison.append(
                {
                    "Setting": setting,
                    "Metric": metric,
                    "Official delta SAIF-RevFilter": official_delta,
                    "Receiver-balanced delta SAIF-RevFilter": balanced_delta,
                    "Balanced minus official delta": balanced_delta - official_delta,
                    "Official direction": "higher" if official_delta > 0 else "lower" if official_delta < 0 else "equal",
                    "Receiver-balanced direction": "higher" if balanced_delta > 0 else "lower" if balanced_delta < 0 else "equal",
                }
            )
    comparison_path = RB_RESULT_ROOT / "official_vs_receiver_balanced_deltas.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    integrity = {
        "phase": "receiver-balanced",
        "frozen_backbone": "128/1/0.3/max/ELU",
        "settings": SETTINGS,
        "methods": list(METHODS),
        "training_seeds": [0, 1, 2],
        "evaluation_seed": 0,
        "instances_per_setting": 256,
        "expected_evaluations": 12,
        "observed_evaluations": len(raw_rows),
        "expected_instance_records": 3072,
        "observed_instance_records": len(records),
        "pool": "balanced_receivers",
        "candidate_hash_assertion": "PASS",
        "same_candidate_instances_across_methods": True,
        "same_candidate_instances_across_three_runs": True,
        "paired_tests": 4,
        "BH_scope": "all four setting-metric comparisons",
        "GPU_ASSERTION": "PASS",
        "CONFIG_ASSERTION": "PASS",
        "FINAL_CHECKPOINT_PROVENANCE": "PASS",
        "code_sha256": code_sha256(),
        "generated_at": now(),
    }
    (RB_RESULT_ROOT / "integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Final receiver-balanced control (d0p3)",
        "",
        "- Evaluations: **12/12**",
        "- Instance records: **3072/3072**",
        "- Candidate matching across methods and seeds: **PASS**",
        "- Two-sided paired Wilcoxon tests; BH over all four comparisons.",
        "",
        "## Paired tests",
        "",
        stats_path.with_suffix(".md").read_text(encoding="utf-8"),
    ]
    (RB_RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"RECEIVER-BALANCED RESULTS READY: {RB_RESULT_ROOT}", flush=True)


def run_receiver_balanced() -> None:
    for method in METHODS:
        for seed in range(3):
            for setting in SETTINGS:
                output = RB_OUTPUT_ROOT / method / f"seed{seed}" / tag(setting)
                log = output / "evaluation.log"
                audit = (
                    RB_RESULT_ROOT
                    / "instance_jsonl"
                    / method
                    / f"seed{seed}"
                    / f"{tag(setting)}.jsonl"
                )
                run_job(
                    method=method,
                    seed=seed,
                    setting=setting,
                    pool="balanced_receivers",
                    profile=False,
                    output=output,
                    log_path=log,
                    audit_path=audit,
                )
    collect_receiver_balanced()


def dry_run(phase: str) -> None:
    phases = ["efficiency", "receiver-balanced"] if phase == "all" else [phase]
    print("DRY_RUN_ONLY")
    print("Frozen backbone: 128/1/0.3/max/ELU")
    print("Methods: RevFilter, SAIF")
    print("Training seeds: 0, 1, 2")
    print("Settings: " + ", ".join(SETTINGS))
    for item in phases:
        print(f"{item}: 12 planned evaluations")
    print("No training, tuning, or TEST-derived model selection is performed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["all", "efficiency", "receiver-balanced"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args.phase)
        return
    assert_checkpoints()
    if args.audit_only:
        if args.phase in {"all", "efficiency"}:
            collect_efficiency()
        if args.phase in {"all", "receiver-balanced"}:
            collect_receiver_balanced()
        print("AUDIT_ONLY = PASS", flush=True)
        return

    cuda = cuda_preflight()
    if args.phase in {"all", "efficiency"}:
        write_environment(EFF_RESULT_ROOT, cuda)
        run_efficiency()
    if args.phase in {"all", "receiver-balanced"}:
        write_environment(RB_RESULT_ROOT, cuda)
        run_receiver_balanced()
    print("AUTHORIZED SUPPORTING EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
