#!/usr/bin/env python3
"""Final candidate-order robustness evaluation for frozen d0p3 checkpoints.

This launcher never trains or selects a model. It evaluates the final
RevFilter/SAIF checkpoints under original ordering and three deterministic
sender/receiver shuffles, verifies membership/order hashes, and produces the
two predeclared paired Wilcoxon/BH families.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
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

CHECKPOINT_ROOT = ROOT / "checkpoints" / "final_revfilter_saif_128_l1_d0p3"
MAIN_RESULT_ROOT = ROOT / "results" / "final_revfilter_saif_128_l1_d0p3"
OUTPUT_ROOT = ROOT / "outputs" / "final_candidate_order_robustness_128_l1_d0p3"
RESULT_ROOT = ROOT / "results" / "final_candidate_order_robustness_128_l1_d0p3"

SETTINGS = ["10+1000@100", "10+10000@100"]
SEEDS = [0, 1, 2]
ORDER_CONDITIONS = {
    "original": ("original", 0),
    "shuffle_1": ("shuffle", 1),
    "shuffle_2": ("shuffle", 2),
    "shuffle_3": ("shuffle", 3),
}
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
]

INSTANCE_FIELDS = [
    "method",
    "training_seed",
    "ckpt",
    "setting",
    "order_condition",
    "order_seed",
    "eval_seed",
    "sample_index",
    "sample_id",
    "candidate_hash",
    "sender_order_hash",
    "receiver_order_hash",
    "candidate_membership_hash",
    "positive_pair_hash",
    "num_senders",
    "num_receivers",
    "initial_candidate_pairs",
    "gt_unique_count",
    "hit_count",
    "HR",
    "NDCG",
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
    files = [
        "scripts/run_final_candidate_order_robustness.py",
        "scripts/paired_instance_wilcoxon_bh.py",
        "algorithms/subgraph/iterative_filtering_algo.py",
        "algorithms/subgraph/models/anchor_double_deep_sets.py",
        "algorithms/subgraph/models/double_deep_sets.py",
        "datasets/elliptic/dataset.py",
        "configurations/algorithm/iterative_filtering.yaml",
        "configurations/algorithm/deepsets.yaml",
        "configurations/dataset/elliptic_recommendation.yaml",
        "configurations/experiment/exp_edge_recommendation.yaml",
    ]
    return {relative: sha256(ROOT / relative) for relative in files}


def checkpoint(method: str, seed: int) -> Path:
    return CHECKPOINT_ROOT / method / f"seed{seed}_stage2.ckpt"


def load_main_provenance() -> dict[tuple[str, int], dict]:
    path = MAIN_RESULT_ROOT / "checkpoint_provenance.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing final checkpoint provenance: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    indexed = {(str(row["method"]), int(row["seed"])): row for row in rows}
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(indexed) != expected:
        raise RuntimeError(
            f"Checkpoint provenance grid mismatch: missing={expected-set(indexed)}, "
            f"extra={set(indexed)-expected}"
        )
    return indexed


def assert_checkpoints() -> dict[tuple[str, int], dict]:
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
        raise RuntimeError(f"Unexpected Main-Table freeze: {integrity}")
    for method in METHODS:
        for seed in SEEDS:
            row = provenance[(method, seed)]
            if row.get("backbone") != "128/1/0.3/max":
                raise RuntimeError(
                    f"Unexpected backbone for {method} seed={seed}: {row.get('backbone')}"
                )
            path = checkpoint(method, seed)
            if not path.is_file():
                raise FileNotFoundError(f"Missing final checkpoint: {path}")
            expected_hash = row["stages"]["stage2"]["sha256"]
            if sha256(path) != expected_hash:
                raise RuntimeError(f"Final checkpoint hash mismatch: {path}")
            if row["stages"]["stage2"].get("test_metric_during_training") is True:
                raise RuntimeError(f"TEST metric appeared during training: {method} seed={seed}")
    print("CHECKPOINT_PROVENANCE = PASS", flush=True)
    print("TEST_DERIVED_MODEL_SELECTION = false", flush=True)
    return provenance


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
        raise RuntimeError("CUDA GPU is required for this formal experiment")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU with CUDA_VISIBLE_DEVICES; "
            f"found {torch.cuda.device_count()} visible devices"
        )
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["current_cuda_device"] = torch.cuda.current_device()
    print(f"GPU name: {info['gpu_name']}", flush=True)
    print("GPU_ASSERTION = PASS", flush=True)
    return info


def write_environment(cuda: dict) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "environment.json").write_text(
        json.dumps(cuda, indent=2) + "\n", encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (RESULT_ROOT / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")


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


def evaluation_paths(
    method: str, seed: int, setting: str, order_condition: str
) -> tuple[Path, Path, Path]:
    output = (
        OUTPUT_ROOT
        / method
        / f"seed{seed}"
        / order_condition
        / tag(setting)
    )
    audit = (
        RESULT_ROOT
        / "instance_jsonl"
        / method
        / f"seed{seed}"
        / order_condition
        / f"{tag(setting)}.jsonl"
    )
    return output, audit, output / "complete.json"


def assert_gpu_log(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "GPU available: True (cuda), used: True" not in text:
        raise RuntimeError(f"GPU-use assertion failed: {path}")


def assert_resolved_config(
    path: Path,
    *,
    method: str,
    candidate_order: str,
    order_seed: int,
) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    assert int(cfg.algorithm.model.input_dim) == 43
    assert int(cfg.algorithm.model.hidden_dim) == 128
    assert int(cfg.algorithm.model.num_layers) == 1
    assert float(cfg.algorithm.model.dropout) == 0.3
    assert cfg.algorithm.model.pool == "max"
    assert cfg.algorithm.model.activation == "ELU"
    assert float(cfg.algorithm.keep_multiplier) == 1.5
    assert float(cfg.dataset.augment.gamma) == 0.4
    assert int(cfg.dataset.augment.min) == 1
    assert int(cfg.dataset.augment.max) == 20
    assert bool(cfg.algorithm.train_with_1_1) is False
    assert cfg.algorithm.candidate_order == candidate_order
    assert int(cfg.algorithm.candidate_order_seed) == order_seed
    assert bool(cfg.algorithm.use_anchor_features) is (method == "SAIF")
    if method == "SAIF":
        assert cfg.algorithm.model.anchor_feature_mode == "full"
        assert int(cfg.algorithm.model.anchor_input_dim) == 6
        assert cfg.algorithm.model.anchor_normalization == "layernorm"
        assert cfg.algorithm.model.anchor_control_mode == "normal"
    assert list(cfg.experiment.tasks) == ["test"]
    assert bool(cfg.experiment.validation.test_during_training) is False
    assert int(cfg.dataset.num_samples) == 256
    assert cfg.dataset.eval_pool_mode == "official"
    assert bool(cfg.dataset.augment.enabled) is False
    assert int(cfg.seed) == 0


def evaluate(
    method: str,
    seed: int,
    setting: str,
    order_condition: str,
    *,
    dry_run: bool,
) -> None:
    candidate_order, order_seed = ORDER_CONDITIONS[order_condition]
    output, audit, marker = evaluation_paths(method, seed, setting, order_condition)
    ckpt = checkpoint(method, seed)
    if marker.is_file() and audit.is_file() and not dry_run:
        rows = [line for line in audit.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if len(rows) != 256:
            raise RuntimeError(f"Incomplete saved audit: {audit}")
        if payload.get("checkpoint_sha256") != sha256(ckpt):
            raise RuntimeError(f"Evaluation checkpoint hash mismatch: {marker}")
        assert_gpu_log(output / "evaluation.log")
        assert_resolved_config(
            output / ".hydra" / "config.yaml",
            method=method,
            candidate_order=candidate_order,
            order_seed=order_seed,
        )
        print(
            f"SKIP completed {method} seed={seed} {setting} {order_condition}",
            flush=True,
        )
        return

    output.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        audit.unlink(missing_ok=True)
    name = f"FinalOrder_{method}_seed{seed}_{order_condition}_{tag(setting)}"
    cmd = base_main(name) + [
        "dataset=elliptic_recommendation",
        "algorithm=iterative_filtering",
        *METHODS[method],
        *FROZEN,
        f"algorithm.candidate_order={candidate_order}",
        f"algorithm.candidate_order_seed={order_seed}",
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
        f"load={ckpt.as_posix()}",
        f"+shortcut={setting}",
        f"+algorithm.audit_output_path={audit.as_posix()}",
        f"+algorithm.audit_method={method}",
        f"+algorithm.audit_training_seed={seed}",
        "+algorithm.audit_eval_seed=0",
        f"+algorithm.audit_setting={tag(setting)}",
        f"hydra.run.dir={output.as_posix()}",
    ]
    print(
        f"START {method} seed={seed} setting={setting} order={order_condition} at {now()}",
        flush=True,
    )
    elapsed = run_logged(cmd, output / "evaluation.log", dry_run=dry_run)
    if dry_run:
        return
    rows = [line for line in audit.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 256:
        raise RuntimeError(f"Expected 256 instance records, found {len(rows)}: {audit}")
    first = json.loads(rows[0])
    required_hashes = {
        "sender_order_hash",
        "receiver_order_hash",
        "candidate_membership_hash",
        "positive_pair_hash",
    }
    if not required_hashes.issubset(first):
        raise RuntimeError(
            "Order-hash instrumentation is missing. Upload the patched "
            "algorithms/subgraph/iterative_filtering_algo.py before running."
        )
    assert_gpu_log(output / "evaluation.log")
    assert_resolved_config(
        output / ".hydra" / "config.yaml",
        method=method,
        candidate_order=candidate_order,
        order_seed=order_seed,
    )
    marker.write_text(
        json.dumps(
            {
                "method": method,
                "training_seed": seed,
                "setting": setting,
                "order_condition": order_condition,
                "candidate_order": candidate_order,
                "order_seed": order_seed,
                "eval_seed": 0,
                "instances": len(rows),
                "checkpoint": str(ckpt),
                "checkpoint_sha256": sha256(ckpt),
                "elapsed_sec": elapsed,
                "instance_jsonl": str(audit),
                "completed_at": now(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE  {method} seed={seed} {setting} {order_condition} "
        f"elapsed={elapsed/60:.1f} min",
        flush=True,
    )


def read_records() -> list[dict]:
    records = []
    required = {
        "sender_order_hash",
        "receiver_order_hash",
        "candidate_membership_hash",
        "positive_pair_hash",
    }
    for method in METHODS:
        for seed in SEEDS:
            for setting in SETTINGS:
                for condition, (_, order_seed) in ORDER_CONDITIONS.items():
                    _, path, _ = evaluation_paths(method, seed, setting, condition)
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
                        if not required.issubset(row):
                            raise RuntimeError(f"Missing order hashes in {path}")
                        row["method"] = method
                        row["training_seed"] = seed
                        row["ckpt"] = str(seed)
                        row["setting"] = setting
                        row["order_condition"] = condition
                        row["order_seed"] = order_seed
                        row["initial_candidate_pairs"] = (
                            int(row["num_senders"]) * int(row["num_receivers"])
                        )
                    records.extend(rows)
    return records


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_main_candidate_hashes() -> dict[tuple[str, int], str]:
    path = MAIN_RESULT_ROOT / "instance_metrics_12288.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Main-Table instance records: {path}")
    hashes: dict[tuple[str, int], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["setting"] not in SETTINGS:
                continue
            key = (row["setting"], int(row["sample_index"]))
            value = row["candidate_hash"]
            if key in hashes and hashes[key] != value:
                raise RuntimeError(f"Main-Table candidate mismatch for {key}")
            hashes[key] = value
    if len(hashes) != len(SETTINGS) * 256:
        raise RuntimeError(f"Expected 512 Main-Table candidates, found {len(hashes)}")
    return hashes


def audit_order_hashes(records: list[dict]) -> list[dict]:
    main_hashes = load_main_candidate_hashes()
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["setting"], int(row["sample_index"]), row["order_condition"])].append(row)
    expected_keys = len(SETTINGS) * 256 * len(ORDER_CONDITIONS)
    if len(grouped) != expected_keys:
        raise RuntimeError(f"Expected {expected_keys} order-instance groups, found {len(grouped)}")

    audit_rows = []
    references: dict[tuple[str, int], dict] = {}
    for setting in SETTINGS:
        for sample_index in range(256):
            original_group = grouped[(setting, sample_index, "original")]
            original = original_group[0]
            references[(setting, sample_index)] = original
            if original["candidate_hash"] != main_hashes[(setting, sample_index)]:
                raise RuntimeError(
                    f"Original membership/ground-truth differs from Main Table: "
                    f"{(setting, sample_index)}"
                )

    for key in sorted(grouped):
        setting, sample_index, condition = key
        rows = grouped[key]
        if len(rows) != 6:
            raise RuntimeError(f"Expected six model/run rows for {key}, found {len(rows)}")
        for field in (
            "sender_order_hash",
            "receiver_order_hash",
            "candidate_membership_hash",
            "positive_pair_hash",
            "candidate_hash",
            "num_senders",
            "num_receivers",
            "initial_candidate_pairs",
        ):
            if len({str(row[field]) for row in rows}) != 1:
                raise RuntimeError(f"Unmatched {field} across methods/runs for {key}")
        row = rows[0]
        original = references[(setting, sample_index)]
        membership_same = row["candidate_membership_hash"] == original["candidate_membership_hash"]
        positives_same = row["positive_pair_hash"] == original["positive_pair_hash"]
        counts_same = (
            int(row["num_senders"]) == int(original["num_senders"])
            and int(row["num_receivers"]) == int(original["num_receivers"])
            and int(row["initial_candidate_pairs"]) == int(original["initial_candidate_pairs"])
        )
        if not membership_same or not positives_same or not counts_same:
            raise RuntimeError(f"Shuffle changed candidate content for {key}")
        sender_changed = row["sender_order_hash"] != original["sender_order_hash"]
        receiver_changed = row["receiver_order_hash"] != original["receiver_order_hash"]
        audit_rows.append(
            {
                "setting": setting,
                "sample_index": sample_index,
                "sample_id": row["sample_id"],
                "order_condition": condition,
                "order_seed": row["order_seed"],
                "num_senders": row["num_senders"],
                "num_receivers": row["num_receivers"],
                "initial_candidate_pairs": row["initial_candidate_pairs"],
                "sender_order_hash": row["sender_order_hash"],
                "receiver_order_hash": row["receiver_order_hash"],
                "candidate_membership_hash": row["candidate_membership_hash"],
                "positive_pair_hash": row["positive_pair_hash"],
                "candidate_hash": row["candidate_hash"],
                "membership_same_as_original": membership_same,
                "positive_pairs_same_as_original": positives_same,
                "counts_same_as_original": counts_same,
                "sender_order_changed_from_original": sender_changed,
                "receiver_order_changed_from_original": receiver_changed,
                "matched_order_across_methods": True,
                "matched_order_across_training_seeds": True,
            }
        )

    # A random permutation may legitimately be the identity for one seed,
    # especially for a two-element side.  Require that the predeclared set of
    # three shuffles perturbs every non-singleton side at least once; never
    # replace or select a seed based on this check.
    for setting in SETTINGS:
        for sample_index in range(256):
            sample_rows = [
                row
                for row in audit_rows
                if row["setting"] == setting
                and int(row["sample_index"]) == sample_index
                and row["order_condition"] != "original"
            ]
            original = references[(setting, sample_index)]
            if int(original["num_senders"]) > 1 and not any(
                row["sender_order_changed_from_original"] for row in sample_rows
            ):
                raise RuntimeError(
                    f"No sender-order perturbation across seeds for {(setting, sample_index)}"
                )
            if int(original["num_receivers"]) > 1 and not any(
                row["receiver_order_changed_from_original"] for row in sample_rows
            ):
                raise RuntimeError(
                    f"No receiver-order perturbation across seeds for {(setting, sample_index)}"
                )
    return audit_rows


def aggregate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[
            (
                row["method"],
                int(row["training_seed"]),
                row["setting"],
                row["order_condition"],
            )
        ].append(row)
    raw = []
    for (method, seed, setting, condition), rows in grouped.items():
        if len(rows) != 256:
            raise RuntimeError(f"Expected 256 rows for {(method, seed, setting, condition)}")
        raw.append(
            {
                "Method": method,
                "Training seed": seed,
                "Setting": setting,
                "Order": condition,
                "Order seed": ORDER_CONDITIONS[condition][1],
                "Eval seed": 0,
                "# instances": 256,
                "HR": statistics.mean(float(row["HR"]) for row in rows),
                "NDCG": statistics.mean(float(row["NDCG"]) for row in rows),
            }
        )
    raw.sort(
        key=lambda row: (
            SETTINGS.index(row["Setting"]),
            list(ORDER_CONDITIONS).index(row["Order"]),
            list(METHODS).index(row["Method"]),
            int(row["Training seed"]),
        )
    )
    summary = []
    for setting in SETTINGS:
        for condition in ORDER_CONDITIONS:
            item: dict[str, str | float] = {"Setting": setting, "Order": condition}
            for method in METHODS:
                method_rows = [
                    row
                    for row in raw
                    if row["Setting"] == setting
                    and row["Order"] == condition
                    and row["Method"] == method
                ]
                if len(method_rows) != 3:
                    raise RuntimeError(f"Expected three runs for {(setting, condition, method)}")
                for metric in ("HR", "NDCG"):
                    values = [float(row[metric]) for row in method_rows]
                    item[f"{method} {metric} mean"] = statistics.mean(values)
                    item[f"{method} {metric} SD"] = statistics.stdev(values)
            item["Delta HR"] = float(item["SAIF HR mean"]) - float(item["RevFilter HR mean"])
            item["Delta NDCG"] = float(item["SAIF NDCG mean"]) - float(item["RevFilter NDCG mean"])
            summary.append(item)
    return raw, summary


def assert_original_replication(raw: list[dict]) -> None:
    path = MAIN_RESULT_ROOT / "raw_48_run_records.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Main-Table aggregate records: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        main = {
            (row["Method"], int(row["Training seed"]), row["Setting"]): row
            for row in csv.DictReader(handle)
            if row["Setting"] in SETTINGS
        }
    original = [row for row in raw if row["Order"] == "original"]
    if len(original) != 12:
        raise RuntimeError(f"Expected 12 original-order aggregate rows, found {len(original)}")
    for row in original:
        key = (row["Method"], int(row["Training seed"]), row["Setting"])
        if key not in main:
            raise RuntimeError(f"Original row absent from Main Table: {key}")
        for metric in ("HR", "NDCG"):
            if abs(float(row[metric]) - float(main[key][metric])) > 5e-4:
                raise RuntimeError(
                    f"Original ordering did not replicate Main Table for {key} {metric}: "
                    f"observed={row[metric]}, expected={main[key][metric]}"
                )
    print("ORIGINAL_MAIN_TABLE_REPLICATION = PASS", flush=True)


def averaged_instances(records: list[dict]) -> dict[tuple[str, str, str, int, str], float]:
    grouped: dict[tuple[str, str, str, int, str], dict[str, float]] = defaultdict(dict)
    for row in records:
        for metric in ("HR", "NDCG"):
            key = (
                row["method"],
                row["order_condition"],
                row["setting"],
                int(row["sample_index"]),
                metric,
            )
            run = str(row["ckpt"])
            if run in grouped[key]:
                raise RuntimeError(f"Duplicate run {run} for {key}")
            grouped[key][run] = float(row[metric])
    averaged = {}
    for key, values in grouped.items():
        if set(values) != {"0", "1", "2"}:
            raise RuntimeError(f"Training-run mismatch for {key}: {sorted(values)}")
        averaged[key] = statistics.mean(values.values())
    return averaged


def test_row(
    *,
    comparison: str,
    setting: str,
    order: str,
    metric: str,
    values: list[float],
    a_mean: float | None = None,
    b_mean: float | None = None,
) -> dict:
    from scripts.paired_instance_wilcoxon_bh import wilcoxon_signed_rank

    if len(values) != 256:
        raise RuntimeError(f"Expected 256 paired values for {comparison}")
    test = wilcoxon_signed_rank(values, alternative="two-sided")
    row = {
        "comparison": comparison,
        "setting": setting,
        "order": order,
        "metric": metric,
        "n": len(values),
        "nonzero_n": test["nonzero_n"],
        "delta_mean": f"{statistics.mean(values):+.10f}",
        "positive": f"{sum(value > 0 for value in values)}/{len(values)}",
        "z": f"{test['z']:.6f}",
        "effect_r": f"{test['effect_r']:.6f}",
        "p": f"{test['p']:.10g}",
        "q_bh": "",
    }
    if a_mean is not None and b_mean is not None:
        row["RevFilter_mean"] = f"{a_mean:.10f}"
        row["SAIF_mean"] = f"{b_mean:.10f}"
    return row


def apply_bh(rows: list[dict]) -> None:
    from scripts.paired_instance_wilcoxon_bh import bh_adjust

    for row, q in zip(rows, bh_adjust([float(row["p"]) for row in rows])):
        row["q_bh"] = f"{q:.10g}"


def statistical_tests(records: list[dict]) -> tuple[list[dict], list[dict]]:
    averaged = averaged_instances(records)
    pairwise = []
    for setting in SETTINGS:
        for condition in ORDER_CONDITIONS:
            for metric in ("HR", "NDCG"):
                rev = [
                    averaged[("RevFilter", condition, setting, idx, metric)]
                    for idx in range(256)
                ]
                saif = [
                    averaged[("SAIF", condition, setting, idx, metric)]
                    for idx in range(256)
                ]
                diffs = [b - a for a, b in zip(rev, saif)]
                pairwise.append(
                    test_row(
                        comparison="SAIF - RevFilter",
                        setting=setting,
                        order=condition,
                        metric=metric,
                        values=diffs,
                        a_mean=statistics.mean(rev),
                        b_mean=statistics.mean(saif),
                    )
                )
    apply_bh(pairwise)

    interactions = []
    for setting in SETTINGS:
        for condition in ("shuffle_1", "shuffle_2", "shuffle_3"):
            for metric in ("HR", "NDCG"):
                values = []
                for idx in range(256):
                    shuffled_delta = (
                        averaged[("SAIF", condition, setting, idx, metric)]
                        - averaged[("RevFilter", condition, setting, idx, metric)]
                    )
                    original_delta = (
                        averaged[("SAIF", "original", setting, idx, metric)]
                        - averaged[("RevFilter", "original", setting, idx, metric)]
                    )
                    values.append(shuffled_delta - original_delta)
                interactions.append(
                    test_row(
                        comparison="Shuffled SAIF effect - original SAIF effect",
                        setting=setting,
                        order=condition,
                        metric=metric,
                        values=values,
                    )
                )
    apply_bh(interactions)
    return pairwise, interactions


def write_stats_markdown(path: Path, title: str, rows: list[dict], *, pairwise: bool) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(
            "Two-sided paired instance-level Wilcoxon signed-rank tests. "
            "The three independently trained checkpoints are averaged per "
            "`(setting, order, sample_index)` before testing. BH correction "
            "covers every row in this file.\n\n"
        )
        if pairwise:
            handle.write(
                "| setting | order | metric | n | nonzero n | RevFilter mean | "
                "SAIF mean | delta | positive | z | effect r | p | BH q |\n"
            )
            handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                handle.write(
                    f"| {row['setting']} | {row['order']} | {row['metric']} | "
                    f"{row['n']} | {row['nonzero_n']} | {row['RevFilter_mean']} | "
                    f"{row['SAIF_mean']} | {row['delta_mean']} | {row['positive']} | "
                    f"{row['z']} | {row['effect_r']} | {row['p']} | {row['q_bh']} |\n"
                )
        else:
            handle.write(
                "| setting | shuffle | metric | n | nonzero n | interaction mean | "
                "positive | z | effect r | p | BH q |\n"
            )
            handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                handle.write(
                    f"| {row['setting']} | {row['order']} | {row['metric']} | "
                    f"{row['n']} | {row['nonzero_n']} | {row['delta_mean']} | "
                    f"{row['positive']} | {row['z']} | {row['effect_r']} | "
                    f"{row['p']} | {row['q_bh']} |\n"
                )


def delta_summary(summary: list[dict]) -> list[dict]:
    rows = []
    for setting in SETTINGS:
        for metric in ("HR", "NDCG"):
            by_order = {
                row["Order"]: float(row[f"Delta {metric}"])
                for row in summary
                if row["Setting"] == setting
            }
            shuffled = [by_order[f"shuffle_{seed}"] for seed in (1, 2, 3)]
            rows.append(
                {
                    "Setting": setting,
                    "Metric": metric,
                    "Original delta": by_order["original"],
                    "Mean shuffled delta": statistics.mean(shuffled),
                    "Min shuffled delta": min(shuffled),
                    "Max shuffled delta": max(shuffled),
                    "Qualitative direction persists": all(
                        value == 0 or by_order["original"] == 0 or (value > 0) == (by_order["original"] > 0)
                        for value in shuffled
                    ),
                }
            )
    return rows


def maximum_absolute_shifts(summary: list[dict]) -> list[dict]:
    rows = []
    for setting in SETTINGS:
        for method in METHODS:
            for metric in ("HR", "NDCG"):
                original = next(
                    float(row[f"{method} {metric} mean"])
                    for row in summary
                    if row["Setting"] == setting and row["Order"] == "original"
                )
                shuffled = [
                    float(row[f"{method} {metric} mean"])
                    for row in summary
                    if row["Setting"] == setting and row["Order"] != "original"
                ]
                rows.append(
                    {
                        "Setting": setting,
                        "Method": method,
                        "Metric": metric,
                        "Original": original,
                        "Min shuffled": min(shuffled),
                        "Max shuffled": max(shuffled),
                        "Max absolute shift from original": max(abs(value - original) for value in shuffled),
                    }
                )
    return rows


def markdown_table_summary(summary: list[dict]) -> str:
    lines = [
        "| Setting | Order | RevFilter HR | RevFilter NDCG | SAIF HR | SAIF NDCG | Delta HR | Delta NDCG |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def ms(method: str, metric: str) -> str:
            return f"{float(row[f'{method} {metric} mean']):.4f}+/-{float(row[f'{method} {metric} SD']):.4f}"
        lines.append(
            f"| {row['Setting']} | {row['Order']} | {ms('RevFilter', 'HR')} | "
            f"{ms('RevFilter', 'NDCG')} | {ms('SAIF', 'HR')} | "
            f"{ms('SAIF', 'NDCG')} | {float(row['Delta HR']):+.4f} | "
            f"{float(row['Delta NDCG']):+.4f} |"
        )
    return "\n".join(lines)


def collect(provenance: dict[tuple[str, int], dict]) -> None:
    records = read_records()
    if len(records) != 12288:
        raise RuntimeError(f"Expected 12,288 instance records, found {len(records)}")
    order_audit = audit_order_hashes(records)
    if len(order_audit) != 2048:
        raise RuntimeError(f"Expected 2,048 order-hash audit rows, found {len(order_audit)}")
    raw, summary = aggregate(records)
    if len(raw) != 48 or len(summary) != 8:
        raise RuntimeError(f"Unexpected aggregate counts: raw={len(raw)}, summary={len(summary)}")
    assert_original_replication(raw)
    deltas = delta_summary(summary)
    shifts = maximum_absolute_shifts(summary)
    pairwise, interactions = statistical_tests(records)
    if len(pairwise) != 16 or len(interactions) != 12:
        raise RuntimeError(
            f"Unexpected test counts: pairwise={len(pairwise)}, interactions={len(interactions)}"
        )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULT_ROOT / "candidate_order_instance_metrics.csv", records, INSTANCE_FIELDS)
    write_csv(RESULT_ROOT / "candidate_order_raw_48.csv", raw)
    write_csv(RESULT_ROOT / "candidate_order_summary.csv", summary)
    write_csv(RESULT_ROOT / "candidate_order_delta_summary.csv", deltas)
    write_csv(RESULT_ROOT / "candidate_order_absolute_shift_summary.csv", shifts)
    write_csv(RESULT_ROOT / "order_hash_audit.csv", order_audit)
    pair_path = RESULT_ROOT / "candidate_order_saif_vs_revfilter_wilcoxon_bh_all16.csv"
    interaction_path = RESULT_ROOT / "candidate_order_interaction_wilcoxon_bh_all12.csv"
    write_csv(pair_path, pairwise)
    write_csv(interaction_path, interactions)
    write_stats_markdown(
        pair_path.with_suffix(".md"),
        "Candidate-Order SAIF vs RevFilter Tests",
        pairwise,
        pairwise=True,
    )
    write_stats_markdown(
        interaction_path.with_suffix(".md"),
        "Candidate-Order Interaction Tests",
        interactions,
        pairwise=False,
    )

    checkpoint_hashes = {
        f"{method}_seed{seed}": sha256(checkpoint(method, seed))
        for method in METHODS
        for seed in SEEDS
    }
    integrity = {
        "phase": "final-candidate-order-robustness",
        "frozen_backbone": "128/1/0.3/max/ELU",
        "methods": list(METHODS),
        "settings": SETTINGS,
        "training_seeds": SEEDS,
        "evaluation_seed": 0,
        "order_conditions": list(ORDER_CONDITIONS),
        "order_seeds": {
            condition: order_seed
            for condition, (_, order_seed) in ORDER_CONDITIONS.items()
        },
        "instances_per_setting": 256,
        "expected_aggregate_record_count": 48,
        "aggregate_record_count": len(raw),
        "expected_instance_record_count": 12288,
        "instance_record_count": len(records),
        "same_candidate_membership": "PASS",
        "same_ground_truth": "PASS",
        "same_candidate_counts": "PASS",
        "matched_order_across_methods": "PASS",
        "matched_order_across_training_seeds": "PASS",
        "original_main_table_replication": "PASS",
        "test_derived_model_selection": False,
        "GPU_ASSERTION": "PASS",
        "CONFIG_ASSERTION": "PASS",
        "CHECKPOINT_PROVENANCE": "PASS",
        "ORDER_HASH_ASSERTION": "PASS",
        "paired_saif_vs_revfilter_tests": len(pairwise),
        "order_interaction_tests": len(interactions),
        "BH_scope_pairwise": "all 16 planned order-setting-metric comparisons",
        "BH_scope_interaction": "all 12 planned shuffle-vs-original interactions",
        "checkpoint_sha256": checkpoint_hashes,
        "code_sha256": code_sha256(),
        "generated_at": now(),
    }
    (RESULT_ROOT / "integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )

    significant_interactions = sum(float(row["q_bh"]) < 0.05 for row in interactions)
    persistent = sum(str(row["Qualitative direction persists"]).lower() == "true" for row in deltas)
    lines = [
        "# Final Candidate-Order Robustness (d0p3)",
        "",
        "## A. Integrity",
        "",
        "- Frozen checkpoints: **PASS**",
        "- Settings: `10+1000@100`, `10+10000@100`",
        "- Orders: `original`, `shuffle_1`, `shuffle_2`, `shuffle_3`",
        "- Same candidate membership and positive pairs: **PASS**",
        "- Matched order across methods and training seeds: **PASS**",
        "- Original Main-Table replication: **PASS**",
        "- Aggregate records: **48/48**",
        "- Instance records: **12,288/12,288**",
        "- GPU assertion: **PASS**",
        "- TEST-derived model selection: **false**",
        "",
        "## B. Absolute results",
        "",
        markdown_table_summary(summary),
        "",
        "## C. SAIF--RevFilter deltas",
        "",
        "The final two columns in the table above report SAIF minus RevFilter.",
        "",
        "## D. Delta stability",
        "",
        "| Setting | Metric | Original delta | Mean shuffled delta | Shuffled range | Direction persists |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['Setting']} | {row['Metric']} | {float(row['Original delta']):+.6f} | "
            f"{float(row['Mean shuffled delta']):+.6f} | "
            f"[{float(row['Min shuffled delta']):+.6f}, {float(row['Max shuffled delta']):+.6f}] | "
            f"{row['Qualitative direction persists']} |"
        )
    lines += [
        "",
        "## E. Paired SAIF-vs-RevFilter tests",
        "",
        pair_path.with_suffix(".md").read_text(encoding="utf-8"),
        "",
        "## F. Order-interaction tests",
        "",
        interaction_path.with_suffix(".md").read_text(encoding="utf-8"),
        "",
        "## G. Mechanical conclusion",
        "",
        f"- Delta direction persisted in **{persistent}/4** setting-metric summaries.",
        f"- Significant shuffled-vs-original interactions after BH: **{significant_interactions}/12**.",
        "- Per-method maximum absolute shifts are reported in `candidate_order_absolute_shift_summary.csv`.",
        "- These are sensitivity results; no claim of mathematical order invariance is made.",
    ]
    (RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"CANDIDATE-ORDER RESULTS READY: {RESULT_ROOT}", flush=True)


def run(*, dry_run: bool) -> None:
    for method in METHODS:
        for seed in SEEDS:
            for setting in SETTINGS:
                for condition in ORDER_CONDITIONS:
                    evaluate(method, seed, setting, condition, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if sum((args.dry_run, args.collect_only, args.audit_only)) > 1:
        parser.error("--dry-run, --collect-only, and --audit-only are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    provenance = assert_checkpoints()
    cuda = cuda_preflight()
    if not args.dry_run:
        write_environment(cuda)
    if args.audit_only:
        collect(provenance)
        print("AUDIT_ONLY = PASS", flush=True)
        return
    if not args.collect_only:
        run(dry_run=args.dry_run)
    if args.dry_run:
        print("DRY_RUN = PASS", flush=True)
        return
    collect(provenance)
    print("CANDIDATE_ORDER_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
