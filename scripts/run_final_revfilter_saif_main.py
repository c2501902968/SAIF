#!/usr/bin/env python3
"""Frozen final RevFilter/SAIF retraining and matched 8-setting evaluation.

The launcher is restartable. Completed stages are identified by a small marker
beside copied final checkpoints; no seed or setting is selected or discarded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "final_revfilter_saif_128_l1_d0p3"
RESULT_ROOT = ROOT / "results" / "final_revfilter_saif_128_l1_d0p3"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "final_revfilter_saif_128_l1_d0p3"
SETTINGS = [
    "1+5@1", "1+10@1", "1+10@3", "1+100@3",
    "3+100@10", "3+1000@10", "10+1000@100", "10+10000@100",
]
METHODS = {
    "RevFilter": [],
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
    "algorithm.candidate_order=original",
    "algorithm.candidate_order_seed=0",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(value: str) -> str:
    return value.replace("+", "p").replace("@", "at").replace(".", "p")


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
        handle.write(f"\nRETURN_CODE={rc}\nELAPSED_SEC={elapsed:.3f}\n")
    if rc != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"Command failed ({rc}): {log_path}\n{tail}")
    return elapsed


def base_main(name: str) -> list[str]:
    return [sys.executable, "-m", "main", f"+name={name}"]


def stage_paths(method: str, seed: int, stage: str) -> tuple[Path, Path, Path]:
    output = OUTPUT_ROOT / "training" / method / f"seed{seed}" / stage
    copied = CHECKPOINT_ROOT / method / f"seed{seed}_{stage}.ckpt"
    marker = copied.with_suffix(".complete.json")
    return output, copied, marker


def find_checkpoint(output: Path) -> Path:
    paths = sorted((output / "checkpoints").glob("*.ckpt"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one best checkpoint in {output}, found {paths}")
    return paths[0]


def train_stage(method: str, seed: int, stage: str, load: Path | None) -> Path:
    output, copied, marker = stage_paths(method, seed, stage)
    if copied.is_file() and marker.is_file():
        print(f"SKIP completed {method} seed={seed} stage={stage}", flush=True)
        return copied
    output.mkdir(parents=True, exist_ok=True)
    copied.parent.mkdir(parents=True, exist_ok=True)
    name = f"Final_{method}_{stage}_seed{seed}"
    cmd = base_main(name) + [
        "dataset=elliptic_recommendation",
        "algorithm=iterative_filtering",
        *METHODS[method],
        *FROZEN,
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
            raise ValueError("Stage 2 requires the Stage-1 checkpoint")
        cmd += [
            "dataset.augment.enabled=true",
            "experiment.training.max_epochs=300",
            "experiment.training.early_stopping.enabled=false",
            f"load={load.as_posix()}",
        ]
    else:
        raise ValueError(stage)
    print(f"START {method} seed={seed} stage={stage} at {now()}", flush=True)
    elapsed = run_logged(cmd, output / "training.log")
    source = find_checkpoint(output)
    shutil.copy2(source, copied)
    payload = {
        "method": method, "seed": seed, "stage": stage,
        "started_from": str(load) if load else None,
        "source_checkpoint": str(source), "copied_checkpoint": str(copied),
        "sha256": sha256(copied), "elapsed_sec": elapsed,
        "resolved_config": str(output / ".hydra" / "config.yaml"),
        "training_log": str(output / "training.log"),
        "completed_at": now(),
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"DONE  {method} seed={seed} stage={stage} elapsed={elapsed/60:.1f} min", flush=True)
    return copied


def evaluate(method: str, seed: int, setting: str, checkpoint: Path) -> None:
    setting_tag = tag(setting)
    output = OUTPUT_ROOT / "evaluation" / method / f"seed{seed}" / setting_tag
    instance_path = RESULT_ROOT / "instance_jsonl" / method / f"seed{seed}" / f"{setting_tag}.jsonl"
    marker = output / "complete.json"
    if marker.is_file() and instance_path.is_file():
        records = instance_path.read_text(encoding="utf-8").splitlines()
        if len(records) == 256:
            print(f"SKIP completed eval {method} seed={seed} setting={setting}", flush=True)
            return
    output.mkdir(parents=True, exist_ok=True)
    instance_path.parent.mkdir(parents=True, exist_ok=True)
    instance_path.unlink(missing_ok=True)
    name = f"Final_{method}_seed{seed}_{setting_tag}"
    cmd = base_main(name) + [
        "dataset=elliptic_recommendation",
        "algorithm=iterative_filtering",
        *METHODS[method],
        *FROZEN,
        "experiment=exp_edge_recommendation",
        "experiment.tasks=[test]",
        "experiment.test.batch_size=16",
        "experiment.test.data.num_workers=0",
        "dataset.num_samples=256",
        "dataset.eval_pool_mode=official",
        "dataset.augment.enabled=false",
        "seed=0",
        "wandb.mode=offline",
        f"load={checkpoint.as_posix()}",
        f"+shortcut={setting}",
        f"+algorithm.audit_output_path={instance_path.as_posix()}",
        f"+algorithm.audit_method={method}",
        f"+algorithm.audit_training_seed={seed}",
        "+algorithm.audit_eval_seed=0",
        f"+algorithm.audit_setting={setting_tag}",
        f"hydra.run.dir={output.as_posix()}",
    ]
    print(f"START eval {method} seed={seed} setting={setting}", flush=True)
    elapsed = run_logged(cmd, output / "evaluation.log")
    count = len(instance_path.read_text(encoding="utf-8").splitlines()) if instance_path.is_file() else 0
    if count != 256:
        raise RuntimeError(f"Expected 256 instance records, found {count}: {instance_path}")
    marker.write_text(json.dumps({
        "method": method, "training_seed": seed, "setting": setting,
        "eval_seed": 0, "instances": count, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "elapsed_sec": elapsed,
        "instance_jsonl": str(instance_path), "completed_at": now(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"DONE  eval {method} seed={seed} setting={setting} elapsed={elapsed/60:.1f} min", flush=True)


def checkpoint_metadata(path: Path) -> dict:
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    callbacks = []
    for key, state in checkpoint.get("callbacks", {}).items():
        callbacks.append({
            "callback": str(key),
            "monitor": state.get("monitor"),
            "best_model_score": float(state["best_model_score"]) if state.get("best_model_score") is not None else None,
            "best_model_path": state.get("best_model_path"),
        })
    return {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "callbacks": callbacks,
    }


def collect_results() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    instance_records = []
    for method in METHODS:
        for seed in range(3):
            for setting in SETTINGS:
                path = RESULT_ROOT / "instance_jsonl" / method / f"seed{seed}" / f"{tag(setting)}.jsonl"
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(rows) != 256:
                    raise RuntimeError(f"Integrity failure: {path} has {len(rows)} rows")
                for row in rows:
                    row["setting"] = setting
                instance_records.extend(rows)

    # Matched candidate assertion across all methods and runs.
    reference = {}
    for row in instance_records:
        key = (row["setting"], row["sample_index"])
        value = row["candidate_hash"]
        if key in reference and reference[key] != value:
            raise RuntimeError(f"Candidate hash mismatch for {key}")
        reference[key] = value
    if len(reference) != len(SETTINGS) * 256:
        raise RuntimeError(f"Expected {len(SETTINGS)*256} unique setting/index candidates, found {len(reference)}")

    instance_csv = RESULT_ROOT / "instance_metrics_12288.csv"
    fields = [
        "method", "training_seed", "setting", "eval_seed", "sample_index", "sample_id",
        "candidate_hash", "num_senders", "num_receivers", "gt_unique_count", "hit_count", "HR", "NDCG",
    ]
    with instance_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(instance_records)

    grouped = {}
    for row in instance_records:
        key = (row["method"], int(row["training_seed"]), row["setting"])
        grouped.setdefault(key, []).append(row)
    raw = []
    for (method, seed, setting), rows in grouped.items():
        raw.append({
            "Method": method, "Training seed": seed, "Setting": setting,
            "Eval seed": 0, "# instances": len(rows),
            "HR": statistics.mean(float(row["HR"]) for row in rows),
            "NDCG": statistics.mean(float(row["NDCG"]) for row in rows),
        })
    raw.sort(key=lambda row: (list(METHODS).index(row["Method"]), row["Training seed"], SETTINGS.index(row["Setting"])))
    raw_csv = RESULT_ROOT / "raw_48_run_records.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)

    summary = []
    for setting in SETTINGS:
        values = {}
        for method in METHODS:
            rows = [row for row in raw if row["Method"] == method and row["Setting"] == setting]
            for metric in ("HR", "NDCG"):
                sample = [float(row[metric]) for row in rows]
                values[f"{method} {metric} mean"] = statistics.mean(sample)
                values[f"{method} {metric} SD"] = statistics.stdev(sample)
        summary.append({
            "Setting": setting, **values,
            "Delta HR": values["SAIF HR mean"] - values["RevFilter HR mean"],
            "Delta NDCG": values["SAIF NDCG mean"] - values["RevFilter NDCG mean"],
            "HR comparison": "higher" if values["SAIF HR mean"] > values["RevFilter HR mean"] else "lower" if values["SAIF HR mean"] < values["RevFilter HR mean"] else "equal",
            "NDCG comparison": "higher" if values["SAIF NDCG mean"] > values["RevFilter NDCG mean"] else "lower" if values["SAIF NDCG mean"] < values["RevFilter NDCG mean"] else "equal",
        })
    summary_csv = RESULT_ROOT / "main_table_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)

    provenance = []
    for method in METHODS:
        for seed in range(3):
            stages = {}
            for stage in ("stage1", "stage2"):
                _, checkpoint, marker = stage_paths(method, seed, stage)
                metadata = checkpoint_metadata(checkpoint)
                metadata["marker"] = json.loads(marker.read_text(encoding="utf-8"))
                log_text = Path(metadata["marker"]["training_log"]).read_text(encoding="utf-8", errors="ignore")
                metadata["test_metric_during_training"] = bool(re.search(r"(?:^|\s)(?:final_test|test)/(?:HR|NDCG|f1|prauc)", log_text))
                stages[stage] = metadata
            provenance.append({"method": method, "seed": seed, "backbone": "128/1/0.3/max", "stages": stages})
    provenance_path = RESULT_ROOT / "checkpoint_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    integrity = {
        "evaluation_seed": 0,
        "instances_per_setting": 256,
        "settings": SETTINGS,
        "expected_aggregate_records": 48,
        "observed_aggregate_records": len(raw),
        "expected_instance_records": 12288,
        "observed_instance_records": len(instance_records),
        "candidate_hash_assertion": "PASS",
        "same_candidate_instances_across_methods": True,
        "same_candidate_instances_across_three_runs": True,
        "test_derived_model_selection": False,
        "frozen": {"input": 43, "hidden": 128, "layers": 1, "dropout": 0.3, "pool": "max", "activation": "ELU", "alpha": 1.5, "gamma": 0.4, "n_merge": [1, 20]},
        "generated_at": now(),
    }
    (RESULT_ROOT / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    print(f"RESULTS READY {RESULT_ROOT}", flush=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    final_checkpoints = {}
    for method in METHODS:
        for seed in range(3):
            stage1 = train_stage(method, seed, "stage1", None)
            stage2 = train_stage(method, seed, "stage2", stage1)
            final_checkpoints[(method, seed)] = stage2
    print("ALL TRAINING COMPLETE", flush=True)
    for method in METHODS:
        for seed in range(3):
            for setting in SETTINGS:
                evaluate(method, seed, setting, final_checkpoints[(method, seed)])
    print("ALL 48 EVALUATIONS COMPLETE", flush=True)
    collect_results()


if __name__ == "__main__":
    main()
