"""Evaluate the validation-selected LightGCN checkpoint on Table 1 settings.

The candidate-instance RNG seed is fixed to zero for every checkpoint, matching
the released multi-setting evaluation YAML and keeping the 256 instances fixed
across the three independently trained models.
"""

from __future__ import annotations

import csv
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs-lightgcn-val-selected-table1"
SETTINGS = [
    "1+5@1",
    "1+10@1",
    "1+10@3",
    "1+100@3",
    "3+100@10",
    "3+1000@10",
    "10+1000@100",
    "10+10000@100",
]
CHECKPOINTS = {
    0: ROOT
    / "outputs/lightgcn_validation/layer2/checkpoints/epoch=148-step=6854.ckpt",
    1: ROOT / "outputs/lightgcn_final/seed1/checkpoints/epoch=142-step=6578.ckpt",
    2: ROOT / "outputs/lightgcn_final/seed2/checkpoints/epoch=142-step=6578.ckpt",
}


def last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None


def run_one(train_seed: int, checkpoint: Path, setting: str) -> dict[str, object]:
    tag = setting.replace("+", "p").replace("@", "at")
    log_path = OUT / f"seed{train_seed}_{tag}.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "AUDIT_RETURN_CODE=0" in text:
            return parse_result(train_seed, checkpoint, setting, log_path, text)

    cmd = [
        sys.executable,
        "-m",
        "main",
        f"+name=LightGCN_VAL2_seed{train_seed}_{tag}",
        "dataset=elliptic_recommendation",
        "algorithm=lightgcn",
        "algorithm.model.num_layers=2",
        "experiment=exp_edge_recommendation",
        "experiment.tasks=[test]",
        "experiment.test.batch_size=16",
        # Evaluation/candidate generation seed, intentionally identical.
        "seed=0",
        "wandb.mode=offline",
        f"load='{checkpoint.as_posix()}'",
        f"+shortcut={setting}",
    ]
    env = os.environ.copy()
    env["WANDB_DATA_DIR"] = str(OUT / "wandb-data")
    env["WANDB_CACHE_DIR"] = str(OUT / "wandb-cache")
    env["WANDB_CONFIG_DIR"] = str(OUT / "wandb-config")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = proc.stdout + f"\nAUDIT_RETURN_CODE={proc.returncode}\n"
    log_path.write_text(text, encoding="utf-8")
    print(f"seed={train_seed} setting={setting} returncode={proc.returncode}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation failed; inspect {log_path}")
    return parse_result(train_seed, checkpoint, setting, log_path, text)


def parse_result(
    train_seed: int, checkpoint: Path, setting: str, log_path: Path, text: str
) -> dict[str, object]:
    hr = last_float(r"final_test/HR\D+([0-9]+(?:\.[0-9]+)?)", text)
    ndcg = last_float(r"final_test/NDCG\D+([0-9]+(?:\.[0-9]+)?)", text)
    density = last_float(r"Avg density:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if hr is None or ndcg is None:
        raise ValueError(f"Missing metrics in {log_path}")
    return {
        "training_seed": train_seed,
        "evaluation_seed": 0,
        "setting": setting,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "HR": hr,
        "NDCG": ndcg,
        "density": density,
        "log": str(log_path.relative_to(ROOT)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for subdir in ("wandb-data", "wandb-cache", "wandb-config"):
        (OUT / subdir).mkdir(exist_ok=True)
    rows = [
        run_one(seed, checkpoint, setting)
        for seed, checkpoint in CHECKPOINTS.items()
        for setting in SETTINGS
    ]

    raw_path = OUT / "lightgcn_val_selected_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUT / "lightgcn_val_selected_summary.md"
    lines = [
        "| Setting | HR (mean +/- sample SD) | NDCG (mean +/- sample SD) |",
        "|---|---:|---:|",
    ]
    for setting in SETTINGS:
        group = [row for row in rows if row["setting"] == setting]
        hr = [float(row["HR"]) for row in group]
        ndcg = [float(row["NDCG"]) for row in group]
        lines.append(
            f"| {setting} | {statistics.mean(hr):.4f} +/- {statistics.stdev(hr):.4f} "
            f"| {statistics.mean(ndcg):.4f} +/- {statistics.stdev(ndcg):.4f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
