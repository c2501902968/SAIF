#!/usr/bin/env python
"""Build Table A1 receiver-balanced control mean +/- std from log files."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs-balanced-receiver-control"
SETTINGS = ["10+1000@100", "10+10000@100"]
METHODS = [
    ("official", "Official RevFilter"),
    ("geometry", "Anchor-only"),
    ("saif", "SAIF"),
]


def last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None


def parse_log(path: Path) -> dict[str, str | float] | None:
    match = re.match(r"(official|geometry|saif)_ckpt(\d+)_(\d+)p(\d+)at(\d+)$", path.stem)
    if not match:
        return None

    method, ckpt, n_pos, n_neg, top_k = match.groups()
    text = path.read_text(encoding="utf-8", errors="ignore")
    hr = last_float(r"final_test/HR\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)
    ndcg = last_float(r"final_test/NDCG\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)
    density = last_float(r"Avg density:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if hr is None or ndcg is None:
        raise ValueError(f"Missing final_test metrics in {path}")

    return {
        "method": method,
        "ckpt": ckpt,
        "setting": f"{n_pos}+{n_neg}@{top_k}",
        "density": density if density is not None else "",
        "HR": hr,
        "NDCG": ndcg,
        "log_path": str(path.relative_to(PROJECT_ROOT)),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def display(values: list[float], digits: int = 4) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def display_ascii(values: list[float], digits: int = 6) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f}+/-{std:.{digits}f}"


def main() -> None:
    rows = []
    for path in sorted(LOG_ROOT.glob("*.log")):
        parsed = parse_log(path)
        if parsed is not None:
            rows.append(parsed)

    raw_path = PROJECT_ROOT / "table_a1_receiver_balanced_raw.csv"
    csv_path = PROJECT_ROOT / "table_a1_receiver_balanced_mean_std.csv"
    md_path = PROJECT_ROOT / "table_a1_receiver_balanced_mean_std.md"

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "ckpt", "setting", "density", "HR", "NDCG", "log_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "setting", "n", "density", "HR_mean_std", "NDCG_mean_std"])
        for method_key, method_name in METHODS:
            for setting in SETTINGS:
                group = [r for r in rows if r["method"] == method_key and r["setting"] == setting]
                if not group:
                    raise ValueError(f"No logs for {method_key} {setting}")
                hr_values = [float(r["HR"]) for r in group]
                ndcg_values = [float(r["NDCG"]) for r in group]
                density_values = [r["density"] for r in group if r["density"] != ""]
                density = density_values[0] if density_values else ""
                writer.writerow(
                    [
                        method_name,
                        setting,
                        len(group),
                        density,
                        display_ascii(hr_values),
                        display_ascii(ndcg_values),
                    ]
                )

    lines = [
        "# Table A1 Receiver-Balanced Control Mean ± Std",
        "",
        "Values are computed from three checkpoint logs per method and setting.",
        "",
        "| Method | 10+1000@100 HR | 10+1000@100 NDCG | 10+10000@100 HR | 10+10000@100 NDCG |",
        "|---|---:|---:|---:|---:|",
    ]
    for method_key, method_name in METHODS:
        values = [method_name]
        for setting in SETTINGS:
            group = [r for r in rows if r["method"] == method_key and r["setting"] == setting]
            values.append(display([float(r["HR"]) for r in group]))
            values.append(display([float(r["NDCG"]) for r in group]))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Log Counts", "", "| Method | Setting | n |", "|---|---|---:|"])
    for method_key, method_name in METHODS:
        for setting in SETTINGS:
            n = sum(1 for r in rows if r["method"] == method_key and r["setting"] == setting)
            lines.append(f"| {method_name} | {setting} | {n} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path)
    print(csv_path)
    print(raw_path)


if __name__ == "__main__":
    main()
