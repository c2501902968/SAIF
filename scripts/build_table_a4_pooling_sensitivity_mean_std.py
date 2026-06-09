#!/usr/bin/env python
"""Build Table A4 pooling sensitivity mean +/- std from log files."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs-pooling-sensitivity"
POOLING_ORDER = ["max", "mean", "sum"]
VARIANTS = [
    ("embedding", "Embedding-only"),
    ("saif", "Full-SAIF"),
]


def last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None


def parse_log(path: Path) -> dict[str, str | float] | None:
    match = re.match(r"(embedding|saif)_(max|mean|sum)_ckpt(\d+)_(\d+)p(\d+)at(\d+)$", path.stem)
    if not match:
        return None

    variant, pooling, ckpt, n_pos, n_neg, top_k = match.groups()
    text = path.read_text(encoding="utf-8", errors="ignore")
    hr = last_float(r"final_test/HR\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)
    ndcg = last_float(r"final_test/NDCG\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)
    density = last_float(r"Avg density:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if hr is None or ndcg is None:
        raise ValueError(f"Missing final_test metrics in {path}")

    return {
        "variant": variant,
        "pooling": pooling,
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


def matching(rows: list[dict[str, str | float]], variant: str, pooling: str):
    return [r for r in rows if r["variant"] == variant and r["pooling"] == pooling]


def main() -> None:
    rows = []
    for path in sorted(LOG_ROOT.glob("*.log")):
        parsed = parse_log(path)
        if parsed is not None:
            rows.append(parsed)

    raw_path = PROJECT_ROOT / "table_a4_pooling_sensitivity_raw.csv"
    csv_path = PROJECT_ROOT / "table_a4_pooling_sensitivity_mean_std.csv"
    md_path = PROJECT_ROOT / "table_a4_pooling_sensitivity_mean_std.md"

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "pooling", "ckpt", "setting", "density", "HR", "NDCG", "log_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pooling",
                "embedding_only_n",
                "embedding_only_HR_mean_std",
                "embedding_only_NDCG_mean_std",
                "full_saif_n",
                "full_saif_HR_mean_std",
                "full_saif_NDCG_mean_std",
            ]
        )
        for pooling in POOLING_ORDER:
            embedding = matching(rows, "embedding", pooling)
            saif = matching(rows, "saif", pooling)
            writer.writerow(
                [
                    pooling,
                    len(embedding),
                    display_ascii([float(r["HR"]) for r in embedding]),
                    display_ascii([float(r["NDCG"]) for r in embedding]),
                    len(saif),
                    display_ascii([float(r["HR"]) for r in saif]),
                    display_ascii([float(r["NDCG"]) for r in saif]),
                ]
            )

    lines = [
        "# Table A4 Pooling Sensitivity Mean ± Std",
        "",
        "Values are computed from six logs per pooling/variant: two sparse settings times three checkpoints.",
        "",
        "| Pooling | Embedding-only HR | Embedding-only NDCG | Full-SAIF HR | Full-SAIF NDCG |",
        "|---|---:|---:|---:|---:|",
    ]
    for pooling in POOLING_ORDER:
        embedding = matching(rows, "embedding", pooling)
        saif = matching(rows, "saif", pooling)
        lines.append(
            "| "
            + " | ".join(
                [
                    pooling.capitalize(),
                    display([float(r["HR"]) for r in embedding]),
                    display([float(r["NDCG"]) for r in embedding]),
                    display([float(r["HR"]) for r in saif]),
                    display([float(r["NDCG"]) for r in saif]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Log Counts", "", "| Variant | Pooling | n |", "|---|---|---:|"])
    for variant, display_name in VARIANTS:
        for pooling in POOLING_ORDER:
            lines.append(f"| {display_name} | {pooling} | {len(matching(rows, variant, pooling))} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path)
    print(csv_path)
    print(raw_path)


if __name__ == "__main__":
    main()
