#!/usr/bin/env python
"""Build the main comparison table with mean +/- std values."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

METHODS = [
    ("MLP", "logs-mlp-table2/revfilter_summary.csv"),
    ("NGCF", "logs-ngcf-table2/revfilter_summary.csv"),
    ("LightGCN", "logs-lightgcn-table2/revfilter_summary.csv"),
    ("RevFilter", "logs-official-revfilter-table2/revfilter_summary.csv"),
    ("SAIF", "logs-anchor-revfilter-table2/revfilter_summary.csv"),
]


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["setting"]: row for row in csv.DictReader(f)}


def display_mean_std(value: str) -> str:
    return value.replace("+/-", " ± ")


def main() -> None:
    summaries = {
        method: read_summary(PROJECT_ROOT / path)
        for method, path in METHODS
    }
    density_source = summaries["MLP"]

    markdown_lines = [
        "# Main Comparison Mean ± Std",
        "",
        "Values are aggregated from 3 log files per method/setting.",
        "",
    ]
    header = ["Method"]
    for setting in SETTINGS:
        header.extend([f"{setting} HR", f"{setting} NDCG"])
    markdown_lines.append("| " + " | ".join(header) + " |")
    markdown_lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")

    for method, _path in METHODS:
        row = [method]
        for setting in SETTINGS:
            item = summaries[method][setting]
            row.extend([display_mean_std(item["HR"]), display_mean_std(item["NDCG"])])
        markdown_lines.append("| " + " | ".join(row) + " |")

    markdown_lines.extend(["", "## Density", "", "| Setting | Density (%) |", "|---|---:|"])
    for setting in SETTINGS:
        density = float(density_source[setting]["density"]) * 100.0
        markdown_lines.append(f"| {setting} | {density:.2f} |")

    md_path = PROJECT_ROOT / "main_comparison_mean_std.md"
    csv_path = PROJECT_ROOT / "main_comparison_mean_std.csv"
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "setting", "density_percent", "HR_mean_std", "NDCG_mean_std"])
        for method, _path in METHODS:
            for setting in SETTINGS:
                item = summaries[method][setting]
                density = float(density_source[setting]["density"]) * 100.0
                writer.writerow(
                    [
                        method,
                        setting,
                        f"{density:.2f}",
                        display_mean_std(item["HR"]),
                        display_mean_std(item["NDCG"]),
                    ]
                )

    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
