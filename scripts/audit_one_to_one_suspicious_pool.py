#!/usr/bin/env python
"""Audit one-to-one suspicious subgraph coverage in the Elliptic raw table."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "elliptic" / "raw" / "data_df.pkl"


def label_value(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def split_value(value) -> str:
    return str(value)


def category(num_senders: int, num_receivers: int) -> str:
    if num_senders == 1 and num_receivers == 1:
        return "one_to_one"
    if num_senders > 1 and num_receivers == 1:
        return "multi_sender"
    if num_senders == 1 and num_receivers > 1:
        return "multi_receiver"
    return "many_to_many"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    data = pd.read_pickle(DATA_PATH)
    rows = []
    for _, row in data.iterrows():
        label = label_value(row.labels)
        num_senders = len(row.senders_mapped)
        num_receivers = len(row.receivers_mapped)
        rows.append(
            {
                "split": split_value(row.split),
                "label": label,
                "num_senders": num_senders,
                "num_receivers": num_receivers,
                "candidate_edges": num_senders * num_receivers,
                "category": category(num_senders, num_receivers),
            }
        )

    suspicious = [row for row in rows if row["label"] == 1]
    categories = ["one_to_one", "multi_sender", "multi_receiver", "many_to_many"]
    splits = ["ALL"] + sorted({row["split"] for row in suspicious})

    summary_rows = []
    for split in splits:
        group = suspicious if split == "ALL" else [row for row in suspicious if row["split"] == split]
        total = len(group)
        for cat in categories:
            subset = [row for row in group if row["category"] == cat]
            summary_rows.append(
                {
                    "split": split,
                    "category": cat,
                    "n": len(subset),
                    "percent_of_suspicious": (len(subset) / total * 100.0) if total else 0.0,
                    "avg_senders": mean([row["num_senders"] for row in subset]),
                    "avg_receivers": mean([row["num_receivers"] for row in subset]),
                    "avg_candidate_edges": mean([row["candidate_edges"] for row in subset]),
                }
            )

    csv_path = PROJECT_ROOT / "one_to_one_suspicious_pool_audit.csv"
    md_path = PROJECT_ROOT / "one_to_one_suspicious_pool_audit.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "category",
                "n",
                "percent_of_suspicious",
                "avg_senders",
                "avg_receivers",
                "avg_candidate_edges",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# One-to-One Suspicious Pool Audit",
        "",
        "| Split | Category | n | % suspicious | Avg senders | Avg receivers | Avg candidate edges |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['category']} | {row['n']} | "
            f"{row['percent_of_suspicious']:.2f} | {row['avg_senders']:.2f} | "
            f"{row['avg_receivers']:.2f} | {row['avg_candidate_edges']:.2f} |"
        )

    one_to_one = [row for row in suspicious if row["category"] == "one_to_one"]
    excluded = [row for row in suspicious if row["category"] != "one_to_one"]
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Total suspicious subgraphs: {len(suspicious)}",
            f"- One-to-one suspicious subgraphs: {len(one_to_one)} ({len(one_to_one) / len(suspicious) * 100.0:.2f}%)",
            f"- Excluded multi-boundary suspicious subgraphs: {len(excluded)} ({len(excluded) / len(suspicious) * 100.0:.2f}%)",
            f"- Excluded avg senders: {mean([row['num_senders'] for row in excluded]):.2f}",
            f"- Excluded avg receivers: {mean([row['num_receivers'] for row in excluded]):.2f}",
            f"- Excluded avg candidate edges: {mean([row['candidate_edges'] for row in excluded]):.2f}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
