#!/usr/bin/env python
"""Summarize SAIF/RevFilter complexity-profile logs.

The corresponding runs should enable ``algorithm.profile_search=true`` so that
``IterativeFilteringAlgo`` logs the internal search counters. The external
``elapsed_sec`` field is written by ``scripts/run_batches.py`` and includes the
whole evaluation process, including dataset construction.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
METHOD_LABELS = {
    "official": "Official RevFilter",
    "saif": "SAIF",
}
PROFILE_KEYS = {
    "HR": "final_test/HR",
    "NDCG": "final_test/NDCG",
    "initial_pairs_per_sample": "final_test/profile_initial_pairs_per_sample",
    "search_rounds_per_sample": "final_test/profile_search_rounds_per_sample",
    "forward_rounds_per_sample": "final_test/profile_forward_rounds_per_sample",
    "scored_regions_per_sample": "final_test/profile_scored_regions_per_sample",
    "region_score_ratio": "final_test/profile_region_score_ratio",
    "scored_sender_tokens_per_sample": "final_test/profile_scored_sender_tokens_per_sample",
    "scored_receiver_tokens_per_sample": "final_test/profile_scored_receiver_tokens_per_sample",
    "scored_node_tokens_per_sample": "final_test/profile_scored_node_tokens_per_sample",
    "scored_edge_volume_per_sample": "final_test/profile_scored_edge_volume_per_sample",
    "max_live_regions_per_sample": "final_test/profile_max_live_regions_per_sample",
    "model_parameters": "final_test/profile_model_parameters",
    "search_elapsed_sec": "final_test/profile_search_elapsed_sec",
}
RAW_FIELDS = [
    "method",
    "ckpt",
    "setting",
    *PROFILE_KEYS.keys(),
    "elapsed_sec",
    "max_mem_kb",
    "log_path",
]
SUMMARY_FIELDS = [
    ("HR", 4),
    ("NDCG", 4),
    ("elapsed_sec", 3),
    ("search_elapsed_sec", 3),
    ("max_mem_kb", 0),
    ("model_parameters", 0),
    ("initial_pairs_per_sample", 2),
    ("scored_regions_per_sample", 2),
    ("region_score_ratio", 4),
    ("scored_node_tokens_per_sample", 2),
    ("scored_edge_volume_per_sample", 2),
    ("search_rounds_per_sample", 2),
    ("forward_rounds_per_sample", 2),
    ("max_live_regions_per_sample", 2),
]
MARKDOWN_FIELDS = [
    ("HR", "HR"),
    ("NDCG", "NDCG"),
    ("elapsed_sec", "total sec"),
    ("search_elapsed_sec", "search sec"),
    ("model_parameters", "params"),
    ("initial_pairs_per_sample", "initial pairs"),
    ("scored_regions_per_sample", "scored regions"),
    ("region_score_ratio", "score ratio"),
    ("scored_node_tokens_per_sample", "encoded nodes"),
    ("search_rounds_per_sample", "rounds"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_root",
        nargs="?",
        default="logs-complexity-profile",
        help="Directory containing complexity-profile .log files.",
    )
    return parser.parse_args()


def last_float_for_key(text: str, key: str) -> float | None:
    matches = re.findall(re.escape(key) + r"\D+" + NUMBER, text)
    return float(matches[-1]) if matches else None


def last_assignment(text: str, key: str) -> float | None:
    matches = re.findall(re.escape(key) + r"=" + NUMBER, text)
    return float(matches[-1]) if matches else None


def parse_log(path: Path) -> dict[str, float | str] | None:
    match = re.match(r"(official|saif)_ckpt([^_]+)_(\d+)p(\d+)at(\d+)$", path.stem)
    if not match:
        print(f"[WARN] skip unrecognized log name: {path.name}")
        return None

    method, ckpt, n_pos, n_neg, top_k = match.groups()
    text = path.read_text(encoding="utf-8", errors="ignore")
    row: dict[str, float | str] = {
        "method": method,
        "ckpt": ckpt,
        "setting": f"{n_pos}+{n_neg}@{top_k}",
        "log_path": str(path),
    }

    for field, key in PROFILE_KEYS.items():
        value = last_float_for_key(text, key)
        row[field] = "" if value is None else value

    elapsed_sec = last_assignment(text, "elapsed_sec")
    max_mem_kb = last_assignment(text, "max_mem_kb")
    row["elapsed_sec"] = "" if elapsed_sec is None else elapsed_sec
    row["max_mem_kb"] = "" if max_mem_kb is None else max_mem_kb
    return row


def mean_std(values: list[float], digits: int) -> str:
    if not values:
        return ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f}+/-{std:.{digits}f}"


def numeric_values(rows: list[dict[str, float | str]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row[field]
        if value != "":
            values.append(float(value))
    return values


def build_summary(rows: list[dict[str, float | str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["setting"]))].append(row)

    summary = []
    for method, setting in sorted(grouped, key=lambda item: (item[0][1], item[0][0])):
        group = grouped[(method, setting)]
        item = {
            "method": METHOD_LABELS.get(method, method),
            "setting": setting,
            "n": str(len(group)),
        }
        for field, digits in SUMMARY_FIELDS:
            item[field] = mean_std(numeric_values(group, field), digits)
        summary.append(item)
    return summary


def write_raw_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(summary: list[dict[str, str]], path: Path) -> None:
    fields = ["method", "setting", "n", *[field for field, _digits in SUMMARY_FIELDS]]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def write_summary_md(summary: list[dict[str, str]], path: Path) -> None:
    headers = ["method", "setting", "n", *[label for _field, label in MARKDOWN_FIELDS]]
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "Note: values are mean+/-std over checkpoints. "
            "`total sec` includes process startup, data loading, candidate construction, "
            "and testing; `search sec` is timed inside the iterative filtering loop.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|---|---|---:" + "|---:" * len(MARKDOWN_FIELDS) + "|\n")
        for row in summary:
            values = [
                row["method"],
                row["setting"],
                row["n"],
                *[row[field] for field, _label in MARKDOWN_FIELDS],
            ]
            f.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    root = Path(args.log_root)
    rows = [row for path in sorted(root.glob("*.log")) if (row := parse_log(path))]
    if not rows:
        raise SystemExit(f"No complexity-profile logs found in {root}")

    raw_path = root / "complexity_profile_raw.csv"
    summary_csv_path = root / "complexity_profile_summary.csv"
    summary_md_path = root / "complexity_profile_summary.md"
    summary = build_summary(rows)

    write_raw_csv(rows, raw_path)
    write_summary_csv(summary, summary_csv_path)
    write_summary_md(summary, summary_md_path)

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_csv_path}")
    print(f"Wrote: {summary_md_path}")
    print(summary_md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
