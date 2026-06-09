#!/usr/bin/env python
"""Summarize RevFilter-style evaluation logs.

The script parses final_test/HR, final_test/NDCG, and Avg density from log files
whose names look like ``0_tuned_10p1000at100.log`` or
``tuned_seed0_10p1000at100.log``.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--prefix", default="revfilter")
    return parser.parse_args()


def last_float(pattern: str, text: str) -> float | None:
    vals = re.findall(pattern, text)
    return float(vals[-1]) if vals else None


def parse_log_name(name: str) -> tuple[str, str, int, int, int] | None:
    match = re.match(r"(?:(.+?)_)?(\d+)p(\d+)at(\d+)$", name)
    if not match:
        return None
    ckpt, n_pos, n_neg, top_k = match.groups()
    ckpt = ckpt if ckpt else "unknown"
    setting = f"{n_pos}+{n_neg}@{top_k}"
    return ckpt, setting, int(n_pos), int(n_neg), int(top_k)


def mean_std(rows: list[dict[str, str]], metric: str) -> str:
    vals = [float(r[metric]) for r in rows if r[metric] != ""]
    if not vals:
        return ""
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.6f}+/-{std:.6f}"


def main() -> None:
    args = parse_args()
    root = args.log_root
    rows = []

    for log_path in sorted(root.glob("*.log")):
        parsed = parse_log_name(log_path.stem)
        if parsed is None:
            print(f"[WARN] cannot parse name: {log_path.name}")
            continue
        ckpt, setting, n_pos, n_neg, top_k = parsed
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        hr = last_float(r"final_test/HR\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)
        ndcg = last_float(r"final_test/NDCG\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)
        density = last_float(r"Avg density:\s*([0-9]+(?:\.[0-9]+)?)", text)

        rows.append(
            {
                "ckpt": ckpt,
                "setting": setting,
                "num_illicits": str(n_pos),
                "num_licits": str(n_neg),
                "top_k": str(top_k),
                "density": "" if density is None else f"{density:.8f}",
                "sparsity": "" if density is None else f"{1 - density:.8f}",
                "HR": "" if hr is None else f"{hr:.6f}",
                "NDCG": "" if ndcg is None else f"{ndcg:.6f}",
                "log_path": str(log_path),
            }
        )

    raw_path = root / f"{args.prefix}_raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "ckpt",
            "setting",
            "num_illicits",
            "num_licits",
            "top_k",
            "density",
            "sparsity",
            "HR",
            "NDCG",
            "log_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    settings = sorted(
        {r["setting"] for r in rows},
        key=lambda s: tuple(int(x) for x in re.split(r"[+@]", s)),
    )
    for setting in settings:
        group = [r for r in rows if r["setting"] == setting]
        group.sort(key=lambda r: r["ckpt"])
        summary_rows.append(
            {
                "setting": setting,
                "n": str(len(group)),
                "density": group[0]["density"],
                "sparsity": group[0]["sparsity"],
                "HR": mean_std(group, "HR"),
                "NDCG": mean_std(group, "NDCG"),
            }
        )

    summary_path = root / f"{args.prefix}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["setting", "n", "density", "sparsity", "HR", "NDCG"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = root / f"{args.prefix}_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| setting | n | density | sparsity | HR | NDCG |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['setting']} | {r['n']} | {r['density']} | {r['sparsity']} | "
                f"{r['HR']} | {r['NDCG']} |\n"
            )

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
