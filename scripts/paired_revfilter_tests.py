#!/usr/bin/env python
"""Paired tests for RevFilter-style raw CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def ckpt_id(text: str) -> str:
    match = re.search(r"(?:seed)?(\d+)", text)
    return match.group(1) if match else text


def read_raw(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["setting"], ckpt_id(row["ckpt"]))
            rows[key] = {"HR": float(row["HR"]), "NDCG": float(row["NDCG"])}
    return rows


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def sample_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def exact_sign_p(diffs: list[float]) -> float:
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = sum(d > 0 for d in nonzero)
    extreme = min(positives, n - positives)
    prob = sum(math.comb(n, k) for k in range(extreme + 1)) / (2**n)
    return min(1.0, 2 * prob)


def wilcoxon_exact_p(diffs: list[float]) -> float:
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    ranked = sorted((abs(d), 1 if d > 0 else -1) for d in nonzero)
    ranks = list(range(1, n + 1))
    observed = sum(rank for rank, (_, sign) in zip(ranks, ranked) if sign > 0)
    total_rank = n * (n + 1) // 2
    lower = min(observed, total_rank - observed)

    counts = {}
    for signs in range(1 << n):
        rank_sum = 0
        for i, rank in enumerate(ranks):
            if signs & (1 << i):
                rank_sum += rank
        counts[rank_sum] = counts.get(rank_sum, 0) + 1
    extreme_count = sum(
        count for rank_sum, count in counts.items() if min(rank_sum, total_rank - rank_sum) <= lower
    )
    return min(1.0, extreme_count / (2**n))


def paired_rows(a_rows, b_rows, setting: str, metric: str) -> tuple[list[str], list[float], list[float], list[float]]:
    seeds = sorted(
        {seed for s, seed in a_rows if s == setting}.intersection(
            {seed for s, seed in b_rows if s == setting}
        ),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    a_vals = [a_rows[(setting, seed)][metric] for seed in seeds]
    b_vals = [b_rows[(setting, seed)][metric] for seed in seeds]
    diffs = [b - a for a, b in zip(a_vals, b_vals)]
    return seeds, a_vals, b_vals, diffs


def main() -> None:
    args = parse_args()
    a_rows = read_raw(args.a)
    b_rows = read_raw(args.b)
    settings = sorted({s for s, _ in a_rows}.intersection({s for s, _ in b_rows}))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ndcg_setting_diffs = []
    hr_setting_diffs = []

    with args.out.open("w", encoding="utf-8") as f:
        f.write(f"# Paired Tests: {args.b_name} - {args.a_name}\n\n")
        f.write(
            "| setting | metric | n | "
            f"{args.a_name} mean | {args.b_name} mean | delta mean | delta std | "
            "positive | sign p | wilcoxon p |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        for setting in settings:
            for metric in ["HR", "NDCG"]:
                seeds, a_vals, b_vals, diffs = paired_rows(a_rows, b_rows, setting, metric)
                if not seeds:
                    continue
                delta = mean(diffs)
                if metric == "NDCG":
                    ndcg_setting_diffs.append(delta)
                else:
                    hr_setting_diffs.append(delta)
                f.write(
                    f"| {setting} | {metric} | {len(seeds)} | "
                    f"{mean(a_vals):.6f} | {mean(b_vals):.6f} | {delta:+.6f} | "
                    f"{sample_std(diffs):.6f} | {sum(d > 0 for d in diffs)}/{len(diffs)} | "
                    f"{exact_sign_p(diffs):.6f} | {wilcoxon_exact_p(diffs):.6f} |\n"
                )

        f.write("\n## Across-setting sign test on mean deltas\n\n")
        f.write("| metric | settings | positive settings | sign p |\n")
        f.write("|---|---:|---:|---:|\n")
        for metric, diffs in [("HR", hr_setting_diffs), ("NDCG", ndcg_setting_diffs)]:
            f.write(
                f"| {metric} | {len(diffs)} | {sum(d > 0 for d in diffs)}/{len(diffs)} | "
                f"{exact_sign_p(diffs):.6f} |\n"
            )

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
