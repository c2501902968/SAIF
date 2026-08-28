#!/usr/bin/env python
"""Paired instance-level Wilcoxon signed-rank tests with BH correction."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.edge_recommendation_analysis_utils import mean, setting_sort_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--a-method", default=None)
    parser.add_argument("--b-method", default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metrics", nargs="+", default=["HR", "NDCG"])
    parser.add_argument("--bh-scope", choices=["all", "metric"], default="all")
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--expected-samples", type=int, default=256)
    parser.add_argument("--expected-settings", type=int, default=8)
    parser.add_argument(
        "--alternative",
        choices=["two-sided", "greater", "less"],
        default="two-sided",
        help="'greater' tests whether B > A.",
    )
    return parser.parse_args()


def read_instance_rows(
    path: Path,
    *,
    method: str | None,
    metrics: list[str],
    expected_runs: int,
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str], set[str]],
]:
    values: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if method is not None and row.get("method") != method:
                continue
            setting = row["setting"]
            sample_id = row.get("sample_id", str(row_idx))
            run_id = row.get("ckpt", "").strip()
            if not run_id:
                raise AssertionError(
                    f"{path}: missing run ID in column 'ckpt' at CSV row {row_idx + 2}"
                )
            key = (setting, sample_id)
            for metric in metrics:
                if row.get(metric, "") == "":
                    raise AssertionError(
                        f"{path}: missing {metric} for setting={setting}, "
                        f"sample_id={sample_id}, run_id={run_id}"
                    )
                if run_id in values[key][metric]:
                    raise AssertionError(
                        f"{path}: duplicate record for setting={setting}, "
                        f"sample_id={sample_id}, run_id={run_id}, metric={metric}"
                    )
                values[key][metric][run_id] = float(row[metric])

    if not values:
        method_note = f" for method={method}" if method is not None else ""
        raise AssertionError(f"{path}: no instance records found{method_note}")

    averaged = {}
    run_ids = {}
    for key, metric_values in values.items():
        metric_run_sets = {metric: set(vals) for metric, vals in metric_values.items()}
        if set(metric_run_sets) != set(metrics):
            raise AssertionError(f"{path}: incomplete metrics for {key}: {metric_run_sets}")
        reference_runs = metric_run_sets[metrics[0]]
        if len(reference_runs) != expected_runs:
            raise AssertionError(
                f"{path}: expected {expected_runs} runs for {key}, "
                f"found {len(reference_runs)}: {sorted(reference_runs)}"
            )
        for metric, metric_runs in metric_run_sets.items():
            if metric_runs != reference_runs:
                raise AssertionError(
                    f"{path}: run-ID mismatch across metrics for {key}; "
                    f"{metric} has {sorted(metric_runs)}, expected {sorted(reference_runs)}"
                )
        averaged[key] = {
            metric: mean(list(vals.values()))
            for metric, vals in metric_values.items()
        }
        run_ids[key] = reference_runs
    return averaged, run_ids


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[indexed[pos][0]] = avg_rank
        start = end
    return ranks


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank(
    diffs: list[float],
    *,
    alternative: str,
) -> dict[str, float | int]:
    nonzero = [diff for diff in diffs if diff != 0.0]
    n = len(nonzero)
    if n == 0:
        return {
            "nonzero_n": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "z": 0.0,
            "p": 1.0,
            "effect_r": 0.0,
        }

    ranks = average_ranks([abs(diff) for diff in nonzero])
    w_plus = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    w_minus = sum(rank for rank, diff in zip(ranks, nonzero) if diff < 0)
    expected = sum(ranks) / 2.0
    variance = sum(rank * rank for rank in ranks) / 4.0
    if variance == 0:
        p = 1.0
        z_signed = 0.0
    else:
        z_signed = (w_plus - expected) / math.sqrt(variance)
        if alternative == "two-sided":
            z_abs = max(0.0, (abs(w_plus - expected) - 0.5) / math.sqrt(variance))
            p = math.erfc(z_abs / math.sqrt(2.0))
        elif alternative == "greater":
            z = (w_plus - expected - 0.5) / math.sqrt(variance)
            p = 1.0 - normal_cdf(z)
        else:
            z = (w_plus - expected + 0.5) / math.sqrt(variance)
            p = normal_cdf(z)

    return {
        "nonzero_n": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "z": z_signed,
        "p": max(0.0, min(1.0, p)),
        "effect_r": z_signed / math.sqrt(n),
    }


def bh_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        q = min(running_min, p_values[idx] * m / rank)
        running_min = q
        adjusted[idx] = min(1.0, q)
    return adjusted


def build_test_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    a_rows, a_run_ids = read_instance_rows(
        args.a,
        method=args.a_method,
        metrics=args.metrics,
        expected_runs=args.expected_runs,
    )
    b_rows, b_run_ids = read_instance_rows(
        args.b,
        method=args.b_method,
        metrics=args.metrics,
        expected_runs=args.expected_runs,
    )
    a_settings = {setting for setting, _ in a_rows}
    b_settings = {setting for setting, _ in b_rows}
    if a_settings != b_settings:
        raise AssertionError(
            f"Setting mismatch: A-only={sorted(a_settings - b_settings)}, "
            f"B-only={sorted(b_settings - a_settings)}"
        )
    if len(a_settings) != args.expected_settings:
        raise AssertionError(
            f"Expected {args.expected_settings} matched settings, "
            f"found {len(a_settings)}: {sorted(a_settings)}"
        )
    settings = sorted(a_settings, key=setting_sort_key)

    result_rows = []
    for setting in settings:
        a_sample_ids = {sample_id for s, sample_id in a_rows if s == setting}
        b_sample_ids = {sample_id for s, sample_id in b_rows if s == setting}
        if a_sample_ids != b_sample_ids:
            raise AssertionError(
                f"Sample mismatch for {setting}: "
                f"A-only={sorted(a_sample_ids - b_sample_ids)}, "
                f"B-only={sorted(b_sample_ids - a_sample_ids)}"
            )
        if len(a_sample_ids) != args.expected_samples:
            raise AssertionError(
                f"Expected {args.expected_samples} matched samples for {setting}, "
                f"found {len(a_sample_ids)}"
            )
        sample_ids = sorted(
            a_sample_ids,
            key=lambda x: int(x) if x.isdigit() else x,
        )
        for sample_id in sample_ids:
            key = (setting, sample_id)
            if a_run_ids[key] != b_run_ids[key]:
                raise AssertionError(
                    f"Run-ID mismatch for {key}: "
                    f"A={sorted(a_run_ids[key])}, B={sorted(b_run_ids[key])}"
                )
        for metric in args.metrics:
            paired = [
                (a_rows[(setting, sample_id)].get(metric), b_rows[(setting, sample_id)].get(metric))
                for sample_id in sample_ids
            ]
            if any(a is None or b is None for a, b in paired):
                raise AssertionError(f"Missing paired {metric} value for {setting}")
            a_vals = [a for a, _ in paired]
            b_vals = [b for _, b in paired]
            diffs = [b - a for a, b in paired]
            test = wilcoxon_signed_rank(diffs, alternative=args.alternative)
            result_rows.append(
                {
                    "setting": setting,
                    "metric": metric,
                    "n": str(len(diffs)),
                    "nonzero_n": str(test["nonzero_n"]),
                    "a_mean": f"{mean(a_vals):.10f}",
                    "b_mean": f"{mean(b_vals):.10f}",
                    "delta_mean": f"{mean(diffs):+.10f}",
                    "positive": f"{sum(diff > 0 for diff in diffs)}/{len(diffs)}",
                    "w_plus": f"{test['w_plus']:.6f}",
                    "w_minus": f"{test['w_minus']:.6f}",
                    "z": f"{test['z']:.6f}",
                    "effect_r": f"{test['effect_r']:.6f}",
                    "p": f"{test['p']:.10g}",
                    "q_bh": "",
                }
            )
    expected_comparisons = args.expected_settings * len(args.metrics)
    if len(result_rows) != expected_comparisons:
        raise AssertionError(
            f"Expected {expected_comparisons} setting-metric comparisons, "
            f"found {len(result_rows)}"
        )
    return result_rows


def apply_bh(rows: list[dict[str, str]], scope: str) -> None:
    if scope == "all":
        q_values = bh_adjust([float(row["p"]) for row in rows])
        for row, q in zip(rows, q_values):
            row["q_bh"] = f"{q:.10g}"
        return

    for metric in sorted({row["metric"] for row in rows}):
        metric_rows = [row for row in rows if row["metric"] == metric]
        q_values = bh_adjust([float(row["p"]) for row in metric_rows])
        for row, q in zip(metric_rows, q_values):
            row["q_bh"] = f"{q:.10g}"


def write_markdown(
    rows: list[dict[str, str]],
    path: Path,
    *,
    a_name: str,
    b_name: str,
    alternative: str,
    bh_scope: str,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Paired Instance-Level Wilcoxon Tests: {b_name} - {a_name}\n\n")
        f.write(
            f"Alternative: `{alternative}`. BH correction scope: `{bh_scope}`. "
            "Rows average the asserted matched independently trained runs per "
            "`(setting, sample_id)` before testing.\n\n"
        )
        f.write(
            "| setting | metric | n | nonzero n | "
            f"{a_name} mean | {b_name} mean | delta mean | positive | "
            "z | effect r | p | BH q |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['setting']} | {row['metric']} | {row['n']} | "
                f"{row['nonzero_n']} | {row['a_mean']} | {row['b_mean']} | "
                f"{row['delta_mean']} | {row['positive']} | {row['z']} | "
                f"{row['effect_r']} | {row['p']} | {row['q_bh']} |\n"
            )


def main() -> None:
    args = parse_args()
    rows = build_test_rows(args)
    apply_bh(rows, args.bh_scope)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting",
        "metric",
        "n",
        "nonzero_n",
        "a_mean",
        "b_mean",
        "delta_mean",
        "positive",
        "w_plus",
        "w_minus",
        "z",
        "effect_r",
        "p",
        "q_bh",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out.with_suffix(".md")
    write_markdown(
        rows,
        md_path,
        a_name=args.a_name,
        b_name=args.b_name,
        alternative=args.alternative,
        bh_scope=args.bh_scope,
    )
    print(f"Wrote: {args.out}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
