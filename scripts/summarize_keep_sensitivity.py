#!/usr/bin/env python
"""Summarize keep-multiplier sensitivity logs.

Expected log names:
  official_km1p5_0_tuned_10p1000at100.log
  saif_km1p5_tuned_seed0_10p1000at100.log
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
    parser.add_argument("--baseline", default="official")
    parser.add_argument("--method", default="saif")
    return parser.parse_args()


def last_float(pattern: str, text: str) -> float | None:
    vals = re.findall(pattern, text)
    return float(vals[-1]) if vals else None


def parse_name(name: str):
    match = re.match(
        r"(?P<method>.+?)_km(?P<km>[0-9p]+)_(?P<ckpt>.+?)_"
        r"(?P<n_pos>\d+)p(?P<n_neg>\d+)at(?P<top_k>\d+)$",
        name,
    )
    if not match:
        return None
    data = match.groupdict()
    data["keep_multiplier"] = data["km"].replace("p", ".")
    data["setting"] = f"{data['n_pos']}+{data['n_neg']}@{data['top_k']}"
    return data


def mean_std(vals: list[float]) -> str:
    if not vals:
        return ""
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.6f}+/-{std:.6f}"


def mean_value(text: str) -> float:
    return float(text.split("+/-")[0])


def main() -> None:
    args = parse_args()
    rows = []
    for log_path in sorted(args.log_root.glob("*.log")):
        parsed = parse_name(log_path.stem)
        if parsed is None:
            print(f"[WARN] cannot parse name: {log_path.name}")
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        hr = last_float(r"final_test/HR\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)
        ndcg = last_float(r"final_test/NDCG\D+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)", text)
        density = last_float(r"Avg density:\s*([0-9]+(?:\.[0-9]+)?)", text)
        rows.append(
            {
                "method": parsed["method"],
                "keep_multiplier": parsed["keep_multiplier"],
                "ckpt": parsed["ckpt"],
                "setting": parsed["setting"],
                "density": "" if density is None else f"{density:.8f}",
                "HR": "" if hr is None else f"{hr:.6f}",
                "NDCG": "" if ndcg is None else f"{ndcg:.6f}",
                "log_path": str(log_path),
            }
        )

    raw_path = args.log_root / "keep_sensitivity_raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "keep_multiplier", "ckpt", "setting", "density", "HR", "NDCG", "log_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    keys = sorted({(r["method"], r["keep_multiplier"], r["setting"]) for r in rows})
    for method, keep_multiplier, setting in keys:
        group = [
            r
            for r in rows
            if r["method"] == method
            and r["keep_multiplier"] == keep_multiplier
            and r["setting"] == setting
        ]
        summary_rows.append(
            {
                "method": method,
                "keep_multiplier": keep_multiplier,
                "setting": setting,
                "n": str(len(group)),
                "density": group[0]["density"],
                "HR": mean_std([float(r["HR"]) for r in group if r["HR"]]),
                "NDCG": mean_std([float(r["NDCG"]) for r in group if r["NDCG"]]),
            }
        )

    summary_path = args.log_root / "keep_sensitivity_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "keep_multiplier", "setting", "n", "density", "HR", "NDCG"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_by_key = {
        (r["method"], r["keep_multiplier"], r["setting"]): r for r in summary_rows
    }
    delta_rows = []
    method_keys = [
        (r["keep_multiplier"], r["setting"])
        for r in summary_rows
        if r["method"] == args.method
    ]
    for keep_multiplier, setting in sorted(set(method_keys)):
        base = summary_by_key.get((args.baseline, keep_multiplier, setting))
        method = summary_by_key.get((args.method, keep_multiplier, setting))
        if not base or not method:
            continue
        base_hr, method_hr = mean_value(base["HR"]), mean_value(method["HR"])
        base_ndcg, method_ndcg = mean_value(base["NDCG"]), mean_value(method["NDCG"])
        delta_rows.append(
            {
                "keep_multiplier": keep_multiplier,
                "setting": setting,
                "baseline_HR": f"{base_hr:.6f}",
                "method_HR": f"{method_hr:.6f}",
                "delta_HR": f"{method_hr - base_hr:+.6f}",
                "baseline_NDCG": f"{base_ndcg:.6f}",
                "method_NDCG": f"{method_ndcg:.6f}",
                "delta_NDCG": f"{method_ndcg - base_ndcg:+.6f}",
            }
        )

    delta_path = args.log_root / "keep_sensitivity_delta.md"
    with delta_path.open("w", encoding="utf-8") as f:
        f.write(
            "| keep_multiplier | setting | RevFilter HR | SAIF HR | delta HR | "
            "RevFilter NDCG | SAIF NDCG | delta NDCG |\n"
        )
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for r in delta_rows:
            f.write(
                f"| {r['keep_multiplier']} | {r['setting']} | {r['baseline_HR']} | "
                f"{r['method_HR']} | {r['delta_HR']} | {r['baseline_NDCG']} | "
                f"{r['method_NDCG']} | {r['delta_NDCG']} |\n"
            )

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {delta_path}")


if __name__ == "__main__":
    main()
