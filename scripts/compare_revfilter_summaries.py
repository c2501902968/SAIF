#!/usr/bin/env python
"""Compare two RevFilter summary CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def mean_value(text: str) -> float:
    return float(text.split("+/-")[0])


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["setting"]: row for row in csv.DictReader(f)}


def main() -> None:
    args = parse_args()
    a = read_summary(args.a)
    b = read_summary(args.b)
    settings = [s for s in a if s in b]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            f"| setting | {args.a_name} HR | {args.b_name} HR | ΔHR | "
            f"{args.a_name} NDCG | {args.b_name} NDCG | ΔNDCG |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for setting in settings:
            a_hr, b_hr = mean_value(a[setting]["HR"]), mean_value(b[setting]["HR"])
            a_ndcg, b_ndcg = mean_value(a[setting]["NDCG"]), mean_value(b[setting]["NDCG"])
            f.write(
                f"| {setting} | {a_hr:.6f} | {b_hr:.6f} | {b_hr - a_hr:+.6f} | "
                f"{a_ndcg:.6f} | {b_ndcg:.6f} | {b_ndcg - a_ndcg:+.6f} |\n"
            )
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
