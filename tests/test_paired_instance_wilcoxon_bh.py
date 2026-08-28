from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.paired_instance_wilcoxon_bh import build_test_rows


FIELDNAMES = ["setting", "sample_id", "ckpt", "HR", "NDCG"]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def fixture_rows(run_ids: tuple[str, ...] = ("0", "1", "2")) -> list[dict[str, str]]:
    rows = []
    for sample_id in ("0", "1"):
        for run_id in run_ids:
            value = 0.1 * (int(sample_id) + 1) + 0.01 * int(run_id)
            rows.append(
                {
                    "setting": "1+5@1",
                    "sample_id": sample_id,
                    "ckpt": run_id,
                    "HR": str(value),
                    "NDCG": str(value / 2),
                }
            )
    return rows


def args_for(a: Path, b: Path) -> argparse.Namespace:
    return argparse.Namespace(
        a=a,
        b=b,
        a_method=None,
        b_method=None,
        metrics=["HR", "NDCG"],
        expected_runs=3,
        expected_samples=2,
        expected_settings=1,
        alternative="two-sided",
    )


class PairedAssertionsTest(unittest.TestCase):
    def test_accepts_exactly_matched_samples_and_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.csv"
            b = root / "b.csv"
            write_rows(a, fixture_rows())
            write_rows(b, fixture_rows())
            rows = build_test_rows(args_for(a, b))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["n"] == "2" for row in rows))

    def test_rejects_run_id_mismatch_even_when_run_count_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.csv"
            b = root / "b.csv"
            write_rows(a, fixture_rows())
            write_rows(b, fixture_rows(("0", "1", "3")))
            with self.assertRaisesRegex(AssertionError, "Run-ID mismatch"):
                build_test_rows(args_for(a, b))

    def test_rejects_missing_matched_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.csv"
            b = root / "b.csv"
            write_rows(a, fixture_rows())
            write_rows(
                b,
                [row for row in fixture_rows() if row["sample_id"] == "0"],
            )
            with self.assertRaisesRegex(AssertionError, "Sample mismatch"):
                build_test_rows(args_for(a, b))


if __name__ == "__main__":
    unittest.main()
