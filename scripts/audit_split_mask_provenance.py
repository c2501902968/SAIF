"""Audit the provenance and integrity of the Elliptic2 TRN/VAL/TST mask."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_DF = ROOT / "data" / "elliptic" / "raw" / "data_df.pkl"
MASK = ROOT / "data" / "elliptic" / "processed" / "mask.pt"
OUT_DIR = ROOT / "results" / "split_mask_provenance"
OFFICIAL_REVTRACK_BLOB_SHA1 = "c707bcff62ef61dcdb23bc5b76aff8b39c5c65a2"
OFFICIAL_REVTRACK_HEAD = "f2111c8a1bafd84ebaa5a04e5caca8f1f0ed7ac0"
RAW_INVENTORY = (
    ROOT
    / "audits"
    / "elliptic2_temporal_feasibility_20260814"
    / "outputs"
    / "inventory"
    / "inventory.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-df", type=Path, default=DATA_DF)
    parser.add_argument("--mask", type=Path, default=MASK)
    parser.add_argument(
        "--reference-data-df",
        type=Path,
        default=None,
        help="Optional independent RevTrack data_df.pkl copy for byte comparison.",
    )
    parser.add_argument(
        "--raw-inventory",
        type=Path,
        default=RAW_INVENTORY,
        help="Optional raw-release schema inventory JSON.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return canonical(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(canonical(item) for item in value))
    if isinstance(value, (list, tuple)):
        return tuple(canonical(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def record_signature(row: pd.Series) -> str:
    fields = (
        "labels",
        "senders",
        "source",
        "sink",
        "receivers",
        "node_ids",
        "edge_index",
    )
    payload = repr(tuple(canonical(row[field]) for field in fields)).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    data_df = args.data_df.resolve()
    mask_path = args.mask.resolve()
    reference_data_df = (
        args.reference_data_df.resolve() if args.reference_data_df is not None else None
    )
    raw_inventory_path = args.raw_inventory.resolve()

    frame = pd.read_pickle(data_df)
    mask = torch.load(mask_path, map_location="cpu", weights_only=False).long()
    mapping = {"TRN": 0, "VAL": 1, "TST": 2}
    expected_mask = torch.tensor([mapping[value] for value in frame["split"]], dtype=torch.long)

    rows = []
    for split in ("TRN", "VAL", "TST"):
        group = frame[frame["split"] == split]
        one_to_one = group["senders_len"].eq(1) & group["receivers_len"].eq(1)
        suffix = group["subg"].str[len(split) :].astype(int)
        rows.append(
            {
                "split": split,
                "total": int(len(group)),
                "licit": int(group["labels"].eq(0).sum()),
                "suspicious": int(group["labels"].eq(1).sum()),
                "one_to_one_licit": int((one_to_one & group["labels"].eq(0)).sum()),
                "one_to_one_suspicious": int((one_to_one & group["labels"].eq(1)).sum()),
                "observed_ratio": float(len(group) / len(frame)),
                "subg_suffix_min": int(suffix.min()),
                "subg_suffix_max": int(suffix.max()),
                "pre_filter_partition_size_implied_by_suffix": int(suffix.max() + 1),
            }
        )

    signatures: dict[str, set[str]] = defaultdict(set)
    for _, row in frame.iterrows():
        signatures[record_signature(row)].add(str(row["split"]))
    cross_split_signatures = {key: sorted(value) for key, value in signatures.items() if len(value) > 1}

    reset = frame.reset_index(names="source_index")
    index_split_counts = reset.groupby("source_index")["split"].nunique()
    local_blob = git_blob_sha1(data_df)
    if raw_inventory_path.is_file():
        raw_inventory = json.loads(raw_inventory_path.read_text(encoding="utf-8"))
        raw_schemas = {item["name"]: item["columns"] for item in raw_inventory["files"]}
    else:
        raw_schemas = {}
    split_tokens = {"split", "mask", "train", "val", "test", "trn", "tst", "fold"}
    raw_split_fields = sorted(
        {
            column
            for columns in raw_schemas.values()
            for column in columns
            if column.lower() in split_tokens
        }
    )
    report = {
        "artifact": {
            "path": str(data_df),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "sha256": sha256(data_df),
            "git_blob_sha1": local_blob,
            "official_revtrack_git_blob_sha1": OFFICIAL_REVTRACK_BLOB_SHA1,
            "official_revtrack_repository": "https://github.com/MITIBMxGraph/RevTrack",
            "official_revtrack_head_inspected": OFFICIAL_REVTRACK_HEAD,
            "byte_identical_to_official_revtrack_release": local_blob == OFFICIAL_REVTRACK_BLOB_SHA1,
            "local_revtrack_copy_path": (
                str(reference_data_df) if reference_data_df is not None else None
            ),
            "local_revtrack_copy_sha256": (
                sha256(reference_data_df)
                if reference_data_df is not None and reference_data_df.exists()
                else None
            ),
            "byte_identical_to_local_revtrack_copy": (
                reference_data_df is not None
                and reference_data_df.exists()
                and sha256(data_df) == sha256(reference_data_df)
            ),
        },
        "original_elliptic2_raw_schema": {
            "inventory_path": str(raw_inventory_path),
            "inventory_available": raw_inventory_path.is_file(),
            "files_and_columns": raw_schemas,
            "split_like_fields_found": raw_split_fields,
            "raw_release_directly_contains_partition": bool(raw_split_fields),
        },
        "mask_consumption": {
            "mask_path": str(mask_path),
            "mask_sha256": sha256(mask_path),
            "mapping": mapping,
            "mask_length": int(mask.numel()),
            "mask_values": sorted(int(value) for value in torch.unique(mask)),
            "mask_exactly_matches_dataframe_split_mapping": bool(torch.equal(mask, expected_mask)),
            "training_code_regenerates_partition": False,
            "loader_operation": "encodes existing data_df.pkl split strings as TRN=0, VAL=1, all other values=2",
        },
        "statistics": rows,
        "record_integrity": {
            "stable_subgraph_id_field": "subg",
            "subg_is_unique": bool(frame["subg"].is_unique),
            "subg_count": int(frame["subg"].nunique()),
            "subg_prefix_matches_split": bool(
                all(str(row.subg).startswith(str(row.split)) for row in frame.itertuples())
            ),
            "source_index_values_assigned_to_multiple_splits": int((index_split_counts > 1).sum()),
            "exact_content_signatures_assigned_to_multiple_splits": len(cross_split_signatures),
            "signature_fields": [
                "labels",
                "senders",
                "source",
                "sink",
                "receivers",
                "node_ids",
                "edge_index",
            ],
        },
        "provenance_conclusion": {
            "provider": "official RevTrack released preprocessed Elliptic2 artifact",
            "published_rule": "random 80:10:10 training/validation/test split",
            "generation_script_available_in_revtrack_repository": False,
            "random_seed_recoverable": False,
            "temporal_split": False,
            "stratification_or_group_constraints_verified": False,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "split_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
