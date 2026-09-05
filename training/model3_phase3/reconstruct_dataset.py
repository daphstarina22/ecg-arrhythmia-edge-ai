"""Validate and stage the Phase 2E.2 calibrated reconstruction.

Raw waveform reconstruction was completed in Phase 2E.2. This script stages
that validated artifact for Phase 3 without touching the original feature
table or any production asset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "mean", "std", "ptp", "zero_crossings", "peak_count", "mean_rr",
    "rr_cv", "qrs_width", "dominant_freq", "vf_band_power_ratio",
]
LABELS = {"VT", "VF"}


def validate(table: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURE_NAMES + ["label", "record_id", "dataset_source"])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if set(table["label"].astype(str)) != LABELS:
        raise ValueError("Labels are not exactly VT and VF")
    if table["label"].astype(str).str.upper().eq("AFIB").any():
        raise ValueError("AFIB is present")
    if not np.isfinite(table[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature values are present")
    if table["record_id"].isna().any() or table["dataset_source"].isna().any():
        raise ValueError("Missing record or source values")
    return table.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage the validated Phase 2E.2 table for Phase 3.")
    parser.add_argument("calibrated_table", type=Path)
    parser.add_argument("--artifact", type=Path, default=Path("training/model3_phase3/artifacts/calibrated_vfvt_feature_table.csv"))
    parser.add_argument("--results", type=Path, default=Path("training/model3_phase3/results"))
    args = parser.parse_args()
    table = validate(pd.read_csv(args.calibrated_table))
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.artifact, index=False)
    inventory = table.groupby(["dataset_source", "label"], observed=True).agg(
        windows=("label", "size"), records=("record_id", "nunique")
    ).reset_index()
    inventory.to_csv(args.results / "raw_data_inventory.csv", index=False)
    summary = {
        "input": str(args.calibrated_table),
        "artifact": str(args.artifact),
        "rows": len(table),
        "records": int(table["record_id"].nunique()),
        "labels": table["label"].value_counts().to_dict(),
        "sources": table["dataset_source"].value_counts().to_dict(),
        "afib_rows": 0,
        "calibration_provenance": "Phase 2E.2 WFDB/Challenge2015 calibrated reconstruction",
        "production_changed": False,
    }
    with (args.results / "raw_data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()