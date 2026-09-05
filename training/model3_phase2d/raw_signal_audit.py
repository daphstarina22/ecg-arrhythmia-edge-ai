"""Audit raw ECG availability and source-scale evidence for Model 3.

This is an analysis-only tool. It never changes feature tables, production
feature extraction, ONNX files, or deployment code. Feature-table statistics
are not treated as a substitute for raw signal calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "mean", "std", "ptp", "zero_crossings", "peak_count", "mean_rr",
    "rr_cv", "qrs_width", "dominant_freq", "vf_band_power_ratio",
]
LABELS = {"VT", "VF"}
RAW_SUFFIXES = {".dat", ".hea", ".atr", ".npy", ".npz", ".mat"}


def validate_table(table: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURE_NAMES + ["label", "record_id"])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    labels = table["label"].astype(str)
    unexpected = sorted(set(labels) - LABELS)
    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")
    if labels.str.upper().eq("AFIB").any():
        raise ValueError("AFIB is present in the feature table")
    if table[FEATURE_NAMES].isna().any().any():
        raise ValueError("NaN feature values are present")
    if not np.isfinite(table[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature values are present")
    result = table.copy()
    result["record_id"] = result["record_id"].astype(str)
    result["dataset_source"] = result["record_id"].str.split("_", n=1).str[0]
    return result


def find_raw_files(raw_roots: list[Path]) -> list[str]:
    files = []
    for root in raw_roots:
        if root.exists():
            files.extend(
                str(path)
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in RAW_SUFFIXES
            )
    return sorted(set(files))


def feature_scale_summary(table: pd.DataFrame) -> pd.DataFrame:
    scale_features = ["mean", "std", "ptp"]
    return table.groupby("dataset_source", observed=True)[scale_features].agg(
        ["mean", "median", "std", "min", "max"]
    )


def record_summary(table: pd.DataFrame) -> pd.DataFrame:
    counts = table.groupby(["dataset_source", "record_id"]).size()
    return counts.groupby(level=0).agg(
        records="count",
        mean_windows="mean",
        median_windows="median",
        std_windows="std",
        min_windows="min",
        max_windows="max",
    ).reset_index()


def run(feature_table: Path, output_dir: Path, raw_roots: list[Path]) -> dict[str, Any]:
    table = validate_table(pd.read_csv(feature_table))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = find_raw_files(raw_roots)

    summary = {
        "feature_table": str(feature_table),
        "rows": int(len(table)),
        "labels": table["label"].value_counts().to_dict(),
        "afib_rows": int(table["label"].astype(str).str.upper().eq("AFIB").sum()),
        "records": int(table["record_id"].nunique()),
        "sources": table["dataset_source"].value_counts().to_dict(),
        "raw_signal_files_found": len(raw_files),
        "raw_signal_files": raw_files,
        "signal_level_normalization_available": bool(raw_files),
        "normalization_note": (
            "Raw signal audit can proceed from the supplied roots."
            if raw_files
            else "Only feature-space evidence is available; true signal-level normalization was not performed."
        ),
    }

    source_scale_summary = feature_scale_summary(table)
    source_scale_summary.to_csv(output_dir / "feature_scale_summary.csv")
    record_summary(table).to_csv(output_dir / "record_summary.csv", index=False)
    table.groupby(["dataset_source", "label"], observed=True).size().reset_index(name="windows").to_csv(
        output_dir / "source_label_counts.csv", index=False
    )
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nFeature-scale summary:")
    print(source_scale_summary.round(4).to_string())
    print("\nRecord summary:")
    print(record_summary(table).round(2).to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw ECG availability for Phase 2D.")
    parser.add_argument("feature_table", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/model3_phase2d/results"))
    parser.add_argument("--raw-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    run(args.feature_table, args.output_dir, args.raw_root)


if __name__ == "__main__":
    main()