"""Reconstruct original Model 3 VF/VT records with calibrated WFDB signals.

This module does not train models, export ONNX, or modify any production or
prior-phase artifacts. It extracts the archives into temporary directories,
rebuilds only record IDs already represented in the original feature table,
and writes research reports under the Phase 2E.2 results directory.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample_poly

from src.features import FEATURE_NAMES, extract_features

SEED = 42
TARGET_FS = 360
WINDOW_SECONDS = 5
STEP_SECONDS = 2.5
LABELS = {"VF", "VT"}


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


def extract_archive(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    return destination


def source_record_ids(table: pd.DataFrame, source: str) -> set[str]:
    prefix = f"{source}_"
    return {
        record_id[len(prefix):]
        for record_id in table.loc[
            table["record_id"].str.startswith(prefix), "record_id"
        ]
    }


def classify_aux(note: str) -> str | None:
    normalized = note.strip().rstrip("\x00")
    if "(VF" in normalized:
        return "VF"
    if "(VT" in normalized:
        return "VT"
    if "AFIB" in normalized.upper() or "(AF" in normalized:
        return None
    return None


def reconstruct_wfdb_record(
    record_path: Path,
    record_id: str,
    source: str,
    allowed_record_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], set[str]]:
    metadata: dict[str, Any] = {
        "source": source,
        "record_id": record_id,
        "expected_in_original": record_id in allowed_record_ids,
    }
    excluded: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    record = wfdb.rdrecord(str(record_path))
    metadata.update({
        "sampling_frequency": float(record.fs),
        "channel_names": list(record.sig_name or []),
        "channel_count": int(record.n_sig),
        "signal_representation": "record.p_signal (WFDB-calibrated physical signal)",
        "units": list(record.units or []),
        "adc_gain": list(record.adc_gain or []),
        "baseline": list(record.baseline or []),
        "adc_zero": list(record.adc_zero or []),
        "duration_seconds": float(record.sig_len / record.fs),
        "target_fs": TARGET_FS,
        "resampling": "resample_poly when source fs differs from 360 Hz",
    })
    if record.p_signal is None:
        raise ValueError(f"No physical p_signal available for {record_id}")
    ecg_indices = [
        index for index, name in enumerate(record.sig_name or [])
        if str(name).upper() in {"ECG", "II"} or "ECG" in str(name).upper()
    ]
    if not ecg_indices:
        excluded.append({"source": source, "record_id": record_id, "reason": "no ECG channel"})
        return [], metadata, excluded, seen_labels
    channel_index = 0 if source == "vfdb" else ecg_indices[0]
    metadata["selected_channel"] = record.sig_name[channel_index]
    metadata["selected_channel_index"] = channel_index
    if source == "vfdb" and record.sig_name[channel_index] != "ECG":
        raise ValueError(f"Unexpected VFDB channel selection for {record_id}")

    annotation = wfdb.rdann(str(record_path), "atr")
    boundaries = list(zip(annotation.sample, annotation.aux_note))
    if source == "cudb":
        boundaries = [
            (sample, note) for sample, note in boundaries
            if str(note).strip().rstrip("\x00")
        ]
    rows: list[dict[str, Any]] = []
    signal = record.p_signal[:, channel_index].astype(np.float64)
    nan_fraction = float(np.isnan(signal).mean())
    metadata["nan_fraction"] = nan_fraction
    if nan_fraction > 0.1:
        excluded.append({
            "source": source,
            "record_id": record_id,
            "reason": f"signal NaN fraction {nan_fraction:.6f} exceeds 0.1",
        })
        return [], metadata, excluded, seen_labels
    if nan_fraction > 0:
        signal = pd.Series(signal).interpolate(limit_direction="both").to_numpy()
        metadata["nan_handling"] = "linear interpolation, limit_direction=both"
    else:
        metadata["nan_handling"] = "none"
    for index, (start_sample, note) in enumerate(boundaries):
        label = classify_aux(str(note))
        if label is None:
            continue
        seen_labels.add(label)
        end_sample = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(signal)
        segment = signal[start_sample:end_sample]
        if len(segment) < record.fs:
            continue
        if record.fs != TARGET_FS:
            segment = resample_poly(segment, TARGET_FS, int(record.fs))
        window_samples = TARGET_FS * WINDOW_SECONDS
        step_samples = int(TARGET_FS * STEP_SECONDS)
        for start in range(0, len(segment) - window_samples, step_samples):
            features = extract_features(segment[start:start + window_samples], TARGET_FS)[0]
            if not np.isfinite(features).all():
                raise ValueError(f"Non-finite features for {source}_{record_id}")
            rows.append({
                **dict(zip(FEATURE_NAMES, features.astype(float))),
                "label": label,
                "record_id": f"{source}_{record_id}",
                "dataset_source": source,
            })
    if not rows and record_id in allowed_record_ids:
        excluded.append({"source": source, "record_id": record_id, "reason": "no eligible VF/VT windows"})
    return rows, metadata, excluded, seen_labels


def feature_summary(table: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = table.groupby(group_columns, observed=True)[FEATURE_NAMES].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    return summary


def compare_feature_summaries(original: pd.DataFrame, calibrated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in sorted(set(original["dataset_source"]) & set(calibrated["dataset_source"])):
        for feature in FEATURE_NAMES:
            old_values = original.loc[original["dataset_source"] == source, feature]
            new_values = calibrated.loc[calibrated["dataset_source"] == source, feature]
            rows.append({
                "dataset_source": source,
                "feature": feature,
                "original_mean": old_values.mean(),
                "calibrated_mean": new_values.mean(),
                "original_median": old_values.median(),
                "calibrated_median": new_values.median(),
                "original_std": old_values.std(),
                "calibrated_std": new_values.std(),
                "original_min": old_values.min(),
                "calibrated_min": new_values.min(),
                "original_max": old_values.max(),
                "calibrated_max": new_values.max(),
                "median_absolute_difference": (
                    new_values.reset_index(drop=True) - old_values.reset_index(drop=True)
                ).abs().median(),
            })
    return pd.DataFrame(rows)


def source_diagnostic(table: pd.DataFrame, split_records: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_records, test_records = split_records
    train_mask = table["record_id"].isin(train_records)
    test_mask = table["record_id"].isin(test_records)
    overlap = sorted(set(train_records) & set(test_records))
    assert not overlap
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )
    model.fit(table.loc[train_mask, FEATURE_NAMES], table.loc[train_mask, "dataset_source"])
    prediction = model.predict(table.loc[test_mask, FEATURE_NAMES])
    truth = table.loc[test_mask, "dataset_source"]
    classes = model[-1].classes_
    matrix = pd.DataFrame(
        confusion_matrix(truth, prediction, labels=classes),
        index=classes,
        columns=classes,
    )
    importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": np.abs(model[-1].coef_).mean(axis=0),
    }).sort_values("importance", ascending=False)
    return {
        "accuracy": accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "record_overlap": overlap,
        "classes": classes.tolist(),
    }, matrix, importance


def plot_source_accuracy(values: dict[str, float], path: Path) -> None:
    axis = pd.Series(values).plot.bar(figsize=(7, 4), ylim=(0, 1), title="Source classification accuracy")
    axis.set_ylabel("Accuracy")
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=140)
    plt.close(axis.figure)


def run(original_path: Path, vfdb_archive: Path, cudb_archive: Path, c2015_path: Path, output_dir: Path) -> None:
    original = validate_table(pd.read_csv(original_path))
    c2015 = validate_table(pd.read_csv(c2015_path))
    if set(c2015["dataset_source"]) != {"c2015"}:
        raise ValueError("Phase 2E.1 table is not exclusively c2015")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model3_phase2e2_") as temporary:
        temporary_path = Path(temporary)
        vfdb_root = extract_archive(vfdb_archive, temporary_path / "vfdb")
        cudb_root = extract_archive(cudb_archive, temporary_path / "cudb")
        vfdb_records = source_record_ids(original, "vfdb")
        cudb_records = source_record_ids(original, "cudb")
        rows: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        decoded_labels: dict[str, list[str]] = {}
        for source, root, allowed in (("vfdb", vfdb_root, vfdb_records), ("cudb", cudb_root, cudb_records)):
            record_root = root
            if source == "cudb":
                record_root = root / "cu-ventricular-tachyarrhythmia-database-1.0.0"
            for record_id in sorted(allowed):
                record_path = record_root / record_id
                try:
                    record_rows, record_metadata, record_excluded, labels = reconstruct_wfdb_record(
                        record_path, record_id, source, allowed
                    )
                except Exception as error:
                    excluded.append({"source": source, "record_id": record_id, "reason": str(error)})
                    continue
                rows.extend(record_rows)
                metadata.append(record_metadata)
                excluded.extend(record_excluded)
                decoded_labels[f"{source}_{record_id}"] = sorted(labels)

        reconstructed_wfdb = pd.DataFrame(rows)
        if reconstructed_wfdb.empty:
            raise RuntimeError("No VFDB/CUDB rows were reconstructed")
        calibrated = pd.concat([c2015, reconstructed_wfdb], ignore_index=True)
        calibrated = calibrated[FEATURE_NAMES + ["label", "record_id", "dataset_source"]]
        calibrated = validate_table(calibrated)
        if calibrated["label"].astype(str).str.upper().eq("AFIB").any():
            raise RuntimeError("AFIB entered calibrated reconstruction")

        all_records = np.array(sorted(original["record_id"].unique()))
        train_index, test_index = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(
                all_records, np.zeros(len(all_records)), groups=all_records
            )
        )
        split_records = (all_records[train_index], all_records[test_index])
        original_diag, original_matrix, original_importance = source_diagnostic(original, split_records)
        calibrated_diag, calibrated_matrix, calibrated_importance = source_diagnostic(calibrated, split_records)

        calibrated.to_csv(output_dir / "calibrated_model3_feature_table.csv", index=False)
        reconstructed_wfdb.to_csv(output_dir / "vfdb_cudb_reconstructed_features.csv", index=False)
        pd.DataFrame(metadata).to_csv(output_dir / "calibration_metadata.csv", index=False)
        pd.DataFrame(excluded).to_csv(output_dir / "excluded_records.csv", index=False)
        pd.DataFrame([
            {"record_id": key, "decoded_labels": "|".join(value)}
            for key, value in sorted(decoded_labels.items())
        ]).to_csv(output_dir / "decoded_labels.csv", index=False)
        feature_summary(original, ["dataset_source", "label"]).to_csv(output_dir / "original_feature_summary.csv")
        feature_summary(calibrated, ["dataset_source", "label"]).to_csv(output_dir / "calibrated_feature_summary.csv")
        compare_feature_summaries(original, calibrated).to_csv(
            output_dir / "original_vs_calibrated_feature_comparison.csv", index=False
        )
        calibrated.groupby(["dataset_source", "label", "record_id"], observed=True).size().reset_index(name="windows").to_csv(output_dir / "record_statistics.csv", index=False)
        calibrated.groupby(["dataset_source", "label"], observed=True).size().reset_index(name="windows").to_csv(output_dir / "source_label_counts.csv", index=False)
        original_matrix.to_csv(output_dir / "source_confusion_matrix_original.csv")
        calibrated_matrix.to_csv(output_dir / "source_confusion_matrix_calibrated.csv")
        original_importance.to_csv(output_dir / "feature_importance_original.csv", index=False)
        calibrated_importance.to_csv(output_dir / "feature_importance_calibrated.csv", index=False)
        source_values = {"original": original_diag["accuracy"], "calibrated": calibrated_diag["accuracy"]}
        plot_source_accuracy(source_values, output_dir / "source_classification_accuracy.png")
        with (output_dir / "source_diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump({"original": original_diag, "calibrated": calibrated_diag}, handle, indent=2)
        with (output_dir / "split_definition.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "random_state": SEED,
                "test_size": 0.2,
                "grouping": "full record_id",
                "train_records": split_records[0].tolist(),
                "test_records": split_records[1].tolist(),
                "overlap": sorted(set(split_records[0]) & set(split_records[1])),
            }, handle, indent=2)
        summary = {
            "sources": ["c2015", "vfdb", "cudb"],
            "original_rows": len(original),
            "calibrated_rows": len(calibrated),
            "original_records": int(original["record_id"].nunique()),
            "calibrated_records": int(calibrated["record_id"].nunique()),
            "labels": calibrated["label"].value_counts().to_dict(),
            "afib_rows": int(calibrated["label"].astype(str).str.upper().eq("AFIB").sum()),
            "record_overlap": sorted(set(split_records[0]) & set(split_records[1])),
            "original_source_diagnostic": original_diag,
            "calibrated_source_diagnostic": calibrated_diag,
            "production_changed": False,
            "onnx_changed": False,
        }
        with (output_dir / "reconstruction_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only calibrated Model 3 reconstruction.")
    parser.add_argument("original_table", type=Path)
    parser.add_argument("vfdb_archive", type=Path)
    parser.add_argument("cudb_archive", type=Path)
    parser.add_argument("c2015_table", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/model3_phase2e2/results"))
    args = parser.parse_args()
    run(args.original_table, args.vfdb_archive, args.cudb_archive, args.c2015_table, args.output_dir)


if __name__ == "__main__":
    main()