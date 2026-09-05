"""Rebuild the Challenge2015 portion of Model 3 with header calibration.

This module intentionally stops before Model 3 training. It reads the nested
archive in memory, creates research-only features, and writes reports under
the requested Phase 2E results directory. Production code, ONNX files, and
the existing feature table are never modified.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_NAMES, extract_features

SEED = 42
TARGET_FS = 360
SOURCE = "c2015"
LABEL_MAP = {
    "Ventricular_Tachycardia": "VT",
    "Ventricular_Flutter_Fib": "VF",
}


def parse_header(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    first = lines[0].split()
    channels = []
    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        gain, units = fields[2].split("/", 1)
        channels.append({
            "file": fields[0],
            "gain": float(gain),
            "units": units,
            "adc_resolution": int(fields[3]),
            "baseline": float(fields[4]),
            "adc_zero": float(fields[5]),
            "initial_value": float(fields[6]),
            "channel": fields[-1],
        })
    return {
        "record_id": first[0],
        "channel_count": int(first[1]),
        "sampling_frequency": float(first[2]),
        "sample_count": int(first[3]),
        "channels": channels,
        "comments": [line[1:] for line in lines if line.startswith("#")],
    }


def load_training_archive(archive_path: Path) -> zipfile.ZipFile:
    outer = zipfile.ZipFile(archive_path)
    training = next(
        item for item in outer.infolist() if item.filename.endswith("/training.zip")
    )
    return zipfile.ZipFile(io.BytesIO(outer.read(training)))


def read_alarms(archive: zipfile.ZipFile) -> dict[str, tuple[str, bool]]:
    alarms = {}
    for line in archive.read("training/ALARMS").decode(errors="replace").splitlines():
        fields = line.strip().split(",")
        if len(fields) != 3:
            continue
        record_id, alarm, flag = fields
        alarms[record_id] = (alarm, flag == "1")
    return alarms


def calibrated_channel(archive: zipfile.ZipFile, record_id: str, channel_name: str = "II") -> tuple[np.ndarray, dict[str, Any]]:
    metadata = parse_header(archive.read(f"training/{record_id}.hea").decode(errors="replace"))
    channels = {channel["channel"]: (index, channel) for index, channel in enumerate(metadata["channels"])}
    if channel_name not in channels:
        raise KeyError(f"Channel {channel_name} is absent for {record_id}")
    channel_index, channel = channels[channel_name]
    values = loadmat(io.BytesIO(archive.read(f"training/{record_id}.mat")))["val"]
    digital = values[channel_index].astype(np.float64)
    physical = (digital - channel["adc_zero"]) / channel["gain"] + channel["baseline"]
    metadata.update({
        "selected_channel": channel_name,
        "channel_index": channel_index,
        "gain": channel["gain"],
        "units": channel["units"],
        "adc_resolution": channel["adc_resolution"],
        "baseline": channel["baseline"],
        "adc_zero": channel["adc_zero"],
        "formula": "(digital - adc_zero) / gain + baseline",
        "raw_min": float(digital.min()),
        "raw_max": float(digital.max()),
        "physical_min": float(physical.min()),
        "physical_max": float(physical.max()),
        "physical_mean": float(physical.mean()),
        "physical_std": float(physical.std()),
        "duration_seconds": float(len(physical) / metadata["sampling_frequency"]),
    })
    return physical, metadata


def window_features(signal: np.ndarray, sampling_frequency: float) -> list[dict[str, float]]:
    resampled = resample_poly(signal, TARGET_FS, int(sampling_frequency))
    windows = []
    window_samples = TARGET_FS * 5
    step_samples = int(TARGET_FS * 2.5)
    for start in range(0, len(resampled) - window_samples, step_samples):
        vector = extract_features(resampled[start:start + window_samples], TARGET_FS)[0]
        windows.append(dict(zip(FEATURE_NAMES, vector.astype(float))))
    return windows


def reconstruct(archive: zipfile.ZipFile, alarms: dict[str, tuple[str, bool]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = [line for line in archive.read("training/RECORDS").decode(errors="replace").splitlines() if line]
    rows = []
    metadata_rows = []
    excluded = []
    for record_id in records:
        alarm, is_true = alarms.get(record_id, (None, False))
        if alarm not in LABEL_MAP or not is_true:
            continue
        try:
            signal, metadata = calibrated_channel(archive, record_id, "II")
        except KeyError as error:
            excluded.append({
                "record_id": record_id,
                "label": LABEL_MAP[alarm],
                "reason": str(error),
            })
            continue
        label = LABEL_MAP[alarm]
        for features in window_features(signal, metadata["sampling_frequency"]):
            rows.append({**features, "label": label, "record_id": f"c2015_{record_id}", "dataset_source": SOURCE})
        metadata_rows.append({"record_id": f"c2015_{record_id}", "label": label, **metadata})
    return pd.DataFrame(rows), pd.DataFrame(metadata_rows), pd.DataFrame(excluded)


def summarize_features(table: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    columns = FEATURE_NAMES
    grouped = table.groupby(group, observed=True)[columns] if group else [("all", table[columns])]
    rows = []
    for name, frame in grouped:
        for feature in columns:
            values = frame[feature]
            rows.append({
                "group": name,
                "feature": feature,
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            })
    return pd.DataFrame(rows)


def compare_old_new(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    old = old[old["record_id"].astype(str).str.startswith("c2015_")].copy()
    new = new.copy()
    old["window_index"] = old.groupby(["record_id", "label"], observed=True).cumcount()
    new["window_index"] = new.groupby(["record_id", "label"], observed=True).cumcount()
    merged = old.merge(
        new,
        on=["record_id", "label", "window_index"],
        suffixes=("_old", "_new"),
        how="inner",
    )
    rows = []
    for feature in FEATURE_NAMES:
        old_values = merged[f"{feature}_old"]
        new_values = merged[f"{feature}_new"]
        ratio = new_values / old_values.replace(0, np.nan)
        rows.append({
            "feature": feature,
            "matched_rows": len(merged),
            "old_mean": old_values.mean(),
            "new_mean": new_values.mean(),
            "old_median": old_values.median(),
            "new_median": new_values.median(),
            "old_std": old_values.std(),
            "new_std": new_values.std(),
            "old_min": old_values.min(),
            "new_min": new_values.min(),
            "old_max": old_values.max(),
            "new_max": new_values.max(),
            "new_old_median_ratio": ratio.median(),
            "median_absolute_difference": (new_values - old_values).abs().median(),
        })
    return pd.DataFrame(rows)


def source_diagnostic(table: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    all_records = np.array(sorted(table["record_id"].unique()))
    record_train, record_test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(
            all_records, np.zeros(len(all_records)), groups=all_records
        )
    )
    train_records = set(all_records[record_train])
    test_records = set(all_records[record_test])
    train = np.flatnonzero(table["record_id"].isin(train_records).to_numpy())
    test = np.flatnonzero(table["record_id"].isin(test_records).to_numpy())
    overlap = sorted(train_records & test_records)
    assert not overlap
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    model.fit(table.iloc[train][FEATURE_NAMES], table.iloc[train]["dataset_source"])
    prediction = model.predict(table.iloc[test][FEATURE_NAMES])
    truth = table.iloc[test]["dataset_source"]
    classes = model[-1].classes_
    matrix = confusion_matrix(truth, prediction, labels=classes)
    importance = pd.DataFrame({"feature": FEATURE_NAMES, "importance": np.abs(model[-1].coef_).mean(axis=0)}).sort_values("importance", ascending=False)
    return {
        "accuracy": accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "record_overlap": overlap,
        "classes": classes.tolist(),
    }, pd.DataFrame(matrix, index=classes, columns=classes), importance


def plot_feature_comparison(old: pd.DataFrame, new: pd.DataFrame, path: Path) -> None:
    summary = pd.DataFrame({"old": old[FEATURE_NAMES].mean(), "new": new[FEATURE_NAMES].mean()})
    axis = summary.plot.bar(figsize=(12, 4), logy=False)
    axis.set_ylabel("Feature mean")
    axis.set_title("Challenge2015 old versus header-calibrated feature means")
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=140)
    plt.close(axis.figure)


def run(feature_table: Path, archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    old = pd.read_csv(feature_table)
    old["dataset_source"] = old["record_id"].astype(str).str.split("_", n=1).str[0]
    archive = load_training_archive(archive_path)
    alarms = read_alarms(archive)
    reconstructed, metadata, excluded = reconstruct(archive, alarms)
    if reconstructed.empty:
        raise RuntimeError("No calibrated VF/VT windows were reconstructed")
    if set(reconstructed["label"]) - {"VF", "VT"}:
        raise RuntimeError("Unexpected labels in reconstructed table")
    if reconstructed[FEATURE_NAMES].isna().any().any() or not np.isfinite(reconstructed[FEATURE_NAMES].to_numpy()).all():
        raise RuntimeError("Non-finite reconstructed feature values")

    old_c2015 = old[old["record_id"].astype(str).str.startswith("c2015_")].copy()
    comparison = compare_old_new(old_c2015, reconstructed)
    combined_old = old.copy()
    combined_new = pd.concat([old[~old["record_id"].astype(str).str.startswith("c2015_")], reconstructed], ignore_index=True)
    old_diag, old_matrix, old_importance = source_diagnostic(combined_old)
    new_diag, new_matrix, new_importance = source_diagnostic(combined_new)

    metadata.to_csv(output_dir / "channel_metadata.csv", index=False)
    metadata.to_csv(output_dir / "calibration_metadata.csv", index=False)
    excluded.to_csv(output_dir / "excluded_records.csv", index=False)
    reconstructed.to_csv(output_dir / "reconstructed_feature_table.csv", index=False)
    reconstructed.groupby(["dataset_source", "label"], observed=True).size().reset_index(name="windows").to_csv(output_dir / "source_label_counts.csv", index=False)
    reconstructed.groupby(["dataset_source", "record_id"], observed=True).size().reset_index(name="windows").to_csv(output_dir / "record_statistics.csv", index=False)
    combined_new.groupby("dataset_source", observed=True)[FEATURE_NAMES].agg(["count", "mean", "median", "std", "min", "max"]).to_csv(output_dir / "source_feature_statistics.csv")
    comparison.to_csv(output_dir / "feature_comparison.csv", index=False)
    pd.concat([summarize_features(old_c2015, "label").assign(condition="old_csv"), summarize_features(reconstructed, "label").assign(condition="header_calibrated")], ignore_index=True).to_csv(output_dir / "old_vs_new_feature_summary.csv", index=False)
    summary = {
        "conversion": "(digital - adc_zero) / gain + baseline",
        "source": SOURCE,
        "sampling_rate_input_hz": 250,
        "sampling_rate_output_hz": TARGET_FS,
        "selected_channel": "II only",
        "old_gain_200_used": False,
        "reconstructed_rows": len(reconstructed),
        "reconstructed_records": int(reconstructed["record_id"].nunique()),
        "excluded_records": excluded["record_id"].tolist(),
        "afib_rows": 0,
        "source_diagnostics": {"old": old_diag, "header_calibrated": new_diag},
        "limitation": "Challenge2015 reconstruction only; VFDB and CUDB raw data are not in this archive.",
    }
    with (output_dir / "source_classification_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary["source_diagnostics"], handle, indent=2)
    old_matrix.to_csv(output_dir / "source_confusion_matrix_old.csv")
    new_matrix.to_csv(output_dir / "source_confusion_matrix_calibrated.csv")
    new_matrix.to_csv(output_dir / "source_confusion_matrix.csv")
    old_importance.to_csv(output_dir / "feature_importance_old.csv", index=False)
    new_importance.to_csv(output_dir / "feature_importance_calibrated.csv", index=False)
    new_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    with (output_dir / "reconstruction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    pd.DataFrame([summary]).drop(columns=["source_diagnostics"]).to_csv(output_dir / "reconstruction_summary.csv", index=False)
    plot_feature_comparison(old_c2015, reconstructed, plots / "old_vs_calibrated_means.png")
    print(json.dumps(summary, indent=2))
    print("\nFeature comparison:")
    print(comparison.to_string(index=False))
    print("\nOld source diagnostic:", old_diag)
    print("Calibrated source diagnostic:", new_diag)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct Challenge2015 Model 3 features with header calibration.")
    parser.add_argument("feature_table", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/model3_phase2e/results"))
    args = parser.parse_args()
    run(args.feature_table, args.archive, args.output_dir)


if __name__ == "__main__":
    main()