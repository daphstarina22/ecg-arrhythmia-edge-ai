"""Controlled Model 3 VF/VT experiments.

This module consumes an existing feature table and writes only diagnostic
reports. It never imports production inference code, writes ONNX files, or
modifies the ten-feature deployment contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "mean", "std", "ptp", "zero_crossings", "peak_count", "mean_rr",
    "rr_cv", "qrs_width", "dominant_freq", "vf_band_power_ratio",
]
REDUCED_FEATURE_NAMES = [name for name in FEATURE_NAMES if name not in {"mean", "std", "ptp"}]
LABELS = ["VT", "VF"]
SEED = 42


def prepare_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy a notebook-created Model 3 table."""
    required = set(FEATURE_NAMES + ["label", "record_id"])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared = table.copy()
    labels = prepared["label"].astype(str)
    unexpected = sorted(set(labels) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected Model 3 labels: {unexpected}")
    if labels.str.upper().eq("AFIB").any():
        raise ValueError("AFIB is present in the Model 3 table")
    if prepared[FEATURE_NAMES].isna().any().any():
        raise ValueError("NaN values are present in the Model 3 features")
    if not np.isfinite(prepared[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values are present in the Model 3 features")

    prepared["record_id"] = prepared["record_id"].astype(str)
    prepared["dataset_source"] = (
        prepared["record_id"].str.split("_", n=1).str[0]
    )
    prepared["target"] = prepared["label"].map({"VT": 0, "VF": 1}).astype(int)
    return prepared


def exact_random_split(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reproduce the notebook's Model 3 record-group split."""
    groups = table["record_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(
        splitter.split(table[FEATURE_NAMES], table["target"], groups=groups)
    )
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = sorted(train_groups & test_groups)
    assert not overlap, f"Record leakage detected: {overlap}"
    return train_idx, test_idx, {
        "split": "GroupShuffleSplit(record_id, test_size=0.25, random_state=42)",
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "train_records": int(len(train_groups)),
        "test_records": int(len(test_groups)),
        "overlap": overlap,
        "train_record_ids": sorted(train_groups),
        "test_record_ids": sorted(test_groups),
    }


def source_split(table: pd.DataFrame, train_sources: set[str], test_source: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create a source-held-out split and verify both classes on each side."""
    train_mask = table["dataset_source"].isin(train_sources)
    test_mask = table["dataset_source"].eq(test_source)
    if (train_mask & test_mask).any():
        raise AssertionError("Source masks overlap")
    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    for name, indices in (("train", train_idx), ("test", test_idx)):
        classes = set(table.iloc[indices]["label"])
        if classes != set(LABELS):
            raise ValueError(f"{name} source-held-out split lacks VF and VT: {classes}")

    train_groups = set(table.iloc[train_idx]["record_id"])
    test_groups = set(table.iloc[test_idx]["record_id"])
    overlap = sorted(train_groups & test_groups)
    assert not overlap, f"Record leakage detected: {overlap}"
    return train_idx, test_idx, {
        "split": f"train={sorted(train_sources)}, test={test_source}",
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "train_records": int(len(train_groups)),
        "test_records": int(len(test_groups)),
        "overlap": overlap,
        "train_record_ids": sorted(train_groups),
        "test_record_ids": sorted(test_groups),
    }


def oversample_training(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically oversample only the minority training class."""
    rng = np.random.default_rng(SEED)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Expected both VT and VF in the training data")
    target_count = int(counts.max())
    pieces_x = [X]
    pieces_y = [y]
    for class_value, count in zip(classes, counts):
        if count < target_count:
            indices = np.flatnonzero(y == class_value)
            sampled = rng.choice(indices, size=target_count - count, replace=True)
            pieces_x.append(X[sampled])
            pieces_y.append(y[sampled])
    order = rng.permutation(sum(len(piece) for piece in pieces_y))
    return np.concatenate(pieces_x)[order], np.concatenate(pieces_y)[order]


def make_xgb(strategy: str, y_train: np.ndarray):
    """Build the notebook's XGBoost family without exporting a model."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("Experiment A-C/D require xgboost, as used by the notebook") from exc

    negative = max(int((y_train == 0).sum()), 1)
    positive = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = negative / positive if strategy == "baseline" else 1.0
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
    )


def metrics_for(model, X_test: np.ndarray, y_test: np.ndarray, experiment: str, features: list[str], split_info: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test, predictions, labels=[0, 1], target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    row = {
        "experiment": experiment,
        "features": ",".join(features),
        "test_strategy": split_info["split"],
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "vf_precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
        "vf_recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
        "vf_f1": f1_score(y_test, predictions, pos_label=1, zero_division=0),
        "vt_precision": precision_score(y_test, predictions, pos_label=0, zero_division=0),
        "vt_recall": recall_score(y_test, predictions, pos_label=0, zero_division=0),
        "vt_f1": f1_score(y_test, predictions, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "train_windows": split_info["train_windows"],
        "test_windows": split_info["test_windows"],
        "train_records": split_info["train_records"],
        "test_records": split_info["test_records"],
        "record_overlap": len(split_info["overlap"]),
    }
    row["classification_report"] = json.dumps(report)
    return row, matrix


def fit_xgb_experiment(table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, features: list[str], strategy: str, name: str, split_info: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    X_train = table.iloc[train_idx][features].to_numpy(dtype=np.float32)
    y_train = table.iloc[train_idx]["target"].to_numpy()
    X_test = table.iloc[test_idx][features].to_numpy(dtype=np.float32)
    y_test = table.iloc[test_idx]["target"].to_numpy()
    model = make_xgb(strategy, y_train)
    if strategy == "oversample":
        X_train, y_train = oversample_training(X_train, y_train)
        model.set_params(scale_pos_weight=1.0)
    if strategy == "class_weight":
        negative = max(int((y_train == 0).sum()), 1)
        positive = max(int((y_train == 1).sum()), 1)
        sample_weights = np.where(y_train == 1, negative / positive, 1.0)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    return metrics_for(model, X_test, y_test, name, features, split_info)


def lightweight_models() -> dict[str, Any]:
    """Return edge-suitable classifier candidates for the frozen split."""
    return {
        "E_logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
        "E_random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "E_extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "E_hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            random_state=SEED,
        ),
    }


def run_lightweight_comparison(table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, split_info: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[list[int]]], pd.DataFrame]:
    features = FEATURE_NAMES
    X_train = table.iloc[train_idx][features].to_numpy(dtype=np.float32)
    y_train = table.iloc[train_idx]["target"].to_numpy()
    X_test = table.iloc[test_idx][features].to_numpy(dtype=np.float32)
    y_test = table.iloc[test_idx]["target"].to_numpy()
    rows = []
    matrices = {}
    importances = []
    for name, model in lightweight_models().items():
        model.fit(X_train, y_train)
        row, matrix = metrics_for(model, X_test, y_test, name, features, split_info)
        rows.append(row)
        matrices[name] = matrix.tolist()
        estimator = model[-1] if hasattr(model, "steps") else model
        if hasattr(estimator, "feature_importances_"):
            importances.extend(
                {"experiment": name, "feature": feature, "importance": float(value)}
                for feature, value in zip(features, estimator.feature_importances_)
            )
    return rows, matrices, pd.DataFrame(importances)


def source_diagnostic(table: pd.DataFrame, features: list[str], train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )
    model.fit(table.iloc[train_idx][features], table.iloc[train_idx]["dataset_source"])
    predictions = model.predict(table.iloc[test_idx][features])
    truth = table.iloc[test_idx]["dataset_source"]
    return {
        "features": ",".join(features),
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=model.classes_).tolist(),
        "classes": model.classes_.tolist(),
    }


def source_label_table(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby(["dataset_source", "label"], observed=True)
        .agg(windows=("label", "size"), records=("record_id", "nunique"))
        .reset_index()
        .assign(percent_of_total=lambda frame: 100 * frame["windows"] / len(table))
    )


def split_source_label_table(table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, split_name: str) -> pd.DataFrame:
    rows = []
    for partition, indices in (("train", train_idx), ("test", test_idx)):
        summary = source_label_table(table.iloc[indices]).assign(
            split=split_name,
            partition=partition,
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run_experiments(table: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    table = prepare_table(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_idx, test_idx, split_info = exact_random_split(table)

    results: list[dict[str, Any]] = []
    matrices: dict[str, list[list[int]]] = {}
    split_distributions = [
        split_source_label_table(table, train_idx, test_idx, "A_B_C_E_random_group_split")
    ]

    for name, strategy in (("A_frozen_baseline", "baseline"), ("B_class_weight", "class_weight"), ("C_train_oversample", "oversample")):
        row, matrix = fit_xgb_experiment(table, train_idx, test_idx, FEATURE_NAMES, strategy, name, split_info)
        results.append(row)
        matrices[name] = matrix.tolist()

    lightweight_rows, lightweight_matrices, importances = run_lightweight_comparison(
        table, train_idx, test_idx, split_info
    )
    results.extend(lightweight_rows)
    matrices.update(lightweight_matrices)
    if not importances.empty:
        importances.to_csv(output_dir / "model3_feature_importance.csv", index=False)

    for train_sources, test_source, name in [
        ({"vfdb", "cudb"}, "c2015", "D1_vfdb_cudb_to_c2015"),
        ({"c2015", "cudb"}, "vfdb", "D2_c2015_cudb_to_vfdb"),
    ]:
        d_train, d_test, d_info = source_split(table, train_sources, test_source)
        split_distributions.append(
            split_source_label_table(table, d_train, d_test, name)
        )
        for feature_name, features in (("full_10", FEATURE_NAMES), ("reduced_no_amplitude", REDUCED_FEATURE_NAMES)):
            name = f"{name}_{feature_name}"
            row, matrix = fit_xgb_experiment(table, d_train, d_test, features, "baseline", name, d_info)
            results.append(row)
            matrices[name] = matrix.tolist()

    comparison = pd.DataFrame(results)
    comparison.to_csv(output_dir / "model3_comparison.csv", index=False)
    source_label_table(table).to_csv(output_dir / "model3_source_label_counts.csv", index=False)
    pd.concat(split_distributions, ignore_index=True).to_csv(
        output_dir / "model3_split_source_label_counts.csv", index=False
    )
    with (output_dir / "model3_confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump(matrices, handle, indent=2)
    with (output_dir / "model3_split_definition.json").open("w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2)

    source_audit = source_diagnostic(table, FEATURE_NAMES, train_idx, test_idx)
    reduced_audit = source_diagnostic(table, REDUCED_FEATURE_NAMES, train_idx, test_idx)
    with (output_dir / "model3_source_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump({"full_10": source_audit, "reduced_no_amplitude": reduced_audit}, handle, indent=2)
    with (output_dir / "model3_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"seed": SEED, "features": FEATURE_NAMES, "reduced_features": REDUCED_FEATURE_NAMES, "labels": LABELS}, handle, indent=2)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run separate Model 3 VF/VT Phase 2B experiments.")
    parser.add_argument("feature_table", type=Path, help="CSV exported from vfvt_df")
    parser.add_argument("--output-dir", type=Path, default=Path("training/results/model3_phase2b"))
    args = parser.parse_args()
    comparison = run_experiments(pd.read_csv(args.feature_table), args.output_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()