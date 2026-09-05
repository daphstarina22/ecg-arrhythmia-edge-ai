"""Phase 2C source-robustness experiments for Model 3 VF versus VT.

This module accepts the notebook's persisted feature table. It does not load
raw ECG windows, modify production feature extraction, export ONNX, or write
under ``models/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import RobustScaler, StandardScaler

FEATURE_NAMES = [
    "mean", "std", "ptp", "zero_crossings", "peak_count", "mean_rr",
    "rr_cv", "qrs_width", "dominant_freq", "vf_band_power_ratio",
]
REDUCED_FEATURE_NAMES = [
    name for name in FEATURE_NAMES if name not in {"mean", "std", "ptp"}
]
LABELS = ["VT", "VF"]
SEED = 42
TEST_SIZE = 0.25


def prepare_table(table: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURE_NAMES + ["label", "record_id"])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared = table.copy()
    labels = prepared["label"].astype(str)
    unexpected = sorted(set(labels) - set(LABELS))
    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")
    if labels.str.upper().eq("AFIB").any():
        raise ValueError("AFIB is present in the feature table")
    if prepared[FEATURE_NAMES].isna().any().any():
        raise ValueError("NaN feature values are present")
    if not np.isfinite(prepared[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature values are present")

    prepared["record_id"] = prepared["record_id"].astype(str)
    if "dataset_source" not in prepared.columns:
        prepared["dataset_source"] = prepared["record_id"].str.split("_", n=1).str[0]
    prepared["target"] = prepared["label"].map({"VT": 0, "VF": 1}).astype(int)
    return prepared


def split_info(table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, name: str) -> dict[str, Any]:
    train_groups = set(table.iloc[train_idx]["record_id"])
    test_groups = set(table.iloc[test_idx]["record_id"])
    overlap = sorted(train_groups & test_groups)
    assert not overlap, f"Record overlap in {name}: {overlap}"
    return {
        "name": name,
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "train_records": int(len(train_groups)),
        "test_records": int(len(test_groups)),
        "record_overlap": overlap,
        "train_record_ids": sorted(train_groups),
        "test_record_ids": sorted(test_groups),
    }


def frozen_random_split(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    groups = table["record_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(table[FEATURE_NAMES], table["target"], groups))
    info = split_info(table, train_idx, test_idx, "random_grouped")
    info["definition"] = "GroupShuffleSplit(record_id, test_size=0.25, random_state=42)"
    return train_idx, test_idx, info


def source_split(table: pd.DataFrame, train_sources: set[str], test_source: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_idx = np.flatnonzero(table["dataset_source"].isin(train_sources).to_numpy())
    test_idx = np.flatnonzero(table["dataset_source"].eq(test_source).to_numpy())
    for partition, indices in (("train", train_idx), ("test", test_idx)):
        labels = set(table.iloc[indices]["label"])
        if labels != set(LABELS):
            raise ValueError(f"{partition} lacks both VF and VT: {labels}")
    info = split_info(table, train_idx, test_idx, f"{sorted(train_sources)}_to_{test_source}")
    info["definition"] = f"train={sorted(train_sources)}, test={test_source}"
    return train_idx, test_idx, info


def make_xgb():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("Phase 2C requires xgboost, matching the notebook classifier") from exc
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
    )


def class_weights(y: np.ndarray) -> np.ndarray:
    negative = max(int((y == 0).sum()), 1)
    positive = max(int((y == 1).sum()), 1)
    return np.where(y == 1, negative / positive, 1.0)


def record_weights(table: pd.DataFrame, train_idx: np.ndarray) -> np.ndarray:
    records = table.iloc[train_idx]["record_id"]
    counts = records.value_counts()
    weights = records.map(lambda record: 1.0 / counts[record]).to_numpy(dtype=float)
    return weights / weights.mean()


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str, features: list[str], strategy: str, test_info: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    predictions = model.predict(X_test)
    classes = list(model.classes_)
    probabilities = model.predict_proba(X_test)[:, classes.index(1)]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test, predictions, labels=[0, 1], target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    return {
        "experiment": name,
        "classifier": type(model[-1] if hasattr(model, "steps") else model).__name__,
        "features": ",".join(features),
        "train_strategy": strategy,
        "test_strategy": test_info["definition"],
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "vt_precision": precision_score(y_test, predictions, pos_label=0, zero_division=0),
        "vt_recall": recall_score(y_test, predictions, pos_label=0, zero_division=0),
        "vt_f1": f1_score(y_test, predictions, pos_label=0, zero_division=0),
        "vf_precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
        "vf_recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
        "vf_f1": f1_score(y_test, predictions, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "train_windows": test_info["train_windows"],
        "test_windows": test_info["test_windows"],
        "train_records": test_info["train_records"],
        "test_records": test_info["test_records"],
        "record_overlap": len(test_info["record_overlap"]),
        "classification_report": json.dumps(report),
    }, matrix


def fit_xgb(table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, features: list[str], name: str, strategy: str, info: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, Any]:
    model = make_xgb()
    X_train = table.iloc[train_idx][features].to_numpy(dtype=np.float32)
    y_train = table.iloc[train_idx]["target"].to_numpy()
    X_test = table.iloc[test_idx][features].to_numpy(dtype=np.float32)
    y_test = table.iloc[test_idx]["target"].to_numpy()
    if strategy == "baseline":
        model.set_params(scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
        model.fit(X_train, y_train)
    elif strategy == "class_weight":
        model.fit(X_train, y_train, sample_weight=class_weights(y_train))
    elif strategy == "record_balanced":
        model.fit(X_train, y_train, sample_weight=record_weights(table, train_idx))
    else:
        raise ValueError(f"Unknown XGBoost strategy: {strategy}")
    row, matrix = evaluate(model, X_test, y_test, name, features, strategy, info)
    return row, matrix, model


def fit_scaled_logistic(table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, features: list[str], scaler: Any, name: str, info: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, Any]:
    model = make_pipeline(
        scaler,
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )
    X_train = table.iloc[train_idx][features]
    y_train = table.iloc[train_idx]["target"].to_numpy()
    X_test = table.iloc[test_idx][features]
    y_test = table.iloc[test_idx]["target"].to_numpy()
    model.fit(X_train, y_train)
    row, matrix = evaluate(model, X_test, y_test, name, features, "feature_space_normalization", info)
    return row, matrix, model


def source_label_distribution(table: pd.DataFrame) -> pd.DataFrame:
    result = (
        table.groupby(["dataset_source", "label"], observed=True)
        .agg(windows=("label", "size"), records=("record_id", "nunique"))
        .reset_index()
    )
    result["percent_of_total"] = 100 * result["windows"] / len(table)
    return result


def split_source_label_distribution(table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, split_name: str) -> pd.DataFrame:
    rows = []
    for partition, indices in (("train", train_idx), ("test", test_idx)):
        summary = source_label_distribution(table.iloc[indices]).assign(
            split=split_name, partition=partition
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


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
        "classes": model[-1].classes_.tolist(),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=model[-1].classes_).tolist(),
    }


def plot_confusion(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(4, 3.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], LABELS)
    axis.set_yticks([0, 1], LABELS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_plots(table: pd.DataFrame, results: list[dict[str, Any]], matrices: dict[str, list[list[int]]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        plot_confusion(np.asarray(matrix), name, plots_dir / f"{name}_confusion.png")

    for feature in FEATURE_NAMES:
        fig, axis = plt.subplots(figsize=(8, 4))
        table.boxplot(column=feature, by="label", ax=axis)
        axis.set_title(f"{feature}: VF versus VT")
        axis.set_xlabel("Label")
        axis.set_ylabel(feature)
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(plots_dir / f"class_{feature}.png", dpi=120)
        plt.close(fig)

    for feature in FEATURE_NAMES:
        fig, axis = plt.subplots(figsize=(8, 4))
        table.boxplot(column=feature, by="dataset_source", ax=axis)
        axis.set_title(f"{feature}: source distribution")
        axis.set_xlabel("Dataset source")
        axis.set_ylabel(feature)
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(plots_dir / f"source_{feature}.png", dpi=120)
        plt.close(fig)

    record_counts = table.groupby(["dataset_source", "record_id"]).size().reset_index(name="windows")
    fig, axis = plt.subplots(figsize=(8, 4))
    record_counts.boxplot(column="windows", by="dataset_source", ax=axis)
    axis.set_title("Windows per record")
    axis.set_xlabel("Dataset source")
    axis.set_ylabel("Windows")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(plots_dir / "record_contribution_distribution.png", dpi=120)
    plt.close(fig)


def run(table: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    table = prepare_table(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    train_idx, test_idx, random_info = frozen_random_split(table)
    results: list[dict[str, Any]] = []
    matrices: dict[str, list[list[int]]] = {}
    split_reports: dict[str, Any] = {"random_grouped": random_info}
    split_distributions = [split_source_label_distribution(table, train_idx, test_idx, "random_grouped")]

    for name, features, strategy in [
        ("A_raw_baseline", FEATURE_NAMES, "baseline"),
        ("B_reduced_baseline", REDUCED_FEATURE_NAMES, "baseline"),
        ("B_reduced_class_weight", REDUCED_FEATURE_NAMES, "class_weight"),
        ("D_raw_record_balanced", FEATURE_NAMES, "record_balanced"),
        ("D_reduced_record_balanced", REDUCED_FEATURE_NAMES, "record_balanced"),
    ]:
        row, matrix, _ = fit_xgb(table, train_idx, test_idx, features, name, strategy, random_info)
        results.append(row)
        matrices[name] = matrix.tolist()

    for name, scaler in (("C_standardized_logistic", StandardScaler()), ("C_robust_scaled_logistic", RobustScaler())):
        row, matrix, _ = fit_scaled_logistic(table, train_idx, test_idx, FEATURE_NAMES, scaler, name, random_info)
        results.append(row)
        matrices[name] = matrix.tolist()

    for train_sources, test_source, prefix in [
        ({"vfdb", "cudb"}, "c2015", "E_vfdb_cudb_to_c2015"),
        ({"c2015", "cudb"}, "vfdb", "E_c2015_cudb_to_vfdb"),
    ]:
        d_train, d_test, d_info = source_split(table, train_sources, test_source)
        split_reports[prefix] = d_info
        split_distributions.append(split_source_label_distribution(table, d_train, d_test, prefix))
        for name, features, strategy in [
            (f"{prefix}_raw", FEATURE_NAMES, "baseline"),
            (f"{prefix}_reduced", REDUCED_FEATURE_NAMES, "baseline"),
            (f"{prefix}_reduced_record_balanced", REDUCED_FEATURE_NAMES, "record_balanced"),
        ]:
            row, matrix, _ = fit_xgb(table, d_train, d_test, features, name, strategy, d_info)
            results.append(row)
            matrices[name] = matrix.tolist()
        for scaler_name, scaler in (("standardized", StandardScaler()), ("robust_scaled", RobustScaler())):
            name = f"{prefix}_{scaler_name}_logistic"
            row, matrix, _ = fit_scaled_logistic(
                table, d_train, d_test, FEATURE_NAMES, scaler, name, d_info
            )
            results.append(row)
            matrices[name] = matrix.tolist()

    comparison = pd.DataFrame(results)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    comparison.to_json(output_dir / "comparison.json", orient="records", indent=2)
    source_label_distribution(table).to_csv(output_dir / "source_label_distributions.csv", index=False)
    pd.concat(split_distributions, ignore_index=True).to_csv(output_dir / "split_source_label_distributions.csv", index=False)
    with (output_dir / "confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump(matrices, handle, indent=2)
    with (output_dir / "split_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(split_reports, handle, indent=2)

    source_diagnostics = {
        "phase2a_reference": 0.955,
        "full_10": source_diagnostic(table, FEATURE_NAMES, train_idx, test_idx),
        "reduced_7": source_diagnostic(table, REDUCED_FEATURE_NAMES, train_idx, test_idx),
    }
    with (output_dir / "source_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(source_diagnostics, handle, indent=2)

    source_features = pd.DataFrame(
        table.groupby("dataset_source", observed=True)[FEATURE_NAMES].agg(["mean", "median", "std", "min", "max"])
    )
    source_features.to_csv(output_dir / "source_feature_distributions.csv")
    class_features = pd.DataFrame(
        table.groupby("label", observed=True)[FEATURE_NAMES].agg(["mean", "median", "std", "min", "max"])
    )
    class_features.to_csv(output_dir / "class_feature_distributions.csv")

    record_summary = table.groupby(["dataset_source", "record_id"]).size().reset_index(name="windows")
    record_summary.to_csv(output_dir / "record_contributions.csv", index=False)

    source_rows = []
    for features_name, features in (("full_10", FEATURE_NAMES), ("reduced_7", REDUCED_FEATURE_NAMES)):
        source_result = source_diagnostics[features_name]
        source_rows.append({"feature_set": features_name, **source_result})
    pd.DataFrame(source_rows).drop(columns=["confusion_matrix", "classes"]).to_csv(output_dir / "source_diagnostics.csv", index=False)

    held_out = comparison[comparison["experiment"].str.startswith("E_")].copy()
    ranking = (
        held_out.groupby(["features", "train_strategy"], as_index=False)
        .agg(
            average_source_heldout_macro_f1=("macro_f1", "mean"),
            average_source_heldout_vf_f1=("vf_f1", "mean"),
            worst_source_heldout_macro_f1=("macro_f1", "min"),
            worst_source_heldout_vf_f1=("vf_f1", "min"),
        )
    )
    ranking["robustness_score"] = (
        0.4 * ranking["average_source_heldout_macro_f1"]
        + 0.4 * ranking["average_source_heldout_vf_f1"]
        + 0.2 * ranking["worst_source_heldout_macro_f1"]
    )
    ranking = ranking.sort_values("robustness_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(output_dir / "robustness_ranking.csv", index=False)

    config = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "labels": {"VT": 0, "VF": 1},
        "features": FEATURE_NAMES,
        "reduced_features": REDUCED_FEATURE_NAMES,
        "record_balancing": "training-only inverse record-frequency sample weights, normalized to mean 1",
        "normalization": "feature-space only; raw ECG windows are unavailable",
        "robustness_formula": "0.4*mean(source-held-out macro F1) + 0.4*mean(source-held-out VF F1) + 0.2*min(source-held-out macro F1)",
        "production_changes": False,
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    write_plots(table, results, matrices, plots_dir)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated Model 3 Phase 2C experiments.")
    parser.add_argument("feature_table", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/model3_phase2c/results"))
    args = parser.parse_args()
    result = run(pd.read_csv(args.feature_table), args.output_dir)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()