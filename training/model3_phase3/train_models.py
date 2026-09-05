"""Train and evaluate experimental calibrated Model 3 candidates only."""

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "mean", "std", "ptp", "zero_crossings", "peak_count", "mean_rr",
    "rr_cv", "qrs_width", "dominant_freq", "vf_band_power_ratio",
]
REDUCED_FEATURES = [name for name in FEATURE_NAMES if name not in {"mean", "std", "ptp"}]
LABELS = ["VT", "VF"]
SEEDS = [42, 123, 456, 789, 2025]
TEST_SIZE = 0.25


def validate(table: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURE_NAMES + ["label", "record_id"])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if set(table["label"].astype(str)) != set(LABELS):
        raise ValueError("Labels must be exactly VT and VF")
    if table["label"].astype(str).str.upper().eq("AFIB").any():
        raise ValueError("AFIB is present")
    if not np.isfinite(table[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature values are present")
    result = table.copy()
    if "dataset_source" not in result.columns:
        result["dataset_source"] = result["record_id"].astype(str).str.split("_", n=1).str[0]
    result["target"] = result["label"].map({"VT": 0, "VF": 1}).astype(int)
    return result


def split(table: pd.DataFrame, seed: int, train_sources: set[str] | None = None, test_source: str | None = None):
    if train_sources is not None and test_source is not None:
        train = np.flatnonzero(table["dataset_source"].isin(train_sources).to_numpy())
        test = np.flatnonzero(table["dataset_source"].eq(test_source).to_numpy())
        definition = f"train={sorted(train_sources)}, test={test_source}"
    else:
        groups = table["record_id"].to_numpy()
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed).split(table, table["target"], groups))
        definition = f"GroupShuffleSplit(record_id, test_size=0.25, random_state={seed})"
    train_records = set(table.iloc[train]["record_id"])
    test_records = set(table.iloc[test]["record_id"])
    overlap = sorted(train_records & test_records)
    assert not overlap, f"Record leakage: {overlap}"
    for name, indexes in (("train", train), ("test", test)):
        if set(table.iloc[indexes]["label"]) != set(LABELS):
            raise ValueError(f"{name} lacks both labels in split {definition}")
    return train, test, {
        "definition": definition,
        "train_windows": int(len(train)),
        "test_windows": int(len(test)),
        "train_records": int(len(train_records)),
        "test_records": int(len(test_records)),
        "overlap": overlap,
    }


def make_model(name: str, y_train: np.ndarray):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is required for XGBoost experiments") from exc
    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    if name == "A_xgb_full":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=ratio, eval_metric="logloss", tree_method="hist", random_state=42)
    if name == "B_xgb_reduced":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=ratio, eval_metric="logloss", tree_method="hist", random_state=42)
    if name == "C_xgb_reduced_class_weight":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=1.0, eval_metric="logloss", tree_method="hist", random_state=42)
    if name == "D_logistic_reduced":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    if name == "E_random_forest_reduced":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    if name == "F_extra_trees_reduced":
        return ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    raise ValueError(name)


def experiment_spec(name: str):
    features = FEATURE_NAMES if name == "A_xgb_full" else REDUCED_FEATURES
    return features


def evaluate_model(model, X_test, y_test, name: str, features: list[str], strategy: str, split_info: dict[str, Any], seed: int) -> tuple[dict[str, Any], np.ndarray]:
    prediction = model.predict(X_test)
    classes = list(model.classes_)
    probability = model.predict_proba(X_test)[:, classes.index(1)]
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    row = {
        "experiment": name,
        "features": ",".join(features),
        "train_strategy": strategy,
        "test_strategy": split_info["definition"],
        "seed": seed,
        "accuracy": accuracy_score(y_test, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
        "vt_precision": precision_score(y_test, prediction, pos_label=0, zero_division=0),
        "vt_recall": recall_score(y_test, prediction, pos_label=0, zero_division=0),
        "vt_f1": f1_score(y_test, prediction, pos_label=0, zero_division=0),
        "vf_precision": precision_score(y_test, prediction, pos_label=1, zero_division=0),
        "vf_recall": recall_score(y_test, prediction, pos_label=1, zero_division=0),
        "vf_f1": f1_score(y_test, prediction, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        "train_windows": split_info["train_windows"],
        "test_windows": split_info["test_windows"],
        "train_records": split_info["train_records"],
        "test_records": split_info["test_records"],
        "record_overlap": len(split_info["overlap"]),
    }
    return row, matrix


def fit_one(table, train_idx, test_idx, name: str, seed: int, source_strategy: str | None = None):
    features = experiment_spec(name)
    model = make_model(name, table.iloc[train_idx]["target"].to_numpy())
    X_train = table.iloc[train_idx][features].to_numpy(dtype=np.float32)
    y_train = table.iloc[train_idx]["target"].to_numpy()
    X_test = table.iloc[test_idx][features].to_numpy(dtype=np.float32)
    y_test = table.iloc[test_idx]["target"].to_numpy()
    strategy = "class_weight" if name == "C_xgb_reduced_class_weight" else "baseline"
    if name == "C_xgb_reduced_class_weight":
        weights = np.where(y_train == 1, (y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, name, features, strategy, source_strategy, seed), model


def source_diagnostic(table: pd.DataFrame, train_records: set[str], test_records: set[str]):
    assert not train_records & test_records
    train = table["record_id"].isin(train_records)
    test = table["record_id"].isin(test_records)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    model.fit(table.loc[train, FEATURE_NAMES], table.loc[train, "dataset_source"])
    prediction = model.predict(table.loc[test, FEATURE_NAMES])
    truth = table.loc[test, "dataset_source"]
    classes = model[-1].classes_
    importance = pd.DataFrame({"feature": FEATURE_NAMES, "importance": np.abs(model[-1].coef_).mean(axis=0)}).sort_values("importance", ascending=False)
    return {
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
        "classes": classes.tolist(),
    }, confusion_matrix(truth, prediction, labels=classes), importance


def source_label_distribution(table: pd.DataFrame) -> pd.DataFrame:
    result = table.groupby(["dataset_source", "label"], observed=True).agg(windows=("label", "size"), records=("record_id", "nunique")).reset_index()
    result["percent_of_total"] = 100 * result["windows"] / len(table)
    return result


def run(table: pd.DataFrame, output_dir: Path) -> None:
    table = validate(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = output_dir.parent / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    results = []
    matrices = {}
    split_reports = {}
    experiment_names = ["A_xgb_full", "B_xgb_reduced", "C_xgb_reduced_class_weight", "D_logistic_reduced", "E_random_forest_reduced", "F_extra_trees_reduced"]

    train, test, info = split(table, 42)
    split_reports["random_seed_42"] = info
    for name in experiment_names:
        (row, matrix), _ = fit_one(table, train, test, name, 42, info)
        results.append(row); matrices[name] = matrix.tolist()

    held_out_specs = [("D1_vfdb_cudb_to_c2015", {"vfdb", "cudb"}, "c2015"), ("D2_c2015_cudb_to_vfdb", {"c2015", "cudb"}, "vfdb")]
    for split_name, train_sources, test_source in held_out_specs:
        train_idx, test_idx, hold_info = split(table, 42, train_sources, test_source)
        split_reports[split_name] = hold_info
        for name in experiment_names:
            (row, matrix), _ = fit_one(table, train_idx, test_idx, name, 42, hold_info)
            row["experiment"] = f"{split_name}_{name}"
            results.append(row); matrices[row["experiment"]] = matrix.tolist()

    repeated = []
    for seed in SEEDS:
        train_idx, test_idx, repeat_info = split(table, seed)
        for name in experiment_names:
            (row, _), _ = fit_one(table, train_idx, test_idx, name, seed, repeat_info)
            repeated.append(row)
    repeated_df = pd.DataFrame(repeated)
    repeated_summary = repeated_df.groupby("experiment").agg({metric: ["mean", "std"] for metric in ["macro_f1", "vf_f1", "balanced_accuracy", "roc_auc"]})
    repeated_summary.to_csv(output_dir / "repeatability_summary.csv")
    repeated_df.to_csv(output_dir / "repeatability_runs.csv", index=False)

    comparison = pd.DataFrame(results)
    comparison.to_csv(output_dir / "final_comparison.csv", index=False)
    comparison.to_json(output_dir / "metrics.json", orient="records", indent=2)
    source_label_distribution(table).to_csv(output_dir / "source_label_distribution.csv", index=False)
    table.groupby(["dataset_source", "label", "record_id"], observed=True).size().reset_index(name="windows").to_csv(output_dir / "record_distribution.csv", index=False)
    with (output_dir / "confusion_matrices.json").open("w", encoding="utf-8") as handle: json.dump(matrices, handle, indent=2)
    with (output_dir / "split_reports.json").open("w", encoding="utf-8") as handle: json.dump(split_reports, handle, indent=2)

    train_records = set(table.iloc[train]["record_id"]); test_records = set(table.iloc[test]["record_id"])
    original_table_path = Path("vfvt_feature_table.csv")
    original = validate(pd.read_csv(original_table_path)) if original_table_path.exists() else None
    diagnostics = {}
    for name, diagnostic_table in [("calibrated", table), ("original", original)]:
        if diagnostic_table is None: continue
        diag, matrix, importance = source_diagnostic(diagnostic_table, train_records, test_records)
        diagnostics[name] = diag
        pd.DataFrame(matrix, index=diag["classes"], columns=diag["classes"]).to_csv(output_dir / f"source_confusion_matrix_{name}.csv")
        importance.to_csv(output_dir / f"source_feature_importance_{name}.csv", index=False)
    with (output_dir / "source_diagnostics.json").open("w", encoding="utf-8") as handle: json.dump(diagnostics, handle, indent=2)

    held = comparison[comparison["experiment"].str.startswith(("D1_", "D2_"))].copy()
    held["candidate"] = held["experiment"].str.replace(
        r"^D1_vfdb_cudb_to_c2015_|^D2_c2015_cudb_to_vfdb_", "", regex=True
    )
    ranking = held.groupby("candidate", as_index=False).agg(
        average_source_heldout_macro_f1=("macro_f1", "mean"),
        average_source_heldout_vf_f1=("vf_f1", "mean"),
        worst_source_heldout_macro_f1=("macro_f1", "min"),
        worst_source_heldout_vf_f1=("vf_f1", "min"),
    )
    ranking["robustness_score"] = 0.4 * ranking["average_source_heldout_macro_f1"] + 0.4 * ranking["average_source_heldout_vf_f1"] + 0.2 * ranking["worst_source_heldout_macro_f1"]
    ranking = ranking.sort_values("robustness_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(output_dir / "robustness_ranking.csv", index=False)
    table.to_csv(artifacts / "calibrated_vfvt_feature_table.csv", index=False)
    config = {"seed": 42, "repeatability_seeds": SEEDS, "test_size": TEST_SIZE, "features": FEATURE_NAMES, "reduced_features": REDUCED_FEATURES, "labels": {"VT": 0, "VF": 1}, "robustness_formula": "0.4*mean(heldout macro F1)+0.4*mean(heldout VF F1)+0.2*worst heldout macro F1", "onnx_exported": False}
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle: json.dump(config, handle, indent=2)
    best = ranking.iloc[0].to_dict()
    report = f"""# Phase 3 Model 3 Report

## Scope

This was an experimental retraining/evaluation run on the calibrated Phase 2E.2
raw-reconstructed table. No production model or ONNX file was changed.

Dataset: {len(table)} windows, {table['record_id'].nunique()} records, labels {table['label'].value_counts().to_dict()}.
AFIB rows: 0. Group overlap: 0.

## Source Bias

Original source-classifier accuracy: {diagnostics.get('original', {}).get('accuracy')}.
Calibrated source-classifier accuracy: {diagnostics.get('calibrated', {}).get('accuracy')}.

## Robustness Selection

Formula: `0.4 * mean held-out macro F1 + 0.4 * mean held-out VF F1 + 0.2 * worst held-out macro F1`.

Best ranked candidate: `{best.get('candidate')}` with robustness score `{best.get('robustness_score')}`.

This ranking is based on source-held-out evaluation, not random-split accuracy.

## Production Decision

**NO MODEL APPROVED FOR PRODUCTION.** Source-held-out performance remains
asymmetric and no ONNX artifact was exported.
"""
    (output_dir / "final_report.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False))
    print("\nSOURCE_DIAGNOSTICS", json.dumps(diagnostics, indent=2))
    print("\nROBUSTNESS", ranking.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experimental calibrated Model 3 Phase 3 training and evaluation.")
    parser.add_argument("feature_table", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/model3_phase3/results"))
    args = parser.parse_args()
    run(pd.read_csv(args.feature_table), args.output_dir)


if __name__ == "__main__":
    main()