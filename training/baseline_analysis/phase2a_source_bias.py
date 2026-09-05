"""Phase 2A source-bias diagnostics for notebook feature tables.

This module is intended to be imported at the end of the Kaggle notebook,
after the existing feature tables have been built. It trains only diagnostic
logistic-regression models; it never writes ONNX files or production models.

Required DataFrame columns:
    mean, std, ptp, zero_crossings, peak_count, mean_rr, rr_cv, qrs_width,
    dominant_freq, vf_band_power_ratio, label, record_id, dataset_source
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_NAMES = [
    "mean", "std", "ptp", "zero_crossings", "peak_count", "mean_rr",
    "rr_cv", "qrs_width", "dominant_freq", "vf_band_power_ratio",
]


def _validate(table: pd.DataFrame) -> None:
    required = set(FEATURE_NAMES + ["label", "record_id", "dataset_source"])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")


def verify_group_split(
    table: pd.DataFrame,
    model_name: str,
    test_size: float,
    random_state: int = 42,
) -> dict[str, Any]:
    """Reproduce the notebook split and assert zero record overlap."""
    _validate(table)
    groups = table["record_id"].astype(str).to_numpy()
    labels = table["label"].to_numpy()
    train_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        ).split(table, labels, groups)
    )
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = sorted(train_groups & test_groups)
    assert not overlap, f"{model_name} record overlap: {overlap}"
    result = {
        "model": model_name,
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "overlap": overlap,
        "train_indices": train_idx,
        "test_indices": test_idx,
    }
    print(
        f"{model_name}: train_groups={len(train_groups)}, "
        f"test_groups={len(test_groups)}, intersection={overlap}"
    )
    return result


def source_label_distribution(table: pd.DataFrame) -> pd.DataFrame:
    """Return window, record, source-percent, and class-percent counts."""
    _validate(table)
    counts = (
        table.groupby(["dataset_source", "label"], observed=True)
        .agg(windows=("label", "size"), unique_records=("record_id", "nunique"))
        .reset_index()
    )
    source_totals = counts.groupby("dataset_source")["windows"].transform("sum")
    class_totals = counts.groupby("label")["windows"].transform("sum")
    counts["percent_of_source"] = 100 * counts["windows"] / source_totals
    counts["percent_of_class"] = 100 * counts["windows"] / class_totals
    return counts.sort_values(["dataset_source", "label"]).reset_index(drop=True)


def _classification_metrics(y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, average="binary", zero_division=0),
        "recall": recall_score(y_true, predictions, average="binary", zero_division=0),
        "f1": f1_score(y_true, predictions, average="binary", zero_division=0),
    }
    if probabilities is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, probabilities)
    return result


def model1_source_only(table: pd.DataFrame, split: dict[str, Any]) -> dict[str, Any]:
    """Measure how well source identity alone predicts shockability."""
    labels = table["label"].to_numpy()
    source = table[["dataset_source"]]
    transformer = ColumnTransformer(
        [("source", OneHotEncoder(handle_unknown="ignore"), ["dataset_source"])],
        remainder="drop",
    )
    model = make_pipeline(
        transformer,
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    train_idx, test_idx = split["train_indices"], split["test_indices"]
    model.fit(source.iloc[train_idx], labels[train_idx])
    predictions = model.predict(source.iloc[test_idx])
    probabilities = model.predict_proba(source.iloc[test_idx])[:, 1]
    result = _classification_metrics(labels[test_idx], predictions, probabilities)
    print("Model 1 source-only metrics:", result)
    return result


def feature_source_classifier(table: pd.DataFrame, split: dict[str, Any]) -> dict[str, Any]:
    """Predict dataset source from the ten baseline ECG features."""
    _validate(table)
    labels = table["dataset_source"].to_numpy()
    train_idx, test_idx = split["train_indices"], split["test_indices"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto"),
    )
    model.fit(table.iloc[train_idx][FEATURE_NAMES], labels[train_idx])
    predictions = model.predict(table.iloc[test_idx][FEATURE_NAMES])
    coefficients = np.abs(model[-1].coef_).mean(axis=0)
    importance = pd.Series(coefficients, index=FEATURE_NAMES).sort_values(ascending=False)
    result = {
        "accuracy": accuracy_score(labels[test_idx], predictions),
        "macro_f1": f1_score(labels[test_idx], predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels[test_idx], predictions, labels=model[-1].classes_),
        "classes": model[-1].classes_,
        "feature_importance": importance,
    }
    print("Feature-to-source accuracy:", result["accuracy"])
    print("Feature-to-source macro F1:", result["macro_f1"])
    print("Feature-to-source confusion matrix:\n", result["confusion_matrix"])
    print("Feature-to-source importance:\n", importance)
    return result


def feature_distributions(table: pd.DataFrame) -> pd.DataFrame:
    """Return count, mean, std, quartiles, and extrema for every source."""
    _validate(table)
    summary = table.groupby("dataset_source", observed=True)[FEATURE_NAMES].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )
    return summary


def run_phase2a(
    model1_table: pd.DataFrame,
    model2_table: pd.DataFrame,
    model3_table: pd.DataFrame,
    combined_source_table: pd.DataFrame,
) -> dict[str, Any]:
    """Run all Phase 2A diagnostics from notebook-created feature tables."""
    split1 = verify_group_split(model1_table, "Model 1", 0.2)
    split2 = verify_group_split(model2_table, "Model 2", 0.2)
    split3 = verify_group_split(model3_table, "Model 3", 0.25)

    distributions = {
        "Model 1": source_label_distribution(model1_table),
        "Model 2": source_label_distribution(model2_table),
        "Model 3": source_label_distribution(model3_table),
    }
    source_only = model1_source_only(model1_table, split1)
    feature_source = feature_source_classifier(combined_source_table, verify_group_split(
        combined_source_table, "combined source classifier", 0.2
    ))
    return {
        "splits": {"Model 1": split1, "Model 2": split2, "Model 3": split3},
        "source_label_distributions": distributions,
        "model1_source_only": source_only,
        "feature_source_classifier": feature_source,
        "feature_distributions": feature_distributions(combined_source_table),
    }