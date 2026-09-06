"""Evaluation utilities, group-aware cross-validation, and record-level metrics."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    pos_label: int = 1,
) -> dict[str, float]:
    """Calculates comprehensive binary classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))

    # Specificity (True Negative Rate)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "f1_score": f1,
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            # Handle 1D or 2D probability array
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob.ravel()
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Calculates multi-class classification metrics including per-class performance."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    unique_classes = sorted(list(set(y_true) | set(y_pred)))
    prec_per_class = precision_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)

    per_class_dict = {}
    for idx, c in enumerate(unique_classes):
        c_name = class_names[c] if class_names and c < len(class_names) else str(c)
        per_class_dict[c_name] = {
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "support": int((y_true == c).sum()),
        }

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_dict,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=unique_classes).tolist(),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            metrics["roc_auc_ovr"] = float("nan")

    return metrics


def compute_record_level_metrics(
    df_predictions: pd.DataFrame,
    record_col: str = "record_id",
    true_col: str = "true_label",
    pred_col: str = "prediction",
    prob_col: str | None = "predicted_probability",
    is_binary: bool = True,
) -> dict[str, Any]:
    """Aggregates window-level predictions per record to compute patient-level metrics."""
    record_rows = []
    for rec_id, group in df_predictions.groupby(record_col):
        true_label = group[true_col].mode().iloc[0]
        # Record-level prediction via majority vote
        pred_label = group[pred_col].mode().iloc[0]
        mean_prob = group[prob_col].mean() if prob_col in group.columns else None

        record_rows.append({
            "record_id": rec_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "mean_prob": mean_prob,
            "total_windows": len(group),
            "correct_windows": int((group[true_col] == group[pred_col]).sum()),
            "record_accuracy": float((group[true_col] == group[pred_col]).mean()),
        })

    rec_df = pd.DataFrame(record_rows)
    y_true_rec = rec_df["true_label"].values
    y_pred_rec = rec_df["pred_label"].values
    y_prob_rec = rec_df["mean_prob"].values if prob_col in rec_df.columns else None

    if is_binary:
        rec_metrics = compute_binary_metrics(y_true_rec, y_pred_rec, y_prob_rec)
    else:
        rec_metrics = compute_multiclass_metrics(y_true_rec, y_pred_rec, None)

    rec_metrics["total_records"] = len(rec_df)
    rec_metrics["per_record_df"] = rec_df
    return rec_metrics
