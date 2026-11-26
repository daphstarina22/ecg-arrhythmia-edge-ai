"""Research-Grade Retraining and Evaluation of Model 2 (5-Class Arrhythmia Classifier).

Evaluates the corrected multi-source research dataset across 5 classes:
NSR, TACHY, BRADYCARDIA, ASYSTOLE, and VENTRICULAR.

All outputs are saved to a separate research-only directory:
training/final_models/results/model2_research_corrected/
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import to_onnx

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.final_models.common.feature_selection import MODEL2_FEATURES

SEED = 42
CLASS_NAMES_5C = ["NSR", "TACHY", "BRADYCARDIA", "ASYSTOLE", "VENTRICULAR"]
OUTPUT_DIR = PROJECT_ROOT / "training" / "final_models" / "results" / "model2_research_corrected"


def get_candidate_models() -> Dict[str, Pipeline]:
    """Returns candidate pipelines with median imputation, standard scaling, and class-weight balancing."""
    return {
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=7,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    max_depth=5,
                    learning_rate=0.05,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]),
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]),
    }


def compute_multiclass_metrics_5c(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES_5C,
) -> Dict[str, Any]:
    """Computes comprehensive 5-class metrics, including per-class precision, recall, f1, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    labels = list(range(len(class_names)))
    prec_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    per_class_dict = {}
    for idx, c_name in enumerate(class_names):
        per_class_dict[c_name] = {
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "support": int((y_true == idx).sum()),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_dict,
        "confusion_matrix": cm,
    }


def compute_record_level_metrics_5c(
    df_preds: pd.DataFrame,
    class_names: List[str] = CLASS_NAMES_5C,
) -> Dict[str, Any]:
    """Aggregates window-level predictions per record using majority voting."""
    record_rows = []
    for rec_id, group in df_preds.groupby("record_id"):
        true_label = group["true_label"].mode().iloc[0]
        pred_label = group["prediction"].mode().iloc[0]
        record_rows.append({
            "record_id": rec_id,
            "dataset_source": group["dataset_source"].iloc[0],
            "true_label": true_label,
            "pred_label": pred_label,
            "total_windows": len(group),
            "correct_windows": int((group["true_label"] == group["prediction"]).sum()),
            "record_accuracy": float((group["true_label"] == group["prediction"]).mean()),
        })

    rec_df = pd.DataFrame(record_rows)
    y_true_rec = rec_df["true_label"].values
    y_pred_rec = rec_df["pred_label"].values

    rec_m = compute_multiclass_metrics_5c(y_true_rec, y_pred_rec, class_names=class_names)
    rec_m["total_records"] = len(rec_df)
    rec_m["per_record_df"] = rec_df
    return rec_m


def run_research_retraining():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    art_dir = PROJECT_ROOT / "training" / "final_models" / "artifacts"
    dataset_path = art_dir / "corrected_master_feature_dataset.csv"

    print("=" * 80)
    print("STEP 2: RESEARCH MODEL 2 RETRAINING (5-CLASS TAXONOMY)")
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Corrected research dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=MODEL2_FEATURES).reset_index(drop=True)

    print(f"\nLoaded {len(df)} supervised windows across {df['record_id'].nunique()} independent records.")
    print("\nClass distribution (5-class):")
    class_counts = df["cardiac_rhythm_5class"].value_counts()
    for c_name in CLASS_NAMES_5C:
        count = class_counts.get(c_name, 0)
        pct = count / len(df) * 100
        print(f"  {c_name:<15}: {count:>5} windows ({pct:>5.2f}%)")

    # Document class imbalance ratio
    max_count = class_counts.max()
    min_count = class_counts.min()
    print(f"\nClass imbalance ratio (max/min): {max_count / min_count:.2f}x ({class_counts.idxmax()}: {max_count} vs {class_counts.idxmin()}: {min_count})")

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES_5C)}
    y = df["cardiac_rhythm_5class"].map(class_to_idx).values.astype(int)
    X = df[MODEL2_FEATURES].values
    groups = df["record_id"].values

    models = get_candidate_models()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Verify zero record leakage across folds
    print("\nVerifying StratifiedGroupKFold record independence...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_recs = set(groups[train_idx])
        val_recs = set(groups[val_idx])
        leakage = train_recs.intersection(val_recs)
        if leakage:
            raise RuntimeError(f"Data leakage detected in fold {fold}: {len(leakage)} overlapping records!")
    print("PASSED: Zero record leakage across all 5 cross-validation folds.")

    comparison_results = []
    best_model_name = None
    best_macro_f1 = -1.0
    best_pipeline = None

    all_models_metrics = {}

    for model_name, pipeline in models.items():
        print(f"\n{'='*60}")
        print(f"Training & Evaluating: {model_name.upper()}")
        print(f"{'='*60}")

        oof_preds = np.zeros(len(df), dtype=int)
        oof_probs = np.zeros((len(df), len(CLASS_NAMES_5C)), dtype=float)
        fold_metrics = []

        t_start = time.time()
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            pipeline.fit(X_train, y_train)
            val_preds = pipeline.predict(X_val)
            val_probs = pipeline.predict_proba(X_val)

            oof_preds[val_idx] = val_preds
            oof_probs[val_idx] = val_probs

            fm = compute_multiclass_metrics_5c(y_val, val_preds, class_names=CLASS_NAMES_5C)
            fold_metrics.append(fm)

        train_time = time.time() - t_start

        # Window-level overall metrics
        overall_window_m = compute_multiclass_metrics_5c(y, oof_preds, class_names=CLASS_NAMES_5C)

        avg_fold_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
        avg_fold_bal_acc = float(np.mean([m["balanced_accuracy"] for m in fold_metrics]))
        avg_fold_macro_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))
        avg_fold_weighted_f1 = float(np.mean([m["weighted_f1"] for m in fold_metrics]))

        print(f"Training Time: {train_time:.1f}s")
        print(f"Cross-Validation Fold Averages -> Accuracy: {avg_fold_acc:.4f}, Balanced Acc: {avg_fold_bal_acc:.4f}, Macro-F1: {avg_fold_macro_f1:.4f}")
        print(f"Overall OOF Window Metrics     -> Accuracy: {overall_window_m['accuracy']:.4f}, Balanced Acc: {overall_window_m['balanced_accuracy']:.4f}, Macro-F1: {overall_window_m['macro_f1']:.4f}")

        print("\nPer-Class Performance (Out-Of-Fold):")
        for c_name, scores in overall_window_m["per_class"].items():
            print(f"  {c_name:<15}: Precision = {scores['precision']:.3f}, Recall = {scores['recall']:.3f}, F1 = {scores['f1']:.3f} (support = {scores['support']})")

        print("\nConfusion Matrix (Rows=True, Cols=Pred):")
        cm_df = pd.DataFrame(overall_window_m["confusion_matrix"], index=CLASS_NAMES_5C, columns=CLASS_NAMES_5C)
        print(cm_df)

        # False Bradycardia rate on Normal Sinus Rhythm
        nsr_idx = class_to_idx["NSR"]
        brady_idx = class_to_idx["BRADYCARDIA"]
        nsr_mask = (y == nsr_idx)
        nsr_total = int(nsr_mask.sum())
        nsr_as_brady = int((oof_preds[nsr_mask] == brady_idx).sum())
        false_brady_rate = (nsr_as_brady / nsr_total) * 100
        print(f"\nFalse Bradycardia Alarms on NSR: {nsr_as_brady} / {nsr_total} windows ({false_brady_rate:.2f}%)")

        # Record-level metrics
        pred_df = df[["record_id", "dataset_source", "cardiac_rhythm_5class", "original_label"]].copy()
        pred_df["true_label"] = y
        pred_df["prediction"] = oof_preds
        for idx, c_name in enumerate(CLASS_NAMES_5C):
            pred_df[f"prob_{c_name}"] = oof_probs[:, idx]

        oof_csv = OUTPUT_DIR / f"research_model2_5class_{model_name}_oof_predictions.csv"
        pred_df.to_csv(oof_csv, index=False)

        rec_m = compute_record_level_metrics_5c(pred_df, class_names=CLASS_NAMES_5C)
        print(f"Record-Level Metrics -> Balanced Acc: {rec_m['balanced_accuracy']:.4f}, Macro-F1: {rec_m['macro_f1']:.4f}, Total Records: {rec_m['total_records']}")

        rec_csv = OUTPUT_DIR / f"research_model2_5class_{model_name}_record_metrics.csv"
        rec_m["per_record_df"].to_csv(rec_csv, index=False)

        # Save confusion matrix and metrics JSON
        cm_path = OUTPUT_DIR / f"research_model2_5class_{model_name}_confusion_matrix.json"
        cm_data = {
            "model": model_name,
            "classes": CLASS_NAMES_5C,
            "window_metrics": overall_window_m,
            "record_metrics": {k: v for k, v in rec_m.items() if k != "per_record_df"},
            "false_brady_on_nsr": {
                "nsr_windows_total": nsr_total,
                "nsr_predicted_brady": nsr_as_brady,
                "false_brady_pct": false_brady_rate,
            },
        }
        with open(cm_path, "w") as f:
            json.dump(cm_data, f, indent=2)

        all_models_metrics[model_name] = cm_data

        res_entry = {
            "model": model_name,
            "window_accuracy": overall_window_m["accuracy"],
            "window_balanced_acc": overall_window_m["balanced_accuracy"],
            "window_macro_f1": overall_window_m["macro_f1"],
            "window_weighted_f1": overall_window_m["weighted_f1"],
            "record_balanced_acc": rec_m["balanced_accuracy"],
            "record_macro_f1": rec_m["macro_f1"],
            "nsr_f1": overall_window_m["per_class"]["NSR"]["f1"],
            "tachy_f1": overall_window_m["per_class"]["TACHY"]["f1"],
            "brady_f1": overall_window_m["per_class"]["BRADYCARDIA"]["f1"],
            "asystole_f1": overall_window_m["per_class"]["ASYSTOLE"]["f1"],
            "ventricular_f1": overall_window_m["per_class"]["VENTRICULAR"]["f1"],
            "false_brady_on_nsr_pct": false_brady_rate,
        }
        comparison_results.append(res_entry)

        if overall_window_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = overall_window_m["macro_f1"]
            best_model_name = model_name
            best_pipeline = pipeline

    # Save comparison summary table
    summary_df = pd.DataFrame(comparison_results)
    summary_csv = OUTPUT_DIR / "research_model2_5class_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved model comparison summary table to: {summary_csv}")

    # Out-of-distribution evaluation on quarantined pre-alarm baseline windows (Challenge 2015)
    quar_path = art_dir / "quarantined_unsupervised_feature_dataset.csv"
    if quar_path.exists():
        print(f"\n{'='*60}")
        print("OUT-OF-DISTRIBUTION EVALUATION: QUARANTINED PRE-ALARM BASELINES")
        print(f"{'='*60}")
        df_quar = pd.read_csv(quar_path)
        pre_alarm_df = df_quar[df_quar["quarantine_status"] == "QUARANTINED_PRE_ALARM"].copy()
        pre_alarm_df = pre_alarm_df.dropna(subset=MODEL2_FEATURES).reset_index(drop=True)
        print(f"Evaluating best model ({best_model_name}) on {len(pre_alarm_df)} pre-alarm baseline windows...")

        # Fit best model on all supervised data for OOD inference
        best_pipeline.fit(X, y)
        X_pre = pre_alarm_df[MODEL2_FEATURES].values
        pre_preds = best_pipeline.predict(X_pre)
        pre_alarm_df["predicted_class"] = [CLASS_NAMES_5C[p] for p in pre_preds]

        pred_dist = pre_alarm_df["predicted_class"].value_counts().to_dict()
        print("\nPre-alarm baseline prediction distribution:")
        for c_name in CLASS_NAMES_5C:
            cnt = pred_dist.get(c_name, 0)
            pct = cnt / len(pre_alarm_df) * 100
            print(f"  {c_name:<15}: {cnt:>5} ({pct:>5.2f}%)")

        ood_summary = {
            "best_model": best_model_name,
            "total_pre_alarm_windows": len(pre_alarm_df),
            "prediction_distribution": pred_dist,
            "bradycardia_count": pred_dist.get("BRADYCARDIA", 0),
            "bradycardia_pct": (pred_dist.get("BRADYCARDIA", 0) / len(pre_alarm_df)) * 100,
        }
        with open(OUTPUT_DIR / "research_model2_pre_alarm_evaluation.json", "w") as f:
            json.dump(ood_summary, f, indent=2)

    # Export best model as research-only candidate ONNX
    print(f"\n{'='*60}")
    print(f"EXPORTING RESEARCH ONNX CANDIDATE ({best_model_name.upper()})")
    print(f"{'='*60}")
    dummy_input = np.zeros((1, len(MODEL2_FEATURES)), dtype=np.float32)
    onnx_path = OUTPUT_DIR / "research_arrhythmia_5class_candidate.onnx"
    try:
        best_pipeline.fit(X, y)
        onx = to_onnx(best_pipeline, dummy_input, target_opset=15)
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"Saved research ONNX candidate to: {onnx_path} ({onnx_path.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"Direct ONNX export failed ({e}). Falling back to Random Forest for research ONNX...")
        rf_pipe = get_candidate_models()["random_forest"]
        rf_pipe.fit(X, y)
        onx = to_onnx(rf_pipe, dummy_input, target_opset=15)
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"Saved research Random Forest ONNX candidate to: {onnx_path} ({onnx_path.stat().st_size:,} bytes)")

    # Save full metrics summary
    full_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": MODEL2_FEATURES,
        "classes": CLASS_NAMES_5C,
        "dataset_windows": len(df),
        "dataset_records": int(df["record_id"].nunique()),
        "class_counts": class_counts.to_dict(),
        "class_imbalance_ratio": float(max_count / min_count),
        "best_model": best_model_name,
        "best_macro_f1": float(best_macro_f1),
        "models": all_models_metrics,
    }
    with open(OUTPUT_DIR / "research_model2_5class_metrics_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\nStep 2 research retraining complete. All artifacts saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    run_research_retraining()
