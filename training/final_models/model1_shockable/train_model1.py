"""Controlled training and evaluation of Model 1 (Shockable vs. Non-Shockable)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import to_onnx

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.final_models.common.evaluation import (
    compute_binary_metrics,
    compute_record_level_metrics,
)
from training.final_models.common.feature_selection import MODEL1_FEATURES
SEED = 42


def get_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
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
                    max_depth=4,
                    learning_rate=0.05,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]),
    }


def run_model1_experiments(dataset_path: Path | None = None) -> dict:
    if dataset_path is None:
        dataset_path = PROJECT_ROOT / "training" / "final_models" / "artifacts" / "model1_shockable_dataset.csv"

    print(f"\n{'='*75}\nRUNNING MODEL 1 (SHOCKABLE VS. NON-SHOCKABLE) CONTROLLED EXPERIMENTS\n{'='*75}")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=MODEL1_FEATURES).reset_index(drop=True)
    print(f"Loaded dataset: {len(df)} windows, {df['record_id'].nunique()} records.")
    print("Class distribution:\n", df["model1_shockable"].value_counts())
    print("Source distribution:\n", df["dataset_source"].value_counts())

    X = df[MODEL1_FEATURES].values
    y = df["model1_shockable"].values.astype(int)
    groups = df["record_id"].values

    results_dir = PROJECT_ROOT / "training" / "final_models" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    export_dir = PROJECT_ROOT / "training" / "final_models" / "export_candidates"
    export_dir.mkdir(parents=True, exist_ok=True)

    models = get_models()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_results = []
    all_predictions = {}
    best_model_name = None
    best_balanced_acc = -1.0
    best_pipeline = None

    for model_name, pipeline in models.items():
        print(f"\n--- Evaluating Model: {model_name} (5-Fold Stratified GroupKFold) ---")
        fold_metrics = []
        oof_preds = np.zeros(len(df), dtype=int)
        oof_probs = np.zeros(len(df), dtype=float)

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            pipeline.fit(X_train, y_train)
            val_preds = pipeline.predict(X_val)
            val_probs = pipeline.predict_proba(X_val)[:, 1]

            oof_preds[val_idx] = val_preds
            oof_probs[val_idx] = val_probs

            metrics = compute_binary_metrics(y_val, val_preds, val_probs)
            fold_metrics.append(metrics)

        # Average window metrics across folds
        avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        print(
            f"Window Metrics -> Balanced Acc: {avg_metrics['balanced_accuracy']:.4f}, "
            f"Macro F1: {avg_metrics['macro_f1']:.4f}, "
            f"Shockable Recall: {avg_metrics['recall_sensitivity']:.4f}, "
            f"Specificity: {avg_metrics['specificity']:.4f}, "
            f"ROC-AUC: {avg_metrics['roc_auc']:.4f}"
        )

        # Save per-window predictions dataframe
        pred_df = df[["record_id", "dataset_source", "original_label"]].copy()
        pred_df["true_label"] = y
        pred_df["prediction"] = oof_preds
        pred_df["predicted_probability"] = oof_probs
        all_predictions[model_name] = pred_df

        pred_csv = results_dir / f"model1_{model_name}_per_window_predictions.csv"
        pred_df.to_csv(pred_csv, index=False)

        # Record-level aggregation metrics
        rec_metrics = compute_record_level_metrics(pred_df)
        print(
            f"Record-Level Metrics -> Balanced Acc: {rec_metrics['balanced_accuracy']:.4f}, "
            f"Macro F1: {rec_metrics['macro_f1']:.4f}, "
            f"Shockable Recall: {rec_metrics['recall_sensitivity']:.4f}, "
            f"Specificity: {rec_metrics['specificity']:.4f}, "
            f"ROC-AUC: {rec_metrics['roc_auc']:.4f}"
        )

        rec_csv = results_dir / f"model1_{model_name}_per_record_summary.csv"
        rec_metrics["per_record_df"].to_csv(rec_csv, index=False)

        res_row = {
            "model": model_name,
            "evaluation": "5fold_stratified_groupcv",
            "window_acc": avg_metrics["accuracy"],
            "window_balanced_acc": avg_metrics["balanced_accuracy"],
            "window_macro_f1": avg_metrics["macro_f1"],
            "window_shockable_precision": avg_metrics["precision"],
            "window_shockable_recall": avg_metrics["recall_sensitivity"],
            "window_specificity": avg_metrics["specificity"],
            "window_shockable_f1": avg_metrics["f1_score"],
            "window_roc_auc": avg_metrics["roc_auc"],
            "record_balanced_acc": rec_metrics["balanced_accuracy"],
            "record_macro_f1": rec_metrics["macro_f1"],
            "record_shockable_recall": rec_metrics["recall_sensitivity"],
            "record_specificity": rec_metrics["specificity"],
            "record_roc_auc": rec_metrics["roc_auc"],
        }
        cv_results.append(res_row)

        if rec_metrics["balanced_accuracy"] > best_balanced_acc:
            best_balanced_acc = rec_metrics["balanced_accuracy"]
            best_model_name = model_name
            best_pipeline = pipeline

    # -------------------------------------------------------------
    # Cross-Database Robustness Evaluation
    # -------------------------------------------------------------
    print(f"\n{'='*50}\nModel 1 Cross-Database Robustness Evaluation\n{'='*50}")
    # D1: Train C2015 -> Test VFDB + CUDB + MIT-BIH
    c2015_mask = df["dataset_source"] == "c2015"
    other_mask = ~c2015_mask

    for model_name, pipeline in models.items():
        # Direction 1: Train C2015 -> Test Others
        pipeline.fit(X[c2015_mask], y[c2015_mask])
        preds_d1 = pipeline.predict(X[other_mask])
        probs_d1 = pipeline.predict_proba(X[other_mask])[:, 1]
        m_d1 = compute_binary_metrics(y[other_mask], preds_d1, probs_d1)

        # Direction 2: Train Others -> Test C2015
        pipeline.fit(X[other_mask], y[other_mask])
        preds_d2 = pipeline.predict(X[c2015_mask])
        probs_d2 = pipeline.predict_proba(X[c2015_mask])[:, 1]
        m_d2 = compute_binary_metrics(y[c2015_mask], preds_d2, probs_d2)

        print(f"[{model_name}] D1 (Train C2015 -> Test Others): Balanced Acc = {m_d1['balanced_accuracy']:.4f}, Sens = {m_d1['recall_sensitivity']:.4f}, Spec = {m_d1['specificity']:.4f}, AUC = {m_d1['roc_auc']:.4f}")
        print(f"[{model_name}] D2 (Train Others -> Test C2015): Balanced Acc = {m_d2['balanced_accuracy']:.4f}, Sens = {m_d2['recall_sensitivity']:.4f}, Spec = {m_d2['specificity']:.4f}, AUC = {m_d2['roc_auc']:.4f}")

        cv_results.append({
            "model": model_name,
            "evaluation": "cross_db_D1_c2015_to_others",
            "window_acc": m_d1["accuracy"],
            "window_balanced_acc": m_d1["balanced_accuracy"],
            "window_macro_f1": m_d1["macro_f1"],
            "window_shockable_precision": m_d1["precision"],
            "window_shockable_recall": m_d1["recall_sensitivity"],
            "window_specificity": m_d1["specificity"],
            "window_shockable_f1": m_d1["f1_score"],
            "window_roc_auc": m_d1["roc_auc"],
            "record_balanced_acc": np.nan,
            "record_macro_f1": np.nan,
            "record_shockable_recall": np.nan,
            "record_specificity": np.nan,
            "record_roc_auc": np.nan,
        })
        cv_results.append({
            "model": model_name,
            "evaluation": "cross_db_D2_others_to_c2015",
            "window_acc": m_d2["accuracy"],
            "window_balanced_acc": m_d2["balanced_accuracy"],
            "window_macro_f1": m_d2["macro_f1"],
            "window_shockable_precision": m_d2["precision"],
            "window_shockable_recall": m_d2["recall_sensitivity"],
            "window_specificity": m_d2["specificity"],
            "window_shockable_f1": m_d2["f1_score"],
            "window_roc_auc": m_d2["roc_auc"],
            "record_balanced_acc": np.nan,
            "record_macro_f1": np.nan,
            "record_shockable_recall": np.nan,
            "record_specificity": np.nan,
            "record_roc_auc": np.nan,
        })

    # Save summary table
    summary_df = pd.DataFrame(cv_results)
    summary_csv = results_dir / "model1_comparison_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nModel 1 summary comparison written to: {summary_csv}")

    # -------------------------------------------------------------
    # Train Best Model on Full Dataset and Export ONNX Candidate
    # -------------------------------------------------------------
    print(f"\nExporting ONNX candidate for best Model 1 ({best_model_name})...")
    best_pipeline.fit(X, y)
    onnx_candidate_path = export_dir / "shockable_classifier_candidate.onnx"

    dummy_input = np.zeros((1, len(MODEL1_FEATURES)), dtype=np.float32)
    onx = to_onnx(best_pipeline, dummy_input, target_opset=15)
    with open(onnx_candidate_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"Exported candidate Model 1 ONNX to: {onnx_candidate_path} ({onnx_candidate_path.stat().st_size:,} bytes)")

    return {
        "best_model": best_model_name,
        "summary": summary_df,
        "onnx_path": str(onnx_candidate_path),
    }


if __name__ == "__main__":
    run_model1_experiments()
