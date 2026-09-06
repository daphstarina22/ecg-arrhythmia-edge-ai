"""Controlled training and evaluation of Model 3 (VT vs. VF/VFL Shockable Subtype Classifier)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import to_onnx

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.final_models.common.data_loader import load_model3_dataset
from training.final_models.common.evaluation import (
    compute_binary_metrics,
    compute_record_level_metrics,
)
from training.final_models.common.feature_selection import (
    MODEL3_SET_A_FULL,
    MODEL3_SET_B_DOMAIN_STABLE,
    MODEL3_SET_C_MINIMAL_EDGE,
)
from sklearn.impute import SimpleImputer
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
                    max_depth=5,
                    min_samples_leaf=10,
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


def run_model3_experiments():
    results_dir = PROJECT_ROOT / "training" / "final_models" / "results"
    export_dir = PROJECT_ROOT / "training" / "final_models" / "export_candidates"
    results_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    feature_subsets = {
        "set_a_full": MODEL3_SET_A_FULL,
        "set_b_domain_stable": MODEL3_SET_B_DOMAIN_STABLE,
        "set_c_minimal_edge": MODEL3_SET_C_MINIMAL_EDGE,
    }

    all_comparison_rows = []
    best_overall_auc = -1.0
    best_candidate_name = None
    best_candidate_pipeline = None
    best_candidate_features = None

    for subset_name, feature_list in feature_subsets.items():
        print(f"\n{'='*75}\nEVALUATING MODEL 3 FEATURE SET: {subset_name} ({len(feature_list)} features)\n{'='*75}")
        df, _ = load_model3_dataset(feature_set=subset_name)

        X = df[feature_list].values
        y = df["target"].values.astype(int)
        groups = df["record_id"].values

        models = get_models()
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

        # ---------------------------------------------------------
        # 1. 5-Fold StratifiedGroupKFold
        # ---------------------------------------------------------
        for model_name, pipeline in models.items():
            print(f"\n--- Model: {model_name} on {subset_name} ---")
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

                m = compute_binary_metrics(y_val, val_preds, val_probs)
                fold_metrics.append(m)

            avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
            print(
                f"Window Metrics -> Bal Acc: {avg_metrics['balanced_accuracy']:.4f}, "
                f"Macro F1: {avg_metrics['macro_f1']:.4f}, "
                f"VF Prec: {avg_metrics['precision']:.4f}, "
                f"VF Rec: {avg_metrics['recall_sensitivity']:.4f}, "
                f"VF F1: {avg_metrics['f1_score']:.4f}, "
                f"ROC-AUC: {avg_metrics['roc_auc']:.4f}"
            )

            # Persist per-window predictions
            pred_df = df[["record_id", "dataset_source", "label", "window_index"]].copy()
            pred_df["true_label"] = y
            pred_df["prediction"] = oof_preds
            pred_df["predicted_probability"] = oof_probs

            pred_csv = results_dir / f"model3_{subset_name}_{model_name}_predictions.csv"
            pred_df.to_csv(pred_csv, index=False)

            # Record-level metrics
            rec_m = compute_record_level_metrics(pred_df)
            print(
                f"Record Metrics -> Bal Acc: {rec_m['balanced_accuracy']:.4f}, "
                f"Macro F1: {rec_m['macro_f1']:.4f}, "
                f"VF Recall: {rec_m['recall_sensitivity']:.4f}, "
                f"VT Specificity: {rec_m['specificity']:.4f}, "
                f"ROC-AUC: {rec_m['roc_auc']:.4f}"
            )

            rec_csv = results_dir / f"model3_{subset_name}_{model_name}_record_summary.csv"
            rec_m["per_record_df"].to_csv(rec_csv, index=False)

            row = {
                "feature_set": subset_name,
                "n_features": len(feature_list),
                "model": model_name,
                "evaluation": "5fold_stratified_groupcv",
                "window_bal_acc": avg_metrics["balanced_accuracy"],
                "window_macro_f1": avg_metrics["macro_f1"],
                "window_vf_precision": avg_metrics["precision"],
                "window_vf_recall": avg_metrics["recall_sensitivity"],
                "window_vf_f1": avg_metrics["f1_score"],
                "window_roc_auc": avg_metrics["roc_auc"],
                "record_bal_acc": rec_m["balanced_accuracy"],
                "record_macro_f1": rec_m["macro_f1"],
                "record_vf_recall": rec_m["recall_sensitivity"],
                "record_vt_specificity": rec_m["specificity"],
                "record_roc_auc": rec_m["roc_auc"],
            }
            all_comparison_rows.append(row)

            # Track best candidate for Domain-Stable Set B
            if subset_name == "set_b_domain_stable" and avg_metrics["roc_auc"] > best_overall_auc:
                best_overall_auc = avg_metrics["roc_auc"]
                best_candidate_name = f"{subset_name}_{model_name}"
                best_candidate_pipeline = pipeline
                best_candidate_features = feature_list

        # ---------------------------------------------------------
        # 2. Cross-Database Evaluation (D1 & D2)
        # ---------------------------------------------------------
        print(f"\n--- Cross-Database Evaluation for {subset_name} ---")
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

            print(f"[{model_name}] D1 (C2015 -> Others): Bal Acc = {m_d1['balanced_accuracy']:.4f}, VF Rec = {m_d1['recall_sensitivity']:.4f}, AUC = {m_d1['roc_auc']:.4f}")
            print(f"[{model_name}] D2 (Others -> C2015): Bal Acc = {m_d2['balanced_accuracy']:.4f}, VF Rec = {m_d2['recall_sensitivity']:.4f}, AUC = {m_d2['roc_auc']:.4f}")

            all_comparison_rows.append({
                "feature_set": subset_name,
                "n_features": len(feature_list),
                "model": model_name,
                "evaluation": "cross_db_D1_c2015_to_others",
                "window_bal_acc": m_d1["balanced_accuracy"],
                "window_macro_f1": m_d1["macro_f1"],
                "window_vf_precision": m_d1["precision"],
                "window_vf_recall": m_d1["recall_sensitivity"],
                "window_vf_f1": m_d1["f1_score"],
                "window_roc_auc": m_d1["roc_auc"],
                "record_bal_acc": np.nan,
                "record_macro_f1": np.nan,
                "record_vf_recall": np.nan,
                "record_vt_specificity": np.nan,
                "record_roc_auc": np.nan,
            })
            all_comparison_rows.append({
                "feature_set": subset_name,
                "n_features": len(feature_list),
                "model": model_name,
                "evaluation": "cross_db_D2_others_to_c2015",
                "window_bal_acc": m_d2["balanced_accuracy"],
                "window_macro_f1": m_d2["macro_f1"],
                "window_vf_precision": m_d2["precision"],
                "window_vf_recall": m_d2["recall_sensitivity"],
                "window_vf_f1": m_d2["f1_score"],
                "window_roc_auc": m_d2["roc_auc"],
                "record_bal_acc": np.nan,
                "record_macro_f1": np.nan,
                "record_vf_recall": np.nan,
                "record_vt_specificity": np.nan,
                "record_roc_auc": np.nan,
            })

    # Save summary table
    summary_df = pd.DataFrame(all_comparison_rows)
    summary_csv = results_dir / "model3_comparison_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nModel 3 comparison summary written to: {summary_csv}")

    # Export ONNX Candidate for best Model 3 (Domain-Stable Set B)
    if best_candidate_pipeline is not None:
        print(f"\nExporting best Model 3 ONNX candidate ({best_candidate_name})...")
        df_full, _ = load_model3_dataset(feature_set="set_b_domain_stable")
        X_full = df_full[best_candidate_features].values
        y_full = df_full["target"].values.astype(int)
        dummy_input = np.zeros((1, len(best_candidate_features)), dtype=np.float32)
        onnx_candidate_path = export_dir / "vf_vt_subtype_classifier_candidate.onnx"
        try:
            best_candidate_pipeline.fit(X_full, y_full)
            onx = to_onnx(best_candidate_pipeline, dummy_input, target_opset=15)
            with open(onnx_candidate_path, "wb") as f:
                f.write(onx.SerializeToString())
            print(f"Exported Model 3 candidate ONNX to: {onnx_candidate_path} ({onnx_candidate_path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"Direct ONNX export failed ({e}). Falling back to Random Forest for Model 3 candidate...")
            rf_pipe = get_models()["random_forest"]
            rf_pipe.fit(X_full, y_full)
            onx = to_onnx(rf_pipe, dummy_input, target_opset=15)
            with open(onnx_candidate_path, "wb") as f:
                f.write(onx.SerializeToString())
            print(f"Exported Model 3 Random Forest candidate ONNX to: {onnx_candidate_path} ({onnx_candidate_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    run_model3_experiments()
