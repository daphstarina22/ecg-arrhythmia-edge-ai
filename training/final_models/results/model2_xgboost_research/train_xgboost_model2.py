"""Research-Grade XGBoost Training and Evaluation for Model 2 (5-Class Arrhythmia Classifier).

Evaluates XGBoost against the corrected multi-source research dataset:
Classes: NSR, TACHY, BRADYCARDIA, ASYSTOLE, VENTRICULAR.

All outputs are saved to a separate research directory:
training/final_models/results/model2_xgboost_research/
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.final_models.common.feature_selection import MODEL2_FEATURES

SEED = 42
CLASS_NAMES_5C = ["NSR", "TACHY", "BRADYCARDIA", "ASYSTOLE", "VENTRICULAR"]
OUTPUT_DIR = Path(__file__).resolve().parent


def compute_multiclass_metrics_5c(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES_5C,
) -> Dict[str, Any]:
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


def run_xgboost_experiments():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    art_dir = PROJECT_ROOT / "training" / "final_models" / "artifacts"
    dataset_path = art_dir / "corrected_master_feature_dataset.csv"

    print("=" * 80)
    print("XGBOOST RESEARCH RETRAINING FOR MODEL 2 (5-CLASS TAXONOMY)")
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=MODEL2_FEATURES).reset_index(drop=True)

    print(f"\nLoaded {len(df)} supervised windows across {df['record_id'].nunique()} independent records.")
    class_counts = df["cardiac_rhythm_5class"].value_counts()
    for c_name in CLASS_NAMES_5C:
        count = class_counts.get(c_name, 0)
        pct = count / len(df) * 100
        print(f"  {c_name:<15}: {count:>5} windows ({pct:>5.2f}%)")

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES_5C)}
    y = df["cardiac_rhythm_5class"].map(class_to_idx).values.astype(int)
    X = df[MODEL2_FEATURES].values
    groups = df["record_id"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1. Verify zero data leakage
    print("\nVerifying StratifiedGroupKFold record independence...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_recs = set(groups[train_idx])
        val_recs = set(groups[val_idx])
        leakage = train_recs.intersection(val_recs)
        if leakage:
            raise RuntimeError(f"Data leakage detected in fold {fold}: {len(leakage)} overlapping records!")
    print("PASSED: Zero record leakage across all 5 cross-validation folds.")

    # 2. Define XGBoost hyperparameter variations
    xgb_configs = {
        "xgboost_depth5_lr005": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": SEED,
            "n_jobs": -1,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
        },
        "xgboost_depth4_lr005": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": SEED,
            "n_jobs": -1,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
        },
        "xgboost_depth6_lr005": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": SEED,
            "n_jobs": -1,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
        },
    }

    all_comparison_rows = []
    best_config_name = None
    best_macro_f1 = -1.0
    best_overall_window_m = None
    all_configs_metrics = {}

    for config_name, params in xgb_configs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating Config: {config_name.upper()}")
        print(f"Parameters: {params}")
        print(f"{'='*60}")

        oof_preds = np.zeros(len(df), dtype=int)
        oof_probs = np.zeros((len(df), len(CLASS_NAMES_5C)), dtype=float)
        fold_metrics = []

        t_start = time.time()
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Fit preprocessing ONLY on training fold
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
            X_val_proc = scaler.transform(imputer.transform(X_val))

            # Compute balanced sample weights for multiclass XGBoost
            sample_weights = compute_sample_weight("balanced", y_train)

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train_proc, y_train, sample_weight=sample_weights)

            val_preds = clf.predict(X_val_proc)
            val_probs = clf.predict_proba(X_val_proc)

            oof_preds[val_idx] = val_preds
            oof_probs[val_idx] = val_probs

            fm = compute_multiclass_metrics_5c(y_val, val_preds, class_names=CLASS_NAMES_5C)
            fold_metrics.append(fm)

        train_time = time.time() - t_start
        overall_window_m = compute_multiclass_metrics_5c(y, oof_preds, class_names=CLASS_NAMES_5C)

        avg_fold_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
        avg_fold_bal_acc = float(np.mean([m["balanced_accuracy"] for m in fold_metrics]))
        avg_fold_macro_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))

        print(f"Training Time: {train_time:.1f}s")
        print(f"Cross-Validation Fold Averages -> Accuracy: {avg_fold_acc:.4f}, Balanced Acc: {avg_fold_bal_acc:.4f}, Macro-F1: {avg_fold_macro_f1:.4f}")
        print(f"Overall OOF Window Metrics     -> Accuracy: {overall_window_m['accuracy']:.4f}, Balanced Acc: {overall_window_m['balanced_accuracy']:.4f}, Macro-F1: {overall_window_m['macro_f1']:.4f}")

        print("\nPer-Class Performance (Out-Of-Fold):")
        for c_name, scores in overall_window_m["per_class"].items():
            print(f"  {c_name:<15}: Precision = {scores['precision']:.3f}, Recall = {scores['recall']:.3f}, F1 = {scores['f1']:.3f} (support = {scores['support']})")

        cm_df = pd.DataFrame(overall_window_m["confusion_matrix"], index=CLASS_NAMES_5C, columns=CLASS_NAMES_5C)
        print("\nConfusion Matrix (Rows=True, Cols=Pred):")
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

        oof_csv = OUTPUT_DIR / f"{config_name}_oof_predictions.csv"
        pred_df.to_csv(oof_csv, index=False)

        rec_m = compute_record_level_metrics_5c(pred_df, class_names=CLASS_NAMES_5C)
        print(f"Record-Level Metrics -> Balanced Acc: {rec_m['balanced_accuracy']:.4f}, Macro-F1: {rec_m['macro_f1']:.4f}")

        rec_csv = OUTPUT_DIR / f"{config_name}_record_metrics.csv"
        rec_m["per_record_df"].to_csv(rec_csv, index=False)

        # Save confusion matrix JSON
        cm_path = OUTPUT_DIR / f"{config_name}_confusion_matrix.json"
        cm_data = {
            "config": config_name,
            "params": params,
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

        all_configs_metrics[config_name] = cm_data

        res_entry = {
            "config": config_name,
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
        all_comparison_rows.append(res_entry)

        if overall_window_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = overall_window_m["macro_f1"]
            best_config_name = config_name
            best_overall_window_m = overall_window_m

    # 3. Add baseline comparison row (HistGradientBoosting from previous run)
    hgb_baseline_path = PROJECT_ROOT / "training" / "final_models" / "results" / "model2_research_corrected" / "research_model2_5class_hist_gradient_boosting_confusion_matrix.json"
    if hgb_baseline_path.exists():
        with open(hgb_baseline_path) as f:
            hgb_data = json.load(f)
        hgb_wm = hgb_data["window_metrics"]
        hgb_rm = hgb_data["record_metrics"]
        hgb_fb = hgb_data["false_brady_on_nsr"]["false_brady_pct"]
        all_comparison_rows.append({
            "config": "BASELINE_HistGradientBoosting",
            "window_accuracy": hgb_wm["accuracy"],
            "window_balanced_acc": hgb_wm["balanced_accuracy"],
            "window_macro_f1": hgb_wm["macro_f1"],
            "window_weighted_f1": hgb_wm["weighted_f1"],
            "record_balanced_acc": hgb_rm["balanced_accuracy"],
            "record_macro_f1": hgb_rm["macro_f1"],
            "nsr_f1": hgb_wm["per_class"]["NSR"]["f1"],
            "tachy_f1": hgb_wm["per_class"]["TACHY"]["f1"],
            "brady_f1": hgb_wm["per_class"]["BRADYCARDIA"]["f1"],
            "asystole_f1": hgb_wm["per_class"]["ASYSTOLE"]["f1"],
            "ventricular_f1": hgb_wm["per_class"]["VENTRICULAR"]["f1"],
            "false_brady_on_nsr_pct": hgb_fb,
        })

    # Save summary table
    summary_df = pd.DataFrame(all_comparison_rows)
    summary_csv = OUTPUT_DIR / "xgboost_vs_baseline_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved comparison summary table to: {summary_csv}")

    # 4. Fit best XGBoost model on full dataset and serialize
    print(f"\n{'='*60}")
    print(f"SERIALIZING BEST XGBOOST MODEL: {best_config_name.upper()}")
    print(f"{'='*60}")
    best_params = xgb_configs[best_config_name]
    full_imputer = SimpleImputer(strategy="median")
    full_scaler = StandardScaler()
    X_full = full_scaler.fit_transform(full_imputer.fit_transform(X))
    full_weights = compute_sample_weight("balanced", y)

    best_xgb = xgb.XGBClassifier(**best_params)
    best_xgb.fit(X_full, y, sample_weight=full_weights)

    # Native JSON model
    native_json_path = OUTPUT_DIR / "research_xgboost_5class_model.json"
    best_xgb.save_model(str(native_json_path))
    print(f"Saved native XGBoost model to: {native_json_path} ({native_json_path.stat().st_size:,} bytes)")

    # Full Joblib pipeline
    full_pipeline = Pipeline([
        ("imputer", full_imputer),
        ("scaler", full_scaler),
        ("clf", best_xgb),
    ])
    joblib_path = OUTPUT_DIR / "research_xgboost_5class_pipeline.joblib"
    joblib.dump(full_pipeline, joblib_path)
    print(f"Saved full pipeline joblib to: {joblib_path} ({joblib_path.stat().st_size:,} bytes)")

    # 5. Out-of-distribution evaluation on quarantined pre-alarm baseline windows
    quar_path = art_dir / "quarantined_unsupervised_feature_dataset.csv"
    if quar_path.exists():
        print(f"\n{'='*60}")
        print("OUT-OF-DISTRIBUTION EVALUATION: QUARANTINED PRE-ALARM BASELINES")
        print(f"{'='*60}")
        df_quar = pd.read_csv(quar_path)
        pre_alarm_df = df_quar[df_quar["quarantine_status"] == "QUARANTINED_PRE_ALARM"].copy()
        pre_alarm_df = pre_alarm_df.dropna(subset=MODEL2_FEATURES).reset_index(drop=True)
        print(f"Evaluating best XGBoost model on {len(pre_alarm_df)} pre-alarm baseline windows...")

        X_pre = full_pipeline.named_steps["scaler"].transform(full_pipeline.named_steps["imputer"].transform(pre_alarm_df[MODEL2_FEATURES].values))
        pre_preds = best_xgb.predict(X_pre)
        pre_alarm_df["predicted_class"] = [CLASS_NAMES_5C[p] for p in pre_preds]

        pred_dist = pre_alarm_df["predicted_class"].value_counts().to_dict()
        print("\nPre-alarm baseline prediction distribution:")
        for c_name in CLASS_NAMES_5C:
            cnt = pred_dist.get(c_name, 0)
            pct = cnt / len(pre_alarm_df) * 100
            print(f"  {c_name:<15}: {cnt:>5} ({pct:>5.2f}%)")

        ood_summary = {
            "best_config": best_config_name,
            "total_pre_alarm_windows": len(pre_alarm_df),
            "prediction_distribution": pred_dist,
            "bradycardia_count": pred_dist.get("BRADYCARDIA", 0),
            "bradycardia_pct": (pred_dist.get("BRADYCARDIA", 0) / len(pre_alarm_df)) * 100,
        }
        with open(OUTPUT_DIR / "xgboost_pre_alarm_evaluation.json", "w") as f:
            json.dump(ood_summary, f, indent=2)

    # 6. Save full experiment summary JSON
    exp_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": MODEL2_FEATURES,
        "classes": CLASS_NAMES_5C,
        "dataset_windows": len(df),
        "dataset_records": int(df["record_id"].nunique()),
        "class_counts": class_counts.to_dict(),
        "best_config": best_config_name,
        "best_macro_f1": float(best_macro_f1),
        "configs": all_configs_metrics,
        "comparison_table": all_comparison_rows,
    }
    with open(OUTPUT_DIR / "xgboost_research_metrics_summary.json", "w") as f:
        json.dump(exp_summary, f, indent=2)

    print(f"\nXGBoost research experiment complete. All artifacts saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    run_xgboost_experiments()
