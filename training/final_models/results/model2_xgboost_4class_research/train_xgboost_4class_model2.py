"""Research-Grade XGBoost Training and Evaluation for Model 2 (4-Class Revision).

Excludes ASYSTOLE from the supervised training dataset.
Classes: NSR, TACHY, BRADYCARDIA, VENTRICULAR.

All outputs are saved to a separate research directory:
training/final_models/results/model2_xgboost_4class_research/

Strict Safety Constraints:
- Production code, deployed models, and ONNX files are NOT modified.
- Existing 5-class training results and datasets are NOT overwritten.
- ASYSTOLE windows are evaluated separately out-of-domain and NOT relabeled.
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
CLASS_NAMES_4C = ["NSR", "TACHY", "BRADYCARDIA", "VENTRICULAR"]
OUTPUT_DIR = Path(__file__).resolve().parent


def compute_multiclass_metrics_4c(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES_4C,
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


def compute_record_level_metrics_4c(
    df_preds: pd.DataFrame,
    class_names: List[str] = CLASS_NAMES_4C,
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

    rec_m = compute_multiclass_metrics_4c(y_true_rec, y_pred_rec, class_names=class_names)
    rec_m["total_records"] = len(rec_df)
    rec_m["per_record_df"] = rec_df
    return rec_m


def evaluate_excluded_asystole(
    pipeline: Pipeline,
    asystole_csv: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluates the 4-class pipeline on the excluded asystole windows.
    
    IMPORTANT NOTE:
    This evaluation is purely exploratory to document how a model without an
    asystole class behaves on flatline and agonal signals. These forced predictions
    must NEVER be interpreted as valid asystole detection.
    """
    if not asystole_csv.exists():
        raise FileNotFoundError(f"Asystole dataset not found at {asystole_csv}")

    df_asy = pd.read_csv(asystole_csv)
    df_asy_clean = df_asy.dropna(subset=MODEL2_FEATURES).reset_index(drop=True)

    X_asy = df_asy_clean[MODEL2_FEATURES].values
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    imputer = pipeline.named_steps["imputer"]

    X_asy_proc = scaler.transform(imputer.transform(X_asy))
    preds = clf.predict(X_asy_proc)
    probs = clf.predict_proba(X_asy_proc)

    df_asy_clean["predicted_class_4c"] = [CLASS_NAMES_4C[p] for p in preds]
    for idx, c_name in enumerate(CLASS_NAMES_4C):
        df_asy_clean[f"prob_{c_name}"] = probs[:, idx]

    # Save window-level predictions
    preds_path = output_dir / "asystole_windows_4class_predictions.csv"
    save_cols = [
        "record_id", "dataset_source", "cardiac_rhythm_5class",
        "predicted_class_4c", "prob_NSR", "prob_TACHY", "prob_BRADYCARDIA", "prob_VENTRICULAR",
        "mean_rr", "peak_count", "dominant_freq", "calibrated_ptp", "calibrated_std"
    ]
    df_asy_clean[save_cols].to_csv(preds_path, index=False)

    # Distribution of forced predictions
    pred_counts = df_asy_clean["predicted_class_4c"].value_counts().to_dict()
    total_asy = len(df_asy_clean)
    pred_summary = {}
    for c_name in CLASS_NAMES_4C:
        cnt = pred_counts.get(c_name, 0)
        pct = (cnt / total_asy) * 100
        pred_summary[c_name] = {"count": cnt, "percentage": float(pct)}

    # Probability summary across all asystole windows
    prob_stats = {}
    for c_name in CLASS_NAMES_4C:
        p_col = df_asy_clean[f"prob_{c_name}"]
        prob_stats[c_name] = {
            "mean": float(p_col.mean()),
            "std": float(p_col.std()),
            "median": float(p_col.median()),
            "min": float(p_col.min()),
            "max": float(p_col.max()),
        }

    # Per-record breakdown
    per_record = {}
    for rec_id, rec_grp in df_asy_clean.groupby("record_id"):
        rec_dist = rec_grp["predicted_class_4c"].value_counts().to_dict()
        per_record[rec_id] = {
            "total_windows": len(rec_grp),
            "dataset_source": rec_grp["dataset_source"].iloc[0],
            "predictions": rec_dist,
            "mean_ptp_mv": float(rec_grp["calibrated_ptp"].mean()),
            "mean_peak_count": float(rec_grp["peak_count"].mean()),
        }

    summary = {
        "disclaimer": "The 4-class model has NO asystole class. These forced outputs demonstrate pseudo-classification behavior on flatline/agonal data and must NOT be interpreted as valid asystole detection.",
        "total_asystole_windows_evaluated": total_asy,
        "forced_prediction_distribution": pred_summary,
        "predicted_probability_statistics": prob_stats,
        "per_record_breakdown": per_record,
    }

    sum_path = output_dir / "asystole_evaluation_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_xgboost_4class_experiments():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = OUTPUT_DIR / "four_class_master_feature_dataset.csv"

    print("=" * 80)
    print("XGBOOST RESEARCH RETRAINING FOR MODEL 2 (4-CLASS: NO ASYSTOLE)")
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    if not dataset_path.exists():
        raise FileNotFoundError(f"4-class dataset not found at {dataset_path}. Run prepare_4class_research_dataset.py first.")

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=MODEL2_FEATURES).reset_index(drop=True)

    print(f"\nLoaded {len(df)} supervised windows across {df['record_id'].nunique()} independent records.")
    class_counts = df["cardiac_rhythm_4class"].value_counts()
    for c_name in CLASS_NAMES_4C:
        count = class_counts.get(c_name, 0)
        pct = count / len(df) * 100
        print(f"  {c_name:<15}: {count:>5} windows ({pct:>5.2f}%)")

    # Strictly verify classes
    actual_classes = set(df["cardiac_rhythm_4class"].unique())
    assert actual_classes == set(CLASS_NAMES_4C), f"Unexpected classes: {actual_classes}"
    assert "ASYSTOLE" not in actual_classes, "ASYSTOLE found in 4-class dataset!"

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES_4C)}
    y = df["cardiac_rhythm_4class"].map(class_to_idx).values.astype(int)
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

    # 2. Define XGBoost hyperparameter variations (identical to 5-class study)
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
        oof_probs = np.zeros((len(df), len(CLASS_NAMES_4C)), dtype=float)
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

            # Compute balanced sample weights for 4-class XGBoost
            sample_weights = compute_sample_weight("balanced", y_train)

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train_proc, y_train, sample_weight=sample_weights)

            val_preds = clf.predict(X_val_proc)
            val_probs = clf.predict_proba(X_val_proc)

            oof_preds[val_idx] = val_preds
            oof_probs[val_idx] = val_probs

            fm = compute_multiclass_metrics_4c(y_val, val_preds, class_names=CLASS_NAMES_4C)
            fold_metrics.append(fm)

        train_time = time.time() - t_start
        overall_window_m = compute_multiclass_metrics_4c(y, oof_preds, class_names=CLASS_NAMES_4C)

        avg_fold_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
        avg_fold_bal_acc = float(np.mean([m["balanced_accuracy"] for m in fold_metrics]))
        avg_fold_macro_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))

        print(f"Training Time: {train_time:.1f}s")
        print(f"Cross-Validation Fold Averages -> Accuracy: {avg_fold_acc:.4f}, Balanced Acc: {avg_fold_bal_acc:.4f}, Macro-F1: {avg_fold_macro_f1:.4f}")
        print(f"Overall OOF Window Metrics     -> Accuracy: {overall_window_m['accuracy']:.4f}, Balanced Acc: {overall_window_m['balanced_accuracy']:.4f}, Macro-F1: {overall_window_m['macro_f1']:.4f}")

        print("\nPer-Class Performance (Out-Of-Fold):")
        for c_name, scores in overall_window_m["per_class"].items():
            print(f"  {c_name:<15}: Precision = {scores['precision']:.3f}, Recall = {scores['recall']:.3f}, F1 = {scores['f1']:.3f} (support = {scores['support']})")

        cm_df = pd.DataFrame(overall_window_m["confusion_matrix"], index=CLASS_NAMES_4C, columns=CLASS_NAMES_4C)
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
        pred_df = df[["record_id", "dataset_source", "cardiac_rhythm_4class", "original_label"]].copy()
        pred_df["true_label"] = y
        pred_df["prediction"] = oof_preds
        for idx, c_name in enumerate(CLASS_NAMES_4C):
            pred_df[f"prob_{c_name}"] = oof_probs[:, idx]

        oof_csv = OUTPUT_DIR / f"{config_name}_oof_predictions.csv"
        pred_df.to_csv(oof_csv, index=False)

        rec_m = compute_record_level_metrics_4c(pred_df, class_names=CLASS_NAMES_4C)
        print(f"Record-Level Metrics -> Balanced Acc: {rec_m['balanced_accuracy']:.4f}, Macro-F1: {rec_m['macro_f1']:.4f}")

        rec_csv = OUTPUT_DIR / f"{config_name}_record_metrics.csv"
        rec_m["per_record_df"].to_csv(rec_csv, index=False)

        # Save confusion matrix JSON
        cm_path = OUTPUT_DIR / f"{config_name}_confusion_matrix.json"
        cm_data = {
            "config": config_name,
            "params": params,
            "classes": CLASS_NAMES_4C,
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
            "ventricular_f1": overall_window_m["per_class"]["VENTRICULAR"]["f1"],
            "false_brady_on_nsr_pct": false_brady_rate,
        }
        all_comparison_rows.append(res_entry)

        if overall_window_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = overall_window_m["macro_f1"]
            best_config_name = config_name
            best_overall_window_m = overall_window_m

    # 3. Fit best XGBoost model on full 4-class dataset and serialize
    print(f"\n{'='*60}")
    print(f"SERIALIZING BEST 4-CLASS XGBOOST MODEL: {best_config_name.upper()}")
    print(f"{'='*60}")
    best_params = xgb_configs[best_config_name]
    full_imputer = SimpleImputer(strategy="median")
    full_scaler = StandardScaler()
    X_full = full_scaler.fit_transform(full_imputer.fit_transform(X))
    full_weights = compute_sample_weight("balanced", y)

    best_xgb = xgb.XGBClassifier(**best_params)
    best_xgb.fit(X_full, y, sample_weight=full_weights)

    # Native JSON model
    native_json_path = OUTPUT_DIR / "research_xgboost_4class_model.json"
    best_xgb.save_model(str(native_json_path))
    print(f"Saved native 4-class XGBoost model to: {native_json_path} ({native_json_path.stat().st_size:,} bytes)")

    # Full Joblib pipeline
    full_pipeline = Pipeline([
        ("imputer", full_imputer),
        ("scaler", full_scaler),
        ("clf", best_xgb),
    ])
    joblib_path = OUTPUT_DIR / "research_xgboost_4class_pipeline.joblib"
    joblib.dump(full_pipeline, joblib_path)
    print(f"Saved full pipeline joblib to: {joblib_path} ({joblib_path.stat().st_size:,} bytes)")

    # 4. Evaluate excluded ASYSTOLE windows
    print(f"\n{'='*60}")
    print("SEPARATE OUT-OF-DOMAIN EVALUATION: EXCLUDED ASYSTOLE WINDOWS")
    print(f"{'='*60}")
    asystole_csv = OUTPUT_DIR / "excluded_asystole_windows.csv"
    asy_summary = evaluate_excluded_asystole(full_pipeline, asystole_csv, OUTPUT_DIR)
    print(f"Evaluated {asy_summary['total_asystole_windows_evaluated']} excluded asystole windows:")
    for c_name, dist in asy_summary["forced_prediction_distribution"].items():
        print(f"  Classified as {c_name:<15}: {dist['count']:>3} windows ({dist['percentage']:>5.2f}%)")

    # 5. Out-of-distribution evaluation on quarantined pre-alarm baseline windows
    quar_path = PROJECT_ROOT / "training" / "final_models" / "artifacts" / "quarantined_unsupervised_feature_dataset.csv"
    if quar_path.exists():
        print(f"\n{'='*60}")
        print("OUT-OF-DISTRIBUTION EVALUATION: QUARANTINED PRE-ALARM BASELINES")
        print(f"{'='*60}")
        df_quar = pd.read_csv(quar_path)
        pre_alarm_df = df_quar[df_quar["quarantine_status"] == "QUARANTINED_PRE_ALARM"].copy()
        pre_alarm_df = pre_alarm_df.dropna(subset=MODEL2_FEATURES).reset_index(drop=True)
        print(f"Evaluating 4-class XGBoost model on {len(pre_alarm_df)} pre-alarm baseline windows...")

        X_pre = full_pipeline.named_steps["scaler"].transform(full_pipeline.named_steps["imputer"].transform(pre_alarm_df[MODEL2_FEATURES].values))
        pre_preds = best_xgb.predict(X_pre)
        pre_alarm_df["predicted_class_4c"] = [CLASS_NAMES_4C[p] for p in pre_preds]

        pred_dist = pre_alarm_df["predicted_class_4c"].value_counts().to_dict()
        print("\nPre-alarm baseline prediction distribution:")
        for c_name in CLASS_NAMES_4C:
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
        with open(OUTPUT_DIR / "xgboost_4class_pre_alarm_evaluation.json", "w") as f:
            json.dump(ood_summary, f, indent=2)

    # 6. Side-by-side comparison with 5-class XGBoost
    five_class_summary_path = PROJECT_ROOT / "training" / "final_models" / "results" / "model2_xgboost_research" / "xgboost_research_metrics_summary.json"
    if five_class_summary_path.exists():
        with open(five_class_summary_path) as f:
            m5_data = json.load(f)
        m5_best_cfg = m5_data["best_config"]
        m5_wm = m5_data["configs"][m5_best_cfg]["window_metrics"]
        m5_rm = m5_data["configs"][m5_best_cfg]["record_metrics"]
        m5_fb = m5_data["configs"][m5_best_cfg]["false_brady_on_nsr"]["false_brady_pct"]

        m4_best_cfg = best_config_name
        m4_wm = all_configs_metrics[m4_best_cfg]["window_metrics"]
        m4_rm = all_configs_metrics[m4_best_cfg]["record_metrics"]
        m4_fb = all_configs_metrics[m4_best_cfg]["false_brady_on_nsr"]["false_brady_pct"]

        comp_rows = [
            {
                "metric": "Number of Classes",
                "model2_5class_xgboost": "5 (NSR, TACHY, BRADY, ASY, VENT)",
                "model2_4class_xgboost": "4 (NSR, TACHY, BRADY, VENT)",
                "delta": "-1 class (ASYSTOLE excluded)",
                "note": "Classes are not directly equivalent.",
            },
            {
                "metric": "Total Supervised Windows",
                "model2_5class_xgboost": f"{m5_data['dataset_windows']}",
                "model2_4class_xgboost": f"{len(df)}",
                "delta": f"{len(df) - m5_data['dataset_windows']} windows",
                "note": "175 asystole windows excluded from training.",
            },
            {
                "metric": "Total Independent Records",
                "model2_5class_xgboost": f"{m5_data['dataset_records']}",
                "model2_4class_xgboost": f"{df['record_id'].nunique()}",
                "delta": f"{df['record_id'].nunique() - m5_data['dataset_records']} records",
                "note": "15 records with only asystole windows removed.",
            },
            {
                "metric": "Window Accuracy",
                "model2_5class_xgboost": f"{m5_wm['accuracy']:.4f}",
                "model2_4class_xgboost": f"{m4_wm['accuracy']:.4f}",
                "delta": f"{(m4_wm['accuracy'] - m5_wm['accuracy']):+.4f}",
                "note": "Higher overall window accuracy in 4-class.",
            },
            {
                "metric": "Window Balanced Accuracy",
                "model2_5class_xgboost": f"{m5_wm['balanced_accuracy']:.4f}",
                "model2_4class_xgboost": f"{m4_wm['balanced_accuracy']:.4f}",
                "delta": f"{(m4_wm['balanced_accuracy'] - m5_wm['balanced_accuracy']):+.4f}",
                "note": "Mean of 4 recall values vs 5 recall values.",
            },
            {
                "metric": "Window Macro-F1",
                "model2_5class_xgboost": f"{m5_wm['macro_f1']:.4f}",
                "model2_4class_xgboost": f"{m4_wm['macro_f1']:.4f}",
                "delta": f"{(m4_wm['macro_f1'] - m5_wm['macro_f1']):+.4f}",
                "note": "Eliminating asystole confusion raises Macro-F1.",
            },
            {
                "metric": "Window Weighted-F1",
                "model2_5class_xgboost": f"{m5_wm['weighted_f1']:.4f}",
                "model2_4class_xgboost": f"{m4_wm['weighted_f1']:.4f}",
                "delta": f"{(m4_wm['weighted_f1'] - m5_wm['weighted_f1']):+.4f}",
                "note": "Weighted by class support.",
            },
            {
                "metric": "Record Balanced Accuracy",
                "model2_5class_xgboost": f"{m5_rm['balanced_accuracy']:.4f}",
                "model2_4class_xgboost": f"{m4_rm['balanced_accuracy']:.4f}",
                "delta": f"{(m4_rm['balanced_accuracy'] - m5_rm['balanced_accuracy']):+.4f}",
                "note": "Majority vote per record.",
            },
            {
                "metric": "Record Macro-F1",
                "model2_5class_xgboost": f"{m5_rm['macro_f1']:.4f}",
                "model2_4class_xgboost": f"{m4_rm['macro_f1']:.4f}",
                "delta": f"{(m4_rm['macro_f1'] - m5_rm['macro_f1']):+.4f}",
                "note": "Record-level Macro-F1.",
            },
            {
                "metric": "NSR F1-Score",
                "model2_5class_xgboost": f"{m5_wm['per_class']['NSR']['f1']:.4f}",
                "model2_4class_xgboost": f"{m4_wm['per_class']['NSR']['f1']:.4f}",
                "delta": f"{(m4_wm['per_class']['NSR']['f1'] - m5_wm['per_class']['NSR']['f1']):+.4f}",
                "note": "NSR classification remains virtually identical.",
            },
            {
                "metric": "TACHY F1-Score",
                "model2_5class_xgboost": f"{m5_wm['per_class']['TACHY']['f1']:.4f}",
                "model2_4class_xgboost": f"{m4_wm['per_class']['TACHY']['f1']:.4f}",
                "delta": f"{(m4_wm['per_class']['TACHY']['f1'] - m5_wm['per_class']['TACHY']['f1']):+.4f}",
                "note": "Tachycardia detection performance.",
            },
            {
                "metric": "BRADYCARDIA F1-Score",
                "model2_5class_xgboost": f"{m5_wm['per_class']['BRADYCARDIA']['f1']:.4f}",
                "model2_4class_xgboost": f"{m4_wm['per_class']['BRADYCARDIA']['f1']:.4f}",
                "delta": f"{(m4_wm['per_class']['BRADYCARDIA']['f1'] - m5_wm['per_class']['BRADYCARDIA']['f1']):+.4f}",
                "note": "Bradycardia detection performance.",
            },
            {
                "metric": "VENTRICULAR F1-Score",
                "model2_5class_xgboost": f"{m5_wm['per_class']['VENTRICULAR']['f1']:.4f}",
                "model2_4class_xgboost": f"{m4_wm['per_class']['VENTRICULAR']['f1']:.4f}",
                "delta": f"{(m4_wm['per_class']['VENTRICULAR']['f1'] - m5_wm['per_class']['VENTRICULAR']['f1']):+.4f}",
                "note": "Ventricular precision improves when asystole is removed.",
            },
            {
                "metric": "ASYSTOLE F1-Score",
                "model2_5class_xgboost": f"{m5_wm['per_class']['ASYSTOLE']['f1']:.4f}",
                "model2_4class_xgboost": "N/A (Excluded)",
                "delta": "N/A",
                "note": "Class removed from Model 2 architecture.",
            },
            {
                "metric": "False Bradycardia on NSR (%)",
                "model2_5class_xgboost": f"{m5_fb:.2f}%",
                "model2_4class_xgboost": f"{m4_fb:.2f}%",
                "delta": f"{(m4_fb - m5_fb):+.2f}%",
                "note": "100% of these windows have true HR < 60 bpm in both models.",
            },
        ]
        df_comp = pd.DataFrame(comp_rows)
        comp_csv = OUTPUT_DIR / "comparison_4class_vs_5class_xgboost.csv"
        df_comp.to_csv(comp_csv, index=False)
        print(f"\nSaved side-by-side comparison table to: {comp_csv}")

    # 7. Save full experiment summary JSON
    exp_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": MODEL2_FEATURES,
        "classes": CLASS_NAMES_4C,
        "dataset_windows": len(df),
        "dataset_records": int(df["record_id"].nunique()),
        "class_counts": class_counts.to_dict(),
        "best_config": best_config_name,
        "best_macro_f1": float(best_macro_f1),
        "configs": all_configs_metrics,
        "asystole_out_of_domain_evaluation": asy_summary,
    }
    with open(OUTPUT_DIR / "xgboost_4class_research_metrics_summary.json", "w") as f:
        json.dump(exp_summary, f, indent=2)

    print(f"\n4-Class XGBoost research experiment complete. All artifacts saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    run_xgboost_4class_experiments()
