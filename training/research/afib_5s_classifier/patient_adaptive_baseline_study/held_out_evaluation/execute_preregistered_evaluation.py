"""
Execution Script: Preregistered Held-Out Evaluation of Causal Patient-Adaptive AFib Classifier
=============================================================================================
Protocol: training/research/afib_5s_classifier/patient_adaptive_baseline_study/PREREGISTERED_HELD_OUT_EVALUATION_PLAN.md

Rules Enforced:
1. Model weights, scaler, and baseline parameters are fitted EXCLUSIVELY on 15 training patients (N=217,851).
2. Held-out test data (4 patients, N=29,260 windows) are evaluated SINGLE-SHOT with ZERO parameter tuning.
3. Causal window processing with strict per-record boundary resets.
4. Comprehensive reporting at frozen thresholds (theta* = 0.560 and theta = 0.760) pooled and per-patient.
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

# Directory Setup
HELD_OUT_DIR = Path(__file__).resolve().parent
STUDY_DIR = HELD_OUT_DIR.parent
RESEARCH_DIR = STUDY_DIR.parent
DATA_DIR = RESEARCH_DIR / "data"

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from baseline_tracker import CausalBaselineTracker, extract_adaptive_features_for_dataframe

TRAIN_CSV = DATA_DIR / "afdb_5s_features_train.csv"
VAL_CSV = DATA_DIR / "afdb_5s_features_val.csv"
TEST_CSV = DATA_DIR / "afdb_5s_features_test.csv"

STRIDE_SEC_TEST = 5.0  # Test set uses 5.0s non-overlapping windows

FEATURE_COLS_ABSOLUTE = [
    "n_beats", "hr_mean_bpm", "hr_std_bpm", "rr_cv", "rr_rmssd", "rr_mad", "rr_pnn50",
    "rr_sample_dispersion", "rr_turning_point_ratio", "alternating_sign_index",
    "f_band_power_ratio", "f_band_flatness", "f_band_peak_purity", "p_band_power_ratio",
    "qrs_power_ratio", "hjorth_mobility", "hjorth_complexity", "ac_std_mv", "ac_ptp_mv",
    "zero_crossings", "lz_complexity"
]

FEATURE_COLS_ADAPTIVE = [
    "delta_hr_bpm",
    "rr_cv_relative",
    "hr_z_robust",
    "afib_index_relative",
    "baseline_warmup_fraction"
]

FEATURE_COLS_COMBINED = FEATURE_COLS_ABSOLUTE + FEATURE_COLS_ADAPTIVE


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, non_afib_hours: float) -> Dict[str, Any]:
    """Computes comprehensive binary classification and false alarm metrics."""
    # Check if single class
    unique_classes = np.unique(y_true)
    is_single_class = (len(unique_classes) < 2)

    if is_single_class:
        cls = unique_classes[0]
        if cls == 1:
            # All positive (like record 07162)
            tp = int(np.sum(y_pred == 1))
            fn = int(np.sum(y_pred == 0))
            fp = 0
            tn = 0
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            return {
                "tp": tp, "fp": 0, "fn": fn, "tn": 0,
                "sensitivity": round(sens, 4),
                "specificity": None,
                "precision": 1.0 if tp > 0 else 0.0,
                "f1_score": round(float(2 * tp / (2 * tp + fn)), 4) if (2 * tp + fn) > 0 else 0.0,
                "balanced_accuracy": None,
                "roc_auc": None,
                "pr_auc": None,
                "fp_per_hour": None,
                "false_alarm_count": 0,
                "non_afib_hours": 0.0,
                "note": "Single-class record (100% AFib). Specificity, ROC-AUC, and FP/hr undefined."
            }
        else:
            # All negative
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
            tp = 0
            fn = 0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            fp_hr = float(fp / non_afib_hours) if non_afib_hours > 0 else 0.0
            return {
                "tp": 0, "fp": fp, "fn": 0, "tn": tn,
                "sensitivity": None,
                "specificity": round(spec, 4),
                "precision": 0.0,
                "f1_score": 0.0,
                "balanced_accuracy": None,
                "roc_auc": None,
                "pr_auc": None,
                "fp_per_hour": round(fp_hr, 2),
                "false_alarm_count": fp,
                "non_afib_hours": round(non_afib_hours, 2),
                "note": "Single-class record (100% Non-AFib). Sensitivity and ROC-AUC undefined."
            }

    # Standard binary case
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    fp_hr = float(fp / non_afib_hours) if non_afib_hours > 0 else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "fp_per_hour": round(fp_hr, 2),
        "false_alarm_count": int(fp),
        "non_afib_hours": round(non_afib_hours, 2),
        "note": "Dual-class record. All metrics defined."
    }


def evaluate_cohort(
    y_true: np.ndarray,
    probs: np.ndarray,
    df_meta: pd.DataFrame,
    stride_sec: float = 5.0
) -> Dict[str, Any]:
    """Evaluates cohort at frozen operating threshold 0.560 and triage threshold 0.760."""
    non_afib_hours = (len(y_true) - y_true.sum()) * stride_sec / 3600.0

    preds_560 = (probs >= 0.560).astype(int)
    preds_760 = (probs >= 0.760).astype(int)

    pooled_560 = compute_binary_metrics(y_true, preds_560, probs, non_afib_hours)
    pooled_760 = compute_binary_metrics(y_true, preds_760, probs, non_afib_hours)

    # Per-patient breakdown
    patient_breakdown = {}
    for rec_id, group in df_meta.groupby("record_id"):
        rec_idx = group.index
        y_p = y_true[rec_idx]
        prob_p = probs[rec_idx]
        p_non_afib_hrs = (len(y_p) - y_p.sum()) * stride_sec / 3600.0

        p_560 = compute_binary_metrics(y_p, (prob_p >= 0.560).astype(int), prob_p, p_non_afib_hrs)
        p_760 = compute_binary_metrics(y_p, (prob_p >= 0.760).astype(int), prob_p, p_non_afib_hrs)

        patient_breakdown[rec_id] = {
            "total_windows": len(y_p),
            "afib_windows": int(y_p.sum()),
            "non_afib_windows": int(len(y_p) - y_p.sum()),
            "afib_prevalence_pct": round(float(y_p.mean() * 100.0), 2),
            "metrics_at_theta_0_560": p_560,
            "metrics_at_theta_0_760": p_760
        }

    # Warmup analysis
    warmup_mask = df_meta["baseline_warmup_fraction"].values < 1.0
    warm_mask = ~warmup_mask
    warmup_analysis = {
        "warmup_windows_count": int(np.sum(warmup_mask)),
        "warmup_windows_pct": round(float(np.mean(warmup_mask) * 100.0), 2),
        "post_warmup_windows_count": int(np.sum(warm_mask)),
        "warmup_subset_metrics_0_560": compute_binary_metrics(
            y_true[warmup_mask],
            preds_560[warmup_mask],
            probs[warmup_mask],
            (len(y_true[warmup_mask]) - y_true[warmup_mask].sum()) * stride_sec / 3600.0
        ) if np.sum(warmup_mask) > 0 else {},
        "post_warmup_subset_metrics_0_560": compute_binary_metrics(
            y_true[warm_mask],
            preds_560[warm_mask],
            probs[warm_mask],
            (len(y_true[warm_mask]) - y_true[warm_mask].sum()) * stride_sec / 3600.0
        )
    }

    return {
        "pooled_fixed_threshold_0_560": pooled_560,
        "pooled_triage_threshold_0_760": pooled_760,
        "per_patient_breakdown": patient_breakdown,
        "warmup_analysis": warmup_analysis
    }


def main():
    print("=" * 80)
    print("PREREGISTERED HELD-OUT TEST EVALUATION: CAUSAL PATIENT-ADAPTIVE AFIB CLASSIFIER")
    print("=" * 80)
    t0 = time.time()

    # Step 1: Pre-flight Audit & Verification of Frozen Specifications
    print("\n[1/5] Pre-Flight Verification & Checksum Audit:")
    for path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        print(f"  {path.name:<30} SHA256: {compute_sha256(path)[:16]}... ({path.stat().st_size:,} bytes)")

    df_train = pd.read_csv(TRAIN_CSV, dtype={"record_id": str})
    df_test = pd.read_csv(TEST_CSV, dtype={"record_id": str})

    print(f"  Training samples: {len(df_train):,} windows across {df_train['record_id'].nunique()} patients")
    print(f"  Held-Out samples: {len(df_test):,} windows across {df_test['record_id'].nunique()} patients")

    train_pts = set(df_train["patient_id"].unique())
    test_pts = set(df_test["patient_id"].unique())
    assert train_pts.isdisjoint(test_pts), "FATAL: Overlap between Train and Held-Out Test patients!"
    print("  [AUDIT PASS] 100% Patient Disjointness between Train and Held-Out Test confirmed.")

    # Step 2: Fit Frozen Models on Training Data
    print("\n[2/5] Fitting Models Strictly on Training Data (Zero Test Exposure):")
    # Extract adaptive features on training data
    df_adapt_train = extract_adaptive_features_for_dataframe(df_train)
    df_train_full = pd.concat([df_train, df_adapt_train], axis=1)
    y_train = df_train_full["is_afib"].values

    # Fit Scalers
    scaler_A = StandardScaler()
    X_train_A = scaler_A.fit_transform(df_train_full[FEATURE_COLS_ABSOLUTE].values)

    scaler_C = StandardScaler()
    X_train_C = scaler_C.fit_transform(df_train_full[FEATURE_COLS_COMBINED].values)

    # Fit Logistic Regression models
    lr_A = LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    lr_A.fit(X_train_A, y_train)

    lr_C = LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    lr_C.fit(X_train_C, y_train)

    # Fit HistGradientBoosting models
    hgb_A = HistGradientBoostingClassifier(max_iter=100, max_depth=6, class_weight="balanced", random_state=42)
    hgb_A.fit(X_train_A, y_train)

    hgb_C = HistGradientBoostingClassifier(max_iter=100, max_depth=6, class_weight="balanced", random_state=42)
    hgb_C.fit(X_train_C, y_train)

    # Freeze Model Package
    frozen_pkg_path = HELD_OUT_DIR / "frozen_models_package.joblib"
    joblib.dump({
        "scaler_Config_A": scaler_A,
        "scaler_Config_C": scaler_C,
        "model_lr_Config_A": lr_A,
        "model_lr_Config_C": lr_C,
        "model_hgb_Config_A": hgb_A,
        "model_hgb_Config_C": hgb_C,
        "feature_cols_A": FEATURE_COLS_ABSOLUTE,
        "feature_cols_C": FEATURE_COLS_COMBINED
    }, frozen_pkg_path)
    print(f"  Frozen model package saved to {frozen_pkg_path.name} (SHA256: {compute_sha256(frozen_pkg_path)[:16]}...)")

    # Step 3: Extract Causal Adaptive Features on Held-Out Test Windows
    print("\n[3/5] Extracting Causal Adaptive Features on Held-Out Test Windows:")
    t_extract0 = time.time()
    df_adapt_test = extract_adaptive_features_for_dataframe(df_test)
    print(f"  Extracted adaptive features on {len(df_adapt_test):,} test windows in {time.time() - t_extract0:.1f}s")

    df_test_full = pd.concat([df_test, df_adapt_test], axis=1)
    y_test = df_test_full["is_afib"].values

    # Step 4: Evaluate Models on Held-Out Test Windows
    print("\n[4/5] Executing Held-Out Test Predictions & Evaluation:")
    X_test_A = scaler_A.transform(df_test_full[FEATURE_COLS_ABSOLUTE].values)
    X_test_C = scaler_C.transform(df_test_full[FEATURE_COLS_COMBINED].values)

    probs_lr_A = lr_A.predict_proba(X_test_A)[:, 1]
    probs_lr_C = lr_C.predict_proba(X_test_C)[:, 1]

    probs_hgb_A = hgb_A.predict_proba(X_test_A)[:, 1]
    probs_hgb_C = hgb_C.predict_proba(X_test_C)[:, 1]

    # Evaluate
    eval_lr_A = evaluate_cohort(y_test, probs_lr_A, df_test_full, stride_sec=STRIDE_SEC_TEST)
    eval_lr_C = evaluate_cohort(y_test, probs_lr_C, df_test_full, stride_sec=STRIDE_SEC_TEST)

    eval_hgb_A = evaluate_cohort(y_test, probs_hgb_A, df_test_full, stride_sec=STRIDE_SEC_TEST)
    eval_hgb_C = evaluate_cohort(y_test, probs_hgb_C, df_test_full, stride_sec=STRIDE_SEC_TEST)

    # Step 5: Save Results and Print Summary
    print("\n[5/5] Saving Evaluation Outputs & Generating Report Data:")
    
    # Save Prediction Table
    pred_table = df_test_full[[
        "record_id", "patient_id", "window_idx", "start_time_sec", "end_time_sec",
        "rhythm_label", "is_afib", "baseline_warmup_fraction",
        "hr_mean_bpm", "delta_hr_bpm", "rr_cv", "rr_cv_relative"
    ]].copy()
    pred_table["prob_lr_Config_A"] = np.round(probs_lr_A, 4)
    pred_table["prob_lr_Config_C"] = np.round(probs_lr_C, 4)
    pred_table["prob_hgb_Config_A"] = np.round(probs_hgb_A, 4)
    pred_table["prob_hgb_Config_C"] = np.round(probs_hgb_C, 4)

    preds_path = HELD_OUT_DIR / "held_out_test_predictions.csv"
    pred_table.to_csv(preds_path, index=False)
    print(f"  Saved prediction table to {preds_path.name}")

    # Save Metrics JSON
    results_json = {
        "protocol": "PREREGISTERED_HELD_OUT_EVALUATION_PLAN.md",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_windows_count": len(y_test),
        "test_afib_count": int(y_test.sum()),
        "test_non_afib_count": int(len(y_test) - y_test.sum()),
        "test_non_afib_hours": round(float((len(y_test) - y_test.sum()) * STRIDE_SEC_TEST / 3600.0), 4),
        "Config_A_Absolute_Only": {
            "logistic_regression": eval_lr_A,
            "hist_gradient_boosting": eval_hgb_A
        },
        "Config_C_Combined_Adaptive": {
            "logistic_regression": eval_lr_C,
            "hist_gradient_boosting": eval_hgb_C
        }
    }

    metrics_path = HELD_OUT_DIR / "held_out_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved metrics to {metrics_path.name}")

    # Print Comparative Results Table
    print("\n" + "=" * 95)
    print("HELD-OUT TEST EMPIRICAL EVALUATION RESULTS (N = 29,260 WINDOWS, 19.24 NON-AFIB HOURS)")
    print("=" * 95)
    print(f"{'Configuration / Model':<35} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Sens(0.56)':<10} | {'Spec(0.56)':<10} | {'FP/hr(0.56)':<10}")
    print("-" * 95)

    m_lr_A = eval_lr_A["pooled_fixed_threshold_0_560"]
    m_lr_C = eval_lr_C["pooled_fixed_threshold_0_560"]
    m_hgb_A = eval_hgb_A["pooled_fixed_threshold_0_560"]
    m_hgb_C = eval_hgb_C["pooled_fixed_threshold_0_560"]

    print(f"{'Config A (Absolute) - LogReg':<35} | {m_lr_A['roc_auc']:<8.4f} | {m_lr_A['pr_auc']:<8.4f} | {m_lr_A['sensitivity']*100:<9.2f}% | {m_lr_A['specificity']*100:<9.2f}% | {m_lr_A['fp_per_hour']:<10.2f}")
    print(f"{'Config C (Combined) - LogReg':<35} | {m_lr_C['roc_auc']:<8.4f} | {m_lr_C['pr_auc']:<8.4f} | {m_lr_C['sensitivity']*100:<9.2f}% | {m_lr_C['specificity']*100:<9.2f}% | {m_lr_C['fp_per_hour']:<10.2f}")
    print("-" * 95)
    print(f"{'Config A (Absolute) - HistGrad':<35} | {m_hgb_A['roc_auc']:<8.4f} | {m_hgb_A['pr_auc']:<8.4f} | {m_hgb_A['sensitivity']*100:<9.2f}% | {m_hgb_A['specificity']*100:<9.2f}% | {m_hgb_A['fp_per_hour']:<10.2f}")
    print(f"{'Config C (Combined) - HistGrad':<35} | {m_hgb_C['roc_auc']:<8.4f} | {m_hgb_C['pr_auc']:<8.4f} | {m_hgb_C['sensitivity']*100:<9.2f}% | {m_hgb_C['specificity']*100:<9.2f}% | {m_hgb_C['fp_per_hour']:<10.2f}")
    print("=" * 95)

    # Per-Patient Breakdown Comparison
    print("\nPER-PATIENT HELD-OUT PERFORMANCE BREAKDOWN (LOGISTIC REGRESSION AT THETA = 0.560):")
    test_records = ["04908", "07162", "08215", "08219"]
    header = f"{'Record':<10} | {'Rhythm Profile':<18} | {'AUC (Orig)':<10} | {'AUC (Adapt)':<11} | {'Sens (Orig)':<11} | {'Sens (Adapt)':<12} | {'FP/hr (Orig)':<12} | {'FP/hr (Adapt)':<13}"
    print(header)
    print("-" * len(header))

    for rec in test_records:
        pa = eval_lr_A["per_patient_breakdown"][rec]["metrics_at_theta_0_560"]
        pc = eval_lr_C["per_patient_breakdown"][rec]["metrics_at_theta_0_560"]
        prev = eval_lr_A["per_patient_breakdown"][rec]["afib_prevalence_pct"]

        auc_a_str = f"{pa['roc_auc']:.4f}" if pa['roc_auc'] is not None else "N/A (100%)"
        auc_c_str = f"{pc['roc_auc']:.4f}" if pc['roc_auc'] is not None else "N/A (100%)"

        sens_a_str = f"{pa['sensitivity']*100:.2f}%" if pa['sensitivity'] is not None else "N/A"
        sens_c_str = f"{pc['sensitivity']*100:.2f}%" if pc['sensitivity'] is not None else "N/A"

        fp_a_str = f"{pa['fp_per_hour']:.2f}" if pa['fp_per_hour'] is not None else "0.00"
        fp_c_str = f"{pc['fp_per_hour']:.2f}" if pc['fp_per_hour'] is not None else "0.00"

        profile = f"AFib: {prev:.1f}%"
        print(f"{rec:<10} | {profile:<18} | {auc_a_str:<10} | {auc_c_str:<11} | {sens_a_str:<11} | {sens_c_str:<12} | {fp_a_str:<12} | {fp_c_str:<13}")

    print("-" * len(header))
    dt = time.time() - t0
    print(f"\nExecution finished in {dt:.1f}s.")


if __name__ == "__main__":
    main()
