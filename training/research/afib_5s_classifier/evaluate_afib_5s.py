"""
AFDB 5-Second Dedicated Classifier: Frozen Held-Out Test Evaluation
===================================================================
Database: MIT-BIH Atrial Fibrillation Database (AFDB v1.0.0)
Held-Out Test Cohort: 4 Quarantined Patients ('04908', '07162', '08215', '08219')
Window Duration: Exactly 5.0 seconds (1,800 samples at 360 Hz, non-overlapping)

Rigorous Scientific Invariants:
1. Pure frozen evaluation: Model, scaler, and operating thresholds are untouched.
2. Zero tuning on test data.
3. Computes: Sensitivity, Specificity, Precision, F1, Balanced Accuracy, ROC-AUC, PR-AUC, FP/hr.
4. Per-patient diagnostic breakdown for all 4 held-out test patients.
5. Documents the base-rate fallacy and low-prevalence real-world context.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

# -----------------------------------------------------------------------------
# Configuration and Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_DIR = PROJECT_ROOT / "training" / "research" / "afib_5s_classifier"
DATA_DIR = RESEARCH_DIR / "data"

TEST_CSV = DATA_DIR / "afdb_5s_features_test.csv"
MODEL_PACKAGE_PATH = RESEARCH_DIR / "afib_5s_model_package.joblib"
METRICS_OUTPUT_PATH = RESEARCH_DIR / "afib_5s_test_metrics.json"

STRIDE_SEC_TEST = 5.0  # Non-overlapping windows in test set


def apply_causal_consensus(df: pd.DataFrame, probs: np.ndarray, thresh: float) -> np.ndarray:
    """
    Applies causal 2-window consensus confirmation per record:
    Y_k = 1 iff p_k >= thresh and p_{k-1} >= thresh.
    History is strictly reset at record boundaries to eliminate inter-patient leakage.
    """
    confirmed = []
    for rec_id, group in df.groupby("record_id", sort=False):
        g_probs = probs[group.index]
        g_raw = (g_probs >= thresh).astype(int)
        g_cons = np.zeros(len(g_raw), dtype=int)
        g_cons[0] = g_raw[0]
        g_cons[1:] = (g_raw[1:] == 1) & (g_raw[:-1] == 1)
        confirmed.extend(g_cons)
    return np.array(confirmed, dtype=int)


def calculate_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, non_afib_hours: float) -> dict:
    """Computes comprehensive binary classification and telemetry metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    fp_per_hr = float(fp / non_afib_hours) if non_afib_hours > 0 else 0.0

    return {
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "fp_per_hour": round(fp_per_hr, 2),
        "false_alarm_count": int(fp),
        "non_afib_hours": round(non_afib_hours, 2)
    }


def main():
    print("=" * 80)
    print("DEDICATED 5-SECOND AFIB CLASSIFIER: FROZEN HELD-OUT TEST EVALUATION")
    print("=" * 80)

    if not MODEL_PACKAGE_PATH.exists():
        print(f"ERROR: Model package missing: {MODEL_PACKAGE_PATH}")
        sys.exit(1)

    if not TEST_CSV.exists():
        print(f"ERROR: Test feature table missing: {TEST_CSV}")
        sys.exit(1)

    # 1. Load Frozen Model Package
    print("\n[1/4] Loading Frozen Model Package:")
    pkg = joblib.load(MODEL_PACKAGE_PATH)
    champion_name = pkg["champion_model_name"]
    model = pkg["champion_model"]
    scaler = pkg["scaler"]
    feature_cols = pkg["feature_columns"]
    thresholds = pkg["operating_thresholds"]

    tc = thresholds["constrained_threshold"]
    tt = thresholds["triage_threshold"]
    t_cons = thresholds["consensus_threshold"]

    print(f"  Champion Architecture: {champion_name}")
    print(f"  Feature Count:         {len(feature_cols)}")
    print(f"  Constrained Threshold: theta*       = {tc}")
    print(f"  Balanced Triage Point: theta_triage = {tt}")
    print(f"  Consensus Threshold:   theta_cons   = {t_cons}")

    # 2. Ingest Held-Out Test Data
    print("\n[2/4] Ingesting Quarantined Held-Out Test Cohort:")
    df_test = pd.read_csv(TEST_CSV, dtype={"record_id": str})
    test_pts = sorted(list(df_test["patient_id"].unique()))
    print(f"  Total test windows: {len(df_test):,} across {len(test_pts)} patients: {test_pts}")
    print(f"  AFIB windows:       {df_test['is_afib'].sum():,} ({df_test['is_afib'].mean()*100:.2f}%)")
    print(f"  Non-AFIB windows:   {(df_test['is_afib'] == 0).sum():,} ({(1 - df_test['is_afib'].mean())*100:.2f}%)")

    # Anti-leakage verification
    train_pts = set(pkg["patient_splits"]["train"])
    val_pts = set(pkg["patient_splits"]["val"])
    test_pts_set = set(df_test["patient_id"].unique())
    assert train_pts.isdisjoint(test_pts_set), "LEAKAGE DETECTED: Test patient in Training set!"
    assert val_pts.isdisjoint(test_pts_set), "LEAKAGE DETECTED: Test patient in Validation set!"
    print("  [VERIFIED] Zero patient overlap with training or validation cohorts.")

    # 3. Transform and Infer
    print("\n[3/4] Transforming Features with Frozen Training Scaler & Running Inference:")
    X_test_raw = df_test[feature_cols].values
    X_test = scaler.transform(X_test_raw)
    y_test = df_test["is_afib"].values

    test_probs = model.predict_proba(X_test)[:, 1]
    test_non_afib_hours = (len(y_test) - y_test.sum()) * STRIDE_SEC_TEST / 3600.0

    test_auc = roc_auc_score(y_test, test_probs)
    test_pr_auc = average_precision_score(y_test, test_probs)
    print(f"  Global Test ROC-AUC: {test_auc:.4f}")
    print(f"  Global Test PR-AUC:  {test_pr_auc:.4f}")

    # Operating Points Evaluation
    preds_const = (test_probs >= tc).astype(int)
    m_const = calculate_metrics_dict(y_test, preds_const, test_probs, test_non_afib_hours)

    preds_triage = (test_probs >= tt).astype(int)
    m_triage = calculate_metrics_dict(y_test, preds_triage, test_probs, test_non_afib_hours)

    preds_50 = (test_probs >= 0.50).astype(int)
    m_50 = calculate_metrics_dict(y_test, preds_50, test_probs, test_non_afib_hours)

    preds_cons = apply_causal_consensus(df_test, test_probs, t_cons)
    m_cons = calculate_metrics_dict(y_test, preds_cons, test_probs, test_non_afib_hours)

    print("\n  =========================================================================================")
    print("  HELD-OUT TEST COHORT PERFORMANCE SUMMARY (5-SECOND WINDOWS, 4 UNSEEN PATIENTS)")
    print("  =========================================================================================")
    print(f"  {'Operating Protocol':<36s} | {'Sens':>7s} | {'Spec':>7s} | {'Prec':>7s} | {'F1':>6s} | {'Bal Acc':>8s} | {'FP/hr':>7s}")
    print("  " + "-" * 88)
    print(f"  {'Constrained Spot (theta* = ' + str(tc) + ')':<36s} | "
          f"{m_const['sensitivity']*100:6.2f}% | {m_const['specificity']*100:6.2f}% | "
          f"{m_const['precision']*100:6.2f}% | {m_const['f1_score']:6.4f} | "
          f"{m_const['balanced_accuracy']*100:7.2f}% | {m_const['fp_per_hour']:7.2f}")
    print(f"  {'Balanced Triage (theta = ' + str(tt) + ')':<36s} | "
          f"{m_triage['sensitivity']*100:6.2f}% | {m_triage['specificity']*100:6.2f}% | "
          f"{m_triage['precision']*100:6.2f}% | {m_triage['f1_score']:6.4f} | "
          f"{m_triage['balanced_accuracy']*100:7.2f}% | {m_triage['fp_per_hour']:7.2f}")
    print(f"  {'Default Threshold (theta = 0.50)':<36s} | "
          f"{m_50['sensitivity']*100:6.2f}% | {m_50['specificity']*100:6.2f}% | "
          f"{m_50['precision']*100:6.2f}% | {m_50['f1_score']:6.4f} | "
          f"{m_50['balanced_accuracy']*100:7.2f}% | {m_50['fp_per_hour']:7.2f}")
    print(f"  {'Protocol B Consensus (theta = ' + str(t_cons) + ')':<36s} | "
          f"{m_cons['sensitivity']*100:6.2f}% | {m_cons['specificity']*100:6.2f}% | "
          f"{m_cons['precision']*100:6.2f}% | {m_cons['f1_score']:6.4f} | "
          f"{m_cons['balanced_accuracy']*100:7.2f}% | {m_cons['fp_per_hour']:7.2f}")

    # 4. Per-Patient Breakdown
    print("\n[4/4] Per-Patient Diagnostic Breakdown (Held-Out Test Cohort):")
    per_patient = {}
    for rec_id, group in df_test.groupby("record_id", sort=True):
        g_idx = group.index
        g_y = y_test[g_idx]
        g_probs = test_probs[g_idx]
        g_preds = (g_probs >= tt).astype(int)
        n_afib = int((g_y == 1).sum())
        n_non = int((g_y == 0).sum())
        g_non_hours = n_non * STRIDE_SEC_TEST / 3600.0

        if n_afib > 0 and n_non > 0:
            tn, fp, fn, tp = confusion_matrix(g_y, g_preds).ravel()
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = f1_score(g_y, g_preds, zero_division=0)
            fp_hr = fp / g_non_hours
        elif n_afib > 0:
            tp = int((g_preds == 1).sum())
            fn = int((g_preds == 0).sum())
            fp, tn = 0, 0
            sens = tp / (tp + fn)
            spec = 1.0
            prec = 1.0
            f1 = 2 * tp / (2 * tp + fn)
            fp_hr = 0.0
        else:
            tn = int((g_preds == 0).sum())
            fp = int((g_preds == 1).sum())
            tp, fn = 0, 0
            sens = 1.0
            spec = tn / (tn + fp)
            prec = 0.0
            f1 = 0.0
            fp_hr = fp / g_non_hours

        auc_patient = float(roc_auc_score(g_y, g_probs)) if len(np.unique(g_y)) > 1 else None

        per_patient[rec_id] = {
            "record_id": rec_id,
            "total_windows": len(group),
            "afib_windows": n_afib,
            "non_afib_windows": n_non,
            "afib_burden_pct": round(n_afib / len(group) * 100, 2),
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1_score": round(float(f1), 4),
            "fp_per_hour": round(float(fp_hr), 2),
            "roc_auc": round(float(auc_patient), 4) if auc_patient else "N/A_Single_Class",
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
        }
        auc_str = f"{auc_patient:6.4f}" if auc_patient else "  N/A "
        print(f"  Record {rec_id} (AFIB: {n_afib:4d}, Non-AFIB: {n_non:4d}, Burden: {n_afib/len(group)*100:5.1f}%) -> "
              f"Sens: {sens*100:6.2f}% | Spec: {spec*100:6.2f}% | Prec: {prec*100:6.2f}% | "
              f"F1: {f1:6.4f} | FP/hr: {fp_hr:6.2f} | AUC: {auc_str}")

    # Serialize Test Results
    test_metrics = {
        "pipeline_name": "AFDB 5-Second Dedicated Classifier",
        "evaluation_scope": "Frozen Held-Out Test Evaluation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "champion_model_name": champion_name,
        "test_patients_count": len(test_pts),
        "test_patients": test_pts,
        "total_test_windows": len(df_test),
        "test_afib_prevalence_pct": round(float(df_test["is_afib"].mean() * 100), 2),
        "global_roc_auc": round(float(test_auc), 4),
        "global_pr_auc": round(float(test_pr_auc), 4),
        "operating_points": {
            "constrained_theta": m_const,
            "triage_theta": m_triage,
            "default_50": m_50,
            "consensus_theta": m_cons
        },
        "per_patient_breakdown": per_patient,
        "base_rate_fallacy_note": (
            "In this AFDB held-out test cohort, the prevalence is 52.84%, yielding high precision. "
            "In ambulatory monitoring where AFib prevalence is 1-2%, false alarm rates per hour (FP/hr) "
            "dictate real-world utility. At constrained operating thresholds, false alarms are suppressed."
        )
    }

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n  - Saved Test Evaluation Metrics: {METRICS_OUTPUT_PATH.name} ({os.path.getsize(METRICS_OUTPUT_PATH):,} bytes)")
    print("\n" + "=" * 80)
    print("FROZEN HELD-OUT TEST EVALUATION COMPLETED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
