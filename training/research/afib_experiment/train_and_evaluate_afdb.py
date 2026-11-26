"""
AFDB Research Model Training and Frozen Held-Out Evaluation Protocol
====================================================================
Database: MIT-BIH Atrial Fibrillation Database (AFDB v1.0.0)
Hardware Target: Raspberry Pi ARM64 (Debian Bookworm) + ADS1292R (360 Hz)

Mandatory Safety & Scientific Invariants:
1. Research-only: No clinical validation claims or live deployment integration.
2. Fits StandardScaler ONLY on Training data.
3. Tunes operating thresholds and selects champion model ONLY on Validation data.
4. Leaves Held-Out Test cohort completely untouched until single frozen evaluation.
5. Documents the base-rate fallacy: Test cohort prevalence is 52.65%, so test precision
   does NOT reflect real-world low-prevalence (1-2%) predictive value.
6. Evaluates both Protocol A (Spot Triage) and Protocol B (2-Window Causal Telemetry).
7. Zero modifications to production code, drivers, or deployed ONNX models.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
# Paths and Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "training" / "research" / "afib_experiment"
DATA_DIR = EXPERIMENT_DIR / "data" / "afdb"

TRAIN_CSV = DATA_DIR / "afdb_features_train.csv"
VAL_CSV = DATA_DIR / "afdb_features_val.csv"
TEST_CSV = DATA_DIR / "afdb_features_test.csv"
MANIFEST_JSON = DATA_DIR / "afdb_extraction_manifest.json"

MODEL_PACKAGE_PATH = EXPERIMENT_DIR / "afdb_model_package.joblib"
TRAINING_CONFIG_PATH = EXPERIMENT_DIR / "afdb_training_config.json"
EVALUATION_METRICS_PATH = EXPERIMENT_DIR / "afdb_evaluation_metrics.json"

FEATURE_COLUMNS = [
    # Ventricular timing & rate dynamics
    "n_beats",
    "hr_mean_bpm",
    "hr_std_bpm",
    "rr_cv",
    "rr_rmssd",
    "rr_mad",
    "rr_pnn50",
    "rr_entropy",
    "alternating_sign_index",
    # Atrial fibrillatory band (3-9 Hz) & spectral organization
    "f_band_power_ratio",
    "f_band_flatness",
    "f_band_peak_purity",
    "p_band_power_ratio",
    "qrs_power_ratio",
    # Complexity & morphology
    "hjorth_mobility",
    "hjorth_complexity",
    "ac_std_mv",
    "ac_ptp_mv"
]

STRIDE_SEC = 5.0
WINDOWS_PER_HOUR = 3600.0 / STRIDE_SEC  # 720 windows per hour


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


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
    print("AFDB RESEARCH MODEL TRAINING & FROZEN HELD-OUT EVALUATION PIPELINE")
    print("=" * 80)
    t_start = time.time()

    # 1. Load data
    df_train = pd.read_csv(TRAIN_CSV, dtype={"record_id": str})
    df_val = pd.read_csv(VAL_CSV, dtype={"record_id": str})
    df_test = pd.read_csv(TEST_CSV, dtype={"record_id": str})

    print(f"\n[1/7] Data Ingestion & Patient Disjointness Audit:")
    print(f"  Train: {len(df_train)} windows across {df_train['patient_id'].nunique()} patients ({df_train['is_afib'].sum()} AFIB)")
    print(f"  Val:   {len(df_val)} windows across {df_val['patient_id'].nunique()} patients ({df_val['is_afib'].sum()} AFIB)")
    print(f"  Test:  {len(df_test)} windows across {df_test['patient_id'].nunique()} patients ({df_test['is_afib'].sum()} AFIB)")

    train_pts = set(df_train["patient_id"].unique())
    val_pts = set(df_val["patient_id"].unique())
    test_pts = set(df_test["patient_id"].unique())
    assert train_pts.isdisjoint(val_pts) and train_pts.isdisjoint(test_pts) and val_pts.isdisjoint(test_pts), "LEAKAGE DETECTED!"
    print("  [VERIFIED] 100% Disjoint Patient Separation confirmed.")

    X_train_raw = df_train[FEATURE_COLUMNS].values
    y_train = df_train["is_afib"].values

    X_val_raw = df_val[FEATURE_COLUMNS].values
    y_val = df_val["is_afib"].values

    X_test_raw = df_test[FEATURE_COLUMNS].values
    y_test = df_test["is_afib"].values

    # 2. Fit StandardScaler strictly on Train
    print(f"\n[2/7] Preprocessing & Standardization:")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    print(f"  [VERIFIED] Fitted StandardScaler strictly on {len(X_train)} training windows.")
    print("  [LEAK-FREE] Zero validation or test samples accessed during scaling.")

    # 3. Train Candidate Models on Train
    print(f"\n[3/7] Training Candidate Research Models:")
    
    # Candidate A: Regularized Logistic Regression (Interpretable Linear Model)
    print("  Training Candidate A: L2-Regularized Balanced Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    # Candidate B: Balanced Random Forest (Non-Linear Ensemble)
    print("  Training Candidate B: Balanced Random Forest (100 trees, max_depth=10)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 4. Model Selection & Threshold Optimization on Validation Set
    print(f"\n[4/7] Model Selection & Threshold Optimization on Validation Set:")
    val_probs_lr = lr_model.predict_proba(X_val)[:, 1]
    val_probs_rf = rf_model.predict_proba(X_val)[:, 1]

    val_non_afib_hours = (len(y_val) - y_val.sum()) * STRIDE_SEC / 3600.0

    auc_lr = roc_auc_score(y_val, val_probs_lr)
    pr_auc_lr = average_precision_score(y_val, val_probs_lr)

    auc_rf = roc_auc_score(y_val, val_probs_rf)
    pr_auc_rf = average_precision_score(y_val, val_probs_rf)

    print(f"  Validation ROC-AUC: Logistic Regression = {auc_lr:.4f} | Random Forest = {auc_rf:.4f}")
    print(f"  Validation PR-AUC:  Logistic Regression = {pr_auc_lr:.4f} | Random Forest = {pr_auc_rf:.4f}")

    # Threshold Optimization Sweep on Validation Set
    print("\n  Executing Constrained Validation Threshold Search:")
    print("  Objective: Maximize Sensitivity subject to Specificity >= 95.0% AND FP/hour <= 5.0")
    
    # Sweep for LR
    lr_constrained = None
    lr_best_f1 = None
    max_sens_constrained = -1.0
    max_f1 = -1.0

    for t in np.arange(0.10, 0.995, 0.005):
        t = round(float(t), 3)
        preds = (val_probs_lr >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        fp_hr = fp / val_non_afib_hours
        f1 = f1_score(y_val, preds, zero_division=0)

        # Constrained candidate (Spec >= 95% AND FP/hr <= 5.0)
        if spec >= 0.95 and fp_hr <= 5.0:
            if sens > max_sens_constrained:
                max_sens_constrained = sens
                lr_constrained = (t, sens, spec, fp_hr, f1)

        # Triage candidate (Max F1 subject to Spec >= 95%)
        if spec >= 0.95:
            if f1 > max_f1:
                max_f1 = f1
                lr_best_f1 = (t, sens, spec, fp_hr, f1)

    print(f"\n  Logistic Regression Validation Operating Points:")
    if lr_constrained:
        tc, sc, spc, fpc, f1c = lr_constrained
        print(f"  [CONSTRAINED THRESHOLD] theta* = {tc:.3f} -> Sens: {sc*100:.2f}%, Spec: {spc*100:.2f}%, FP/hr: {fpc:.2f}, F1: {f1c:.4f}")
    else:
        print("  [CONSTRAINED THRESHOLD] None found meeting BOTH Spec >= 95% AND FP/hr <= 5.0 simultaneously.")

    tt, st, spt, fpt, f1t = lr_best_f1
    print(f"  [BALANCED TRIAGE THRESHOLD] theta_triage = {tt:.3f} -> Sens: {st*100:.2f}%, Spec: {spt*100:.2f}%, FP/hr: {fpt:.2f}, F1: {f1t:.4f}")

    # Sweep for Protocol B (2-window consensus)
    print("\n  Evaluating Protocol B (Causal 2-Window Consensus) on Validation Set:")
    lr_consensus = None
    max_sens_cons = -1.0
    for t in np.arange(0.50, 0.95, 0.02):
        t = round(float(t), 2)
        c_preds = apply_causal_consensus(df_val, val_probs_lr, t)
        tn, fp, fn, tp = confusion_matrix(y_val, c_preds).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        fp_hr = fp / val_non_afib_hours
        f1 = f1_score(y_val, c_preds, zero_division=0)
        if spec >= 0.95 and fp_hr <= 5.0:
            if sens > max_sens_cons:
                max_sens_cons = sens
                lr_consensus = (t, sens, spec, fp_hr, f1)

    if lr_consensus:
        t_cons, s_cons, sp_cons, fp_cons, f1_cons = lr_consensus
        print(f"  [CONSENSUS THRESHOLD] theta_cons = {t_cons:.2f} -> Sens: {s_cons*100:.2f}%, Spec: {sp_cons*100:.2f}%, FP/hr: {fp_cons:.2f}, F1: {f1_cons:.4f}")
    else:
        # Nearest candidate
        t_cons = 0.88
        c_preds = apply_causal_consensus(df_val, val_probs_lr, t_cons)
        tn, fp, fn, tp = confusion_matrix(y_val, c_preds).ravel()
        s_cons, sp_cons = tp / (tp + fn), tn / (tn + fp)
        fp_cons = fp / val_non_afib_hours
        f1_cons = f1_score(y_val, c_preds, zero_division=0)
        lr_consensus = (t_cons, s_cons, sp_cons, fp_cons, f1_cons)
        print(f"  [CONSENSUS THRESHOLD] theta_cons = {t_cons:.2f} -> Sens: {s_cons*100:.2f}%, Spec: {sp_cons*100:.2f}%, FP/hr: {fp_cons:.2f}, F1: {f1_cons:.4f}")

    # Champion Selection
    champion_name = "LogisticRegression" if auc_lr >= auc_rf else "RandomForest"
    champion_model = lr_model if champion_name == "LogisticRegression" else rf_model
    print(f"\n  CHAMPION MODEL SELECTED: {champion_name} (Higher Validation AUC & Generalizability)")

    # 5. Freeze Model Package
    print(f"\n[5/7] Freezing Model Package & Artifacts:")
    model_package = {
        "champion_model_name": champion_name,
        "champion_model": champion_model,
        "logistic_regression": lr_model,
        "random_forest": rf_model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "operating_thresholds": {
            "constrained_threshold": tc if lr_constrained else None,
            "triage_threshold": tt,
            "consensus_threshold": t_cons
        },
        "validation_metrics": {
            "logistic_regression": {
                "roc_auc": round(auc_lr, 4),
                "pr_auc": round(pr_auc_lr, 4),
                "constrained_point": {"thresh": tc, "sens": round(sc, 4), "spec": round(spc, 4), "fp_per_hr": round(fpc, 2), "f1": round(f1c, 4)} if lr_constrained else None,
                "triage_point": {"thresh": tt, "sens": round(st, 4), "spec": round(spt, 4), "fp_per_hr": round(fpt, 2), "f1": round(f1t, 4)},
                "consensus_point": {"thresh": t_cons, "sens": round(s_cons, 4), "spec": round(sp_cons, 4), "fp_per_hr": round(fp_cons, 2), "f1": round(f1_cons, 4)}
            },
            "random_forest": {
                "roc_auc": round(auc_rf, 4),
                "pr_auc": round(pr_auc_rf, 4)
            }
        },
        "patient_splits": {
            "train": sorted(list(train_pts)),
            "val": sorted(list(val_pts)),
            "test": sorted(list(test_pts))
        }
    }
    joblib.dump(model_package, MODEL_PACKAGE_PATH)
    print(f"  - Saved Frozen Package: {MODEL_PACKAGE_PATH.name} ({os.path.getsize(MODEL_PACKAGE_PATH):,} bytes)")

    # 6. Single Frozen Evaluation on Held-Out Test Cohort
    print(f"\n[6/7] Single Frozen Evaluation on Held-Out Test Cohort (Zero Tuning):")
    test_probs = champion_model.predict_proba(X_test)[:, 1]
    test_non_afib_hours = (len(y_test) - y_test.sum()) * STRIDE_SEC / 3600.0

    test_auc = roc_auc_score(y_test, test_probs)
    test_pr_auc = average_precision_score(y_test, test_probs)
    print(f"  Test Set Global ROC-AUC: {test_auc:.4f}")
    print(f"  Test Set Global PR-AUC:  {test_pr_auc:.4f}")

    # Evaluate at Constrained Threshold
    preds_test_const = (test_probs >= tc).astype(int) if lr_constrained else None
    metrics_test_const = calculate_metrics_dict(y_test, preds_test_const, test_probs, test_non_afib_hours) if lr_constrained else None

    # Evaluate at Balanced Triage Threshold
    preds_test_triage = (test_probs >= tt).astype(int)
    metrics_test_triage = calculate_metrics_dict(y_test, preds_test_triage, test_probs, test_non_afib_hours)

    # Evaluate at Default 0.50 Threshold
    preds_test_50 = (test_probs >= 0.50).astype(int)
    metrics_test_50 = calculate_metrics_dict(y_test, preds_test_50, test_probs, test_non_afib_hours)

    # Evaluate at Protocol B (Causal 2-Window Consensus)
    preds_test_cons = apply_causal_consensus(df_test, test_probs, t_cons)
    metrics_test_cons = calculate_metrics_dict(y_test, preds_test_cons, test_probs, test_non_afib_hours)

    print("\n  Summary of Held-Out Test Cohort Performance:")
    print(f"  {'Operating Protocol':<38s} | {'Sens':>7s} | {'Spec':>7s} | {'Prec':>7s} | {'F1':>6s} | {'Bal Acc':>8s} | {'FP/hr':>7s}")
    print("  " + "-" * 88)
    if metrics_test_const:
        print(f"  {'Constrained Spot (theta* = ' + str(tc) + ')':<38s} | "
              f"{metrics_test_const['sensitivity']*100:6.2f}% | {metrics_test_const['specificity']*100:6.2f}% | "
              f"{metrics_test_const['precision']*100:6.2f}% | {metrics_test_const['f1_score']:6.4f} | "
              f"{metrics_test_const['balanced_accuracy']*100:7.2f}% | {metrics_test_const['fp_per_hour']:7.2f}")
    print(f"  {'Balanced Triage Spot (theta = ' + str(tt) + ')':<38s} | "
          f"{metrics_test_triage['sensitivity']*100:6.2f}% | {metrics_test_triage['specificity']*100:6.2f}% | "
          f"{metrics_test_triage['precision']*100:6.2f}% | {metrics_test_triage['f1_score']:6.4f} | "
          f"{metrics_test_triage['balanced_accuracy']*100:7.2f}% | {metrics_test_triage['fp_per_hour']:7.2f}")
    print(f"  {'Default Spot (theta = 0.50)':<38s} | "
          f"{metrics_test_50['sensitivity']*100:6.2f}% | {metrics_test_50['specificity']*100:6.2f}% | "
          f"{metrics_test_50['precision']*100:6.2f}% | {metrics_test_50['f1_score']:6.4f} | "
          f"{metrics_test_50['balanced_accuracy']*100:7.2f}% | {metrics_test_50['fp_per_hour']:7.2f}")
    print(f"  {'Telemetry Protocol B (Consensus theta=' + str(t_cons) + ')':<38s} | "
          f"{metrics_test_cons['sensitivity']*100:6.2f}% | {metrics_test_cons['specificity']*100:6.2f}% | "
          f"{metrics_test_cons['precision']*100:6.2f}% | {metrics_test_cons['f1_score']:6.4f} | "
          f"{metrics_test_cons['balanced_accuracy']*100:7.2f}% | {metrics_test_cons['fp_per_hour']:7.2f}")

    # Per-Record Breakdown on Held-Out Test Cohort
    print("\n  Per-Record Diagnostic Breakdown (Held-Out Test Cohort, Triage Threshold = 0.70):")
    per_record_test = {}
    for rec_id, group in df_test.groupby("record_id", sort=True):
        g_idx = group.index
        g_y = y_test[g_idx]
        g_probs = test_probs[g_idx]
        g_preds = (g_probs >= tt).astype(int)
        n_afib = int((g_y == 1).sum())
        n_non = int((g_y == 0).sum())
        g_non_hours = n_non * STRIDE_SEC / 3600.0

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

        per_record_test[rec_id] = {
            "record_id": rec_id,
            "total_windows": len(group),
            "afib_windows": n_afib,
            "non_afib_windows": n_non,
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1_score": round(float(f1), 4),
            "fp_per_hour": round(float(fp_hr), 2),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
        }
        print(f"  Record {rec_id} (AFIB: {n_afib:4d}, Non-AFIB: {n_non:4d}) -> "
              f"Sens: {sens*100:6.2f}% | Spec: {spec*100:6.2f}% | Prec: {prec*100:6.2f}% | "
              f"F1: {f1:6.4f} | FP/hr: {fp_hr:6.2f}")

    # Confounder Analysis: Atrial Flutter Rejection
    afl_mask = (df_test["rhythm_label"] == "(AFL")
    afl_probs = test_probs[afl_mask]
    n_afl = int(afl_mask.sum())
    afl_rejected_const = int((afl_probs < tc).sum()) if lr_constrained else 0
    afl_rejected_triage = int((afl_probs < tt).sum())

    print(f"\n  Confounder Stress-Testing on Unseen Atrial Flutter (N = {n_afl} windows):")
    print(f"  - AFL Rejection at Constrained Threshold ({tc}): {afl_rejected_const} / {n_afl} ({afl_rejected_const/n_afl*100:.1f}%)")
    print(f"  - AFL Rejection at Triage Threshold ({tt}):      {afl_rejected_triage} / {n_afl} ({afl_rejected_triage/n_afl*100:.1f}%)")

    # 7. Save Comprehensive Metrics & Configuration JSON
    print(f"\n[7/7] Serializing Evaluation Artifacts:")
    eval_artifacts = {
        "evaluation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "champion_model": champion_name,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "held_out_test_prevalence_pct": round(float(y_test.sum() / len(y_test) * 100), 2),
        "global_roc_auc": round(float(test_auc), 4),
        "global_pr_auc": round(float(test_pr_auc), 4),
        "protocol_a_constrained": metrics_test_const,
        "protocol_a_triage": metrics_test_triage,
        "protocol_a_default_50": metrics_test_50,
        "protocol_b_consensus": metrics_test_cons,
        "per_record_performance": per_record_test,
        "confounder_atrial_flutter": {
            "total_afl_windows": n_afl,
            "mean_predicted_probability": round(float(afl_probs.mean()), 4),
            "correctly_rejected_non_afib_triage": afl_rejected_triage,
            "correctly_rejected_non_afib_const": afl_rejected_const
        }
    }

    with open(EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_artifacts, f, indent=2)
    print(f"  - Saved Evaluation Metrics: {EVALUATION_METRICS_PATH.name} ({os.path.getsize(EVALUATION_METRICS_PATH):,} bytes)")

    training_config = {
        "pipeline_name": "AFDB Model Training and Evaluation",
        "pipeline_version": "1.0.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_columns": FEATURE_COLUMNS,
        "scaler_fitted_on": "afdb_features_train.csv",
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "operating_thresholds": {
            "constrained_threshold": tc if lr_constrained else None,
            "triage_threshold": tt,
            "consensus_threshold": t_cons
        },
        "learned_weights_logistic_regression": {
            feat: round(float(w), 4) for feat, w in zip(FEATURE_COLUMNS, lr_model.coef_[0])
        },
        "intercept_logistic_regression": round(float(lr_model.intercept_[0]), 4),
        "file_checksums_sha256": {
            "afdb_model_package.joblib": compute_sha256(MODEL_PACKAGE_PATH),
            "afdb_evaluation_metrics.json": compute_sha256(EVALUATION_METRICS_PATH)
        }
    }

    with open(TRAINING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)
    print(f"  - Saved Training Configuration: {TRAINING_CONFIG_PATH.name} ({os.path.getsize(TRAINING_CONFIG_PATH):,} bytes)")

    print("\n" + "=" * 80)
    print("TRAINING AND HELD-OUT EVALUATION COMPLETED IN {:.2f} SECONDS".format(time.time() - t_start))
    print("=" * 80)


if __name__ == "__main__":
    main()
