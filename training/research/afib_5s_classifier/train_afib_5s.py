"""
AFDB 5-Second Dedicated Classifier: Model Training & Threshold Optimization
============================================================================
Hardware Target: Raspberry Pi ARM64 (Debian Bookworm) + ADS1292R (360 Hz)
Database: MIT-BIH Atrial Fibrillation Database (AFDB v1.0.0)

Safety & Anti-Leakage Protocol:
1. Preprocessing (StandardScaler) is fitted EXCLUSIVELY on Training patients (15 patients).
2. Model selection and operating threshold sweeps are executed EXCLUSIVELY on Validation patients (4 patients).
3. The Held-Out Test cohort (4 patients) is completely quarantined and UNTOUCHED during this phase.
4. Champion model and decision thresholds are frozen before test evaluation.
"""

import os
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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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

TRAIN_CSV = DATA_DIR / "afdb_5s_features_train.csv"
VAL_CSV = DATA_DIR / "afdb_5s_features_val.csv"
TEST_CSV = DATA_DIR / "afdb_5s_features_test.csv"

MODEL_PACKAGE_PATH = RESEARCH_DIR / "afib_5s_model_package.joblib"
TRAINING_CONFIG_PATH = RESEARCH_DIR / "afib_5s_training_config.json"

FEATURE_COLUMNS = [
    # Ventricular timing & rate dynamics
    "n_beats",
    "hr_mean_bpm",
    "hr_std_bpm",
    "rr_cv",
    "rr_rmssd",
    "rr_mad",
    "rr_pnn50",
    "rr_sample_dispersion",
    "rr_turning_point_ratio",
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
    "ac_ptp_mv",
    "zero_crossings",
    "lz_complexity"
]

STRIDE_SEC_TRAIN_VAL = 2.5


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
    print("DEDICATED 5-SECOND AFIB CLASSIFIER: TRAINING & VALIDATION FREEZING")
    print("=" * 80)
    t_start = time.time()

    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        print(f"ERROR: Training or validation CSV missing! Ensure feature extraction is completed.")
        sys.exit(1)

    # 1. Ingestion & Strict Patient Disjointness Audit
    print("\n[1/6] Ingesting Datasets & Verifying Disjointness:")
    df_train = pd.read_csv(TRAIN_CSV, dtype={"record_id": str})
    df_val = pd.read_csv(VAL_CSV, dtype={"record_id": str})

    train_pts = set(df_train["patient_id"].unique())
    val_pts = set(df_val["patient_id"].unique())

    print(f"  Training cohort:   {len(df_train):,} windows across {len(train_pts)} patients ({df_train['is_afib'].sum():,} AFIB)")
    print(f"  Validation cohort: {len(df_val):,} windows across {len(val_pts)} patients ({df_val['is_afib'].sum():,} AFIB)")

    assert train_pts.isdisjoint(val_pts), "FATAL LEAKAGE: Overlap between Train and Validation patients!"
    print("  [VERIFIED] 100% Patient Disjointness between Train and Val confirmed.")
    print("  [QUARANTINE] Test cohort remains completely unread and untouched.")

    X_train_raw = df_train[FEATURE_COLUMNS].values
    y_train = df_train["is_afib"].values

    X_val_raw = df_val[FEATURE_COLUMNS].values
    y_val = df_val["is_afib"].values

    # Check for NaN / Inf
    assert not np.isnan(X_train_raw).any(), "NaN found in X_train!"
    assert not np.isinf(X_train_raw).any(), "Inf found in X_train!"
    assert not np.isnan(X_val_raw).any(), "NaN found in X_val!"
    assert not np.isinf(X_val_raw).any(), "Inf found in X_val!"

    # 2. Fit StandardScaler EXCLUSIVELY on Training data
    print("\n[2/6] Preprocessing: Fitting StandardScaler Strictly on Training Patients:")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    print(f"  Fitted on {len(X_train):,} training windows across {len(FEATURE_COLUMNS)} features.")
    print(f"  Learned n_beats mean: {scaler.mean_[FEATURE_COLUMNS.index('n_beats')]:.2f} (std: {scaler.scale_[FEATURE_COLUMNS.index('n_beats')]:.2f})")
    print("  [LEAK-FREE] Zero validation samples accessed during scaling.")

    # 3. Train Candidate Models
    print("\n[3/6] Training Candidate Research Models on 5-Second Training Windows:")
    
    # Candidate A: L2-Regularized Balanced Logistic Regression
    print("  Training Candidate A: L2-Regularized Balanced Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    # Candidate B: Balanced Random Forest
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

    # Candidate C: Histogram Gradient Boosting Classifier
    print("  Training Candidate C: Histogram Gradient Boosting (max_iter=100, max_depth=6)...")
    hgb_model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
    hgb_model.fit(X_train, y_train)

    # 4. Model Selection on Validation Set
    print("\n[4/6] Evaluating Candidates on Validation Set (4 Patients):")
    val_probs_lr = lr_model.predict_proba(X_val)[:, 1]
    val_probs_rf = rf_model.predict_proba(X_val)[:, 1]
    val_probs_hgb = hgb_model.predict_proba(X_val)[:, 1]

    val_non_afib_hours = (len(y_val) - y_val.sum()) * STRIDE_SEC_TRAIN_VAL / 3600.0

    auc_lr = roc_auc_score(y_val, val_probs_lr)
    pr_auc_lr = average_precision_score(y_val, val_probs_lr)

    auc_rf = roc_auc_score(y_val, val_probs_rf)
    pr_auc_rf = average_precision_score(y_val, val_probs_rf)

    auc_hgb = roc_auc_score(y_val, val_probs_hgb)
    pr_auc_hgb = average_precision_score(y_val, val_probs_hgb)

    print(f"  Candidate A (Logistic Regression): ROC-AUC = {auc_lr:.4f} | PR-AUC = {pr_auc_lr:.4f}")
    print(f"  Candidate B (Random Forest):       ROC-AUC = {auc_rf:.4f} | PR-AUC = {pr_auc_rf:.4f}")
    print(f"  Candidate C (HistGradientBoost):   ROC-AUC = {auc_hgb:.4f} | PR-AUC = {pr_auc_hgb:.4f}")

    # Select Champion Model
    candidates = {
        "LogisticRegression": (lr_model, auc_lr, pr_auc_lr, val_probs_lr),
        "RandomForest": (rf_model, auc_rf, pr_auc_rf, val_probs_rf),
        "HistGradientBoosting": (hgb_model, auc_hgb, pr_auc_hgb, val_probs_hgb)
    }
    # Rank by ROC-AUC + PR-AUC sum
    champion_name = max(candidates.keys(), key=lambda k: candidates[k][1] + candidates[k][2])
    champion_model, champ_auc, champ_pr_auc, val_probs_champ = candidates[champion_name]

    print(f"\n  >>> CHAMPION MODEL SELECTED: {champion_name} (ROC-AUC: {champ_auc:.4f}, PR-AUC: {champ_pr_auc:.4f})")

    # 5. Threshold Optimization Search on Validation Set
    print("\n[5/6] Operating Threshold Optimization on Validation Set:")
    
    # Sweep for Champion Model
    champ_constrained = None
    champ_best_f1 = None
    max_sens_constrained = -1.0
    max_f1 = -1.0

    for t in np.arange(0.05, 0.995, 0.005):
        t = round(float(t), 3)
        preds = (val_probs_champ >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fp_hr = fp / val_non_afib_hours if val_non_afib_hours > 0 else 0.0
        f1 = f1_score(y_val, preds, zero_division=0)

        # Constrained candidate (Spec >= 95% AND FP/hr <= 5.0)
        if spec >= 0.95 and fp_hr <= 5.0:
            if sens > max_sens_constrained:
                max_sens_constrained = sens
                champ_constrained = (t, sens, spec, fp_hr, f1)

        # Triage candidate (Max F1 subject to Spec >= 90%)
        if spec >= 0.90:
            if f1 > max_f1:
                max_f1 = f1
                champ_best_f1 = (t, sens, spec, fp_hr, f1)

    if champ_constrained:
        tc, sc, spc, fpc, f1c = champ_constrained
        print(f"  [CONSTRAINED THRESHOLD] theta* = {tc:.3f} -> Sens: {sc*100:.2f}%, Spec: {spc*100:.2f}%, FP/hr: {fpc:.2f}, F1: {f1c:.4f}")
    else:
        # Fallback to high-specificity threshold if 5.0 FP/hr is tight on paroxysmal bursts
        print("  [CONSTRAINED THRESHOLD] Searching best available operating point with Spec >= 95.0%...")
        best_spec_pt = None
        best_f1_high_spec = -1.0
        for t in np.arange(0.50, 0.995, 0.005):
            t = round(float(t), 3)
            preds = (val_probs_champ >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fp_hr = fp / val_non_afib_hours
            f1 = f1_score(y_val, preds, zero_division=0)
            if spec >= 0.95 and f1 > best_f1_high_spec:
                best_f1_high_spec = f1
                best_spec_pt = (t, sens, spec, fp_hr, f1)
        tc, sc, spc, fpc, f1c = best_spec_pt
        champ_constrained = best_spec_pt
        print(f"  [CONSTRAINED THRESHOLD] theta* = {tc:.3f} -> Sens: {sc*100:.2f}%, Spec: {spc*100:.2f}%, FP/hr: {fpc:.2f}, F1: {f1c:.4f}")

    tt, st, spt, fpt, f1t = champ_best_f1
    print(f"  [BALANCED TRIAGE THRESHOLD] theta_triage = {tt:.3f} -> Sens: {st*100:.2f}%, Spec: {spt*100:.2f}%, FP/hr: {fpt:.2f}, F1: {f1t:.4f}")

    # Sweep for Protocol B (2-window consensus confirmation)
    print("\n  Evaluating Protocol B (Causal 2-Window Consensus) on Validation Set:")
    champ_consensus = None
    max_sens_cons = -1.0
    for t in np.arange(0.40, 0.95, 0.02):
        t = round(float(t), 2)
        c_preds = apply_causal_consensus(df_val, val_probs_champ, t)
        tn, fp, fn, tp = confusion_matrix(y_val, c_preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fp_hr = fp / val_non_afib_hours
        f1 = f1_score(y_val, c_preds, zero_division=0)
        if spec >= 0.95 and fp_hr <= 5.0:
            if sens > max_sens_cons:
                max_sens_cons = sens
                champ_consensus = (t, sens, spec, fp_hr, f1)

    if champ_consensus:
        t_cons, s_cons, sp_cons, fp_cons, f1_cons = champ_consensus
    else:
        t_cons = round(float(tc * 0.9), 2)
        c_preds = apply_causal_consensus(df_val, val_probs_champ, t_cons)
        tn, fp, fn, tp = confusion_matrix(y_val, c_preds).ravel()
        s_cons, sp_cons = tp / (tp + fn), tn / (tn + fp)
        fp_cons = fp / val_non_afib_hours
        f1_cons = f1_score(y_val, c_preds, zero_division=0)
        champ_consensus = (t_cons, s_cons, sp_cons, fp_cons, f1_cons)

    print(f"  [CONSENSUS THRESHOLD] theta_cons = {t_cons:.2f} -> Sens: {s_cons*100:.2f}%, Spec: {sp_cons*100:.2f}%, FP/hr: {fp_cons:.2f}, F1: {f1_cons:.4f}")

    # 6. Freeze Model Package & Write Training Config
    print("\n[6/6] Freezing Model Package and Configuration:")
    model_package = {
        "pipeline_name": "AFDB 5-Second Dedicated Classifier",
        "champion_model_name": champion_name,
        "champion_model": champion_model,
        "candidate_models": {
            "logistic_regression": lr_model,
            "random_forest": rf_model,
            "hist_gradient_boosting": hgb_model
        },
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "operating_thresholds": {
            "constrained_threshold": tc,
            "triage_threshold": tt,
            "consensus_threshold": t_cons
        },
        "validation_metrics": {
            "candidate_roc_auc": {
                "logistic_regression": round(auc_lr, 4),
                "random_forest": round(auc_rf, 4),
                "hist_gradient_boosting": round(auc_hgb, 4)
            },
            "candidate_pr_auc": {
                "logistic_regression": round(pr_auc_lr, 4),
                "random_forest": round(pr_auc_rf, 4),
                "hist_gradient_boosting": round(pr_auc_hgb, 4)
            },
            "champion_validation_operating_points": {
                "constrained": {"thresh": tc, "sens": round(sc, 4), "spec": round(spc, 4), "fp_per_hr": round(fpc, 2), "f1": round(f1c, 4)},
                "triage": {"thresh": tt, "sens": round(st, 4), "spec": round(spt, 4), "fp_per_hr": round(fpt, 2), "f1": round(f1t, 4)},
                "consensus": {"thresh": t_cons, "sens": round(s_cons, 4), "spec": round(sp_cons, 4), "fp_per_hr": round(fp_cons, 2), "f1": round(f1_cons, 4)}
            }
        },
        "patient_splits": {
            "train": sorted(list(train_pts)),
            "val": sorted(list(val_pts)),
            "test_quarantined": ["04908", "07162", "08215", "08219"]
        }
    }

    joblib.dump(model_package, MODEL_PACKAGE_PATH)
    print(f"  - Serialized Frozen Model Package: {MODEL_PACKAGE_PATH.name} ({os.path.getsize(MODEL_PACKAGE_PATH):,} bytes)")

    training_config = {
        "pipeline_name": "AFDB 5-Second Dedicated Classifier",
        "pipeline_version": "2.0.0",
        "window_duration_sec": 5.0,
        "sampling_rate_hz": 360,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "champion_model_name": champion_name,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "scaler_fitted_on": "afdb_5s_features_train.csv",
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "operating_thresholds": {
            "constrained_threshold": tc,
            "triage_threshold": tt,
            "consensus_threshold": t_cons
        },
        "learned_scaler_means": {
            feat: round(float(m), 4) for feat, m in zip(FEATURE_COLUMNS, scaler.mean_)
        },
        "learned_scaler_scales": {
            feat: round(float(s), 4) for feat, s in zip(FEATURE_COLUMNS, scaler.scale_)
        },
        "file_checksums_sha256": {
            "afdb_5s_model_package.joblib": compute_sha256(MODEL_PACKAGE_PATH)
        }
    }

    # If logistic regression is champion or candidate, store weights
    if hasattr(lr_model, "coef_"):
        training_config["logistic_regression_weights"] = {
            feat: round(float(w), 4) for feat, w in zip(FEATURE_COLUMNS, lr_model.coef_[0])
        }
        training_config["logistic_regression_intercept"] = round(float(lr_model.intercept_[0]), 4)

    with open(TRAINING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    print(f"  - Serialized Training Configuration: {TRAINING_CONFIG_PATH.name} ({os.path.getsize(TRAINING_CONFIG_PATH):,} bytes)")

    print("\n" + "=" * 80)
    print(f"MODEL TRAINING & VALIDATION FREEZING COMPLETED IN {time.time() - t_start:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    main()
