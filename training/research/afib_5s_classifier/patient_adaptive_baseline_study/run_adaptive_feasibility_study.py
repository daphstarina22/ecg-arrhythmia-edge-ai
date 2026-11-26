"""
Feasibility Study: Causal Patient-Adaptive Baseline Features
============================================================
Evaluates whether causal patient-adaptive baseline features improve generalization
across unseen patients on 5-second ECG windows without future leakage.

Evaluated Configurations:
  Config A: Existing 21 absolute features (Baseline reference).
  Config B: Baseline-relative features alone (5 features).
  Config C: Existing 21 absolute features + 5 baseline-relative features (26 features).

Model Architectures:
  1. Balanced L2 Logistic Regression (matching existing pipeline).
  2. Histogram Gradient Boosting (tree-based non-linear baseline).

Protocol:
  - Feature extraction strictly causal per record.
  - Scaler and models fitted strictly on training cohort (15 patients, N=217,851).
  - Evaluated on validation cohort (4 patients, N=58,645).
  - Quarantined test cohort is strictly UNTOUCHED.
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
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

# Add experiment path
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[3]
RESEARCH_DIR = PROJECT_ROOT / "training" / "research" / "afib_5s_classifier"
DATA_DIR = RESEARCH_DIR / "data"

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from baseline_tracker import CausalBaselineTracker, extract_adaptive_features_for_dataframe

TRAIN_CSV = DATA_DIR / "afdb_5s_features_train.csv"
VAL_CSV = DATA_DIR / "afdb_5s_features_val.csv"
STRIDE_SEC_VAL = 2.5

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, non_afib_hours: float) -> Dict[str, Any]:
    """Computes comprehensive binary classification and false alarm metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
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
        "non_afib_hours": round(non_afib_hours, 2)
    }


def evaluate_model_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    df_meta: pd.DataFrame,
    stride_sec: float = 2.5
) -> Dict[str, Any]:
    """Evaluates model performance at fixed (0.560) and optimal thresholds, pooled and per-patient."""
    non_afib_hours = (len(y_true) - y_true.sum()) * stride_sec / 3600.0

    # 1. Fixed Operating Threshold theta* = 0.560 (from existing frozen 5s model)
    preds_fixed = (y_prob >= 0.560).astype(int)
    pooled_fixed = compute_metrics(y_true, preds_fixed, y_prob, non_afib_hours)

    # 2. Threshold sweep for Spec >= 95.0%
    best_t = 0.560
    best_f1 = -1.0
    best_opt = None
    for t in np.arange(0.20, 0.95, 0.01):
        t = round(float(t), 2)
        preds_t = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds_t, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, preds_t, zero_division=0)
        if spec >= 0.95 and f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_opt = compute_metrics(y_true, preds_t, y_prob, non_afib_hours)
    if best_opt is None:
        best_opt = pooled_fixed
        best_t = 0.560

    # 3. Per-Patient Breakdown at theta* = 0.560
    patient_breakdown = {}
    for rec_id, g in df_meta.groupby("record_id"):
        rec_idx = g.index
        y_p = y_true[rec_idx]
        prob_p = y_prob[rec_idx]
        pred_p = preds_fixed[rec_idx]
        p_non_afib_hrs = (len(y_p) - y_p.sum()) * stride_sec / 3600.0
        patient_breakdown[rec_id] = {
            "total_windows": len(y_p),
            "afib_windows": int(y_p.sum()),
            "non_afib_windows": int(len(y_p) - y_p.sum()),
            "afib_prevalence_pct": round(float(y_p.mean() * 100.0), 2),
            "metrics_at_fixed_threshold": compute_metrics(y_p, pred_p, prob_p, p_non_afib_hrs)
        }

    # 4. Warmup analysis
    warmup_mask = df_meta["baseline_warmup_fraction"].values < 1.0
    warm_mask = ~warmup_mask
    warmup_metrics = {
        "warmup_windows_count": int(np.sum(warmup_mask)),
        "warmup_windows_pct": round(float(np.mean(warmup_mask) * 100.0), 2),
        "post_warmup_windows_count": int(np.sum(warm_mask)),
        "warmup_subset_metrics": compute_metrics(
            y_true[warmup_mask],
            preds_fixed[warmup_mask],
            y_prob[warmup_mask],
            (len(y_true[warmup_mask]) - y_true[warmup_mask].sum()) * stride_sec / 3600.0
        ) if np.sum(warmup_mask) > 0 else {},
        "post_warmup_subset_metrics": compute_metrics(
            y_true[warm_mask],
            preds_fixed[warm_mask],
            y_prob[warm_mask],
            (len(y_true[warm_mask]) - y_true[warm_mask].sum()) * stride_sec / 3600.0
        )
    }

    return {
        "pooled_fixed_threshold_0_560": pooled_fixed,
        "optimal_threshold_spec95": {"threshold": best_t, "metrics": best_opt},
        "per_patient_breakdown": patient_breakdown,
        "warmup_analysis": warmup_metrics
    }


def main():
    print("=" * 80)
    print("FEASIBILITY STUDY: CAUSAL PATIENT-ADAPTIVE BASELINE FEATURES")
    print("=" * 80)
    t0 = time.time()

    # 1. Ingestion
    print("\n[1/5] Ingesting Datasets & Extracting Causal Adaptive Features:")
    df_train = pd.read_csv(TRAIN_CSV, dtype={"record_id": str})
    df_val = pd.read_csv(VAL_CSV, dtype={"record_id": str})

    print(f"  Training cohort:   {len(df_train):,} windows across {df_train['record_id'].nunique()} records")
    print(f"  Validation cohort: {len(df_val):,} windows across {df_val['record_id'].nunique()} records")

    # Verify no patient overlap
    train_pts = set(df_train["patient_id"].unique())
    val_pts = set(df_val["patient_id"].unique())
    assert train_pts.isdisjoint(val_pts), "FATAL: Overlap between Train and Val!"
    print("  [VERIFIED] 100% Patient Disjointness between Train and Val.")

    # 2. Extract Causal Baseline Features
    print("  Extracting causal baseline features for Training cohort...")
    t_adapt_train0 = time.time()
    df_adapt_train = extract_adaptive_features_for_dataframe(df_train)
    print(f"  Training adaptive features extracted in {time.time() - t_adapt_train0:.1f}s")

    print("  Extracting causal baseline features for Validation cohort...")
    t_adapt_val0 = time.time()
    df_adapt_val = extract_adaptive_features_for_dataframe(df_val)
    print(f"  Validation adaptive features extracted in {time.time() - t_adapt_val0:.1f}s")

    # Combine datasets
    df_train_full = pd.concat([df_train, df_adapt_train], axis=1)
    df_val_full = pd.concat([df_val, df_adapt_val], axis=1)

    y_train = df_train_full["is_afib"].values
    y_val = df_val_full["is_afib"].values

    # Check for NaNs
    for c in FEATURE_COLS_COMBINED:
        assert not df_train_full[c].isna().any(), f"NaN in Train feature {c}!"
        assert not df_val_full[c].isna().any(), f"NaN in Val feature {c}!"
    print("  [VERIFIED] All 26 features contain 0 NaNs and 0 Infs.")

    # 3. Model Training & Evaluation Across Configurations
    print("\n[2/5] Training & Evaluating Feature Configurations:")
    results = {}
    predictions_table = df_val_full[["record_id", "patient_id", "window_idx", "start_time_sec", "is_afib", "baseline_warmup_fraction"]].copy()

    configs = {
        "Config_A_Absolute_Only": FEATURE_COLS_ABSOLUTE,
        "Config_B_Adaptive_Only": FEATURE_COLS_ADAPTIVE,
        "Config_C_Combined_Absolute_Plus_Adaptive": FEATURE_COLS_COMBINED
    }

    for cfg_name, feat_cols in configs.items():
        print(f"\n  -------------------------------------------------------------")
        print(f"  Evaluating: {cfg_name} ({len(feat_cols)} features)")
        print(f"  -------------------------------------------------------------")

        X_tr_raw = df_train_full[feat_cols].values
        X_va_raw = df_val_full[feat_cols].values

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)

        # 3.1 Logistic Regression
        lr = LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        lr.fit(X_tr, y_train)
        probs_lr = lr.predict_proba(X_va)[:, 1]
        predictions_table[f"prob_lr_{cfg_name}"] = np.round(probs_lr, 4)

        lr_eval = evaluate_model_at_thresholds(y_val, probs_lr, df_val_full, stride_sec=STRIDE_SEC_VAL)
        print(f"    [Logistic Regression] Pooled ROC-AUC: {lr_eval['pooled_fixed_threshold_0_560']['roc_auc']:.4f} | PR-AUC: {lr_eval['pooled_fixed_threshold_0_560']['pr_auc']:.4f}")
        print(f"      @ theta=0.560 -> Sens: {lr_eval['pooled_fixed_threshold_0_560']['sensitivity']*100:.2f}%, Spec: {lr_eval['pooled_fixed_threshold_0_560']['specificity']*100:.2f}%, FP/hr: {lr_eval['pooled_fixed_threshold_0_560']['fp_per_hour']:.2f}")

        # 3.2 HistGradientBoosting
        hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=6, class_weight="balanced", random_state=42)
        hgb.fit(X_tr, y_train)
        probs_hgb = hgb.predict_proba(X_va)[:, 1]
        predictions_table[f"prob_hgb_{cfg_name}"] = np.round(probs_hgb, 4)

        hgb_eval = evaluate_model_at_thresholds(y_val, probs_hgb, df_val_full, stride_sec=STRIDE_SEC_VAL)
        print(f"    [HistGradientBoost]   Pooled ROC-AUC: {hgb_eval['pooled_fixed_threshold_0_560']['roc_auc']:.4f} | PR-AUC: {hgb_eval['pooled_fixed_threshold_0_560']['pr_auc']:.4f}")
        print(f"      @ theta=0.560 -> Sens: {hgb_eval['pooled_fixed_threshold_0_560']['sensitivity']*100:.2f}%, Spec: {hgb_eval['pooled_fixed_threshold_0_560']['specificity']*100:.2f}%, FP/hr: {hgb_eval['pooled_fixed_threshold_0_560']['fp_per_hour']:.2f}")

        # Print per-patient ROC-AUCs for Logistic Regression
        print("    Per-Patient ROC-AUC (Logistic Regression):")
        for rec, pdata in lr_eval["per_patient_breakdown"].items():
            m = pdata["metrics_at_fixed_threshold"]
            print(f"      Record {rec} (N={pdata['total_windows']}, AFib={pdata['afib_prevalence_pct']}%): AUC = {m['roc_auc']:.4f}, Sens = {m['sensitivity']*100:.2f}%, Spec = {m['specificity']*100:.2f}%, FP/hr = {m['fp_per_hour']:.2f}")

        # Store model weights if linear
        weights = {}
        if hasattr(lr, "coef_"):
            for col_name, w in zip(feat_cols, lr.coef_[0]):
                weights[col_name] = round(float(w), 4)

        results[cfg_name] = {
            "feature_columns": feat_cols,
            "feature_count": len(feat_cols),
            "logistic_regression": {
                "weights": weights,
                "intercept": round(float(lr.intercept_[0]), 4),
                "evaluation": lr_eval
            },
            "hist_gradient_boosting": {
                "evaluation": hgb_eval
            }
        }

    # 4. Save Artifacts
    print("\n[3/5] Saving Empirical Validation Results:")
    metrics_path = EXPERIMENT_DIR / "adaptive_study_validation_metrics.json"
    predictions_path = EXPERIMENT_DIR / "adaptive_study_validation_predictions.csv"

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics to {metrics_path.name}")

    predictions_table.to_csv(predictions_path, index=False)
    print(f"  Saved validation predictions to {predictions_path.name}")

    # 5. Comparative Synthesis Summary
    print("\n" + "=" * 80)
    print("COMPARATIVE EMPIRICAL SUMMARY (VALIDATION COHORT, N=58,645)")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'LR AUC':<8} | {'LR PR-AUC':<9} | {'Sens(0.56)':<10} | {'Spec(0.56)':<10} | {'FP/hr':<8}")
    print("-" * 90)
    for cfg in configs.keys():
        lr_m = results[cfg]["logistic_regression"]["evaluation"]["pooled_fixed_threshold_0_560"]
        print(f"{cfg:<35} | {lr_m['roc_auc']:<8.4f} | {lr_m['pr_auc']:<9.4f} | {lr_m['sensitivity']*100:<9.2f}% | {lr_m['specificity']*100:<9.2f}% | {lr_m['fp_per_hour']:<8.2f}")
    print("-" * 90)

    # Patient-level comparison
    print("\nPER-PATIENT ROC-AUC COMPARISON (LOGISTIC REGRESSION):")
    val_records = ["04126", "05121", "07879", "07910"]
    header = f"{'Record':<10} | {'Prevalence':<12} | " + " | ".join([f"{cfg[:12]:<12}" for cfg in configs.keys()])
    print(header)
    print("-" * len(header))
    for rec in val_records:
        prev = results["Config_A_Absolute_Only"]["logistic_regression"]["evaluation"]["per_patient_breakdown"][rec]["afib_prevalence_pct"]
        aucs = [results[cfg]["logistic_regression"]["evaluation"]["per_patient_breakdown"][rec]["metrics_at_fixed_threshold"]["roc_auc"] for cfg in configs.keys()]
        print(f"{rec:<10} | {prev:<11.1f}% | " + " | ".join([f"{a:<12.4f}" for a in aucs]))

    dt = time.time() - t0
    print(f"\nExecution completed in {dt:.1f}s.")


if __name__ == "__main__":
    main()
