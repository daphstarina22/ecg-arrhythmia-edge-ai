"""
Empirical Audit of 5-Second AFib Held-Out Test Discrepancy
==========================================================
Investigates:
1. Validation vs Held-Out Test ROC-AUC gap (0.9832 vs 0.6615).
2. Per-patient vs Global Pooled ROC-AUC (Simpson's Paradox / Baseline shift).
3. Record 07162 vs other records: feature distributions, heart rates, variability.
4. Window-level sample weighting across records (duration and window count).
5. False alarms per hour and threshold consistency.
6. Feature distribution shifts between Train, Val, and Test patients.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESEARCH_DIR = PROJECT_ROOT / "training" / "research" / "afib_5s_classifier"
DATA_DIR = RESEARCH_DIR / "data"
AUDIT_DIR = RESEARCH_DIR / "test_performance_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PACKAGE_PATH = RESEARCH_DIR / "afib_5s_model_package.joblib"
TRAIN_CSV = DATA_DIR / "afdb_5s_features_train.csv"
VAL_CSV = DATA_DIR / "afdb_5s_features_val.csv"
TEST_CSV = DATA_DIR / "afdb_5s_features_test.csv"


def main():
    print("=" * 80)
    print("AUDITING HELD-OUT TEST PERFORMANCE DISCREPANCY")
    print("=" * 80)

    # 1. Load model package
    pkg = joblib.load(MODEL_PACKAGE_PATH)
    model = pkg["champion_model"]
    scaler = pkg["scaler"]
    feature_cols = pkg["feature_columns"]
    thresholds = pkg["operating_thresholds"]
    tc = thresholds["constrained_threshold"]
    tt = thresholds["triage_threshold"]

    # 2. Ingest Data
    df_train = pd.read_csv(TRAIN_CSV, dtype={"record_id": str})
    df_val = pd.read_csv(VAL_CSV, dtype={"record_id": str})
    df_test = pd.read_csv(TEST_CSV, dtype={"record_id": str})

    # Predictions on Val
    X_val = scaler.transform(df_val[feature_cols].values)
    y_val = df_val["is_afib"].values
    probs_val = model.predict_proba(X_val)[:, 1]

    # Predictions on Test
    X_test = scaler.transform(df_test[feature_cols].values)
    y_test = df_test["is_afib"].values
    probs_test = model.predict_proba(X_test)[:, 1]

    df_test["prob_afib"] = probs_test
    df_val["prob_afib"] = probs_val

    # Audit 1: Validation per-patient vs pooled AUC
    print("\n[Audit 1] Validation Per-Record Analysis:")
    val_records = {}
    for rec_id, grp in df_val.groupby("record_id", sort=True):
        y_g = grp["is_afib"].values
        p_g = grp["prob_afib"].values
        auc_g = roc_auc_score(y_g, p_g) if len(np.unique(y_g)) > 1 else None
        n_afib = int((y_g == 1).sum())
        n_non = int((y_g == 0).sum())
        val_records[rec_id] = {
            "total_windows": len(grp),
            "afib_windows": n_afib,
            "non_afib_windows": n_non,
            "afib_prevalence_pct": round(n_afib / len(grp) * 100, 2),
            "auc": round(auc_g, 4) if auc_g else None,
            "mean_prob_afib_class": round(float(p_g[y_g == 1].mean()), 4) if n_afib > 0 else None,
            "mean_prob_non_afib_class": round(float(p_g[y_g == 0].mean()), 4) if n_non > 0 else None,
        }
        auc_str = f"{auc_g:.4f}" if auc_g else "N/A"
        print(f"  Val Record {rec_id}: N={len(grp):5d} (AFIB: {n_afib:5d}, Non: {n_non:5d}, Prev: {n_afib/len(grp)*100:5.1f}%) -> AUC: {auc_str}")

    # Audit 2: Test per-patient vs pooled AUC
    print("\n[Audit 2] Test Per-Record Analysis:")
    test_records = {}
    for rec_id, grp in df_test.groupby("record_id", sort=True):
        y_g = grp["is_afib"].values
        p_g = grp["prob_afib"].values
        auc_g = roc_auc_score(y_g, p_g) if len(np.unique(y_g)) > 1 else None
        n_afib = int((y_g == 1).sum())
        n_non = int((y_g == 0).sum())
        test_records[rec_id] = {
            "total_windows": len(grp),
            "afib_windows": n_afib,
            "non_afib_windows": n_non,
            "afib_prevalence_pct": round(n_afib / len(grp) * 100, 2),
            "auc": round(auc_g, 4) if auc_g else None,
            "mean_prob_afib_class": round(float(p_g[y_g == 1].mean()), 4) if n_afib > 0 else None,
            "mean_prob_non_afib_class": round(float(p_g[y_g == 0].mean()), 4) if n_non > 0 else None,
        }
        auc_str = f"{auc_g:.4f}" if auc_g else "N/A"
        print(f"  Test Record {rec_id}: N={len(grp):5d} (AFIB: {n_afib:5d}, Non: {n_non:5d}, Prev: {n_afib/len(grp)*100:5.1f}%) -> AUC: {auc_str} | "
              f"Mean P(AFIB): {test_records[rec_id]['mean_prob_afib_class']} vs Mean P(Non): {test_records[rec_id]['mean_prob_non_afib_class']}")

    # Audit 3: Global Pooled vs Record 07162 Removal
    print("\n[Audit 3] Impact of Record 07162 on Pooled Test Metrics:")
    df_test_no_07162 = df_test[df_test["record_id"] != "07162"]
    auc_no_07162 = roc_auc_score(df_test_no_07162["is_afib"], df_test_no_07162["prob_afib"])
    pr_auc_no_07162 = average_precision_score(df_test_no_07162["is_afib"], df_test_no_07162["prob_afib"])

    # Metrics with and without 07162
    preds_orig = (df_test["prob_afib"] >= tc).astype(int)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(df_test["is_afib"], preds_orig).ravel()
    sens_o = tp_o / (tp_o + fn_o)
    spec_o = tn_o / (tn_o + fp_o)

    preds_no = (df_test_no_07162["prob_afib"] >= tc).astype(int)
    tn_n, fp_n, fn_n, tp_n = confusion_matrix(df_test_no_07162["is_afib"], preds_no).ravel()
    sens_no = tp_n / (tp_n + fn_n)
    spec_no = tn_n / (tn_n + fp_n)

    print(f"  Full Test Set (4 patients, N=29,260):")
    print(f"    Global ROC-AUC: {roc_auc_score(y_test, probs_test):.4f} | PR-AUC: {average_precision_score(y_test, probs_test):.4f}")
    print(f"    Sensitivity: {sens_o*100:.2f}% | Specificity: {spec_o*100:.2f}% (TP={tp_o}, FN={fn_o}, FP={fp_o}, TN={tn_o})")
    print(f"\n  Without Record 07162 (3 patients: 04908, 08215, 08219, N=21,897):")
    print(f"    Global ROC-AUC: {auc_no_07162:.4f} | PR-AUC: {pr_auc_no_07162:.4f}")
    print(f"    Sensitivity: {sens_no*100:.2f}% | Specificity: {spec_no*100:.2f}% (TP={tp_n}, FN={fn_n}, FP={fp_n}, TN={tn_n})")
    print(f"    [OBSERVATION] Removing 07162 increases sensitivity from {sens_o*100:.2f}% to {sens_no*100:.2f}%! But does global AUC change?")

    # Audit 4: Why is Global AUC 0.6615 when individual records have AUC 0.9686, 0.9363, 0.8944?
    # This is textbook Simpson's Paradox / Cross-Patient Score Calibration Shift!
    print("\n[Audit 4] Cross-Patient Probability Calibration & Simpson's Paradox:")
    print("  Checking the distribution of predicted probabilities by patient and class:")
    for rec_id, grp in df_test.groupby("record_id", sort=True):
        p_afib = grp[grp["is_afib"] == 1]["prob_afib"]
        p_non = grp[grp["is_afib"] == 0]["prob_afib"]
        q_afib = [p_afib.quantile(q) for q in [0.1, 0.5, 0.9]] if len(p_afib) > 0 else [None]*3
        q_non = [p_non.quantile(q) for q in [0.1, 0.5, 0.9]] if len(p_non) > 0 else [None]*3
        print(f"  Record {rec_id}:")
        if len(p_afib) > 0:
            print(f"    AFIB (N={len(p_afib):4d}):     10th={q_afib[0]:.3f}, Median={q_afib[1]:.3f}, 90th={q_afib[2]:.3f}, Mean={p_afib.mean():.3f}")
        if len(p_non) > 0:
            print(f"    Non-AFIB (N={len(p_non):4d}): 10th={q_non[0]:.3f}, Median={q_non[1]:.3f}, 90th={q_non[2]:.3f}, Mean={p_non.mean():.3f}")

    # Notice: Record 08219 Non-AFIB mean prob vs Record 07162 AFIB mean prob!
    p_non_08219 = df_test[(df_test["record_id"] == "08219") & (df_test["is_afib"] == 0)]["prob_afib"]
    p_afib_07162 = df_test[(df_test["record_id"] == "07162") & (df_test["is_afib"] == 1)]["prob_afib"]
    p_afib_08215 = df_test[(df_test["record_id"] == "08215") & (df_test["is_afib"] == 1)]["prob_afib"]
    print(f"\n  CRITICAL COMPARISON:")
    print(f"  Record 08219 Non-AFIB (True Negative) Mean P(AFib): {p_non_08219.mean():.4f} (Median: {p_non_08219.median():.4f})")
    print(f"  Record 07162 AFIB     (True Positive) Mean P(AFib): {p_afib_07162.mean():.4f} (Median: {p_afib_07162.median():.4f})")
    print(f"  Record 08215 AFIB     (True Positive) Mean P(AFib): {p_afib_08215.mean():.4f} (Median: {p_afib_08215.median():.4f})")

    # Intersite / Cross-record inversions:
    inversion_count = 0
    # For how many pairs (x_afib from 07162, y_non from 08219) is prob(x_afib) < prob(y_non)?
    # That directly destroys pooled AUC!
    p_afib_07162_vals = p_afib_07162.values
    p_non_08219_vals = p_non_08219.values
    inversions = (p_afib_07162_vals[:, None] < p_non_08219_vals[None, :]).mean()
    print(f"  Inversion fraction: In {inversions*100:.2f}% of pairs, Patient 07162's true AFib had a LOWER probability than Patient 08219's Non-AFib!")

    # Audit 5: Feature Distribution Shifts
    print("\n[Audit 5] Feature Distribution Shifts:")
    key_features = ["hr_mean_bpm", "rr_cv", "rr_sample_dispersion", "f_band_power_ratio", "ac_ptp_mv", "n_beats"]
    print(f"  {'Feature':<22s} | {'Train Mean':>10s} | {'Val Mean':>10s} | {'04908':>8s} | {'07162':>8s} | {'08215':>8s} | {'08219':>8s}")
    print("  " + "-" * 88)
    for feat in key_features:
        tr_m = df_train[feat].mean()
        val_m = df_val[feat].mean()
        f_04908 = df_test[df_test["record_id"] == "04908"][feat].mean()
        f_07162 = df_test[df_test["record_id"] == "07162"][feat].mean()
        f_08215 = df_test[df_test["record_id"] == "08215"][feat].mean()
        f_08219 = df_test[df_test["record_id"] == "08219"][feat].mean()
        print(f"  {feat:<22s} | {tr_m:10.2f} | {val_m:10.2f} | {f_04908:8.2f} | {f_07162:8.2f} | {f_08215:8.2f} | {f_08219:8.2f}")

    # Audit 6: Heart Rate & RR CV comparison for AFIB only
    print("\n[Audit 6] AFIB-Only Feature Comparison (Why did 07162 and 08215 fail?):")
    print(f"  {'Record':<12s} | {'N (AFIB)':>8s} | {'HR Mean':>8s} | {'RR CV':>8s} | {'Dispersion':>10s} | {'f-band ratio':>12s} | {'Pred Sens @ tc':>14s}")
    print("  " + "-" * 80)
    for rec_id in ["04908", "07162", "08215", "08219"]:
        sub = df_test[(df_test["record_id"] == rec_id) & (df_test["is_afib"] == 1)]
        n = len(sub)
        hr_m = sub["hr_mean_bpm"].mean()
        cv_m = sub["rr_cv"].mean()
        disp_m = sub["rr_sample_dispersion"].mean()
        f_m = sub["f_band_power_ratio"].mean()
        sens = (sub["prob_afib"] >= tc).mean() * 100.0
        print(f"  {rec_id:<12s} | {n:8d} | {hr_m:8.1f} | {cv_m:8.4f} | {disp_m:10.4f} | {f_m:12.4f} | {sens:13.1f}%")

    # Audit 7: Non-AFIB Comparison (Why did 08219 have 39% false alarms?)
    print("\n[Audit 7] Non-AFIB Feature Comparison (Why did 08219 have high false alarms?):")
    print(f"  {'Record':<12s} | {'N (Non)':>8s} | {'HR Mean':>8s} | {'RR CV':>8s} | {'Dispersion':>10s} | {'f-band ratio':>12s} | {'FP Rate @ tc':>14s}")
    print("  " + "-" * 80)
    for rec_id in ["04908", "08215", "08219"]:
        sub = df_test[(df_test["record_id"] == rec_id) & (df_test["is_afib"] == 0)]
        n = len(sub)
        hr_m = sub["hr_mean_bpm"].mean()
        cv_m = sub["rr_cv"].mean()
        disp_m = sub["rr_sample_dispersion"].mean()
        f_m = sub["f_band_power_ratio"].mean()
        fp_rate = (sub["prob_afib"] >= tc).mean() * 100.0
        print(f"  {rec_id:<12s} | {n:8d} | {hr_m:8.1f} | {cv_m:8.4f} | {disp_m:10.4f} | {f_m:12.4f} | {fp_rate:13.1f}%")

    # Check weights of the Logistic Regression model
    print("\n[Audit 8] Logistic Regression Learned Coefficients:")
    coef_dict = dict(zip(feature_cols, model.coef_[0]))
    sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for f, w in sorted_coefs:
        print(f"  {f:<25s}: {w:+.4f}")

    # Check False Alarms per hour formula consistency
    print("\n[Audit 9] False Alarms per Hour Consistency Check:")
    stride_sec = 5.0
    tot_non_afib = (y_test == 0).sum()
    hours_non_afib = tot_non_afib * stride_sec / 3600.0
    fp_count = int(((probs_test >= tc) & (y_test == 0)).sum())
    fp_per_hr = fp_count / hours_non_afib
    print(f"  Non-AFIB Windows: {tot_non_afib}")
    print(f"  Non-AFIB Hours:   {hours_non_afib:.2f} hours")
    print(f"  False Alarms:     {fp_count}")
    print(f"  Calculated FP/hr: {fp_per_hr:.2f} (reported in metrics: 157.45)")


if __name__ == "__main__":
    main()
