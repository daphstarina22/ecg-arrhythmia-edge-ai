"""
AFib Rigorous Evaluation and Confounder Analysis Protocol
==========================================================
Author: Senior ECG ML Researcher & Edge-AI Deployment Engineer
Target: Raspberry Pi ARM64 (Debian Bookworm)

Evaluates the frozen candidate baseline model strictly on the held-out Test cohort:
1. Full confusion matrix, sensitivity, specificity, precision, F1, balanced accuracy.
2. Per-subject and per-record breakdown.
3. False positives per hour of continuous monitoring (FP/hour).
4. Detection latency across episode transitions.
5. 95% Confidence intervals via non-parametric bootstrapping (N=1000).
6. Slice-based error analysis against confounding rhythms (AFL, Bigeminy, Pacemaker, Bradycardia).
7. Comparison against a naive RR-CV heuristic baseline.
"""

import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

# Paths
EXPERIMENT_DIR = Path(__file__).resolve().parent
TEST_CSV = EXPERIMENT_DIR / "features_test.csv"
CONF_CSV = EXPERIMENT_DIR / "features_confounders.csv"
PACKAGE_PATH = EXPERIMENT_DIR / "afib_baseline_package.joblib"
RESULTS_JSON = EXPERIMENT_DIR / "evaluation_test_results.json"
CONFOUNDER_JSON = EXPERIMENT_DIR / "evaluation_confounder_results.json"


def run_evaluation():
    print("=" * 70)
    print("Phase D: Rigorous AFib Evaluation & Confounder Stress-Testing")
    print("=" * 70)

    # 1. Load frozen model package
    if not PACKAGE_PATH.exists():
        raise FileNotFoundError(f"Model package missing at {PACKAGE_PATH}. Run train_afib_baseline.py first!")

    pkg = joblib.load(PACKAGE_PATH)
    scaler = pkg["scaler"]
    features = pkg["feature_columns"]
    operating_thresh = pkg["operating_threshold"]
    rf_model = pkg["random_forest"]
    lr_model = pkg["logistic_regression"]
    champion_name = pkg["champion_model_name"]

    print(f"\n[LOADED] Champion Model: {champion_name} (Operating Threshold: {operating_thresh:.2f})")
    print(f"Features ({len(features)}): {features}")

    # 2. Load held-out test cohort
    df_test = pd.read_csv(TEST_CSV)
    print(f"\n[TEST COHORT] Loaded {len(df_test)} test windows across {df_test['subject_id'].nunique()} independent subjects:")
    for subj, group in df_test.groupby("subject_id"):
        n_afib = (group["is_afib"] == 1).sum()
        n_non_afib = (group["is_afib"] == 0).sum()
        print(f"  Subject {subj:<14} (Record {group['record_id'].iloc[0]}): {n_afib:>4} AFib | {n_non_afib:>4} Non-AFib windows")

    X_test_raw = df_test[features].values
    y_test = df_test["is_afib"].values

    # Transform test features using the FROZEN training scaler
    X_test = scaler.transform(X_test_raw)

    # 3. Model Predictions on Test Cohort
    test_probs_rf = rf_model.predict_proba(X_test)[:, 1]
    test_probs_lr = lr_model.predict_proba(X_test)[:, 1]

    test_preds_rf = (test_probs_rf >= operating_thresh).astype(int)
    # Default 0.50 threshold for logistic regression comparison
    test_preds_lr = (test_probs_lr >= 0.50).astype(int)

    # 4. Simple Heuristic Baseline (RR Coefficient of Variation > 0.12)
    test_preds_heuristic = (df_test["rr_cv"].values >= 0.12).astype(int)

    # -------------------------------------------------------------------------
    # Core Evaluation Metrics
    # -------------------------------------------------------------------------
    def compute_all_metrics(y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        
        # Temporal False Alarm Rate (FP per hour of non-AFib monitoring)
        # Stride is 5 seconds per window
        non_afib_hours = (len(y_true[y_true == 0]) * 5.0) / 3600.0
        fp_per_hour = fp / non_afib_hours if non_afib_hours > 0 else 0.0

        return {
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1_score": round(float(f1), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "fp_per_hour": round(float(fp_per_hour), 2),
            "non_afib_monitoring_hours": round(float(non_afib_hours), 2)
        }

    metrics_rf = compute_all_metrics(y_test, test_preds_rf, test_probs_rf)
    metrics_lr = compute_all_metrics(y_test, test_preds_lr, test_probs_lr)
    metrics_heur = compute_all_metrics(y_test, test_preds_heuristic, df_test["rr_cv"].values)

    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Metric':<28} | {'RandomForest (Champion)':<23} | {'Logistic Regression':<20} | {'RR-CV Heuristic':<15}")
    print("-" * 92)
    print(f"{'Confusion Matrix (TP/FP/TN/FN)':<28} | {metrics_rf['confusion_matrix']['TP']}/{metrics_rf['confusion_matrix']['FP']}/{metrics_rf['confusion_matrix']['TN']}/{metrics_rf['confusion_matrix']['FN']:<18} | {metrics_lr['confusion_matrix']['TP']}/{metrics_lr['confusion_matrix']['FP']}/{metrics_lr['confusion_matrix']['TN']}/{metrics_lr['confusion_matrix']['FN']:<15} | {metrics_heur['confusion_matrix']['TP']}/{metrics_heur['confusion_matrix']['FP']}/{metrics_heur['confusion_matrix']['TN']}/{metrics_heur['confusion_matrix']['FN']}")
    print(f"{'Sensitivity (Recall)':<28} | {metrics_rf['sensitivity']*100:>6.2f}%{' ':>16} | {metrics_lr['sensitivity']*100:>6.2f}%{' ':>13} | {metrics_heur['sensitivity']*100:>6.2f}%")
    print(f"{'Specificity':<28} | {metrics_rf['specificity']*100:>6.2f}%{' ':>16} | {metrics_lr['specificity']*100:>6.2f}%{' ':>13} | {metrics_heur['specificity']*100:>6.2f}%")
    print(f"{'Precision (PPV)':<28} | {metrics_rf['precision']*100:>6.2f}%{' ':>16} | {metrics_lr['precision']*100:>6.2f}%{' ':>13} | {metrics_heur['precision']*100:>6.2f}%")
    print(f"{'F1-Score':<28} | {metrics_rf['f1_score']:>6.4f}{' ':>16} | {metrics_lr['f1_score']:>6.4f}{' ':>13} | {metrics_heur['f1_score']:>6.4f}")
    print(f"{'Balanced Accuracy':<28} | {metrics_rf['balanced_accuracy']*100:>6.2f}%{' ':>16} | {metrics_lr['balanced_accuracy']*100:>6.2f}%{' ':>13} | {metrics_heur['balanced_accuracy']*100:>6.2f}%")
    print(f"{'ROC-AUC':<28} | {metrics_rf['roc_auc']:>6.4f}{' ':>16} | {metrics_lr['roc_auc']:>6.4f}{' ':>13} | {metrics_heur['roc_auc']:>6.4f}")
    print(f"{'PR-AUC':<28} | {metrics_rf['pr_auc']:>6.4f}{' ':>16} | {metrics_lr['pr_auc']:>6.4f}{' ':>13} | {metrics_heur['pr_auc']:>6.4f}")
    print(f"{'False Positives / Hour':<28} | {metrics_rf['fp_per_hour']:>6.2f} FP/hr{' ':>13} | {metrics_lr['fp_per_hour']:>6.2f} FP/hr{' ':>10} | {metrics_heur['fp_per_hour']:>6.2f} FP/hr")
    print(f"{'Non-AFib Monitoring Time':<28} | {metrics_rf['non_afib_monitoring_hours']:>6.2f} hours{' ':>14} | {metrics_lr['non_afib_monitoring_hours']:>6.2f} hours{' ':>11} | {metrics_heur['non_afib_monitoring_hours']:>6.2f} hours")

    # -------------------------------------------------------------------------
    # Non-Parametric Bootstrap Confidence Intervals (N=1000)
    # -------------------------------------------------------------------------
    print("\nComputing 95% Bootstrap Confidence Intervals (N=1000 resamples)...")
    np.random.seed(42)
    n_resamples = 1000
    boot_metrics = {"sensitivity": [], "specificity": [], "precision": [], "f1": [], "balanced_acc": []}

    indices = np.arange(len(y_test))
    for _ in range(n_resamples):
        boot_idx = np.random.choice(indices, size=len(indices), replace=True)
        y_b = y_test[boot_idx]
        p_b = test_preds_rf[boot_idx]

        # Only evaluate if both classes are present in sample
        if len(np.unique(y_b)) < 2:
            continue

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_b, p_b).ravel()
        boot_metrics["sensitivity"].append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0)
        boot_metrics["specificity"].append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0)
        boot_metrics["precision"].append(tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0)
        boot_metrics["f1"].append(f1_score(y_b, p_b, zero_division=0))
        boot_metrics["balanced_acc"].append(balanced_accuracy_score(y_b, p_b))

    ci_95 = {}
    print("\n95% Bootstrap Confidence Intervals for Champion Model:")
    for k, v in boot_metrics.items():
        low = np.percentile(v, 2.5)
        high = np.percentile(v, 97.5)
        ci_95[k] = (round(float(low), 4), round(float(high), 4))
        print(f"  {k:<18}: {np.mean(v):.4f}  [95% CI: {low:.4f} - {high:.4f}]")

    # -------------------------------------------------------------------------
    # Per-Subject and Per-Record Performance Breakdown
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-SUBJECT PERFORMANCE AUDIT ON TEST COHORT")
    print("=" * 70)
    per_subject_results = []
    print(f"{'Subject ID':<14} | {'Record':<6} | {'True Rhythm':<12} | {'Windows':<8} | {'Sens (%)':<9} | {'Spec (%)':<9} | {'FP Count':<8}")
    print("-" * 75)

    for subj, group in df_test.groupby("subject_id"):
        s_idx = group.index
        y_subj = y_test[s_idx]
        p_subj = test_preds_rf[s_idx]
        rec_id = group["record_id"].iloc[0]

        is_afib_rec = (y_subj == 1).sum() > 0
        rhythm_type = "AFIB" if is_afib_rec else "NSR"

        if is_afib_rec:
            sens_subj = recall_score(y_subj, p_subj, zero_division=0) * 100.0
            spec_subj = 100.0  # No non-AFib in this record
            fp_subj = 0
            sens_str = f"{sens_subj:6.2f}%"
            spec_str = "N/A"
        else:
            sens_subj = 100.0
            spec_subj = (p_subj == 0).sum() / len(p_subj) * 100.0
            fp_subj = int((p_subj == 1).sum())
            sens_str = "N/A"
            spec_str = f"{spec_subj:6.2f}%"

        print(f"{subj:<14} | {rec_id:<6} | {rhythm_type:<12} | {len(y_subj):<8} | {sens_str:<9} | {spec_str:<9} | {fp_subj:<8}")

        per_subject_results.append({
            "subject_id": subj,
            "record_id": str(rec_id),
            "primary_rhythm": rhythm_type,
            "total_windows": len(y_subj),
            "sensitivity_pct": round(sens_subj, 2) if is_afib_rec else None,
            "specificity_pct": round(spec_subj, 2) if not is_afib_rec else None,
            "fp_count": int(fp_subj)
        })

    # -------------------------------------------------------------------------
    # Detection Latency Analysis (Record 221 AFib Episodes)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EPISODE DETECTION LATENCY AUDIT (Record 221 AFib)")
    print("=" * 70)
    # Filter only record 221 AFib windows sorted by start time
    df_221 = df_test[df_test["record_id"].astype(str) == "221"].sort_values("start_sample")
    probs_221 = rf_model.predict_proba(scaler.transform(df_221[features].values))[:, 1]
    preds_221 = (probs_221 >= operating_thresh).astype(int)

    # In Record 221, AFib starts at sample 28 (0.08s) and continues continuously until interrupted by trigeminy
    # The first window starts at sample 928 (after 2.5s guard).
    first_detected_idx = np.where(preds_221 == 1)[0]
    if len(first_detected_idx) > 0:
        first_win = df_221.iloc[first_detected_idx[0]]
        detection_latency_sec = float(first_win["end_time_sec"])
        print(f"Record 221 Episode 1 Onset: 0.08s")
        print(f"First Detected Window End Time: {detection_latency_sec:.2f}s")
        print(f"Measured Detection Latency: {detection_latency_sec:.2f} seconds")
    else:
        detection_latency_sec = -1.0
        print("Record 221: No detection emitted!")

    # -------------------------------------------------------------------------
    # Temporal Smoothing & Consecutive Confirmation Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEMPORAL SMOOTHING: 2-CONSECUTIVE WINDOW CONFIRMATION")
    print("=" * 70)
    # Check impact of requiring 2 consecutive positive windows within each record
    smoothed_preds = np.zeros_like(test_preds_rf)
    for rec_id, group in df_test.groupby("record_id", sort=False):
        indices = group.index.values
        sub_p = test_preds_rf[indices]
        for i in range(len(indices)):
            if i == 0:
                smoothed_preds[indices[i]] = sub_p[i]
            else:
                smoothed_preds[indices[i]] = 1 if (sub_p[i] == 1 and sub_p[i-1] == 1) else 0

    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_test, smoothed_preds).ravel()
    sens_s = tp_s / (tp_s + fn_s)
    spec_s = tn_s / (tn_s + fp_s)
    fp_rate_s = fp_s / metrics_rf["non_afib_monitoring_hours"]

    print(f"Raw 1-Window Decision   -> Sens: {metrics_rf['sensitivity']*100:.2f}%, Spec: {metrics_rf['specificity']*100:.2f}%, FP/hr: {metrics_rf['fp_per_hour']:.2f}")
    print(f"2-Window Confirmation   -> Sens: {sens_s*100:.2f}%, Spec: {spec_s*100:.2f}%, FP/hr: {fp_rate_s:.2f}")

    # -------------------------------------------------------------------------
    # Slice-Based Error Analysis on Confounders (features_confounders.csv)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SLICE-BASED ERROR ANALYSIS: CONVOLUTED / CONFOUNDING RHYTHMS")
    print("=" * 70)
    confounder_metrics = {}
    if CONF_CSV.exists():
        df_conf = pd.read_csv(CONF_CSV)
        X_conf = scaler.transform(df_conf[features].values)
        probs_conf = rf_model.predict_proba(X_conf)[:, 1]
        preds_conf = (probs_conf >= operating_thresh).astype(int)

        df_conf["pred_afib"] = preds_conf
        df_conf["prob_afib"] = probs_conf

        print(f"{'Rhythm Slice':<26} | {'Windows':<8} | {'AFib Alarms':<12} | {'Alarm Rate (%)':<14} | {'Clinical Interpretation'}")
        print("-" * 95)

        for cohort_name, grp in df_conf.groupby("cohort"):
            n_tot = len(grp)
            n_pos = grp["pred_afib"].sum()
            pos_pct = (n_pos / n_tot) * 100.0
            mean_p = grp["prob_afib"].mean()

            interp = ""
            if cohort_name == "confounder_afl":
                interp = "Atrial Flutter: Should be flagged as non-AFib or specialized"
            elif cohort_name == "confounder_bigeminy":
                interp = "Frequent PVCs: Tests resistance to irregular RR confounding"
            elif cohort_name == "confounder_bradycardia":
                interp = "Sinus Bradycardia: Low HR must not trigger false AFib"
            elif cohort_name == "confounder_paced_normal":
                interp = "Pacemaker spikes: Evaluates artificial hardware distortion"
            elif cohort_name == "confounder_paced_afib":
                interp = "Paced patient with true paroxysmal AFib"

            print(f"{cohort_name:<26} | {n_tot:<8} | {n_pos:<12} | {pos_pct:>6.2f}%{' ':>7} | {interp}")

            confounder_metrics[cohort_name] = {
                "total_windows": int(n_tot),
                "afib_alarms_emitted": int(n_pos),
                "alarm_rate_pct": round(float(pos_pct), 2),
                "mean_afib_probability": round(float(mean_p), 4),
                "clinical_interpretation": interp
            }

    # -------------------------------------------------------------------------
    # Save Structured Reports
    # -------------------------------------------------------------------------
    final_results = {
        "metadata": {
            "target_system": "Raspberry Pi ARM64 + ADS1292R (360 Hz)",
            "champion_model": champion_name,
            "operating_threshold": operating_thresh,
            "window_duration_sec": 10.0,
            "stride_sec": 5.0
        },
        "test_cohort_summary": {
            "total_windows": len(df_test),
            "afib_windows": int((y_test == 1).sum()),
            "non_afib_windows": int((y_test == 0).sum()),
            "afib_subjects_count": int(df_test[df_test['is_afib'] == 1]['subject_id'].nunique()),
            "control_subjects_count": int(df_test[df_test['is_afib'] == 0]['subject_id'].nunique())
        },
        "champion_metrics": metrics_rf,
        "alternative_logistic_regression": metrics_lr,
        "rr_cv_heuristic_baseline": metrics_heur,
        "bootstrap_95_ci": ci_95,
        "per_subject_breakdown": per_subject_results,
        "detection_latency_sec": detection_latency_sec,
        "temporal_smoothing_2window": {
            "sensitivity": round(float(sens_s), 4),
            "specificity": round(float(spec_s), 4),
            "fp_per_hour": round(float(fp_rate_s), 2)
        }
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(final_results, f, indent=2)

    with open(CONFOUNDER_JSON, "w") as f:
        json.dump(confounder_metrics, f, indent=2)

    print(f"\n[SUCCESS] Exported full evaluation reports to:\n  {RESULTS_JSON}\n  {CONFOUNDER_JSON}")


if __name__ == "__main__":
    run_evaluation()
