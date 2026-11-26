"""
Audit Script: Reconciliation of Patient 04908 Metrics, False Alarms, and Denominators
======================================================================================
Reconciles the reported false-positive count, specificity, and false-alarm rates for
Record 04908 from the saved prediction tables and ground-truth rhythm annotations.

Calculates both:
  1. False alarms per eligible non-AFib hour.
  2. False alarms per total eligible recording hour.

Explains rhythm breakdown (Normal Sinus vs. Atrial Flutter) and guard-band impacts.
Preserves all original reports and metrics untouched.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[4]
DATA_DIR = PROJECT_ROOT / "training" / "research" / "afib_5s_classifier" / "data"
EVAL_DIR = PROJECT_ROOT / "training" / "research" / "afib_5s_classifier" / "patient_adaptive_baseline_study" / "held_out_evaluation"

PREDS_CSV = EVAL_DIR / "held_out_test_predictions.csv"
REJECTED_CSV = DATA_DIR / "afdb_5s_rejected_windows.csv"
SUMMARY_JSON = DATA_DIR / "afdb_5s_extraction_summary.json"


def reconcile_04908():
    df_preds = pd.read_csv(PREDS_CSV, dtype={"record_id": str})
    df_04908 = df_preds[df_preds["record_id"] == "04908"].copy()

    # Rejected windows
    df_rej = pd.read_csv(REJECTED_CSV, dtype={"record_id": str})
    rej_04908 = df_rej[df_rej["record_id"] == "04908"]

    # Ingestion metadata
    with open(SUMMARY_JSON, "r") as f:
        sum_data = json.load(f)
    meta_04908 = next(item for item in sum_data["per_record_summaries"] if item["record_id"] == "04908")

    stride_sec = 5.0
    total_kept_windows = len(df_04908)  # 7,333
    total_rejected_windows = len(rej_04908)  # 31
    total_candidate_windows = meta_04908["total_windows"]  # 7,364
    raw_duration_hours = meta_04908["duration_hours"]  # 10.23 hours

    n_afib = int((df_04908["is_afib"] == 1).sum())  # 600
    n_non_afib = int((df_04908["is_afib"] == 0).sum())  # 6,733

    # Breakdown by rhythm label
    rhythm_counts = df_04908["rhythm_label"].value_counts().to_dict()

    # Model Decisions at theta = 0.560
    p_A = (df_04908["prob_lr_Config_A"] >= 0.560).astype(int)
    p_C = (df_04908["prob_lr_Config_C"] >= 0.560).astype(int)

    tp_A = int(((p_A == 1) & (df_04908["is_afib"] == 1)).sum())
    fp_A = int(((p_A == 1) & (df_04908["is_afib"] == 0)).sum())
    tn_A = int(((p_A == 0) & (df_04908["is_afib"] == 0)).sum())
    fn_A = int(((p_A == 0) & (df_04908["is_afib"] == 1)).sum())

    tp_C = int(((p_C == 1) & (df_04908["is_afib"] == 1)).sum())
    fp_C = int(((p_C == 1) & (df_04908["is_afib"] == 0)).sum())
    tn_C = int(((p_C == 0) & (df_04908["is_afib"] == 0)).sum())
    fn_C = int(((p_C == 0) & (df_04908["is_afib"] == 1)).sum())

    # Rhythm-specific false alarm breakdown
    fp_rhythm_A = {}
    fp_rhythm_C = {}
    for r, group in df_04908.groupby("rhythm_label"):
        if group["is_afib"].iloc[0] == 0:
            fp_rhythm_A[r] = int(((group["prob_lr_Config_A"] >= 0.560)).sum())
            fp_rhythm_C[r] = int(((group["prob_lr_Config_C"] >= 0.560)).sum())

    # Denominators
    non_afib_eligible_hours = n_non_afib * stride_sec / 3600.0  # 9.3514 hours
    total_eligible_hours = total_kept_windows * stride_sec / 3600.0  # 10.1847 hours
    raw_recording_hours = raw_duration_hours  # 10.23 hours

    # False alarm rates
    rates_A = {
        "fp_per_eligible_non_afib_hour": round(fp_A / non_afib_eligible_hours, 2),
        "fp_per_total_eligible_hour": round(fp_A / total_eligible_hours, 2),
        "fp_per_raw_recording_hour": round(fp_A / raw_recording_hours, 2)
    }

    rates_C = {
        "fp_per_eligible_non_afib_hour": round(fp_C / non_afib_eligible_hours, 2),
        "fp_per_total_eligible_hour": round(fp_C / total_eligible_hours, 2),
        "fp_per_raw_recording_hour": round(fp_C / raw_recording_hours, 2)
    }

    reconciliation = {
        "record_id": "04908",
        "cohort": "test",
        "stride_sec": stride_sec,
        "window_counts": {
            "total_candidate_windows": total_candidate_windows,
            "kept_usable_windows": total_kept_windows,
            "rejected_guard_band_windows": total_rejected_windows,
            "afib_positive_windows": n_afib,
            "non_afib_negative_windows": n_non_afib,
            "rhythm_composition": rhythm_counts
        },
        "durations_in_hours": {
            "eligible_non_afib_hours": round(non_afib_eligible_hours, 4),
            "total_eligible_recording_hours": round(total_eligible_hours, 4),
            "raw_total_recording_hours": round(raw_recording_hours, 4)
        },
        "model_Config_A_Original": {
            "confusion_matrix": {"tp": tp_A, "fp": fp_A, "fn": fn_A, "tn": tn_A},
            "sensitivity": round(tp_A / (tp_A + fn_A), 4),
            "specificity": round(tn_A / (tn_A + fp_A), 4),
            "false_positives_by_rhythm": fp_rhythm_A,
            "false_alarm_rates": rates_A
        },
        "model_Config_C_Adaptive": {
            "confusion_matrix": {"tp": tp_C, "fp": fp_C, "fn": fn_C, "tn": tn_C},
            "sensitivity": round(tp_C / (tp_C + fn_C), 4),
            "specificity": round(tn_C / (tn_C + fp_C), 4),
            "false_positives_by_rhythm": fp_rhythm_C,
            "false_alarm_rates": rates_C
        },
        "reconciliation_explanation": (
            "The discrepancy stems from three factors: "
            "(1) Denominator definition: False alarms per eligible non-AFib hour (9.35h) yields 81.16 FP/hr, "
            "whereas false alarms per total eligible recording hour (10.18h) yields 74.52 FP/hr. "
            "(2) Confounder composition: Of the 759 false positives in Config A, 42 occur during Atrial Flutter (AFL) "
            "segments (42/44 = 95.5% flagged as AFib), while 717 occur during Normal Sinus (N) rhythm. "
            "(3) Guard band rejection: 31 windows (155s) fell within 2.5s transition guard bands and were excluded."
        )
    }

    output_path = STUDY_DIR / "audit_04908_reconciliation.json"
    with open(output_path, "w") as f:
        json.dump(reconciliation, f, indent=2)

    print("=" * 80)
    print("PATIENT 04908 AUDIT RECONCILIATION")
    print("=" * 80)
    print(f"Windows: Kept = {total_kept_windows}, Rejected = {total_rejected_windows}, Total = {total_candidate_windows}")
    print(f"Rhythms: AFIB = {rhythm_counts.get('(AFIB', 0)}, N = {rhythm_counts.get('(N', 0)}, AFL = {rhythm_counts.get('(AFL', 0)}")
    print(f"\nDurations:")
    print(f"  Eligible Non-AFib Duration: {non_afib_eligible_hours:.4f} hours")
    print(f"  Total Eligible Duration:    {total_eligible_hours:.4f} hours")
    print(f"  Raw Physical Duration:      {raw_recording_hours:.4f} hours")
    print(f"\nConfig A (Original Model):")
    print(f"  TP: {tp_A}, FP: {fp_A}, FN: {fn_A}, TN: {tn_A}")
    print(f"  Sensitivity: {tp_A/(tp_A+fn_A)*100:.2f}% | Specificity: {tn_A/(tn_A+fp_A)*100:.2f}%")
    print(f"  False Alarms in Sinus (N):   {fp_rhythm_A.get('(N', 0)} / {rhythm_counts.get('(N', 0)} ({fp_rhythm_A.get('(N', 0)/rhythm_counts.get('(N', 1)*100:.1f}%)")
    print(f"  False Alarms in Flutter (AFL): {fp_rhythm_A.get('(AFL', 0)} / {rhythm_counts.get('(AFL', 0)} ({fp_rhythm_A.get('(AFL', 0)/rhythm_counts.get('(AFL', 1)*100:.1f}%)")
    print(f"  Rate (per non-AFib hour): {rates_A['fp_per_eligible_non_afib_hour']:.2f} FP/hr")
    print(f"  Rate (per total hour):    {rates_A['fp_per_total_eligible_hour']:.2f} FP/hr")
    print(f"\nConfig C (Adaptive Model):")
    print(f"  TP: {tp_C}, FP: {fp_C}, FN: {fn_C}, TN: {tn_C}")
    print(f"  Sensitivity: {tp_C/(tp_C+fn_C)*100:.2f}% | Specificity: {tn_C/(tn_C+fp_C)*100:.2f}%")
    print(f"  False Alarms in Sinus (N):   {fp_rhythm_C.get('(N', 0)} / {rhythm_counts.get('(N', 0)} ({fp_rhythm_C.get('(N', 0)/rhythm_counts.get('(N', 1)*100:.1f}%)")
    print(f"  False Alarms in Flutter (AFL): {fp_rhythm_C.get('(AFL', 0)} / {rhythm_counts.get('(AFL', 0)} ({fp_rhythm_C.get('(AFL', 0)/rhythm_counts.get('(AFL', 1)*100:.1f}%)")
    print(f"  Rate (per non-AFib hour): {rates_C['fp_per_eligible_non_afib_hour']:.2f} FP/hr")
    print(f"  Rate (per total hour):    {rates_C['fp_per_total_eligible_hour']:.2f} FP/hr")
    print(f"\nSaved reconciliation audit to: {output_path.name}")


if __name__ == "__main__":
    reconcile_04908()
