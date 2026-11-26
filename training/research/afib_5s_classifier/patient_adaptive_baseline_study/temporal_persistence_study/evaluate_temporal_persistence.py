"""
Evaluation Script: Causal Temporal Persistence & Multi-Window Consensus Study
=============================================================================
Protocol: Evaluates causal temporal persistence filters on patient-disjoint
validation data (Records: 04126, 05121, 07879, 07910; N = 58,645 windows).

Evaluated Rules:
  1. unfiltered (Baseline reference, theta = 0.560)
  2. consecutive_2 (2 consecutive positive windows)
  3. consecutive_3 (3 consecutive positive windows)
  4. consecutive_4 (4 consecutive positive windows)
  5. majority_2_of_3 (2 of last 3 windows)
  6. majority_3_of_4 (3 of last 4 windows)
  7. hysteresis_on3_off2 (Onset: 3 consecutive positives; Offset: 2 consecutive negatives)

Calculates:
  - Window-level metrics: Sensitivity, Specificity, Precision, F1, Balanced Accuracy.
  - False alarm rate per eligible non-AFib hour (26.1861 hrs).
  - False alarm rate per total eligible recording hour (40.7257 hrs).
  - Episode-level metrics: Detected episodes, missed episodes, detection delay (median/mean sec).
  - Contiguous false alarm events (pager burst count).
  - Per-patient breakdown across all 4 validation records.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import wfdb

STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[4]
RESEARCH_DIR = PROJECT_ROOT / "training" / "research" / "afib_5s_classifier"
DATA_DIR = RESEARCH_DIR / "data"
RAW_AFDB_DIR = PROJECT_ROOT / "training" / "research" / "afib_experiment" / "data" / "afdb" / "raw"

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from persistence_filter import CausalTemporalPersistenceFilter, apply_persistence_filter

VAL_PREDS_CSV = RESEARCH_DIR / "patient_adaptive_baseline_study" / "adaptive_study_validation_predictions.csv"
VAL_FEATURES_CSV = DATA_DIR / "afdb_5s_features_val.csv"

STRIDE_SEC_VAL = 2.5
VALIDATION_RECORDS = ["04126", "05121", "07879", "07910"]


def parse_ground_truth_episodes(raw_afdb_dir: Path, record_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts reference AFib episode intervals from WFDB .atr annotations."""
    record_episodes = {}
    fs = 250

    for rec in record_ids:
        rec_path = str(raw_afdb_dir / rec)
        ann = wfdb.rdann(rec_path, "atr")
        header = wfdb.rdheader(rec_path)
        total_samples = header.sig_len

        episodes = []
        curr_rhythm = None
        curr_start = None

        for s, aux in zip(ann.sample, ann.aux_note):
            rhythm = aux.strip("\x00").strip()
            if rhythm != curr_rhythm:
                if curr_rhythm == "(AFIB":
                    episodes.append({
                        "start_sec": float(curr_start / fs),
                        "end_sec": float(s / fs),
                        "duration_sec": float((s - curr_start) / fs)
                    })
                curr_rhythm = rhythm
                curr_start = s

        if curr_rhythm == "(AFIB":
            episodes.append({
                "start_sec": float(curr_start / fs),
                "end_sec": float(total_samples / fs),
                "duration_sec": float((total_samples - curr_start) / fs)
            })

        record_episodes[rec] = episodes

    return record_episodes


def compute_contiguous_events(binary_series: np.ndarray) -> int:
    """Counts contiguous runs of 1s (episodes of alerts or false alarms)."""
    if len(binary_series) == 0:
        return 0
    diff = np.diff(np.pad(binary_series, (1, 1), mode="constant", constant_values=0))
    event_starts = np.sum(diff == 1)
    return int(event_starts)


def evaluate_episode_level_performance(
    df_rec: pd.DataFrame,
    alerts: np.ndarray,
    episodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates episode detection rate and alert latency for true AFib episodes.
    """
    total_episodes = len(episodes)
    usable_episodes = 0
    detected_episodes = 0
    missed_episodes = 0
    delays_sec = []

    times = df_rec["start_time_sec"].values

    for ep in episodes:
        s = ep["start_sec"]
        e = ep["end_sec"]

        # Windows whose interval [start_time, end_time] falls inside episode
        ep_mask = (times >= s) & (times <= e)
        ep_alerts = alerts[ep_mask]
        ep_times = times[ep_mask]

        if len(ep_alerts) == 0:
            # Micro-burst episode absorbed by transition guard band
            continue

        usable_episodes += 1
        if np.any(ep_alerts == 1):
            detected_episodes += 1
            first_alert_idx = np.where(ep_alerts == 1)[0][0]
            first_alert_time = ep_times[first_alert_idx]
            delay = max(0.0, float(first_alert_time - s))
            delays_sec.append(delay)
        else:
            missed_episodes += 1

    return {
        "total_wfdb_episodes": total_episodes,
        "usable_episodes": usable_episodes,
        "unusable_micro_episodes": total_episodes - usable_episodes,
        "detected_episodes": detected_episodes,
        "missed_episodes": missed_episodes,
        "episode_sensitivity": round(float(detected_episodes / usable_episodes), 4) if usable_episodes > 0 else 0.0,
        "mean_delay_sec": round(float(np.mean(delays_sec)), 2) if len(delays_sec) > 0 else None,
        "median_delay_sec": round(float(np.median(delays_sec)), 2) if len(delays_sec) > 0 else None,
        "delays_sec": [round(d, 2) for d in delays_sec]
    }


def compute_metrics(
    y_true: np.ndarray,
    alerts: np.ndarray,
    non_afib_hours: float,
    total_hours: float
) -> Dict[str, Any]:
    """Computes comprehensive window-level and rate metrics."""
    tp = int(np.sum((alerts == 1) & (y_true == 1)))
    fp = int(np.sum((alerts == 1) & (y_true == 0)))
    tn = int(np.sum((alerts == 0) & (y_true == 0)))
    fn = int(np.sum((alerts == 0) & (y_true == 1)))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    bal_acc = float(0.5 * (sens + spec))

    fp_rate_non_afib = float(fp / non_afib_hours) if non_afib_hours > 0 else 0.0
    fp_rate_total = float(fp / total_hours) if total_hours > 0 else 0.0

    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "fp_per_eligible_non_afib_hour": round(fp_rate_non_afib, 2),
        "fp_per_total_eligible_hour": round(fp_rate_total, 2),
        "false_alarm_windows_count": fp,
        "false_alarm_events_count": compute_contiguous_events((alerts == 1) & (y_true == 0))
    }


def main():
    print("=" * 80)
    print("CAUSAL TEMPORAL PERSISTENCE & MULTI-WINDOW CONSENSUS STUDY")
    print("=========================================================")
    print("Target: Validation Cohort (Records: 04126, 05121, 07879, 07910; N=58,645)")
    print("Frozen Classifier: Config C (Combined Absolute + Adaptive, LogReg)")
    print("=" * 80)
    t0 = time.time()

    # 1. Ingest Predictions & Ground Truth
    print("\n[1/4] Ingesting Validation Predictions & WFDB Episode Annotations:")
    df_val = pd.read_csv(VAL_PREDS_CSV)
    df_val["record_id"] = df_val["record_id"].astype(str).str.zfill(5)
    print(f"  Ingested {len(df_val):,} validation windows across {df_val['record_id'].nunique()} records")

    val_episodes = parse_ground_truth_episodes(RAW_AFDB_DIR, VALIDATION_RECORDS)
    total_val_eps = sum(len(eps) for eps in val_episodes.values())
    print(f"  Parsed {total_val_eps} ground-truth AFib episodes from WFDB annotations")

    # Overall validation durations
    y_val = df_val["is_afib"].values
    n_total_windows = len(y_val)  # 58,645
    n_non_afib_windows = int(len(y_val) - y_val.sum())  # 37,708
    n_afib_windows = int(y_val.sum())  # 20,937

    non_afib_hours = n_non_afib_windows * STRIDE_SEC_VAL / 3600.0  # 26.1861 h
    total_hours = n_total_windows * STRIDE_SEC_VAL / 3600.0        # 40.7257 h

    print(f"  Total Duration: {total_hours:.2f} hours ({n_total_windows:,} windows)")
    print(f"  Non-AFib Duration: {non_afib_hours:.2f} hours ({n_non_afib_windows:,} windows)")
    print(f"  AFib Duration: {(total_hours - non_afib_hours):.2f} hours ({n_afib_windows:,} windows)")

    # 2. Evaluate Candidate Persistence Rules
    print("\n[2/4] Evaluating Candidate Temporal Persistence Rules on Config C:")
    rules_to_evaluate = CausalTemporalPersistenceFilter.SUPPORTED_RULES
    results = {}

    df_alerts_table = df_val[["record_id", "patient_id", "window_idx", "start_time_sec", "is_afib"]].copy()

    for rule in rules_to_evaluate:
        print(f"\n  Evaluating Rule: '{rule}'...")
        alerts = apply_persistence_filter(
            df_val,
            rule_name=rule,
            prob_col="prob_lr_Config_C_Combined_Absolute_Plus_Adaptive",
            threshold=0.560,
            max_gap_sec=6.0
        )
        df_alerts_table[f"alert_{rule}"] = alerts

        # Pooled window metrics
        pooled_metrics = compute_metrics(y_val, alerts, non_afib_hours, total_hours)

        # Pooled episode metrics
        all_delays = []
        pooled_usable_eps = 0
        pooled_detected_eps = 0
        pooled_missed_eps = 0

        # Per-patient metrics
        patient_results = {}
        for rec in VALIDATION_RECORDS:
            rec_mask = (df_val["record_id"] == rec).values
            y_rec = y_val[rec_mask]
            alerts_rec = alerts[rec_mask]
            df_rec = df_val[rec_mask]

            rec_non_afib_h = (len(y_rec) - y_rec.sum()) * STRIDE_SEC_VAL / 3600.0
            rec_total_h = len(y_rec) * STRIDE_SEC_VAL / 3600.0

            rec_win_metrics = compute_metrics(y_rec, alerts_rec, rec_non_afib_h, rec_total_h)
            rec_ep_metrics = evaluate_episode_level_performance(df_rec, alerts_rec, val_episodes[rec])

            pooled_usable_eps += rec_ep_metrics["usable_episodes"]
            pooled_detected_eps += rec_ep_metrics["detected_episodes"]
            pooled_missed_eps += rec_ep_metrics["missed_episodes"]
            all_delays.extend(rec_ep_metrics["delays_sec"])

            patient_results[rec] = {
                "window_metrics": rec_win_metrics,
                "episode_metrics": rec_ep_metrics
            }

        pooled_ep_metrics = {
            "total_wfdb_episodes": total_val_eps,
            "usable_episodes": pooled_usable_eps,
            "detected_episodes": pooled_detected_eps,
            "missed_episodes": pooled_missed_eps,
            "episode_sensitivity": round(pooled_detected_eps / pooled_usable_eps, 4) if pooled_usable_eps > 0 else 0.0,
            "median_delay_sec": round(float(np.median(all_delays)), 2) if len(all_delays) > 0 else None,
            "mean_delay_sec": round(float(np.mean(all_delays)), 2) if len(all_delays) > 0 else None
        }

        results[rule] = {
            "rule_name": rule,
            "pooled_window_metrics": pooled_metrics,
            "pooled_episode_metrics": pooled_ep_metrics,
            "per_patient_breakdown": patient_results
        }

        print(f"    Window Sens: {pooled_metrics['sensitivity']*100:.2f}% | Spec: {pooled_metrics['specificity']*100:.2f}%")
        print(f"    False Alarms (per non-AFib hr): {pooled_metrics['fp_per_eligible_non_afib_hour']:.2f} FP/hr (Events: {pooled_metrics['false_alarm_events_count']})")
        print(f"    Episode Sens: {pooled_ep_metrics['episode_sensitivity']*100:.2f}% ({pooled_detected_eps}/{pooled_usable_eps}) | Median Delay: {pooled_ep_metrics['median_delay_sec']}s")

    # 3. Save Artifacts
    print("\n[3/4] Saving Persistence Study Artifacts:")
    metrics_path = STUDY_DIR / "persistence_study_validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics to {metrics_path.name}")

    preds_path = STUDY_DIR / "persistence_study_validation_predictions.csv"
    df_alerts_table.to_csv(preds_path, index=False)
    print(f"  Saved predictions table to {preds_path.name}")

    # 4. Print Comprehensive Comparative Summary
    print("\n" + "=" * 105)
    print("COMPARATIVE PERSISTENCE FILTER EVALUATION ON VALIDATION COHORT (N=58,645 WINDOWS, 30 USABLE EPISODES)")
    print("=" * 105)
    header = f"{'Persistence Rule':<22} | {'Win Sens':<9} | {'Win Spec':<9} | {'FP/non-AF h':<11} | {'FP/tot h':<9} | {'FA Events':<9} | {'Ep Sens':<9} | {'Med Delay':<9}"
    print(header)
    print("-" * len(header))

    for rule in rules_to_evaluate:
        wm = results[rule]["pooled_window_metrics"]
        em = results[rule]["pooled_episode_metrics"]
        med_del_str = f"{em['median_delay_sec']}s" if em['median_delay_sec'] is not None else "N/A"
        print(
            f"{rule:<22} | {wm['sensitivity']*100:<8.2f}% | {wm['specificity']*100:<8.2f}% | "
            f"{wm['fp_per_eligible_non_afib_hour']:<11.2f} | {wm['fp_per_total_eligible_hour']:<9.2f} | "
            f"{wm['false_alarm_events_count']:<9} | {em['episode_sensitivity']*100:<8.2f}% | {med_del_str:<9}"
        )
    print("=" * 105)

    # Per-Patient Breakdown Comparison
    print("\nPER-PATIENT FALSE ALARMS (FP PER ELIGIBLE NON-AFIB HOUR) ACROSS RULES:")
    p_header = f"{'Record':<10} | {'Unfiltered':<11} | {'Consec-2':<10} | {'Consec-3':<10} | {'Consec-4':<10} | {'Maj 2-of-3':<11} | {'Hysteresis':<11}"
    print(p_header)
    print("-" * len(p_header))
    for rec in VALIDATION_RECORDS:
        r_unf = results["unfiltered"]["per_patient_breakdown"][rec]["window_metrics"]["fp_per_eligible_non_afib_hour"]
        r_c2 = results["consecutive_2"]["per_patient_breakdown"][rec]["window_metrics"]["fp_per_eligible_non_afib_hour"]
        r_c3 = results["consecutive_3"]["per_patient_breakdown"][rec]["window_metrics"]["fp_per_eligible_non_afib_hour"]
        r_c4 = results["consecutive_4"]["per_patient_breakdown"][rec]["window_metrics"]["fp_per_eligible_non_afib_hour"]
        r_m2 = results["majority_2_of_3"]["per_patient_breakdown"][rec]["window_metrics"]["fp_per_eligible_non_afib_hour"]
        r_hys = results["hysteresis_on3_off2"]["per_patient_breakdown"][rec]["window_metrics"]["fp_per_eligible_non_afib_hour"]
        print(f"{rec:<10} | {r_unf:<11.2f} | {r_c2:<10.2f} | {r_c3:<10.2f} | {r_c4:<10.2f} | {r_m2:<11.2f} | {r_hys:<11.2f}")
    print("-" * len(p_header))

    print("\nPER-PATIENT WINDOW SENSITIVITY ACROSS RULES:")
    s_header = f"{'Record':<10} | {'Unfiltered':<11} | {'Consec-2':<10} | {'Consec-3':<10} | {'Consec-4':<10} | {'Maj 2-of-3':<11} | {'Hysteresis':<11}"
    print(s_header)
    print("-" * len(s_header))
    for rec in VALIDATION_RECORDS:
        s_unf = results["unfiltered"]["per_patient_breakdown"][rec]["window_metrics"]["sensitivity"] * 100
        s_c2 = results["consecutive_2"]["per_patient_breakdown"][rec]["window_metrics"]["sensitivity"] * 100
        s_c3 = results["consecutive_3"]["per_patient_breakdown"][rec]["window_metrics"]["sensitivity"] * 100
        s_c4 = results["consecutive_4"]["per_patient_breakdown"][rec]["window_metrics"]["sensitivity"] * 100
        s_m2 = results["majority_2_of_3"]["per_patient_breakdown"][rec]["window_metrics"]["sensitivity"] * 100
        s_hys = results["hysteresis_on3_off2"]["per_patient_breakdown"][rec]["window_metrics"]["sensitivity"] * 100
        print(f"{rec:<10} | {s_unf:<10.2f}% | {s_c2:<9.2f}% | {s_c3:<9.2f}% | {s_c4:<9.2f}% | {s_m2:<10.2f}% | {s_hys:<10.2f}%")
    print("-" * len(s_header))

    dt = time.time() - t0
    print(f"\nExecution finished in {dt:.1f}s.")


if __name__ == "__main__":
    main()
