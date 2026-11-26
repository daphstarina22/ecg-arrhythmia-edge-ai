"""
Protocol-Integrity Audit: Threshold and Timing Audit for Temporal Persistence.
Target: Validation Cohort (Records 04126, 05121, 07879, 07910; N=58,645 windows)
Frozen Classifier: Config C (Combined Absolute + Causal Adaptive, Logistic Regression)

This script:
1. Compares validation performance across all candidate persistence rules at both threshold 0.500 and 0.560.
2. Corrects the causal alert availability timing:
   - Window [start_time_sec, start_time_sec + 5.0s] completes and becomes available at start_time_sec + 5.0s.
   - Successive overlapping windows are separated by stride (2.5s), not 5.0s.
   - Causal alert delay = (first_alert_window_start + 5.0s) - episode_start.
3. Audits per-patient window metrics, episode detection, detection delays, and false alarm rates.
4. Generates corrected_validation_metrics.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import wfdb

AUDIT_DIR = Path(__file__).resolve().parent
PERSISTENCE_DIR = AUDIT_DIR.parent
ADAPTIVE_STUDY_DIR = PERSISTENCE_DIR.parent
RESEARCH_DIR = ADAPTIVE_STUDY_DIR.parent
PROJECT_ROOT = RESEARCH_DIR.parents[2]

if str(PERSISTENCE_DIR) not in sys.path:
    sys.path.insert(0, str(PERSISTENCE_DIR))

from persistence_filter import CausalTemporalPersistenceFilter, apply_persistence_filter

VAL_PREDS_CSV = ADAPTIVE_STUDY_DIR / "adaptive_study_validation_predictions.csv"
RAW_AFDB_DIR = PROJECT_ROOT / "training" / "research" / "afib_experiment" / "data" / "afdb" / "raw"

VALIDATION_RECORDS = ["04126", "05121", "07879", "07910"]
WINDOW_SEC = 5.0
STRIDE_SEC = 2.5


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


def evaluate_episode_level_performance_causal(
    df_rec: pd.DataFrame,
    alerts: np.ndarray,
    episodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates episode detection rate and alert latency under both:
      1. Nominal timing (window start time, used in previous report).
      2. Causal physical timing (window end time, when prediction is actually available).
    """
    total_episodes = len(episodes)
    usable_episodes = 0
    detected_episodes = 0
    missed_episodes = 0
    nominal_delays = []
    causal_delays = []

    times = df_rec["start_time_sec"].values

    for ep in episodes:
        s = ep["start_sec"]
        e = ep["end_sec"]

        ep_mask = (times >= s) & (times <= e)
        ep_alerts = alerts[ep_mask]
        ep_times = times[ep_mask]

        if len(ep_alerts) == 0:
            continue

        usable_episodes += 1
        if np.any(ep_alerts == 1):
            detected_episodes += 1
            first_idx = np.where(ep_alerts == 1)[0][0]
            first_start_time = ep_times[first_idx]

            # Nominal delay (reported previously: window start time - episode start)
            nom_delay = max(0.0, float(first_start_time - s))
            nominal_delays.append(nom_delay)

            # Causal physical delay (prediction available at window end: start + 5.0s - episode start)
            causal_avail_time = first_start_time + WINDOW_SEC
            causal_delay = max(0.0, float(causal_avail_time - s))
            causal_delays.append(causal_delay)
        else:
            missed_episodes += 1

    return {
        "total_wfdb_episodes": total_episodes,
        "usable_episodes": usable_episodes,
        "unusable_micro_episodes": total_episodes - usable_episodes,
        "detected_episodes": detected_episodes,
        "missed_episodes": missed_episodes,
        "episode_sensitivity": round(float(detected_episodes / usable_episodes), 4) if usable_episodes > 0 else 0.0,
        "nominal_median_delay_sec": round(float(np.median(nominal_delays)), 2) if len(nominal_delays) > 0 else None,
        "nominal_mean_delay_sec": round(float(np.mean(nominal_delays)), 2) if len(nominal_delays) > 0 else None,
        "causal_median_delay_sec": round(float(np.median(causal_delays)), 2) if len(causal_delays) > 0 else None,
        "causal_mean_delay_sec": round(float(np.mean(causal_delays)), 2) if len(causal_delays) > 0 else None,
        "causal_delays_sec": [round(d, 2) for d in causal_delays]
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


def run_audit():
    print("=" * 80)
    print("PROTOCOL-INTEGRITY AUDIT: THRESHOLD & TIMING VERIFICATION")
    print("=" * 80)

    df_val = pd.read_csv(VAL_PREDS_CSV)
    df_val["record_id"] = df_val["record_id"].astype(str).str.zfill(5)
    print(f"Loaded {len(df_val):,} validation windows across records: {df_val['record_id'].unique().tolist()}")

    episodes = parse_ground_truth_episodes(RAW_AFDB_DIR, VALIDATION_RECORDS)
    total_eps = sum(len(e) for e in episodes.values())
    print(f"Loaded {total_eps} ground truth AFib episodes across {len(VALIDATION_RECORDS)} validation records.")

    y_val = df_val["is_afib"].values
    n_non_afib = int(len(y_val) - y_val.sum())
    n_total = len(y_val)

    non_afib_hours = n_non_afib * STRIDE_SEC / 3600.0  # 26.1861 h
    total_hours = n_total * STRIDE_SEC / 3600.0        # 40.7257 h

    thresholds = [0.500, 0.560]
    rules = CausalTemporalPersistenceFilter.SUPPORTED_RULES

    audit_results = {
        "audit_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target": "Validation Cohort (04126, 05121, 07879, 07910)",
            "total_windows": n_total,
            "window_sec": WINDOW_SEC,
            "stride_sec": STRIDE_SEC,
            "non_afib_hours": round(non_afib_hours, 4),
            "total_hours": round(total_hours, 4),
            "thresholds_audited": thresholds
        },
        "results_by_threshold": {}
    }

    for th in thresholds:
        print(f"\n--- AUDITING PROBABILITY THRESHOLD: {th:.3f} ---")
        th_dict = {}

        for rule in rules:
            alerts = apply_persistence_filter(
                df_val,
                rule_name=rule,
                prob_col="prob_lr_Config_C_Combined_Absolute_Plus_Adaptive",
                threshold=th,
                max_gap_sec=6.0
            )

            pooled_win = compute_metrics(y_val, alerts, non_afib_hours, total_hours)

            # Pooled episode evaluation
            all_causal_delays = []
            all_nom_delays = []
            pooled_usable = 0
            pooled_det = 0
            pooled_missed = 0

            per_patient = {}

            for rec in VALIDATION_RECORDS:
                mask = (df_val["record_id"] == rec).values
                df_rec = df_val[mask]
                rec_alerts = alerts[mask]
                rec_y = y_val[mask]
                rec_non_afib_h = (len(rec_y) - rec_y.sum()) * STRIDE_SEC / 3600.0
                rec_tot_h = len(rec_y) * STRIDE_SEC / 3600.0

                p_win = compute_metrics(rec_y, rec_alerts, rec_non_afib_h, rec_tot_h)
                p_ep = evaluate_episode_level_performance_causal(df_rec, rec_alerts, episodes[rec])

                per_patient[rec] = {
                    "window_metrics": p_win,
                    "episode_metrics": p_ep
                }

                pooled_usable += p_ep["usable_episodes"]
                pooled_det += p_ep["detected_episodes"]
                pooled_missed += p_ep["missed_episodes"]
                all_causal_delays.extend(p_ep["causal_delays_sec"])

            pooled_ep = {
                "total_wfdb_episodes": sum(len(episodes[r]) for r in VALIDATION_RECORDS),
                "usable_episodes": pooled_usable,
                "detected_episodes": pooled_det,
                "missed_episodes": pooled_missed,
                "episode_sensitivity": round(float(pooled_det / pooled_usable), 4) if pooled_usable > 0 else 0.0,
                "causal_median_delay_sec": round(float(np.median(all_causal_delays)), 2) if len(all_causal_delays) > 0 else None,
                "causal_mean_delay_sec": round(float(np.mean(all_causal_delays)), 2) if len(all_causal_delays) > 0 else None
            }

            th_dict[rule] = {
                "pooled_window_metrics": pooled_win,
                "pooled_episode_metrics": pooled_ep,
                "per_patient_breakdown": per_patient
            }

            print(f"  Rule: {rule:<20s} | Sens: {pooled_win['sensitivity']:.4f} | Spec: {pooled_win['specificity']:.4f} | FP/non-AF: {pooled_win['fp_per_eligible_non_afib_hour']:5.2f} | Ep Sens: {pooled_ep['episode_sensitivity']:.1%} ({pooled_ep['detected_episodes']}/{pooled_ep['usable_episodes']}) | Causal Delay: {pooled_ep['causal_median_delay_sec']}s")

        audit_results["results_by_threshold"][f"threshold_{th:.3f}"] = th_dict

    out_file = AUDIT_DIR / "corrected_validation_metrics.json"
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nSaved corrected validation metrics to: {out_file}")


if __name__ == "__main__":
    run_audit()
