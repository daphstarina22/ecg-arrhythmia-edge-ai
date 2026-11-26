"""Independent 5-Second Model 2 Validation on CUDB (cu01 and cu04).

Evaluates the existing four-class XGBoost pipeline on completely independent, unseen
Creighton University Ventricular Tachyarrhythmia Database (CUDB) records:
- cu01: Sustained coarse Ventricular Fibrillation (VFIB)
- cu04: Recurrent paroxysmal Ventricular Tachycardia & Flutter (VT/VFL)

Strict Safety & Scope Constraints:
- The trained Model 2 pipeline (research_xgboost_4class_pipeline.joblib) is NOT retrained or modified.
- Feature definitions, preprocessing, decision thresholds, and class mappings remain strictly unchanged.
- Production files, deployed ONNX models, and previous research artifacts are preserved.
- Resampling from 250 Hz to 360 Hz preserves physical calibration (mV).
- Evaluates non-overlapping 5-second windows for primary metrics.
- Evaluates 1-second sliding windows for secondary onset and termination latency analysis.
- Unambiguous intervals, sinus intervals, and transition boundaries are strictly separated.
- Not clinical validation: explicitly documented as a limited two-record independent benchmark.
"""

import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly, welch
import wfdb

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.final_models.common.feature_selection import MODEL2_FEATURES

TARGET_FS = 360
SOURCE_FS = 250
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)  # 1800 samples at 360 Hz
STEP_SAMPLES_5S = WINDOW_SAMPLES              # 1800 samples (non-overlapping)
STEP_SAMPLES_1S = int(TARGET_FS * 1.0)        # 360 samples (1-second sliding stride)

CLASS_NAMES_4C = ["NSR", "TACHY", "BRADYCARDIA", "VENTRICULAR"]
OUTPUT_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = PROJECT_ROOT / "training" / "final_models" / "results" / "model2_xgboost_4class_research" / "research_xgboost_4class_pipeline.joblib"
CUDB_ZIP_PATH = PROJECT_ROOT / "cu-ventricular-tachyarrhythmia-database-1.0.0.zip"
TRAINING_CSV_PATH = PROJECT_ROOT / "training" / "final_models" / "results" / "model2_xgboost_4class_research" / "four_class_master_feature_dataset.csv"


# ==============================================================================
# 1. INDEPENDENCE & ANTI-LEAKAGE VERIFICATION
# ==============================================================================

def verify_record_independence(rec_ids: List[str]) -> Dict[str, Any]:
    """Verifies that none of the target evaluation records were present in training data."""
    audit_results = {}
    if not TRAINING_CSV_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at: {TRAINING_CSV_PATH}")

    df_train = pd.read_csv(TRAINING_CSV_PATH)
    train_records = set(df_train["record_id"].unique())

    for rid in rec_ids:
        # Check both raw id ('cu01') and prefixed key ('cudb_cu01')
        raw_match = rid in train_records
        prefixed_match = f"cudb_{rid}" in train_records
        contained_matches = [r for r in train_records if rid in str(r).lower()]

        is_present = raw_match or prefixed_match or len(contained_matches) > 0
        audit_results[rid] = {
            "record_id": rid,
            "raw_match_in_training": raw_match,
            "prefixed_match_in_training": prefixed_match,
            "matching_training_records": contained_matches,
            "training_windows_count": int(df_train["record_id"].isin([rid, f"cudb_{rid}"]).sum()),
            "is_completely_independent": not is_present,
        }
        if is_present:
            raise ValueError(f"DATA LEAKAGE DETECTED: Record {rid} matches training data: {contained_matches}")

    return {
        "status": "PASSED_ZERO_LEAKAGE",
        "total_training_records_in_dataset": len(train_records),
        "total_training_windows_in_dataset": len(df_train),
        "records_audited": audit_results,
    }


# ==============================================================================
# 2. SIGNAL SAFEGUARDS & TRAINING-CONCORDANT FEATURE EXTRACTION
# ==============================================================================

def butter_bandpass(
    sig: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: int = TARGET_FS,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter matching Model 2 training exactly."""
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)


def evaluate_signal_quality(w: np.ndarray) -> Tuple[str, bool, str]:
    """Evaluates raw calibrated window for non-finite values, flatline, or severe saturation."""
    if np.any(np.isnan(w)) or np.any(np.isinf(w)):
        return "BAD_SIGNAL_NONFINITE", False, "Non-finite values (NaN/Inf) detected in window"

    ptp = float(np.ptp(w))
    std = float(np.std(w))

    if ptp < 0.005 or std < 0.001:
        return "BAD_SIGNAL_LEAD_OFF", False, f"Dead flatline or electrode detachment: ptp={ptp:.4f} mV, std={std:.4f} mV"

    if ptp > 15.0 or std > 4.0:
        return "BAD_SIGNAL_SATURATED", False, f"Severe saturation or baseline wander: ptp={ptp:.2f} mV, std={std:.2f} mV"

    return "SIGNAL_OK", True, "Signal within physiological bounds"


def extract_features_single_window(sig: np.ndarray, fs: int = TARGET_FS) -> Dict[str, float]:
    """Training-concordant feature extractor with documented flatline safeguard and noise floor."""
    ptp_val = float(np.ptp(sig))
    std_val = float(np.std(sig))
    zero_cross = int(np.sum(np.diff(sig > 0) != 0))

    nperseg = min(len(sig), int(fs * 2))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    valid_mask = (freqs >= 0.5) & (freqs <= 30.0)
    freqs_band = freqs[valid_mask]
    psd_band = psd[valid_mask]

    total_power = np.sum(psd_band) + 1e-12
    vf_mask = (freqs_band >= 3.0) & (freqs_band <= 9.0)
    vf_power = np.sum(psd_band[vf_mask])
    vf_ratio = float(vf_power / total_power)

    dom_idx = np.argmax(psd_band)
    dom_freq = float(freqs_band[dom_idx])

    psd_norm = psd_band / total_power
    psd_norm = psd_norm[psd_norm > 0]
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm) + 1e-12))
    spec_purity = float(psd_band[dom_idx] / (np.median(psd_band) + 1e-12))

    diff1 = np.diff(sig)
    diff2 = np.diff(diff1)
    var_zero = std_val**2 + 1e-12
    var_d1 = float(np.var(diff1)) + 1e-12
    var_d2 = float(np.var(diff2)) + 1e-12

    mobility = float(np.sqrt(var_d1 / var_zero))
    complexity = float(np.sqrt(var_d2 / var_d1) / mobility)

    # LZ complexity (downsampled to 90 Hz, N=450)
    step = max(1, int(round(fs / 90.0)))
    sig_down = sig[::step]
    med = np.median(sig_down)
    binary_seq = (sig_down > med).astype(np.int8)

    words = set()
    w = ""
    lz_count = 0
    for b in binary_seq:
        w += str(b)
        if w not in words:
            words.add(w)
            lz_count += 1
            w = ""
    if w:
        lz_count += 1
    n_seq = len(binary_seq)
    lz_norm = float(lz_count * np.log2(n_seq) / n_seq) if n_seq > 0 else 0.0

    # Pan-Tompkins Peak Detection with near-flatline safeguard and noise floor
    if ptp_val < 0.15 or std_val < 0.025:
        peak_count = 0
        mean_rr = 5.0
        rr_cv = 0.0
        qrs_width = 0.0
        hr_bpm = 0.0
    else:
        diff_sq = diff1**2
        int_len = int(fs * 0.150)
        integrated = np.convolve(diff_sq, np.ones(int_len) / int_len, mode="same")
        max_int = float(np.max(integrated))
        threshold = max(0.3 * max_int, 0.00030) if max_int > 0 else 1.0

        refractory = int(fs * 0.300)
        peaks = []
        last_p = -refractory
        for p in range(1, len(integrated) - 1):
            if integrated[p] > integrated[p - 1] and integrated[p] > integrated[p + 1] and integrated[p] > threshold:
                if p - last_p >= refractory:
                    peaks.append(p)
                    last_p = p

        peak_count = len(peaks)
        if peak_count >= 2:
            rr_intervals = np.diff(peaks) / fs
            mean_rr = float(np.mean(rr_intervals))
            rr_cv = float(np.std(rr_intervals) / (mean_rr + 1e-12))
            qrs_width = 0.080
            hr_bpm = 60.0 / mean_rr
        else:
            mean_rr = 5.0
            rr_cv = 0.0
            qrs_width = 0.0
            hr_bpm = 0.0

    return {
        "mean_rr": mean_rr,
        "rr_cv": rr_cv,
        "qrs_width": qrs_width,
        "peak_count": peak_count,
        "dominant_freq": dom_freq,
        "vf_band_power_ratio": vf_ratio,
        "spectral_entropy": spec_entropy,
        "zero_crossings": zero_cross,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "calibrated_ptp": ptp_val,
        "calibrated_std": std_val,
        "heart_rate_bpm": hr_bpm,
    }


# ==============================================================================
# 3. CUDB RECORD LOADER & ANNOTATION PARSER
# ==============================================================================

def load_cudb_record(rec_id: str) -> Dict[str, Any]:
    """Loads a CUDB record from the workspace zip archive, extracting signal & reference annotations."""
    if not CUDB_ZIP_PATH.exists():
        raise FileNotFoundError(f"CUDB archive not found at: {CUDB_ZIP_PATH}")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(CUDB_ZIP_PATH) as z:
            for item in z.namelist():
                if f"/{rec_id}." in item or item.startswith(f"{rec_id}."):
                    fname = os.path.basename(item)
                    with open(os.path.join(tmpdir, fname), "wb") as out_f:
                        out_f.write(z.read(item))

        rec_path = os.path.join(tmpdir, rec_id)
        rec_obj = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")

        # Channel 0 calibrated physical signal in mV
        raw_sig_250 = rec_obj.p_signal[:, 0].astype(np.float32)
        raw_fs = int(rec_obj.fs)
        if raw_fs != SOURCE_FS:
            raise ValueError(f"Expected source fs {SOURCE_FS} Hz, got {raw_fs} Hz for {rec_id}")

        # Polyphase FIR resampling from 250 Hz to TARGET_FS (360 Hz)
        # resample_poly upsamples by 36 and downsamples by 25 (360/250 = 36/25)
        sig_calib_360 = resample_poly(raw_sig_250, TARGET_FS, SOURCE_FS).astype(np.float32)
        sig_filt_360 = butter_bandpass(sig_calib_360, fs=TARGET_FS)

        # Parse rhythm episodes (start marked with '[', termination marked with ']')
        episodes = []
        curr_start = None
        for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
            t_sec = s / SOURCE_FS
            if sym == "[":
                curr_start = (s, t_sec, aux.strip())
            elif sym == "]" and curr_start is not None:
                episodes.append({
                    "start_sample_250": curr_start[0],
                    "end_sample_250": s,
                    "start_time_sec": curr_start[1],
                    "end_time_sec": t_sec,
                    "duration_sec": t_sec - curr_start[1],
                    "aux_note": curr_start[2],
                })
                curr_start = None

        # Parse individual beat annotations
        beats = []
        for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
            beats.append({
                "sample_250": s,
                "time_sec": s / SOURCE_FS,
                "symbol": sym,
                "aux_note": aux.strip(),
            })

    return {
        "record_id": rec_id,
        "source_fs": raw_fs,
        "target_fs": TARGET_FS,
        "sig_len_250": len(raw_sig_250),
        "sig_len_360": len(sig_calib_360),
        "duration_sec": len(raw_sig_250) / SOURCE_FS,
        "adc_gain": float(rec_obj.adc_gain[0]),
        "baseline": float(rec_obj.baseline[0]),
        "units": rec_obj.units[0],
        "sig_name": rec_obj.sig_name[0],
        "calibrated_raw_signal_360": sig_calib_360,
        "calibrated_filtered_signal_360": sig_filt_360,
        "episodes": episodes,
        "beats": beats,
    }


# ==============================================================================
# 4. WINDOW EVALUATION & ANNOTATION ALIGNMENT ENGINE
# ==============================================================================

def classify_window_ground_truth(
    t_start: float,
    t_end: float,
    episodes: List[Dict[str, Any]],
    record_id: str,
) -> Tuple[str, str, float]:
    """Classifies a window into UNAMBIGUOUS_VENTRICULAR, UNAMBIGUOUS_SINUS, or TRANSITION."""
    win_dur = t_end - t_start
    vent_overlap_sec = 0.0

    for ep in episodes:
        overlap = max(0.0, min(t_end, ep["end_time_sec"]) - max(t_start, ep["start_time_sec"]))
        vent_overlap_sec += overlap

    vent_fraction = min(1.0, vent_overlap_sec / win_dur)

    if vent_fraction >= 0.999:
        category = "UNAMBIGUOUS_VENTRICULAR"
        ref_rhythm = "VENTRICULAR"
    elif vent_fraction <= 0.001:
        category = "UNAMBIGUOUS_SINUS"
        ref_rhythm = "NSR"
    else:
        category = "TRANSITION"
        ref_rhythm = f"TRANSITION_VENT_{vent_fraction:.2f}"

    return category, ref_rhythm, float(vent_fraction)


def run_windowed_evaluation(
    rec_data: Dict[str, Any],
    pipeline: Any,
    step_samples: int,
    stride_mode: str = "5s_non_overlapping",
) -> pd.DataFrame:
    """Executes Model 2 feature extraction and inference across windowed ECG signal."""
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    imputer = pipeline.named_steps["imputer"]

    sig_calib = rec_data["calibrated_raw_signal_360"]
    sig_filt = rec_data["calibrated_filtered_signal_360"]
    episodes = rec_data["episodes"]
    rec_id = rec_data["record_id"]

    total_samples = len(sig_filt)
    n_windows = (total_samples - WINDOW_SAMPLES) // step_samples + 1

    rows = []
    for win_idx in range(n_windows):
        start_samp = win_idx * step_samples
        end_samp = start_samp + WINDOW_SAMPLES
        t_start = start_samp / TARGET_FS
        t_end = end_samp / TARGET_FS

        w_calib = sig_calib[start_samp:end_samp]
        w_filt = sig_filt[start_samp:end_samp]

        # 1. Quality evaluation on calibrated raw window
        sq_status, is_valid, sq_msg = evaluate_signal_quality(w_calib)

        # 2. Ground-truth annotation alignment
        category, ref_rhythm, vent_frac = classify_window_ground_truth(t_start, t_end, episodes, rec_id)

        row_dict = {
            "record_id": f"cudb_{rec_id}",
            "raw_record_id": rec_id,
            "stride_mode": stride_mode,
            "window_index": win_idx,
            "time_start_sec": round(t_start, 3),
            "time_end_sec": round(t_end, 3),
            "ventricular_overlap_fraction": round(vent_frac, 3),
            "ground_truth_category": category,
            "reference_rhythm": ref_rhythm,
            "signal_quality": sq_status,
            "is_evaluable": is_valid,
            "signal_quality_detail": sq_msg,
        }

        if not is_valid:
            for f in MODEL2_FEATURES:
                row_dict[f] = np.nan
            row_dict["heart_rate_bpm"] = np.nan
            row_dict["predicted_class_4c"] = "UNEVALUABLE"
            for c_name in CLASS_NAMES_4C:
                row_dict[f"prob_{c_name}"] = np.nan
        else:
            # 3. Extract 12 Model 2 features
            feats = extract_features_single_window(w_filt, fs=TARGET_FS)
            for f in MODEL2_FEATURES:
                row_dict[f] = feats[f]
            row_dict["heart_rate_bpm"] = feats["heart_rate_bpm"]

            # 4. Pipeline inference
            feat_vec = np.array([[feats[f] for f in MODEL2_FEATURES]], dtype=np.float32)
            X_proc = scaler.transform(imputer.transform(feat_vec))
            pred_idx = int(clf.predict(X_proc)[0])
            probs = clf.predict_proba(X_proc)[0]

            row_dict["predicted_class_4c"] = CLASS_NAMES_4C[pred_idx]
            for c_idx, c_name in enumerate(CLASS_NAMES_4C):
                row_dict[f"prob_{c_name}"] = float(probs[c_idx])

        rows.append(row_dict)

    return pd.DataFrame(rows)


# ==============================================================================
# 5. METRICS CALCULATION ENGINE
# ==============================================================================

def compute_record_metrics(df_5s: pd.DataFrame, df_1s: Optional[pd.DataFrame], episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive non-overlapping metrics, confusion matrices, and latency metrics."""
    total_wins = len(df_5s)
    evaluable_wins = int(df_5s["is_evaluable"].sum())

    # Category counts
    cat_counts = df_5s["ground_truth_category"].value_counts().to_dict()
    pred_counts = df_5s["predicted_class_4c"].value_counts().to_dict()

    # Filter to unambiguous windows for primary statistical metrics
    df_unambig = df_5s[df_5s["ground_truth_category"].isin(["UNAMBIGUOUS_SINUS", "UNAMBIGUOUS_VENTRICULAR"])].copy()
    df_sinus = df_5s[df_5s["ground_truth_category"] == "UNAMBIGUOUS_SINUS"].copy()
    df_vent = df_5s[df_5s["ground_truth_category"] == "UNAMBIGUOUS_VENTRICULAR"].copy()
    df_trans = df_5s[df_5s["ground_truth_category"] == "TRANSITION"].copy()

    # Unambiguous Ventricular Performance (Sensitivity / Recall)
    n_vent = len(df_vent)
    tp_vent = int((df_vent["predicted_class_4c"] == "VENTRICULAR").sum()) if n_vent > 0 else 0
    fn_vent = n_vent - tp_vent
    vent_recall = float(tp_vent / n_vent) if n_vent > 0 else 0.0

    # Unambiguous Sinus Performance (Specificity and False Alarm Rate)
    n_sinus = len(df_sinus)
    tn_sinus = int((df_sinus["predicted_class_4c"] == "NSR").sum()) if n_sinus > 0 else 0
    fp_vent_on_sinus = int((df_sinus["predicted_class_4c"] == "VENTRICULAR").sum()) if n_sinus > 0 else 0
    fp_brady_on_sinus = int((df_sinus["predicted_class_4c"] == "BRADYCARDIA").sum()) if n_sinus > 0 else 0
    fp_tachy_on_sinus = int((df_sinus["predicted_class_4c"] == "TACHY").sum()) if n_sinus > 0 else 0

    sinus_nsr_concordance = float(tn_sinus / n_sinus) if n_sinus > 0 else 0.0
    sinus_vent_fpr = float(fp_vent_on_sinus / n_sinus) if n_sinus > 0 else 0.0
    sinus_any_alarm_rate = float((n_sinus - tn_sinus) / n_sinus) if n_sinus > 0 else 0.0

    # Confusion Matrix on unambiguous windows
    conf_matrix = {}
    for true_cat in ["UNAMBIGUOUS_SINUS", "UNAMBIGUOUS_VENTRICULAR"]:
        sub = df_unambig[df_unambig["ground_truth_category"] == true_cat]
        conf_matrix[true_cat] = {c: int((sub["predicted_class_4c"] == c).sum()) for c in CLASS_NAMES_4C}

    # Probability summaries for Ventricular class
    vent_probs_during_vent = df_vent["prob_VENTRICULAR"].dropna().tolist()
    vent_probs_during_sinus = df_sinus["prob_VENTRICULAR"].dropna().tolist()

    prob_summary = {
        "ventricular_prob_during_ventricular": {
            "mean": float(np.mean(vent_probs_during_vent)) if vent_probs_during_vent else 0.0,
            "median": float(np.median(vent_probs_during_vent)) if vent_probs_during_vent else 0.0,
            "min": float(np.min(vent_probs_during_vent)) if vent_probs_during_vent else 0.0,
            "max": float(np.max(vent_probs_during_vent)) if vent_probs_during_vent else 0.0,
        },
        "ventricular_prob_during_sinus": {
            "mean": float(np.mean(vent_probs_during_sinus)) if vent_probs_during_sinus else 0.0,
            "median": float(np.median(vent_probs_during_sinus)) if vent_probs_during_sinus else 0.0,
            "min": float(np.min(vent_probs_during_sinus)) if vent_probs_during_sinus else 0.0,
            "max": float(np.max(vent_probs_during_sinus)) if vent_probs_during_sinus else 0.0,
        },
    }

    # Transition windows analysis
    transition_breakdown = df_trans["predicted_class_4c"].value_counts().to_dict() if len(df_trans) > 0 else {}

    # Episode-level onset and termination timing (using 1s sliding windows if provided)
    episode_latencies = []
    if df_1s is not None and episodes:
        for ep_idx, ep in enumerate(episodes, 1):
            t_on = ep["start_time_sec"]
            t_off = ep["end_time_sec"]

            # Onset search: windows ending after onset
            # First window where predicted_class_4c == 'VENTRICULAR' on or after onset
            post_onset_wins = df_1s[(df_1s["time_end_sec"] >= t_on) & (df_1s["time_start_sec"] <= t_off)].copy()
            vent_wins = post_onset_wins[post_onset_wins["predicted_class_4c"] == "VENTRICULAR"]

            if len(vent_wins) > 0:
                first_vent_win = vent_wins.iloc[0]
                first_vent_time = first_vent_win["time_start_sec"]
                onset_latency = max(0.0, first_vent_time - t_on)
                onset_detected = True
            else:
                onset_latency = None
                onset_detected = False

            # Termination search: windows starting after termination
            # First window where predicted_class_4c != 'VENTRICULAR' on or after termination
            post_off_wins = df_1s[df_1s["time_start_sec"] >= t_off].copy()
            non_vent_wins = post_off_wins[post_off_wins["predicted_class_4c"] != "VENTRICULAR"]

            if len(non_vent_wins) > 0:
                first_non_vent_win = non_vent_wins.iloc[0]
                first_non_vent_time = first_non_vent_win["time_start_sec"]
                term_latency = max(0.0, first_non_vent_time - t_off)
                term_deescalated = True
            else:
                term_latency = None
                term_deescalated = False

            episode_latencies.append({
                "episode_index": ep_idx,
                "start_time_sec": t_on,
                "end_time_sec": t_off,
                "duration_sec": ep["duration_sec"],
                "aux_note": ep["aux_note"],
                "onset_detected": onset_detected,
                "onset_detection_latency_sec_1s_stride": round(onset_latency, 2) if onset_latency is not None else None,
                "termination_deescalated": term_deescalated,
                "termination_latency_sec_1s_stride": round(term_latency, 2) if term_latency is not None else None,
            })

    return {
        "total_windows_5s": total_wins,
        "evaluable_windows_5s": evaluable_wins,
        "ground_truth_category_counts": cat_counts,
        "predicted_class_counts_overall": pred_counts,
        "unambiguous_evaluation": {
            "total_unambiguous_windows": len(df_unambig),
            "unambiguous_ventricular_windows": n_vent,
            "ventricular_true_positives": tp_vent,
            "ventricular_false_negatives": fn_vent,
            "ventricular_recall": round(vent_recall, 4),
            "unambiguous_sinus_windows": n_sinus,
            "sinus_true_negatives_nsr": tn_sinus,
            "sinus_nsr_concordance_rate": round(sinus_nsr_concordance, 4),
            "sinus_ventricular_false_alarm_rate": round(sinus_vent_fpr, 4),
            "sinus_predicted_bradycardia_count": fp_brady_on_sinus,
            "sinus_predicted_tachycardia_count": fp_tachy_on_sinus,
            "sinus_predicted_ventricular_count": fp_vent_on_sinus,
            "confusion_matrix_unambiguous": conf_matrix,
        },
        "probability_distributions": prob_summary,
        "transition_windows": {
            "total_transition_windows": len(df_trans),
            "transition_predictions_breakdown": transition_breakdown,
        },
        "episode_timing_and_latencies": episode_latencies,
    }


# ==============================================================================
# 6. MAIN ORCHESTRATOR
# ==============================================================================

def main():
    print("=" * 80)
    print("INDEPENDENT 5-SECOND MODEL 2 VALIDATION ON CUDB (cu01 and cu04)")
    print("=" * 80)

    # Step 1: Independence & Anti-Leakage Verification
    target_records = ["cu01", "cu04"]
    print(f"\n[Step 1] Verifying data independence against 4-class training dataset...")
    leakage_audit = verify_record_independence(target_records)
    print(f"Status: {leakage_audit['status']}")
    for r, info in leakage_audit["records_audited"].items():
        print(f"  - Record {r}: independent={info['is_completely_independent']}, training rows={info['training_windows_count']}")

    # Step 2: Load Model Pipeline
    print(f"\n[Step 2] Loading trained 4-class Model 2 pipeline...")
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Trained pipeline artifact missing: {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
    print("Pipeline loaded successfully. Steps:", list(pipeline.named_steps.keys()))

    # Step 3: Evaluate cu01 and cu04
    all_5s_dfs = []
    all_1s_dfs = []
    record_metrics_all = {}

    for rec_id in target_records:
        print(f"\n[Step 3] Loading & processing CUDB record: {rec_id}...")
        rec_data = load_cudb_record(rec_id)
        print(f"  Signal length: {rec_data['sig_len_250']} samples @ 250 Hz ({rec_data['duration_sec']:.2f} s)")
        print(f"  Resampled length: {rec_data['sig_len_360']} samples @ 360 Hz")
        print(f"  Calibrated units: {rec_data['units']}, gain: {rec_data['adc_gain']} counts/mV, baseline: {rec_data['baseline']}")
        print(f"  Ventricular episodes found: {len(rec_data['episodes'])}")
        for ep in rec_data["episodes"]:
            print(f"    Episode: {ep['start_time_sec']:.2f}s .. {ep['end_time_sec']:.2f}s ({ep['duration_sec']:.2f}s) | aux='{ep['aux_note']}'")

        # 5-second non-overlapping evaluation
        print(f"\n  Running primary non-overlapping 5-second window evaluation...")
        df_5s = run_windowed_evaluation(rec_data, pipeline, STEP_SAMPLES_5S, stride_mode="5s_non_overlapping")
        csv_5s_path = OUTPUT_DIR / f"cudb_{rec_id}_5s_predictions.csv"
        df_5s.to_csv(csv_5s_path, index=False)
        print(f"  Saved 5s predictions ({len(df_5s)} windows) to: {csv_5s_path.name}")

        # 1-second sliding evaluation for timing
        print(f"  Running secondary 1-second sliding window timing evaluation...")
        df_1s = run_windowed_evaluation(rec_data, pipeline, STEP_SAMPLES_1S, stride_mode="1s_sliding")
        csv_1s_path = OUTPUT_DIR / f"cudb_{rec_id}_1s_sliding_predictions.csv"
        df_1s.to_csv(csv_1s_path, index=False)
        print(f"  Saved 1s sliding predictions ({len(df_1s)} windows) to: {csv_1s_path.name}")

        # Compute record metrics
        rec_metrics = compute_record_metrics(df_5s, df_1s, rec_data["episodes"])
        record_metrics_all[f"cudb_{rec_id}"] = rec_metrics
        all_5s_dfs.append(df_5s)
        all_1s_dfs.append(df_1s)

    # Combined 5-second dataset & combined metrics
    print(f"\n[Step 4] Computing combined benchmark metrics...")
    df_combined_5s = pd.concat(all_5s_dfs, ignore_index=True)
    combined_csv_path = OUTPUT_DIR / "cudb_combined_5s_predictions.csv"
    df_combined_5s.to_csv(combined_csv_path, index=False)
    print(f"Saved combined 5s predictions ({len(df_combined_5s)} windows) to: {combined_csv_path.name}")

    combined_metrics = compute_record_metrics(df_combined_5s, None, [])
    # Aggregate latencies from individual records
    combined_latencies = []
    for r_k, r_m in record_metrics_all.items():
        combined_latencies.extend(r_m["episode_timing_and_latencies"])
    combined_metrics["episode_timing_and_latencies"] = combined_latencies

    final_results_payload = {
        "benchmark_metadata": {
            "evaluation_title": "Independent 5-Second Model 2 Validation on CUDB (cu01 and cu04)",
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_model": "research_xgboost_4class_pipeline.joblib",
            "classes": CLASS_NAMES_4C,
            "target_database": "Creighton University Ventricular Tachyarrhythmia Database (CUDB)",
            "source_archive": "cu-ventricular-tachyarrhythmia-database-1.0.0.zip",
            "records_evaluated": target_records,
            "source_sampling_rate_hz": SOURCE_FS,
            "resampled_sampling_rate_hz": TARGET_FS,
            "window_length_sec": WINDOW_SEC,
            "window_length_samples_360hz": WINDOW_SAMPLES,
            "primary_stride_sec": 5.0,
            "secondary_timing_stride_sec": 1.0,
        },
        "anti_leakage_audit": leakage_audit,
        "combined_benchmark_metrics": combined_metrics,
        "per_record_metrics": record_metrics_all,
    }

    metrics_json_path = OUTPUT_DIR / "cudb_validation_metrics.json"
    with open(metrics_json_path, "w") as jf:
        json.dump(final_results_payload, jf, indent=2)
    print(f"Saved structured metrics JSON to: {metrics_json_path.name}")

    print("\n" + "=" * 80)
    print("CUDB INDEPENDENT VALIDATION SUMMARY")
    print("=" * 80)
    for rec_key, m in record_metrics_all.items():
        unamb = m["unambiguous_evaluation"]
        print(f"\n--- {rec_key.upper()} ---")
        print(f"Total 5s Windows: {m['total_windows_5s']} (Evaluable: {m['evaluable_windows_5s']})")
        print(f"Ventricular Recall (Sensitivity): {unamb['ventricular_recall']:.2%} ({unamb['ventricular_true_positives']}/{unamb['unambiguous_ventricular_windows']})")
        print(f"Sinus NSR Concordance: {unamb['sinus_nsr_concordance_rate']:.2%} ({unamb['sinus_true_negatives_nsr']}/{unamb['unambiguous_sinus_windows']})")
        print(f"Sinus Ventricular False Alarm Rate: {unamb['sinus_ventricular_false_alarm_rate']:.2%} ({unamb['sinus_predicted_ventricular_count']}/{unamb['unambiguous_sinus_windows']})")
        print(f"Sinus Bradycardia Predictions: {unamb['sinus_predicted_bradycardia_count']}")
        print(f"Confusion Matrix (Unambiguous): {unamb['confusion_matrix_unambiguous']}")
        print(f"Episode Timing:")
        for ep in m["episode_timing_and_latencies"]:
            print(f"  Ep {ep['episode_index']} ({ep['duration_sec']:.1f}s): Onset latency={ep['onset_detection_latency_sec_1s_stride']}s, Term latency={ep['termination_latency_sec_1s_stride']}s")

    comb_unamb = combined_metrics["unambiguous_evaluation"]
    print(f"\n--- COMBINED CUDB COHORT (cu01 + cu04) ---")
    print(f"Total Unambiguous Windows: {comb_unamb['total_unambiguous_windows']}")
    print(f"Combined Ventricular Recall: {comb_unamb['ventricular_recall']:.2%} ({comb_unamb['ventricular_true_positives']}/{comb_unamb['unambiguous_ventricular_windows']})")
    print(f"Combined Sinus NSR Concordance: {comb_unamb['sinus_nsr_concordance_rate']:.2%} ({comb_unamb['sinus_true_negatives_nsr']}/{comb_unamb['unambiguous_sinus_windows']})")
    print(f"Combined Sinus Ventricular False Alarm Rate: {comb_unamb['sinus_ventricular_false_alarm_rate']:.2%} ({comb_unamb['sinus_predicted_ventricular_count']}/{comb_unamb['unambiguous_sinus_windows']})")
    print(f"Combined Confusion Matrix:\n{json.dumps(comb_unamb['confusion_matrix_unambiguous'], indent=2)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
