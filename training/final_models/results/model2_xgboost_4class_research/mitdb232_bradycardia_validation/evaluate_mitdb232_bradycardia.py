"""Evaluation of Four-Class Model 2 Pipeline on MIT-BIH Record 232 (Sinus Bradycardia).

Conducts a full-record 30-minute evaluation of the unchanged four-class Model 2 XGBoost pipeline
(research_xgboost_4class_pipeline.joblib) on MIT-BIH record 232 (mitdb_232).

Strict Provenance & Disclosure Constraints:
- mitdb_232 contributed 121 windows to the supervised 4-class training dataset.
- This is a full-record evaluation of a training-overlap record, NOT independent validation.
- All 361 non-overlapping 5-second windows across the 30-minute recording are evaluated.
- Windows are partitioned into in-sample training windows vs. out-of-training windows.
- Rate behavior and beat-level annotations (frequent APBs/PACs with RBBB aberration) are explicitly tracked.
- Zero models retrained; zero features modified; zero thresholds changed; zero production code touched.
"""

import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.final_models.common.feature_selection import MODEL2_FEATURES

TARGET_FS = 360
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)  # 1800 samples at 360 Hz
STEP_SAMPLES = WINDOW_SAMPLES                 # Non-overlapping 5-second windows

CLASS_NAMES_4C = ["NSR", "TACHY", "BRADYCARDIA", "VENTRICULAR"]
OUTPUT_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = PROJECT_ROOT / "training" / "final_models" / "results" / "model2_xgboost_4class_research" / "research_xgboost_4class_pipeline.joblib"
TRAINING_CSV_PATH = PROJECT_ROOT / "training" / "final_models" / "results" / "model2_xgboost_4class_research" / "four_class_master_feature_dataset.csv"
MITBIH_ZIP_PATH = PROJECT_ROOT / "MITBIH.zip"


# ==============================================================================
# 1. SIGNAL SAFEGUARDS & TRAINING-CONCORDANT FEATURE EXTRACTION
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
# 2. PROVENANCE AUDIT & TRAINING OVERLAP IDENTIFIER
# ==============================================================================

def audit_record_provenance() -> Dict[str, Any]:
    """Audits the training representation of mitdb_232 in the 4-class dataset."""
    if not TRAINING_CSV_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAINING_CSV_PATH}")

    df_train = pd.read_csv(TRAINING_CSV_PATH)
    sub232 = df_train[df_train["record_id"].isin(["232", "mitdb_232", "mitbih_232"])].copy()

    training_win_indices = sorted(sub232["window_index"].astype(int).tolist())
    class_counts = sub232["cardiac_rhythm_4class"].value_counts().to_dict()

    return {
        "record_id": "mitdb_232",
        "provenance_status": "TRAINING_OVERLAP_RECORD",
        "is_independent_validation": False,
        "total_windows_in_training_dataset": len(sub232),
        "training_class_breakdown": class_counts,
        "training_window_indices": training_win_indices,
        "training_time_range_sec": [
            float(sub232["time_start_sec"].min()),
            float(sub232["time_end_sec"].max()),
        ] if len(sub232) > 0 else [],
    }


# ==============================================================================
# 3. MIT-BIH RECORD 232 LOADER & ANNOTATION PARSER
# ==============================================================================

def load_mitbih_232() -> Dict[str, Any]:
    """Loads signal and beat annotations for record 232 from MITBIH.zip."""
    if not MITBIH_ZIP_PATH.exists():
        raise FileNotFoundError(f"MITBIH archive not found at: {MITBIH_ZIP_PATH}")

    with zipfile.ZipFile(MITBIH_ZIP_PATH) as z:
        # Load signal CSV
        with z.open("232.csv") as f:
            df_csv = pd.read_csv(f)

        mlii_cols = [c for c in df_csv.columns if "MLII" in c]
        if not mlii_cols:
            raise ValueError("Lead MLII not found in 232.csv")
        lead_name = mlii_cols[0]
        raw_sig = df_csv[lead_name].values.astype(np.float32)

        # Standard MIT-BIH calibration: 200 counts/mV, baseline 1024
        sig_calib = ((raw_sig - 1024.0) / 200.0).astype(np.float32)
        sig_filt = butter_bandpass(sig_calib, fs=TARGET_FS)

        # Load annotations text
        with z.open("232annotations.txt") as f:
            ann_text = f.read().decode("utf-8", errors="replace")

    # Parse annotations
    rhythm_markers = []
    beat_annotations = []
    beat_type_counts = {}

    for line in ann_text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                s_idx = int(parts[1])
                sym = parts[2]
                aux_note = ""
                for p in parts[3:]:
                    if p.startswith("("):
                        aux_note = p
                        rhythm_markers.append((s_idx, p))

                beat_annotations.append((s_idx, sym, aux_note))
                beat_type_counts[sym] = beat_type_counts.get(sym, 0) + 1
            except ValueError:
                pass

    return {
        "record_id": "mitdb_232",
        "fs": TARGET_FS,
        "lead_name": lead_name,
        "total_samples": len(sig_calib),
        "duration_sec": len(sig_calib) / TARGET_FS,
        "calibrated_signal": sig_calib,
        "filtered_signal": sig_filt,
        "rhythm_markers": rhythm_markers,
        "beat_annotations": beat_annotations,
        "beat_type_counts": beat_type_counts,
    }


# ==============================================================================
# 4. FULL-RECORD EVALUATION ENGINE
# ==============================================================================

def evaluate_mitdb_232(
    rec_data: Dict[str, Any],
    pipeline: Any,
    training_win_indices: List[int],
) -> pd.DataFrame:
    """Evaluates all non-overlapping 5-second windows of record 232."""
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    imputer = pipeline.named_steps["imputer"]

    sig_calib = rec_data["calibrated_signal"]
    sig_filt = rec_data["filtered_signal"]
    beat_annotations = rec_data["beat_annotations"]
    train_indices_set = set(training_win_indices)

    total_samples = len(sig_filt)
    n_windows = total_samples // WINDOW_SAMPLES

    rows = []
    for win_idx in range(n_windows):
        start_samp = win_idx * STEP_SAMPLES
        end_samp = start_samp + WINDOW_SAMPLES
        t_start = start_samp / TARGET_FS
        t_end = end_samp / TARGET_FS

        w_calib = sig_calib[start_samp:end_samp]
        w_filt = sig_filt[start_samp:end_samp]

        # 1. Quality evaluation
        sq_status, is_valid, sq_msg = evaluate_signal_quality(w_calib)

        # 2. Extract features
        feats = extract_features_single_window(w_filt, fs=TARGET_FS)

        # 3. Count annotated beats in window
        window_beats = [sym for s, sym, aux in beat_annotations if start_samp <= s < end_samp]
        n_a_beats = sum(1 for sym in window_beats if sym == "A")  # APB with RBBB aberration
        n_r_beats = sum(1 for sym in window_beats if sym == "R")  # Conducted RBBB sinus beat
        n_noise_markers = sum(1 for sym in window_beats if sym == "~")
        total_annotated_beats = len(window_beats)

        # 4. Model inference
        feat_vec = np.array([[feats[f] for f in MODEL2_FEATURES]], dtype=np.float32)
        X_proc = scaler.transform(imputer.transform(feat_vec))
        pred_idx = int(clf.predict(X_proc)[0])
        probs = clf.predict_proba(X_proc)[0]
        pred_class = CLASS_NAMES_4C[pred_idx]

        # 5. Categorize rate and clinical behavior
        # In record 232, global rhythm is (SBR. But frequent PAC runs elevate window heart rate.
        hr = feats["heart_rate_bpm"]
        peaks = feats["peak_count"]
        is_brady_rate = (hr < 60.0) and (peaks <= 5) and (sq_status == "SIGNAL_OK")

        if is_brady_rate:
            window_category = "RATE_CONFIRMED_BRADYCARDIA"
            reference_label = "BRADYCARDIA"
        else:
            window_category = "PAC_BURST_ELEVATED_RATE"
            reference_label = "SINUS_BRADY_WITH_PAC_BURST"

        is_in_training = win_idx in train_indices_set

        row_dict = {
            "record_id": "mitdb_232",
            "window_index": win_idx,
            "time_start_sec": round(t_start, 2),
            "time_end_sec": round(t_end, 2),
            "in_4class_training": is_in_training,
            "reference_rhythm": "(SBR",
            "reference_category": window_category,
            "reference_label": reference_label,
            "is_brady_rate": is_brady_rate,
            "signal_quality": sq_status,
            "is_evaluable": is_valid,
            "annotated_beats_total": total_annotated_beats,
            "r_beats_count": n_r_beats,
            "a_beats_count": n_a_beats,
            "noise_markers_count": n_noise_markers,
            "pan_tompkins_peaks": peaks,
            "heart_rate_bpm": round(hr, 2),
            "predicted_class_4c": pred_class,
            "prob_NSR": float(probs[0]),
            "prob_TACHY": float(probs[1]),
            "prob_BRADYCARDIA": float(probs[2]),
            "prob_VENTRICULAR": float(probs[3]),
        }
        for f in MODEL2_FEATURES:
            row_dict[f] = feats[f]

        rows.append(row_dict)

    return pd.DataFrame(rows)


# ==============================================================================
# 5. METRICS AGGREGATION & REPORTING
# ==============================================================================

def compute_mitdb_232_metrics(df: pd.DataFrame, prov_audit: Dict[str, Any]) -> Dict[str, Any]:
    """Computes comprehensive performance metrics for record 232."""
    total_wins = len(df)
    evaluable_wins = int(df["is_evaluable"].sum())

    # Overall predicted class distribution
    overall_counts = df["predicted_class_4c"].value_counts().to_dict()

    # In-training subset (the 121 windows used during training)
    df_train = df[df["in_4class_training"]].copy()
    n_train = len(df_train)
    train_pred_counts = df_train["predicted_class_4c"].value_counts().to_dict()
    train_brady_tp = int((df_train["predicted_class_4c"] == "BRADYCARDIA").sum())
    train_brady_recall = float(train_brady_tp / n_train) if n_train > 0 else 0.0

    # Out-of-training subset (the 240 windows excluded from training)
    df_out = df[~df["in_4class_training"]].copy()
    n_out = len(df_out)
    out_pred_counts = df_out["predicted_class_4c"].value_counts().to_dict()

    # Rate-confirmed bradycardia subset (HR < 60 bpm, peak_count <= 5)
    df_brady_rate = df[df["is_brady_rate"]].copy()
    n_brady_rate = len(df_brady_rate)
    brady_rate_tp = int((df_brady_rate["predicted_class_4c"] == "BRADYCARDIA").sum())
    brady_rate_recall = float(brady_rate_tp / n_brady_rate) if n_brady_rate > 0 else 0.0

    # Elevated-rate PAC burst subset (HR >= 60 bpm)
    df_pac_burst = df[~df["is_brady_rate"]].copy()
    n_pac_burst = len(df_pac_burst)
    pac_burst_pred_counts = df_pac_burst["predicted_class_4c"].value_counts().to_dict()

    # NSR Leakage
    nsr_leakage_count = int((df["predicted_class_4c"] == "NSR").sum())
    nsr_leakage_rate_overall = float(nsr_leakage_count / total_wins) if total_wins > 0 else 0.0

    # Ventricular False Alarms
    vent_fa_count = int((df["predicted_class_4c"] == "VENTRICULAR").sum())
    vent_fa_rate_overall = float(vent_fa_count / total_wins) if total_wins > 0 else 0.0

    # Probability summary across all windows
    prob_stats = {}
    for c_name in CLASS_NAMES_4C:
        vals = df[f"prob_{c_name}"].dropna().tolist()
        prob_stats[c_name] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "median": float(np.median(vals)) if vals else 0.0,
            "min": float(np.min(vals)) if vals else 0.0,
            "max": float(np.max(vals)) if vals else 0.0,
            "q25": float(np.percentile(vals, 25)) if vals else 0.0,
            "q75": float(np.percentile(vals, 75)) if vals else 0.0,
        }

    return {
        "record_provenance": prov_audit,
        "total_windows_evaluated": total_wins,
        "evaluable_windows": evaluable_wins,
        "predicted_class_counts_overall": overall_counts,
        "in_training_subset_metrics": {
            "total_in_training_windows": n_train,
            "predicted_class_breakdown": train_pred_counts,
            "in_sample_bradycardia_recall": round(train_brady_recall, 4),
            "in_sample_bradycardia_accuracy": round(train_brady_recall, 4),
        },
        "out_of_training_subset_metrics": {
            "total_out_of_training_windows": n_out,
            "predicted_class_breakdown": out_pred_counts,
            "predicted_nsr_leakage_count": int((df_out["predicted_class_4c"] == "NSR").sum()),
            "predicted_ventricular_false_alarm_count": int((df_out["predicted_class_4c"] == "VENTRICULAR").sum()),
            "predicted_bradycardia_count": int((df_out["predicted_class_4c"] == "BRADYCARDIA").sum()),
        },
        "rate_behavior_concordance": {
            "rate_confirmed_bradycardia_windows": n_brady_rate,
            "rate_confirmed_bradycardia_recall": round(brady_rate_recall, 4),
            "pac_burst_elevated_rate_windows": n_pac_burst,
            "pac_burst_predicted_class_breakdown": pac_burst_pred_counts,
        },
        "error_and_leakage_analysis": {
            "total_nsr_leakage_windows": nsr_leakage_count,
            "nsr_leakage_rate_overall": round(nsr_leakage_rate_overall, 4),
            "total_ventricular_false_alarm_windows": vent_fa_count,
            "ventricular_false_alarm_rate_overall": round(vent_fa_rate_overall, 4),
        },
        "probability_distributions": prob_stats,
    }


# ==============================================================================
# 6. MAIN ORCHESTRATOR
# ==============================================================================

def main():
    print("=" * 80)
    print("FOUR-CLASS MODEL 2 EVALUATION ON MIT-BIH RECORD 232 (SINUS BRADYCARDIA)")
    print("=" * 80)

    # Step 1: Provenance Audit
    print("\n[Step 1] Auditing record provenance in 4-class training dataset...")
    prov_audit = audit_record_provenance()
    print(f"Status: {prov_audit['provenance_status']}")
    print(f"Disclosed: mitdb_232 contributed {prov_audit['total_windows_in_training_dataset']} windows to training.")
    print(f"Is Independent Validation: {prov_audit['is_independent_validation']} (TRAINING-OVERLAP RECORD)")

    # Step 2: Load Model Pipeline
    print("\n[Step 2] Loading trained 4-class Model 2 pipeline...")
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Trained pipeline artifact missing: {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
    print("Pipeline loaded successfully. Steps:", list(pipeline.named_steps.keys()))

    # Step 3: Load Record 232
    print("\n[Step 3] Loading record 232 from MITBIH.zip...")
    rec_data = load_mitbih_232()
    print(f"  Signal length: {rec_data['total_samples']} samples @ {rec_data['fs']} Hz ({rec_data['duration_sec']:.1f} s)")
    print(f"  Lead: {rec_data['lead_name']}")
    print(f"  Rhythm markers: {rec_data['rhythm_markers']}")
    print(f"  Total beat annotations: {len(rec_data['beat_annotations'])}")
    print(f"  Beat types: {rec_data['beat_type_counts']}")

    # Step 4: Evaluate Windows
    print("\n[Step 4] Running 5-second non-overlapping evaluation across all 30 minutes...")
    df_eval = evaluate_mitdb_232(rec_data, pipeline, prov_audit["training_window_indices"])
    csv_path = OUTPUT_DIR / "mitdb_232_5s_predictions.csv"
    df_eval.to_csv(csv_path, index=False)
    print(f"Saved predictions ({len(df_eval)} windows) to: {csv_path.name}")

    # Step 5: Metrics & JSON Export
    print("\n[Step 5] Computing validation metrics & error analysis...")
    metrics = compute_mitdb_232_metrics(df_eval, prov_audit)
    json_path = OUTPUT_DIR / "mitdb_232_validation_metrics.json"
    with open(json_path, "w") as jf:
        json.dump(metrics, jf, indent=2)
    print(f"Saved metrics JSON to: {json_path.name}")

    # Summary Display
    print("\n" + "=" * 80)
    print("MIT-BIH RECORD 232 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total 5s Windows Evaluated: {metrics['total_windows_evaluated']} ({rec_data['duration_sec']:.1f} s)")
    print(f"Overall Predicted Class Counts: {metrics['predicted_class_counts_overall']}")
    print("\n--- IN-TRAINING SUBSET (121 windows in training dataset) ---")
    train_m = metrics["in_training_subset_metrics"]
    print(f"In-Sample Bradycardia Recall: {train_m['in_sample_bradycardia_recall']:.2%} ({train_m['in_sample_bradycardia_accuracy']})")
    print(f"Predicted Class Breakdown: {train_m['predicted_class_breakdown']}")

    print("\n--- OUT-OF-TRAINING SUBSET (240 windows excluded from training) ---")
    out_m = metrics["out_of_training_subset_metrics"]
    print(f"Predicted NSR Leakage: {out_m['predicted_nsr_leakage_count']} / {out_m['total_out_of_training_windows']}")
    print(f"Predicted Ventricular False Alarms: {out_m['predicted_ventricular_false_alarm_count']} / {out_m['total_out_of_training_windows']}")
    print(f"Predicted Bradycardia: {out_m['predicted_bradycardia_count']} / {out_m['total_out_of_training_windows']}")

    print("\n--- RATE BEHAVIOR CONCORDANCE ---")
    rate_m = metrics["rate_behavior_concordance"]
    print(f"Rate-Confirmed Bradycardia (HR < 60 bpm): {rate_m['rate_confirmed_bradycardia_windows']} windows")
    print(f"  Recall on Rate-Confirmed Bradycardia: {rate_m['rate_confirmed_bradycardia_recall']:.2%}")
    print(f"PAC-Burst Elevated-Rate Windows (HR >= 60 bpm): {rate_m['pac_burst_elevated_rate_windows']} windows")
    print(f"  Predictions during PAC bursts: {rate_m['pac_burst_predicted_class_breakdown']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
