"""
AFDB 5-Second Window Feature Extraction Pipeline
=================================================
Target Hardware: Raspberry Pi ARM64 + ADS1292R (360 Hz)
Window Duration: Exactly 5.0 seconds (1,800 samples at 360 Hz)
Resampling: Rational polyphase (250 Hz -> 360 Hz, up=36, down=25) on continuous Lead II signal (mV).
Strides:
  - Train & Val: 2.5 seconds (900 samples at 360 Hz)
  - Held-out Test: 5.0 seconds (1,800 samples at 360 Hz, non-overlapping)
Transition Guard Band: 2.5 seconds (900 samples at 360 Hz) around any rhythm transition.
Signal Rejection:
  - Continuous flatline zero run >= 1.0 s
  - Calibrated peak-to-peak amplitude Vpp < 0.05 mV
  - Non-finite (NaN / Inf)

Feature Set (21 features):
  - Ventricular Timing & Rate: n_beats, hr_mean_bpm, hr_std_bpm, rr_cv, rr_rmssd, rr_mad, rr_pnn50,
                               rr_sample_dispersion, rr_turning_point_ratio, alternating_sign_index
  - Spectral & Fibrillatory Band: f_band_power_ratio (3-9 Hz), f_band_flatness, f_band_peak_purity,
                                  p_band_power_ratio (1-3 Hz), qrs_power_ratio (10-25 Hz)
  - Morphology & Dynamics: hjorth_mobility, hjorth_complexity, ac_std_mv, ac_ptp_mv,
                           zero_crossings, lz_complexity
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from itertools import groupby
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

import numpy as np
import scipy.signal as signal
import pandas as pd
import wfdb

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FS_SOURCE = 250
FS_TARGET = 360
RESAMPLE_UP = 36
RESAMPLE_DOWN = 25

WINDOW_SEC = 5.0
TRAIN_STRIDE_SEC = 2.5
TEST_STRIDE_SEC = 5.0
GUARD_SEC = 2.5

WINDOW_SAMPLES_360 = int(WINDOW_SEC * FS_TARGET)         # 1800
TRAIN_STRIDE_SAMPLES_360 = int(TRAIN_STRIDE_SEC * FS_TARGET)  # 900
TEST_STRIDE_SAMPLES_360 = int(TEST_STRIDE_SEC * FS_TARGET)    # 1800
GUARD_SAMPLES_360 = int(GUARD_SEC * FS_TARGET)           # 900

WINDOW_SAMPLES_250 = int(WINDOW_SEC * FS_SOURCE)         # 1250
TRAIN_STRIDE_SAMPLES_250 = int(TRAIN_STRIDE_SEC * FS_SOURCE)  # 625
TEST_STRIDE_SAMPLES_250 = int(TEST_STRIDE_SEC * FS_SOURCE)    # 1250
GUARD_SAMPLES_250 = int(GUARD_SEC * FS_SOURCE)           # 625

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_AFDB_DIR = PROJECT_ROOT / "training" / "research" / "afib_experiment" / "data" / "afdb" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "training" / "research" / "afib_5s_classifier" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": [
        "04015", "04043", "04048", "04746", "04936",
        "05091", "05261", "06426", "06453", "06995",
        "07859", "08378", "08405", "08434", "08455"
    ],
    "val": [
        "04126", "05121", "07879", "07910"
    ],
    "test": [
        "04908", "07162", "08215", "08219"
    ]
}

COHORT_MAP = {}
for cohort, recs in SPLITS.items():
    for r in recs:
        COHORT_MAP[r] = cohort


# -----------------------------------------------------------------------------
# Signal Processing & Feature Extraction
# -----------------------------------------------------------------------------
def butter_bandpass_filter(sig: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: float = FS_TARGET, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter matching deploy/edge_runtime.py."""
    nyquist = 0.5 * fs
    low = max(0.001, lowcut / nyquist)
    high = min(0.999, highcut / nyquist)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, sig)


def detect_r_peaks(sig_filt: np.ndarray, fs: float = FS_TARGET) -> List[int]:
    """Robust QRS detector using bandpass (5-18 Hz), derivative, squaring, and integration."""
    nyq = 0.5 * fs
    b_qrs, a_qrs = signal.butter(2, [5.0 / nyq, 18.0 / nyq], btype="band")
    sig_qrs = signal.filtfilt(b_qrs, a_qrs, sig_filt)
    diff = np.diff(sig_qrs)
    squared = diff ** 2
    int_win = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(int_win) / int_win, mode="same")

    thresh = float(np.percentile(integrated, 85) * 0.45)
    min_dist = int(0.200 * fs)  # 200 ms refractory period (max 300 bpm)
    peaks, _ = signal.find_peaks(integrated, height=thresh, distance=min_dist)

    refined_peaks = []
    search_rad = int(0.050 * fs)  # +/- 50 ms search radius
    for p in peaks:
        start_idx = max(0, p - search_rad)
        end_idx = min(len(sig_filt), p + search_rad)
        if end_idx > start_idx:
            local_max = start_idx + int(np.argmax(sig_filt[start_idx:end_idx]))
            refined_peaks.append(local_max)

    return sorted(list(set(refined_peaks)))


def compute_asi(valid_rrs: np.ndarray, deadband_ms: float = 20.0) -> float:
    """Computes Alternating Sign Index (ASI) from valid RR intervals in ms."""
    if len(valid_rrs) < 3:
        return 0.0
    diff_rrs = np.diff(valid_rrs)
    signs = np.zeros(len(diff_rrs), dtype=int)
    signs[diff_rrs > deadband_ms] = 1
    signs[diff_rrs < -deadband_ms] = -1

    valid_transitions = 0
    alternating_flips = 0
    for i in range(len(signs) - 1):
        prod = signs[i] * signs[i + 1]
        if prod != 0:
            valid_transitions += 1
            if prod == -1:
                alternating_flips += 1
    if valid_transitions == 0:
        return 0.0
    return float(alternating_flips / valid_transitions)


def compute_turning_point_ratio(rrs: np.ndarray) -> float:
    """Computes turning point ratio (proportion of local peaks/troughs in RR sequence)."""
    if len(rrs) < 3:
        return 0.0
    tps = 0
    for i in range(1, len(rrs) - 1):
        if (rrs[i] > rrs[i - 1] and rrs[i] > rrs[i + 1]) or (rrs[i] < rrs[i - 1] and rrs[i] < rrs[i + 1]):
            tps += 1
    return float(tps / (len(rrs) - 2))


def compute_lempel_ziv(sig_filt: np.ndarray, fs: float = FS_TARGET) -> float:
    """Computes Lempel-Ziv complexity normalized on downsampled binary sequence."""
    step = max(1, int(round(fs / 90.0)))
    sig_down = sig_filt[::step]
    n = len(sig_down)
    if n < 10:
        return 0.0
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
    b_n = n / np.log2(n)
    return float(lz_count / (b_n + 1e-12))


def extract_5s_features(sig_window: np.ndarray, fs: float = FS_TARGET) -> Dict[str, float]:
    """
    Extracts physically justified 21-feature vector from a 5-second ECG window:
    1. Ventricular timing dynamics: Heart rate, RR CV, RMSSD, MAD, pNN50, Sample Dispersion, TPR, ASI.
    2. Atrial fibrillatory band (3-9 Hz): Welch PSD with nperseg=720 (4 segments), Flatness, Peak purity.
    3. Morphology & Complexity: Hjorth parameters, QRS power ratio, AC std, AC ptp, Zero Crossings, LZ complexity.
    """
    sig_filt = butter_bandpass_filter(sig_window, lowcut=0.5, highcut=40.0, fs=fs, order=2)
    ac_ptp = float(np.ptp(sig_filt))
    ac_std = float(np.std(sig_filt))
    zero_crossings = int(np.sum(np.diff(sig_filt > 0) != 0))

    # 1. R-peaks & RR dynamics
    peaks = detect_r_peaks(sig_filt, fs=fs)
    n_beats = len(peaks)

    if n_beats >= 3:
        raw_rrs = np.diff(peaks) / fs * 1000.0  # ms
        valid_rrs = raw_rrs[(raw_rrs >= 200.0) & (raw_rrs <= 2500.0)]
    else:
        valid_rrs = np.array([])

    if len(valid_rrs) >= 2:
        hr_mean = float(60000.0 / np.mean(valid_rrs))
        hr_std = float(np.std(60000.0 / valid_rrs))
        rr_cv = float(np.std(valid_rrs) / np.mean(valid_rrs))
        diff_rrs = np.diff(valid_rrs)
        rr_rmssd = float(np.sqrt(np.mean(diff_rrs ** 2))) if len(diff_rrs) > 0 else 0.0
        rr_mad = float(np.median(np.abs(diff_rrs - np.median(diff_rrs)))) if len(diff_rrs) > 0 else 0.0
        rr_pnn50 = float(np.sum(np.abs(diff_rrs) > 50.0) / len(diff_rrs)) if len(diff_rrs) > 0 else 0.0

        # Dispersion: IQR / median
        q75, q25 = np.percentile(valid_rrs, [75, 25])
        iqr = q75 - q25
        med_rr = np.median(valid_rrs)
        rr_dispersion = float(iqr / (med_rr + 1e-12))
        rr_tpr = compute_turning_point_ratio(valid_rrs)
        asi = compute_asi(valid_rrs, deadband_ms=20.0)
    else:
        hr_mean = 0.0
        hr_std = 0.0
        rr_cv = 0.0
        rr_rmssd = 0.0
        rr_mad = 0.0
        rr_pnn50 = 0.0
        rr_dispersion = 0.0
        rr_tpr = 0.0
        asi = 0.0

    # 2. Welch PSD with nperseg=720 (2.0s), 50% overlap -> 4 segments on 1800 samples
    nperseg = min(len(sig_filt), int(2.0 * fs))  # 720
    freqs, psd = signal.welch(sig_filt, fs=fs, nperseg=nperseg, noverlap=int(nperseg // 2))
    total_power = float(np.sum(psd)) + 1e-12

    # Fibrillatory band: 3.0 to 9.0 Hz
    f_mask = (freqs >= 3.0) & (freqs <= 9.0)
    f_psd = psd[f_mask]
    f_power = float(np.sum(f_psd))
    f_band_power_ratio = float(f_power / total_power)

    geo_mean = float(np.exp(np.mean(np.log(f_psd + 1e-12))))
    arith_mean = float(np.mean(f_psd)) + 1e-12
    f_band_flatness = float(geo_mean / arith_mean)
    f_band_peak_purity = float(np.max(f_psd) / arith_mean) if len(f_psd) > 0 else 1.0

    # P-wave band: 1.0 to 3.0 Hz
    p_mask = (freqs >= 1.0) & (freqs < 3.0)
    p_band_power_ratio = float(np.sum(psd[p_mask]) / total_power)

    # QRS band: 10.0 to 25.0 Hz
    qrs_mask = (freqs >= 10.0) & (freqs <= 25.0)
    qrs_power_ratio = float(np.sum(psd[qrs_mask]) / total_power)

    # 3. Hjorth parameters
    dx = np.diff(sig_filt)
    ddx = np.diff(dx)
    var_x = float(np.var(sig_filt)) + 1e-12
    var_dx = float(np.var(dx)) + 1e-12
    var_ddx = float(np.var(ddx)) + 1e-12

    mobility = float(np.sqrt(var_dx / var_x))
    complexity = float(np.sqrt(var_ddx / var_dx) / mobility) if mobility > 0 else 0.0

    # 4. Lempel-Ziv complexity
    lz_norm = compute_lempel_ziv(sig_filt, fs=fs)

    return {
        "n_beats": int(n_beats),
        "hr_mean_bpm": round(hr_mean, 2),
        "hr_std_bpm": round(hr_std, 2),
        "rr_cv": round(rr_cv, 4),
        "rr_rmssd": round(rr_rmssd, 2),
        "rr_mad": round(rr_mad, 2),
        "rr_pnn50": round(rr_pnn50, 4),
        "rr_sample_dispersion": round(rr_dispersion, 4),
        "rr_turning_point_ratio": round(rr_tpr, 4),
        "alternating_sign_index": round(asi, 4),
        "f_band_power_ratio": round(f_band_power_ratio, 4),
        "f_band_flatness": round(f_band_flatness, 4),
        "f_band_peak_purity": round(f_band_peak_purity, 3),
        "p_band_power_ratio": round(p_band_power_ratio, 4),
        "qrs_power_ratio": round(qrs_power_ratio, 4),
        "hjorth_mobility": round(mobility, 4),
        "hjorth_complexity": round(complexity, 4),
        "ac_std_mv": round(ac_std, 4),
        "ac_ptp_mv": round(ac_ptp, 4),
        "zero_crossings": int(zero_crossings),
        "lz_complexity": round(lz_norm, 4)
    }


# -----------------------------------------------------------------------------
# Record-Level Processing
# -----------------------------------------------------------------------------
def process_single_afdb_record(rec_id: str) -> Dict[str, Any]:
    """Processes a single AFDB record, extracting 5-second windows with transition guard bands."""
    t0 = time.time()
    rec_path = str(RAW_AFDB_DIR / rec_id)
    cohort = COHORT_MAP[rec_id]
    patient_id = f"AFDB_{rec_id}"

    # Determine stride based on cohort
    # Train and Val: 2.5s stride
    # Test: 5.0s non-overlapping stride
    if cohort in ["train", "val"]:
        stride_sec = TRAIN_STRIDE_SEC
        stride_samples_250 = TRAIN_STRIDE_SAMPLES_250
        stride_samples_360 = TRAIN_STRIDE_SAMPLES_360
    else:
        stride_sec = TEST_STRIDE_SEC
        stride_samples_250 = TEST_STRIDE_SAMPLES_250
        stride_samples_360 = TEST_STRIDE_SAMPLES_360

    header = wfdb.rdheader(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    record = wfdb.rdrecord(rec_path, physical=True)

    sig_250 = record.p_signal[:, 1]  # Channel 1: Lead II analog, physical mV
    n_samples_250 = header.sig_len

    # 1. Parse rhythm transitions
    t_samples = []
    t_rhythms = []
    for s, aux in zip(ann.sample, ann.aux_note):
        clean_aux = aux.strip('\x00').strip()
        t_samples.append(int(s))
        t_rhythms.append(clean_aux)

    t_samples = np.array(t_samples, dtype=np.int64)
    first_start = t_samples[0]

    # 2. Continuous zero-run mask for dropout detection (at 250 Hz)
    is_zero = (sig_250 == 0.0)
    long_zero_mask = np.zeros(n_samples_250, dtype=bool)
    idx = 0
    for k, g in groupby(is_zero):
        run_len = sum(1 for _ in g)
        if k and run_len >= int(1.0 * FS_SOURCE):
            long_zero_mask[idx:idx + run_len] = True
        idx += run_len
    cumsum_long_zero = np.pad(np.cumsum(long_zero_mask), (1, 0))

    # 3. Rational polyphase anti-aliasing resampling to 360 Hz
    sig_360 = signal.resample_poly(sig_250, up=RESAMPLE_UP, down=RESAMPLE_DOWN).astype(np.float32)

    # 4. Windowing calculations
    w_starts_250 = np.arange(0, n_samples_250 - WINDOW_SAMPLES_250 + 1, stride_samples_250)
    w_ends_250 = w_starts_250 + WINDOW_SAMPLES_250
    n_total_windows = len(w_starts_250)

    # Vectorized Guard Band & Transition Intersection Check
    reject_transition = np.zeros(n_total_windows, dtype=bool)
    for t in t_samples:
        reject_transition |= ((w_starts_250 - GUARD_SAMPLES_250 < t) & (t < w_ends_250 + GUARD_SAMPLES_250))

    reject_unknown = (w_starts_250 < first_start)
    w_dropout_samples = cumsum_long_zero[w_ends_250] - cumsum_long_zero[w_starts_250]
    reject_dropout = (w_dropout_samples > 0)
    seg_idx = np.searchsorted(t_samples, w_starts_250, side='right') - 1

    candidate_mask = (~reject_transition) & (~reject_unknown) & (~reject_dropout) & (seg_idx >= 0)

    kept_features = []
    rejected_windows = []

    for w_idx in range(n_total_windows):
        ws_250 = int(w_starts_250[w_idx])
        we_250 = int(w_ends_250[w_idx])
        ws_360 = int(w_idx * stride_samples_360)
        we_360 = int(ws_360 + WINDOW_SAMPLES_360)
        start_time_s = round(w_idx * stride_sec, 4)
        end_time_s = round(start_time_s + WINDOW_SEC, 4)

        if not candidate_mask[w_idx]:
            if reject_unknown[w_idx]:
                reason = "pre_annotation_unknown"
                detail = f"Window starts at {start_time_s:.2f}s before first annotation at {first_start/FS_SOURCE:.2f}s"
            elif reject_transition[w_idx]:
                reason = "transition_guard_violation"
                detail = f"Window within 2.5s guard band of rhythm transition"
            elif reject_dropout[w_idx]:
                reason = "flatline_zero_dropout"
                detail = f"Window contains samples from >= 1.0s continuous flat zero run"
            else:
                reason = "unassigned_segment"
                detail = "Window cannot be assigned to valid rhythm segment"

            rejected_windows.append({
                "record_id": rec_id,
                "patient_id": patient_id,
                "window_idx": w_idx,
                "start_time_sec": start_time_s,
                "end_time_sec": end_time_s,
                "rejection_reason": reason,
                "rejection_detail": detail,
                "cohort": cohort
            })
            continue

        # Check amplitude on resampled 360 Hz window
        w_sig_360 = sig_360[ws_360:we_360]
        if len(w_sig_360) < WINDOW_SAMPLES_360:
            continue

        v_pp = float(np.ptp(w_sig_360))
        if v_pp < 0.05:
            rejected_windows.append({
                "record_id": rec_id,
                "patient_id": patient_id,
                "window_idx": w_idx,
                "start_time_sec": start_time_s,
                "end_time_sec": end_time_s,
                "rejection_reason": "low_amplitude_flatline",
                "rejection_detail": f"Peak-to-peak amplitude {v_pp:.4f} mV < 0.05 mV threshold",
                "cohort": cohort
            })
            continue

        # Extract complete feature set
        rhythm_label = t_rhythms[seg_idx[w_idx]]
        is_afib = 1 if rhythm_label == "(AFIB" else 0

        feats = extract_5s_features(w_sig_360, fs=FS_TARGET)
        feats.update({
            "record_id": rec_id,
            "patient_id": patient_id,
            "window_idx": w_idx,
            "start_time_sec": start_time_s,
            "end_time_sec": end_time_s,
            "rhythm_label": rhythm_label,
            "is_afib": is_afib,
            "cohort": cohort
        })
        kept_features.append(feats)

    dt = time.time() - t0
    n_kept = len(kept_features)
    n_afib = sum(1 for f in kept_features if f["is_afib"] == 1)
    n_non_afib = n_kept - n_afib
    n_afl = sum(1 for f in kept_features if f["rhythm_label"] == "(AFL")

    summary = {
        "record_id": rec_id,
        "cohort": cohort,
        "stride_sec": stride_sec,
        "duration_hours": round(n_samples_250 / FS_SOURCE / 3600.0, 2),
        "total_windows": n_total_windows,
        "kept_windows": n_kept,
        "afib_windows": n_afib,
        "non_afib_windows": n_non_afib,
        "afl_windows": n_afl,
        "rejected_windows": len(rejected_windows),
        "processing_time_sec": round(dt, 2)
    }

    return {
        "record_id": rec_id,
        "cohort": cohort,
        "kept_features": kept_features,
        "rejected_windows": rejected_windows,
        "summary": summary
    }


def main():
    print("=" * 80)
    print("DEDICATED 5-SECOND AFDB FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    t_start = time.time()

    all_records = SPLITS["train"] + SPLITS["val"] + SPLITS["test"]
    print(f"Ingesting {len(all_records)} records across 3 disjoint cohorts...")
    print(f"  - Train (15 records, 2.5s stride)")
    print(f"  - Val   (4 records, 2.5s stride)")
    print(f"  - Test  (4 records, 5.0s non-overlapping stride)")

    all_train_features = []
    all_val_features = []
    all_test_features = []
    all_rejected = []
    record_summaries = []

    n_workers = min(12, os.cpu_count() or 4)
    print(f"Spawning {n_workers} parallel workers across {len(all_records)} records...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_rec = {executor.submit(process_single_afdb_record, rec_id): rec_id for rec_id in all_records}
        for future in as_completed(future_to_rec):
            rec_id = future_to_rec[future]
            try:
                res = future.result()
                s = res["summary"]
                print(f"[{rec_id} | {s['cohort'].upper():5s}] Done in {s['processing_time_sec']:5.1f}s | "
                      f"Kept: {s['kept_windows']:5d} (AFIB: {s['afib_windows']:4d}, "
                      f"Non-AFIB: {s['non_afib_windows']:4d}, AFL: {s['afl_windows']:3d}) | "
                      f"Rej: {s['rejected_windows']:4d}", flush=True)

                cohort = res["cohort"]
                if cohort == "train":
                    all_train_features.extend(res["kept_features"])
                elif cohort == "val":
                    all_val_features.extend(res["kept_features"])
                elif cohort == "test":
                    all_test_features.extend(res["kept_features"])

                all_rejected.extend(res["rejected_windows"])
                record_summaries.append(s)
            except Exception as e:
                print(f"ERROR processing record {rec_id}: {e}", file=sys.stderr)
                raise e

    # Convert to DataFrames and serialize
    print("\nSerializing feature tables to CSV...")
    df_train = pd.DataFrame(all_train_features)
    df_val = pd.DataFrame(all_val_features)
    df_test = pd.DataFrame(all_test_features)
    df_rej = pd.DataFrame(all_rejected)

    train_csv = OUTPUT_DIR / "afdb_5s_features_train.csv"
    val_csv = OUTPUT_DIR / "afdb_5s_features_val.csv"
    test_csv = OUTPUT_DIR / "afdb_5s_features_test.csv"
    rej_csv = OUTPUT_DIR / "afdb_5s_rejected_windows.csv"
    summary_json = OUTPUT_DIR / "afdb_5s_extraction_summary.json"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    df_rej.to_csv(rej_csv, index=False)

    total_kept = len(df_train) + len(df_val) + len(df_test)
    summary_data = {
        "pipeline": "AFDB Dedicated 5-Second Feature Extractor",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_processing_time_sec": round(time.time() - t_start, 2),
        "cohort_counts": {
            "train": {
                "patients": len(SPLITS["train"]),
                "windows": len(df_train),
                "afib_windows": int(df_train["is_afib"].sum()),
                "non_afib_windows": int((df_train["is_afib"] == 0).sum()),
                "afib_prevalence_pct": round(float(df_train["is_afib"].mean() * 100), 2)
            },
            "val": {
                "patients": len(SPLITS["val"]),
                "windows": len(df_val),
                "afib_windows": int(df_val["is_afib"].sum()),
                "non_afib_windows": int((df_val["is_afib"] == 0).sum()),
                "afib_prevalence_pct": round(float(df_val["is_afib"].mean() * 100), 2)
            },
            "test": {
                "patients": len(SPLITS["test"]),
                "windows": len(df_test),
                "afib_windows": int(df_test["is_afib"].sum()),
                "non_afib_windows": int((df_test["is_afib"] == 0).sum()),
                "afib_prevalence_pct": round(float(df_test["is_afib"].mean() * 100), 2)
            }
        },
        "rejected_windows_total": len(df_rej),
        "per_record_summaries": record_summaries,
        "feature_columns": [c for c in df_train.columns if c not in ["record_id", "patient_id", "window_idx", "start_time_sec", "end_time_sec", "rhythm_label", "is_afib", "cohort"]],
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print("5-SECOND FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
    print(f"  Train CSV: {train_csv.name} ({len(df_train):,} windows, {os.path.getsize(train_csv):,} bytes)")
    print(f"  Val CSV:   {val_csv.name} ({len(df_val):,} windows, {os.path.getsize(val_csv):,} bytes)")
    print(f"  Test CSV:  {test_csv.name} ({len(df_test):,} windows, {os.path.getsize(test_csv):,} bytes)")
    print(f"  Rej CSV:   {rej_csv.name} ({len(df_rej):,} windows, {os.path.getsize(rej_csv):,} bytes)")
    print(f"  Total Kept Windows: {total_kept:,}")
    print(f"  Elapsed Time: {time.time() - t_start:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
