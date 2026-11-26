"""
Reproducible Offline AFDB Feature Extraction Pipeline
=====================================================
Target: Raspberry Pi ARM64 (Debian Bookworm) + ADS1292R Target (360 Hz)
Database: MIT-BIH Atrial Fibrillation Database (AFDB v1.0.0)

Rigorous Engineering & Clinical Safeguards:
1. Strictly isolated patient cohorts (Train: 15 patients, Val: 4 patients, Test: 4 patients).
2. Primary signal channel: Lead II analog (Channel 1 'ECG2') preserved in physical millivolts (mV).
3. Rational polyphase anti-aliasing resampling (250 Hz -> 360 Hz, up=36, down=25) on continuous records.
4. 10.0-second windows with 5.0-second stride (3,600 samples, 1,800 sample step at 360 Hz).
5. 2.5-second transition guard bands (900 samples at 360 Hz / 625 samples at 250 Hz).
6. Quality rejection for flatline zero dropouts (continuous run >= 1.0s) and low amplitude (Vpp < 0.05 mV).
7. Feature set: 16 timing/spectral features + Alternating Sign Index (ASI) + n_beats.
8. Complete audit manifest and SHA256 checksum generation.
9. ZERO MODEL TRAINING, ZERO SCALER FITTING, ZERO LEAKAGE.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from itertools import groupby
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.signal as signal
import pandas as pd
import wfdb

# -----------------------------------------------------------------------------
# Configuration & Hardware Target Specifications
# -----------------------------------------------------------------------------
FS_SOURCE = 250
FS_TARGET = 360
RESAMPLE_UP = 36
RESAMPLE_DOWN = 25

WINDOW_SEC = 10.0
STRIDE_SEC = 5.0
GUARD_SEC = 2.5

WINDOW_SAMPLES_360 = int(WINDOW_SEC * FS_TARGET)  # 3,600
STRIDE_SAMPLES_360 = int(STRIDE_SEC * FS_TARGET)  # 1,800
GUARD_SAMPLES_360 = int(GUARD_SEC * FS_TARGET)    # 900

WINDOW_SAMPLES_250 = int(WINDOW_SEC * FS_SOURCE)  # 2,500
STRIDE_SAMPLES_250 = int(STRIDE_SEC * FS_SOURCE)  # 1,250
GUARD_SAMPLES_250 = int(GUARD_SEC * FS_SOURCE)    # 625

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AFDB_RAW_DIR = PROJECT_ROOT / "training" / "research" / "afib_experiment" / "data" / "afdb" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "training" / "research" / "afib_experiment" / "data" / "afdb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECORD_IDS = [
    "04015", "04043", "04048", "04126", "04746", "04908", "04936",
    "05091", "05121", "05261", "06426", "06453", "06995", "07162",
    "07859", "07879", "07910", "08215", "08219", "08378", "08405",
    "08434", "08455"
]

SPLITS = {
    "train": ["04015", "04043", "04048", "04746", "04936", "05091", "05261",
              "06426", "06453", "06995", "07859", "08378", "08405", "08434", "08455"],
    "val": ["04126", "05121", "07879", "07910"],
    "test": ["04908", "07162", "08215", "08219"]
}

COHORT_MAP = {}
for cohort, recs in SPLITS.items():
    for r in recs:
        COHORT_MAP[r] = cohort


# -----------------------------------------------------------------------------
# Signal Preprocessing and Feature Extraction Functions
# -----------------------------------------------------------------------------
def butter_bandpass_filter(sig: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: float = FS_TARGET, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter matching deploy/edge_runtime.py."""
    nyquist = 0.5 * fs
    low = max(0.001, lowcut / nyquist)
    high = min(0.999, highcut / nyquist)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, sig)


def detect_r_peaks(sig_filt: np.ndarray, fs: float = FS_TARGET) -> list[int]:
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
    """
    Computes Alternating Sign Index (ASI) from valid RR intervals in ms.
    Formula: fraction of consecutive non-zero successive differences that change sign.
    Deadband epsilon = 20 ms suppresses baseline sinus jitter.
    """
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


def extract_window_features(sig_window: np.ndarray, fs: float = FS_TARGET) -> dict:
    """
    Extracts physically justified features from a 10-second ECG window:
    1. Ventricular timing dynamics: Heart rate, RR CV, RMSSD, MAD, Shannon entropy, ASI.
    2. Atrial fibrillatory band (3-9 Hz): Relative power, spectral flatness (Wiener entropy), peak purity.
    3. Morphological & Complexity: Hjorth parameters, QRS power ratio, AC std, AC ptp.
    """
    sig_filt = butter_bandpass_filter(sig_window, lowcut=0.5, highcut=40.0, fs=fs, order=2)
    ac_ptp = float(np.ptp(sig_filt))
    ac_std = float(np.std(sig_filt))

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

        hist, _ = np.histogram(valid_rrs, bins=12, range=(200.0, 2000.0), density=True)
        hist = hist[hist > 0]
        rr_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0
        asi = compute_asi(valid_rrs, deadband_ms=20.0)
    else:
        hr_mean = 0.0
        hr_std = 0.0
        rr_cv = 0.0
        rr_rmssd = 0.0
        rr_mad = 0.0
        rr_pnn50 = 0.0
        rr_entropy = 0.0
        asi = 0.0

    # 2. Welch PSD (0.5 - 40 Hz)
    nperseg = min(len(sig_filt), int(4.0 * fs))
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

    return {
        "n_beats": int(n_beats),
        "hr_mean_bpm": round(hr_mean, 2),
        "hr_std_bpm": round(hr_std, 2),
        "rr_cv": round(rr_cv, 4),
        "rr_rmssd": round(rr_rmssd, 2),
        "rr_mad": round(rr_mad, 2),
        "rr_pnn50": round(rr_pnn50, 4),
        "rr_entropy": round(rr_entropy, 4),
        "alternating_sign_index": round(asi, 4),
        "f_band_power_ratio": round(f_band_power_ratio, 4),
        "f_band_flatness": round(f_band_flatness, 4),
        "f_band_peak_purity": round(f_band_peak_purity, 3),
        "p_band_power_ratio": round(p_band_power_ratio, 4),
        "qrs_power_ratio": round(qrs_power_ratio, 4),
        "hjorth_mobility": round(mobility, 4),
        "hjorth_complexity": round(complexity, 4),
        "ac_std_mv": round(ac_std, 4),
        "ac_ptp_mv": round(ac_ptp, 4)
    }


# -----------------------------------------------------------------------------
# Record Processing Pipeline (Executed in Parallel per Record)
# -----------------------------------------------------------------------------
def process_single_record(rec_id: str) -> dict:
    """
    Reads, resamples, audits, and extracts features for a single AFDB record.
    Returns:
      - record_id
      - kept_features: list of dicts for valid windows
      - rejected_windows: list of dicts for rejected windows
      - summary_stats: per-record count dictionary
    """
    t0 = time.time()
    rec_path = str(AFDB_RAW_DIR / rec_id)
    cohort = COHORT_MAP[rec_id]
    patient_id = f"AFDB_{rec_id}"

    header = wfdb.rdheader(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    record = wfdb.rdrecord(rec_path, physical=True)

    sig_250 = record.p_signal[:, 1]  # Channel 1: Lead II analog, mV
    n_samples_250 = header.sig_len

    # 1. Parse rhythm transitions
    t_samples = []
    t_rhythms = []
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
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
    n_samples_360 = len(sig_360)

    # 4. Windowing calculations
    w_starts_250 = np.arange(0, n_samples_250 - WINDOW_SAMPLES_250 + 1, STRIDE_SAMPLES_250)
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

    # 5. Extract features and identify flatlines
    kept_features = []
    rejected_windows = []

    for w_idx in range(n_total_windows):
        ws_250 = int(w_starts_250[w_idx])
        we_250 = int(w_ends_250[w_idx])
        ws_360 = int(w_idx * STRIDE_SAMPLES_360)
        we_360 = int(ws_360 + WINDOW_SAMPLES_360)
        start_time_s = round(w_idx * STRIDE_SEC, 4)
        end_time_s = round(start_time_s + WINDOW_SEC, 4)

        if not candidate_mask[w_idx]:
            # Record rejection reason
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
                "start_sample_250": ws_250,
                "end_sample_250": we_250,
                "start_sample_360": ws_360,
                "end_sample_360": we_360,
                "start_time_sec": start_time_s,
                "end_time_sec": end_time_s,
                "rejection_reason": reason,
                "rejection_detail": detail,
                "cohort": cohort
            })
            continue

        # Candidate window - check amplitude on resampled 360 Hz window
        w_sig_360 = sig_360[ws_360:we_360]
        v_pp = float(np.ptp(w_sig_360))

        if v_pp < 0.05:
            rejected_windows.append({
                "record_id": rec_id,
                "patient_id": patient_id,
                "window_idx": w_idx,
                "start_sample_250": ws_250,
                "end_sample_250": we_250,
                "start_sample_360": ws_360,
                "end_sample_360": we_360,
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

        feats = extract_window_features(w_sig_360, fs=FS_TARGET)
        feats.update({
            "record_id": rec_id,
            "patient_id": patient_id,
            "window_idx": w_idx,
            "start_sample_250": ws_250,
            "end_sample_250": we_250,
            "start_sample_360": ws_360,
            "end_sample_360": we_360,
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
    n_rej_trans = sum(1 for r in rejected_windows if r["rejection_reason"] == "transition_guard_violation")
    n_rej_qual = len(rejected_windows) - n_rej_trans

    summary = {
        "record_id": rec_id,
        "cohort": cohort,
        "duration_hours": round(n_samples_250 / FS_SOURCE / 3600.0, 4),
        "total_windows": n_total_windows,
        "kept_windows": n_kept,
        "afib_windows": n_afib,
        "non_afib_windows": n_non_afib,
        "afl_windows": n_afl,
        "rejected_transition": n_rej_trans,
        "rejected_quality": n_rej_qual,
        "processing_time_s": round(dt, 2)
    }

    return {
        "record_id": rec_id,
        "kept_features": kept_features,
        "rejected_windows": rejected_windows,
        "summary": summary
    }


def compute_file_sha256(filepath: Path) -> str:
    """Calculates SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("STARTING AFDB REPRODUCIBLE FEATURE EXTRACTION PIPELINE")
    print(f"Primary Channel: Lead II Analog (Channel 1 'ECG2') in physical mV")
    print(f"Resampling: {FS_SOURCE} Hz -> {FS_TARGET} Hz via polyphase FIR (up={RESAMPLE_UP}, down={RESAMPLE_DOWN})")
    print(f"Segmentation: {WINDOW_SEC}s window ({WINDOW_SAMPLES_360} samples), {STRIDE_SEC}s stride, {GUARD_SEC}s guard band")
    print(f"Destination: {OUTPUT_DIR}")
    print("=" * 80)

    t_start = time.time()

    # Parallel processing across records using ProcessPoolExecutor
    n_workers = min(12, os.cpu_count() or 4)
    print(f"Spawning {n_workers} worker processes across {len(RECORD_IDS)} records...")

    all_kept_train = []
    all_kept_val = []
    all_kept_test = []
    all_rejected = []
    per_record_summaries = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_rec = {executor.submit(process_single_record, rec_id): rec_id for rec_id in RECORD_IDS}
        for future in as_completed(future_to_rec):
            rec_id = future_to_rec[future]
            try:
                res = future.result()
                summary = res["summary"]
                per_record_summaries.append(summary)
                all_rejected.extend(res["rejected_windows"])

                cohort = summary["cohort"]
                if cohort == "train":
                    all_kept_train.extend(res["kept_features"])
                elif cohort == "val":
                    all_kept_val.extend(res["kept_features"])
                elif cohort == "test":
                    all_kept_test.extend(res["kept_features"])

                print(f"[{rec_id} | {cohort.upper():5s}] Done in {summary['processing_time_s']:5.1f}s | "
                      f"Kept: {summary['kept_windows']:5d} (AFIB: {summary['afib_windows']:4d}, "
                      f"Non-AFIB: {summary['non_afib_windows']:4d}, AFL: {summary['afl_windows']:3d}) | "
                      f"Rej Trans: {summary['rejected_transition']:3d}, Rej Qual: {summary['rejected_quality']:2d}", flush=True)
            except Exception as e:
                print(f"ERROR processing record {rec_id}: {e}", file=sys.stderr)
                raise e

    print("=" * 80)
    print("ALL 23 RECORDS PROCESSED SUCCESSFULLY. ASSEMBLING DATASETS...")
    print("=" * 80)

    # Sort record summaries by record_id
    per_record_summaries.sort(key=lambda s: s["record_id"])

    df_train = pd.DataFrame(all_kept_train)
    df_val = pd.DataFrame(all_kept_val)
    df_test = pd.DataFrame(all_kept_test)
    df_rejected = pd.DataFrame(all_rejected)

    # Verify column order
    feature_cols = [
        "n_beats", "hr_mean_bpm", "hr_std_bpm", "rr_cv", "rr_rmssd", "rr_mad", "rr_pnn50", "rr_entropy",
        "alternating_sign_index", "f_band_power_ratio", "f_band_flatness", "f_band_peak_purity",
        "p_band_power_ratio", "qrs_power_ratio", "hjorth_mobility", "hjorth_complexity", "ac_std_mv", "ac_ptp_mv"
    ]
    meta_cols = [
        "record_id", "patient_id", "window_idx", "start_sample_250", "end_sample_250",
        "start_sample_360", "end_sample_360", "start_time_sec", "end_time_sec",
        "rhythm_label", "is_afib", "cohort"
    ]
    full_cols = feature_cols + meta_cols

    # Reorder columns
    df_train = df_train[full_cols].sort_values(["record_id", "window_idx"]).reset_index(drop=True)
    df_val = df_val[full_cols].sort_values(["record_id", "window_idx"]).reset_index(drop=True)
    df_test = df_test[full_cols].sort_values(["record_id", "window_idx"]).reset_index(drop=True)
    df_rejected = df_rejected.sort_values(["record_id", "window_idx"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Comprehensive Verification Checks Prior to Saving
    # -------------------------------------------------------------------------
    print("\nExecuting Pre-Save Verification Invariants:")
    
    # 1. Total window counts
    total_kept = len(df_train) + len(df_val) + len(df_test)
    total_rejected = len(df_rejected)
    total_windows = total_kept + total_rejected
    print(f"1. Window Count Audit: Total={total_windows} (Kept={total_kept}, Rejected={total_rejected})")
    assert total_windows == 168645, f"Expected 168,645 total windows, got {total_windows}"
    assert total_kept == 166939, f"Expected 166,939 kept windows, got {total_kept}"
    assert total_rejected == 1706, f"Expected 1,706 rejected windows, got {total_rejected}"
    assert len(df_train) == 108522, f"Expected 108,522 train windows, got {len(df_train)}"
    assert len(df_val) == 29255, f"Expected 29,255 val windows, got {len(df_val)}"
    assert len(df_test) == 29162, f"Expected 29,162 test windows, got {len(df_test)}"
    print("   [PASS] Exact window counts verified across all cohorts.")

    # 2. Null / NaN / Inf check
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        n_nans = int(df.isna().sum().sum())
        n_infs = int(np.isinf(df[feature_cols].values).sum())
        assert n_nans == 0, f"{name} contains {n_nans} NaNs!"
        assert n_infs == 0, f"{name} contains {n_infs} Infs!"
    print("   [PASS] Zero NaNs and zero Infs verified across all feature columns.")

    # 3. Patient-level split isolation
    train_patients = set(df_train["patient_id"].unique())
    val_patients = set(df_val["patient_id"].unique())
    test_patients = set(df_test["patient_id"].unique())
    assert train_patients.isdisjoint(val_patients), "Train and Val share patients!"
    assert train_patients.isdisjoint(test_patients), "Train and Test share patients!"
    assert val_patients.isdisjoint(test_patients), "Val and Test share patients!"
    print("   [PASS] 100% Disjoint Patient Separation confirmed (Train: 15, Val: 4, Test: 4).")

    # 4. Class counts
    total_afib = int(df_train["is_afib"].sum() + df_val["is_afib"].sum() + df_test["is_afib"].sum())
    total_non_afib = total_kept - total_afib
    assert total_afib == 66425, f"Expected 66,425 AFIB windows, got {total_afib}"
    assert total_non_afib == 100514, f"Expected 100,514 Non-AFIB windows, got {total_non_afib}"
    assert int(df_train["is_afib"].sum()) == 40633, "Train AFIB mismatch"
    assert int(df_val["is_afib"].sum()) == 10438, "Val AFIB mismatch"
    assert int(df_test["is_afib"].sum()) == 15354, "Test AFIB mismatch"
    print("   [PASS] Class counts match audit exactly (AFIB: 66,425, Non-AFIB: 100,514).")

    # -------------------------------------------------------------------------
    # Save Datasets to Disk
    # -------------------------------------------------------------------------
    print("\nWriting Feature Datasets to Disk:")
    train_csv = OUTPUT_DIR / "afdb_features_train.csv"
    val_csv = OUTPUT_DIR / "afdb_features_val.csv"
    test_csv = OUTPUT_DIR / "afdb_features_test.csv"
    rejected_csv = OUTPUT_DIR / "afdb_rejected_windows.csv"
    manifest_json = OUTPUT_DIR / "afdb_extraction_manifest.json"

    df_train.to_csv(train_csv, index=False)
    print(f" - Saved: {train_csv.name} ({len(df_train)} rows, {os.path.getsize(train_csv):,} bytes)")

    df_val.to_csv(val_csv, index=False)
    print(f" - Saved: {val_csv.name} ({len(df_val)} rows, {os.path.getsize(val_csv):,} bytes)")

    df_test.to_csv(test_csv, index=False)
    print(f" - Saved: {test_csv.name} ({len(df_test)} rows, {os.path.getsize(test_csv):,} bytes)")

    df_rejected.to_csv(rejected_csv, index=False)
    print(f" - Saved: {rejected_csv.name} ({len(df_rejected)} rows, {os.path.getsize(rejected_csv):,} bytes)")

    # Compute checksums
    checksums = {
        "afdb_features_train.csv": compute_file_sha256(train_csv),
        "afdb_features_val.csv": compute_file_sha256(val_csv),
        "afdb_features_test.csv": compute_file_sha256(test_csv),
        "afdb_rejected_windows.csv": compute_file_sha256(rejected_csv)
    }

    # Build comprehensive extraction manifest
    manifest = {
        "pipeline_name": "AFDB Offline Feature Extraction",
        "pipeline_version": "1.0.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_duration_sec": round(time.time() - t_start, 2),
        "sampling_rates": {
            "source_fs_hz": FS_SOURCE,
            "target_fs_hz": FS_TARGET,
            "polyphase_up": RESAMPLE_UP,
            "polyphase_down": RESAMPLE_DOWN
        },
        "windowing_config": {
            "window_duration_sec": WINDOW_SEC,
            "window_stride_sec": STRIDE_SEC,
            "guard_band_sec": GUARD_SEC,
            "window_samples_360": WINDOW_SAMPLES_360,
            "stride_samples_360": STRIDE_SAMPLES_360,
            "guard_samples_360": GUARD_SAMPLES_360
        },
        "signal_channel": {
            "channel_index": 1,
            "channel_name": "ECG2",
            "clinical_lead_analog": "Lead II",
            "physical_units": "millivolts (mV)"
        },
        "quality_rejection_criteria": {
            "continuous_zero_run_threshold_sec": 1.0,
            "min_peak_to_peak_mv": 0.05
        },
        "feature_specifications": {
            "feature_count": len(feature_cols),
            "feature_names": feature_cols,
            "asi_deadband_ms": 20.0
        },
        "patient_splits": SPLITS,
        "summary_counts": {
            "total_potential_windows": total_windows,
            "total_kept_windows": total_kept,
            "total_rejected_windows": total_rejected,
            "train_kept_windows": len(df_train),
            "train_afib_windows": int(df_train["is_afib"].sum()),
            "train_non_afib_windows": len(df_train) - int(df_train["is_afib"].sum()),
            "val_kept_windows": len(df_val),
            "val_afib_windows": int(df_val["is_afib"].sum()),
            "val_non_afib_windows": len(df_val) - int(df_val["is_afib"].sum()),
            "test_kept_windows": len(df_test),
            "test_afib_windows": int(df_test["is_afib"].sum()),
            "test_non_afib_windows": len(df_test) - int(df_test["is_afib"].sum())
        },
        "per_record_summary": per_record_summaries,
        "file_checksums_sha256": checksums
    }

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f" - Saved: {manifest_json.name} ({os.path.getsize(manifest_json):,} bytes)")

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETED SUCCESSFULLY IN {:.2f} SECONDS".format(time.time() - t_start))
    print("ALL VERIFICATION INVARIANTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
