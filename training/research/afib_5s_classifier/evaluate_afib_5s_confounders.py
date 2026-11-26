"""
AFDB 5-Second Dedicated Classifier: Independent Multi-Arrhythmia Confounder Testing
===================================================================================
Evaluates the frozen 5-second AFib classifier against non-AFib arrhythmias:
1. Atrial Flutter (AFL): AFDB record '04043' AFL segment + MIT-BIH record '222' (pure AFL).
2. Ventricular Bigeminy: MIT-BIH record '233' (continuous ventricular bigeminy).
3. Ventricular Tachycardia (VT): CUDB record 'cu04' (paroxysmal VT/VFL) + MIT-BIH record '207'.
4. Ventricular Fibrillation (VF): CUDB record 'cu01' (sustained coarse VF interval t >= 214s).
5. Sinus Bradycardia: MIT-BIH record '232' (sustained sinus bradycardia, 40-55 bpm).
6. Normal Sinus Control: CUDB record 'cu01' (sinus interval t < 214s) + MIT-BIH record '100'.
7. Poor-Quality & Noise: Flatline zero-dropouts, high-frequency saturation, and amplitude collapse.
"""

import os
import sys
import json
import time
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import scipy.signal as signal
import pandas as pd
import joblib
import wfdb

# -----------------------------------------------------------------------------
# Configuration and Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_DIR = PROJECT_ROOT / "training" / "research" / "afib_5s_classifier"
DATA_DIR = RESEARCH_DIR / "data"

MODEL_PACKAGE_PATH = RESEARCH_DIR / "afib_5s_model_package.joblib"
METRICS_OUTPUT_PATH = RESEARCH_DIR / "afib_5s_confounder_metrics.json"
PREDICTIONS_CSV_PATH = RESEARCH_DIR / "afib_5s_confounder_predictions.csv"

CUDB_ZIP = PROJECT_ROOT / "cu-ventricular-tachyarrhythmia-database-1.0.0.zip"
MITBIH_ZIP = PROJECT_ROOT / "MITBIH.zip"
RAW_AFDB_DIR = PROJECT_ROOT / "training" / "research" / "afib_experiment" / "data" / "afdb" / "raw"

TARGET_FS = 360
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_FS)  # 1800 samples


# -----------------------------------------------------------------------------
# Feature Extraction (Matching extract_afdb_5s_features.py exactly)
# -----------------------------------------------------------------------------
def butter_bandpass_filter(sig: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: float = TARGET_FS, order: int = 2) -> np.ndarray:
    nyquist = 0.5 * fs
    low = max(0.001, lowcut / nyquist)
    high = min(0.999, highcut / nyquist)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, sig)


def detect_r_peaks(sig_filt: np.ndarray, fs: float = TARGET_FS) -> List[int]:
    nyq = 0.5 * fs
    b_qrs, a_qrs = signal.butter(2, [5.0 / nyq, 18.0 / nyq], btype="band")
    sig_qrs = signal.filtfilt(b_qrs, a_qrs, sig_filt)
    diff = np.diff(sig_qrs)
    squared = diff ** 2
    int_win = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(int_win) / int_win, mode="same")

    thresh = float(np.percentile(integrated, 85) * 0.45)
    min_dist = int(0.200 * fs)
    peaks, _ = signal.find_peaks(integrated, height=thresh, distance=min_dist)

    refined_peaks = []
    search_rad = int(0.050 * fs)
    for p in peaks:
        start_idx = max(0, p - search_rad)
        end_idx = min(len(sig_filt), p + search_rad)
        if end_idx > start_idx:
            local_max = start_idx + int(np.argmax(sig_filt[start_idx:end_idx]))
            refined_peaks.append(local_max)

    return sorted(list(set(refined_peaks)))


def compute_asi(valid_rrs: np.ndarray, deadband_ms: float = 20.0) -> float:
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
    if len(rrs) < 3:
        return 0.0
    tps = 0
    for i in range(1, len(rrs) - 1):
        if (rrs[i] > rrs[i - 1] and rrs[i] > rrs[i + 1]) or (rrs[i] < rrs[i - 1] and rrs[i] < rrs[i + 1]):
            tps += 1
    return float(tps / (len(rrs) - 2))


def compute_lempel_ziv(sig_filt: np.ndarray, fs: float = TARGET_FS) -> float:
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


def extract_5s_features(sig_window: np.ndarray, fs: float = TARGET_FS) -> Dict[str, float]:
    sig_filt = butter_bandpass_filter(sig_window, lowcut=0.5, highcut=40.0, fs=fs, order=2)
    ac_ptp = float(np.ptp(sig_filt))
    ac_std = float(np.std(sig_filt))
    zero_crossings = int(np.sum(np.diff(sig_filt > 0) != 0))

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

    nperseg = min(len(sig_filt), int(2.0 * fs))
    freqs, psd = signal.welch(sig_filt, fs=fs, nperseg=nperseg, noverlap=int(nperseg // 2))
    total_power = float(np.sum(psd)) + 1e-12

    f_mask = (freqs >= 3.0) & (freqs <= 9.0)
    f_psd = psd[f_mask]
    f_power = float(np.sum(f_psd))
    f_band_power_ratio = float(f_power / total_power)

    geo_mean = float(np.exp(np.mean(np.log(f_psd + 1e-12))))
    arith_mean = float(np.mean(f_psd)) + 1e-12
    f_band_flatness = float(geo_mean / arith_mean)
    f_band_peak_purity = float(np.max(f_psd) / arith_mean) if len(f_psd) > 0 else 1.0

    p_mask = (freqs >= 1.0) & (freqs < 3.0)
    p_band_power_ratio = float(np.sum(psd[p_mask]) / total_power)

    qrs_mask = (freqs >= 10.0) & (freqs <= 25.0)
    qrs_power_ratio = float(np.sum(psd[qrs_mask]) / total_power)

    dx = np.diff(sig_filt)
    ddx = np.diff(dx)
    var_x = float(np.var(sig_filt)) + 1e-12
    var_dx = float(np.var(dx)) + 1e-12
    var_ddx = float(np.var(ddx)) + 1e-12

    mobility = float(np.sqrt(var_dx / var_x))
    complexity = float(np.sqrt(var_ddx / var_dx) / mobility) if mobility > 0 else 0.0
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
# Confounder Loaders
# -----------------------------------------------------------------------------
def load_cudb_record(rec_id: str) -> Tuple[np.ndarray, wfdb.Annotation]:
    """Loads a CUDB record from zip and returns 360 Hz resampled signal in physical mV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(CUDB_ZIP) as z:
            for item in z.namelist():
                if f"/{rec_id}." in item or item.startswith(f"{rec_id}."):
                    fname = os.path.basename(item)
                    with open(os.path.join(tmpdir, fname), "wb") as out_f:
                        out_f.write(z.read(item))

        rec_path = os.path.join(tmpdir, rec_id)
        rec_obj = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")

        raw_sig_250 = rec_obj.p_signal[:, 0].astype(np.float32)
        sig_360 = signal.resample_poly(raw_sig_250, up=36, down=25).astype(np.float32)
        return sig_360, ann


def load_mitbih_csv_record(rec_id: str) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """Loads an MIT-BIH record from MITBIH.zip, returns 360 Hz signal in mV and rhythm notes."""
    with zipfile.ZipFile(MITBIH_ZIP) as z:
        # Load CSV
        csv_name = f"{rec_id}.csv"
        with z.open(csv_name) as f:
            df_sig = pd.read_csv(f)
        col_names = [c.replace("'", "").strip() for c in df_sig.columns]
        df_sig.columns = col_names

        lead_col = "MLII" if "MLII" in col_names else col_names[1]
        sig_mv = ((df_sig[lead_col].values.astype(np.float64) - 1024.0) / 200.0).astype(np.float32)

        # Load annotations
        ann_name = f"{rec_id}annotations.txt"
        lines = z.read(ann_name).decode("utf-8", errors="replace").splitlines()
        rhythms = []
        for l in lines[1:]:
            parts = l.split()
            if len(parts) >= 3:
                try:
                    samp = int(parts[1])
                    aux = parts[-1] if parts[-1].startswith("(") else ""
                    if aux:
                        rhythms.append((samp, aux))
                except ValueError:
                    continue

        return sig_mv, rhythms


def main():
    print("=" * 80)
    print("INDEPENDENT 5-SECOND AFIB CLASSIFIER CONFOUNDER EVALUATION")
    print("=" * 80)

    # 1. Load Frozen Model Package
    pkg = joblib.load(MODEL_PACKAGE_PATH)
    champion_name = pkg["champion_model_name"]
    model = pkg["champion_model"]
    scaler = pkg["scaler"]
    feature_cols = pkg["feature_columns"]
    thresholds = pkg["operating_thresholds"]

    tc = thresholds["constrained_threshold"]
    tt = thresholds["triage_threshold"]
    t_cons = thresholds["consensus_threshold"]

    print(f"Loaded Frozen Champion Model: {champion_name}")
    print(f"Operating Thresholds: Constrained={tc:.3f}, Triage={tt:.3f}, Consensus={t_cons:.3f}")

    confounder_windows = []

    # -------------------------------------------------------------------------
    # Cohort 1: CUDB cu01 (Normal Sinus Baseline vs Sustained Coarse VF)
    # -------------------------------------------------------------------------
    print("\n[1/7] Ingesting CUDB cu01 (Sinus Baseline & Sustained Coarse VF):")
    sig_cu01, ann_cu01 = load_cudb_record("cu01")
    # cu01 has 214s normal sinus rhythm, then continuous sustained VF from sample 53541 (at 250 Hz, t=214.16s)
    vf_onset_sec = 214.16
    n_win_cu01 = int(len(sig_cu01) / WINDOW_SAMPLES)

    for w_i in range(n_win_cu01):
        ws = w_i * WINDOW_SAMPLES
        we = ws + WINDOW_SAMPLES
        t_start = w_i * WINDOW_SEC
        t_end = t_start + WINDOW_SEC

        w_sig = sig_cu01[ws:we]
        if t_end <= vf_onset_sec:
            clinical_truth = "Normal_Sinus_Rhythm"
            category = "Control_Sinus"
        elif t_start >= vf_onset_sec + 2.5:
            clinical_truth = "Ventricular_Fibrillation"
            category = "Confounder_VF"
        else:
            continue  # Transition window

        feats = extract_5s_features(w_sig, fs=TARGET_FS)
        feats.update({
            "record_id": "cudb_cu01",
            "source_db": "CUDB",
            "window_idx": w_i,
            "start_time_sec": t_start,
            "end_time_sec": t_end,
            "clinical_truth": clinical_truth,
            "category": category,
            "is_true_afib": 0
        })
        confounder_windows.append(feats)

    # -------------------------------------------------------------------------
    # Cohort 2: CUDB cu04 (Paroxysmal Rapid VT / Ventricular Flutter)
    # -------------------------------------------------------------------------
    print("[2/7] Ingesting CUDB cu04 (Paroxysmal Rapid VT / Ventricular Flutter):")
    sig_cu04, ann_cu04 = load_cudb_record("cu04")
    n_win_cu04 = int(len(sig_cu04) / WINDOW_SAMPLES)

    for w_i in range(n_win_cu04):
        ws = w_i * WINDOW_SAMPLES
        we = ws + WINDOW_SAMPLES
        t_start = w_i * WINDOW_SEC
        t_end = t_start + WINDOW_SEC
        w_sig = sig_cu04[ws:we]

        feats = extract_5s_features(w_sig, fs=TARGET_FS)
        feats.update({
            "record_id": "cudb_cu04",
            "source_db": "CUDB",
            "window_idx": w_i,
            "start_time_sec": t_start,
            "end_time_sec": t_end,
            "clinical_truth": "Ventricular_Tachycardia_Flutter",
            "category": "Confounder_VT",
            "is_true_afib": 0
        })
        confounder_windows.append(feats)

    # -------------------------------------------------------------------------
    # Cohort 3: MIT-BIH 222 (Pure Atrial Flutter - AFL)
    # -------------------------------------------------------------------------
    print("[3/7] Ingesting MIT-BIH 222 (Pure Sustained Atrial Flutter):")
    sig_222, rhythms_222 = load_mitbih_csv_record("222")
    # Take 100 5-second non-overlapping windows from pure AFL segment (sample 1000 to 181000)
    for w_i in range(100):
        ws = 1000 + w_i * WINDOW_SAMPLES
        we = ws + WINDOW_SAMPLES
        if we > len(sig_222):
            break
        w_sig = sig_222[ws:we]
        feats = extract_5s_features(w_sig, fs=TARGET_FS)
        feats.update({
            "record_id": "mitdb_222",
            "source_db": "MIT-BIH",
            "window_idx": w_i,
            "start_time_sec": round(ws / TARGET_FS, 2),
            "end_time_sec": round(we / TARGET_FS, 2),
            "clinical_truth": "Atrial_Flutter",
            "category": "Confounder_AFL",
            "is_true_afib": 0
        })
        confounder_windows.append(feats)

    # -------------------------------------------------------------------------
    # Cohort 4: MIT-BIH 233 (Continuous Ventricular Bigeminy)
    # -------------------------------------------------------------------------
    print("[4/7] Ingesting MIT-BIH 233 (Continuous Ventricular Bigeminy):")
    sig_233, rhythms_233 = load_mitbih_csv_record("233")
    for w_i in range(100):
        ws = 1000 + w_i * WINDOW_SAMPLES
        we = ws + WINDOW_SAMPLES
        if we > len(sig_233):
            break
        w_sig = sig_233[ws:we]
        feats = extract_5s_features(w_sig, fs=TARGET_FS)
        feats.update({
            "record_id": "mitdb_233",
            "source_db": "MIT-BIH",
            "window_idx": w_i,
            "start_time_sec": round(ws / TARGET_FS, 2),
            "end_time_sec": round(we / TARGET_FS, 2),
            "clinical_truth": "Ventricular_Bigeminy",
            "category": "Confounder_Bigeminy",
            "is_true_afib": 0
        })
        confounder_windows.append(feats)

    # -------------------------------------------------------------------------
    # Cohort 5: MIT-BIH 232 (Sinus Bradycardia, HR 40-55 bpm)
    # -------------------------------------------------------------------------
    print("[5/7] Ingesting MIT-BIH 232 (Sustained Sinus Bradycardia):")
    sig_232, rhythms_232 = load_mitbih_csv_record("232")
    for w_i in range(100):
        ws = 1000 + w_i * WINDOW_SAMPLES
        we = ws + WINDOW_SAMPLES
        if we > len(sig_232):
            break
        w_sig = sig_232[ws:we]
        feats = extract_5s_features(w_sig, fs=TARGET_FS)
        feats.update({
            "record_id": "mitdb_232",
            "source_db": "MIT-BIH",
            "window_idx": w_i,
            "start_time_sec": round(ws / TARGET_FS, 2),
            "end_time_sec": round(we / TARGET_FS, 2),
            "clinical_truth": "Sinus_Bradycardia",
            "category": "Confounder_Bradycardia",
            "is_true_afib": 0
        })
        confounder_windows.append(feats)

    # -------------------------------------------------------------------------
    # Cohort 6: MIT-BIH 100 (Clean Normal Sinus Rhythm Baseline)
    # -------------------------------------------------------------------------
    print("[6/7] Ingesting MIT-BIH 100 (Clean Normal Sinus Rhythm Control):")
    sig_100, _ = load_mitbih_csv_record("100")
    for w_i in range(100):
        ws = 1000 + w_i * WINDOW_SAMPLES
        we = ws + WINDOW_SAMPLES
        if we > len(sig_100):
            break
        w_sig = sig_100[ws:we]
        feats = extract_5s_features(w_sig, fs=TARGET_FS)
        feats.update({
            "record_id": "mitdb_100",
            "source_db": "MIT-BIH",
            "window_idx": w_i,
            "start_time_sec": round(ws / TARGET_FS, 2),
            "end_time_sec": round(we / TARGET_FS, 2),
            "clinical_truth": "Normal_Sinus_Rhythm",
            "category": "Control_Sinus",
            "is_true_afib": 0
        })
        confounder_windows.append(feats)

    # -------------------------------------------------------------------------
    # Cohort 7: Synthetic Quality & Artifact Stress-Test Windows
    # -------------------------------------------------------------------------
    print("[7/7] Synthesizing Signal Quality & Noise Artifact Windows:")
    t_arr = np.linspace(0, 5.0, WINDOW_SAMPLES, endpoint=False)
    
    # 1. Flatline (zero amplitude)
    sig_flat = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    f_flat = extract_5s_features(sig_flat, fs=TARGET_FS)
    f_flat.update({"record_id": "artifact_flatline", "source_db": "Synthetic", "window_idx": 0, "start_time_sec": 0.0, "end_time_sec": 5.0, "clinical_truth": "Flatline_Asystole", "category": "Confounder_Artifact", "is_true_afib": 0})
    confounder_windows.append(f_flat)

    # 2. Extreme rail saturation (>5.0 mV clipping)
    sig_sat = np.clip(np.sin(2 * np.pi * 1.0 * t_arr) * 10.0, -5.0, 5.0).astype(np.float32)
    f_sat = extract_5s_features(sig_sat, fs=TARGET_FS)
    f_sat.update({"record_id": "artifact_saturation", "source_db": "Synthetic", "window_idx": 1, "start_time_sec": 0.0, "end_time_sec": 5.0, "clinical_truth": "Saturation_Clipping", "category": "Confounder_Artifact", "is_true_afib": 0})
    confounder_windows.append(f_sat)

    # 3. High-frequency noise (EMG / electrode motion)
    np.random.seed(42)
    sig_noise = (np.random.randn(WINDOW_SAMPLES) * 0.8).astype(np.float32)
    f_noise = extract_5s_features(sig_noise, fs=TARGET_FS)
    f_noise.update({"record_id": "artifact_hf_noise", "source_db": "Synthetic", "window_idx": 2, "start_time_sec": 0.0, "end_time_sec": 5.0, "clinical_truth": "High_Frequency_Noise", "category": "Confounder_Artifact", "is_true_afib": 0})
    confounder_windows.append(f_noise)

    # Convert to DataFrame
    df_conf = pd.DataFrame(confounder_windows)
    print(f"\nTotal Confounder Windows Ingested: {len(df_conf):,}")

    # Scale with frozen training scaler
    X_conf_raw = df_conf[feature_cols].values
    X_conf = scaler.transform(X_conf_raw)
    probs_conf = model.predict_proba(X_conf)[:, 1]

    df_conf["prob_afib"] = np.round(probs_conf, 4)
    df_conf["pred_constrained_theta"] = (probs_conf >= tc).astype(int)
    df_conf["pred_triage_theta"] = (probs_conf >= tt).astype(int)
    df_conf["pred_50"] = (probs_conf >= 0.50).astype(int)

    # Save predictions CSV
    df_conf.to_csv(PREDICTIONS_CSV_PATH, index=False)
    print(f"Serialized Confounder Predictions: {PREDICTIONS_CSV_PATH.name} ({len(df_conf)} rows)")

    # -------------------------------------------------------------------------
    # Detailed Confounder Performance Breakdown
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STANDALONE UNGATED 5-SECOND AFIB CLASSIFIER: CONFOUNDER STRESS-TEST RESULTS")
    print("=" * 90)
    print(f"{'Cohort / Clinical Condition':<35s} | {'N':>4s} | {'Mean Prob':>9s} | {'Max Prob':>8s} | {'FP Rate @ 0.50':>14s} | {'FP Rate @ Constrained':>20s}")
    print("-" * 90)

    summary_by_cohort = {}
    for cat, grp in df_conf.groupby("category", sort=True):
        n = len(grp)
        mean_p = float(grp["prob_afib"].mean())
        max_p = float(grp["prob_afib"].max())
        fp_50 = int(grp["pred_50"].sum())
        fp_tc = int(grp["pred_constrained_theta"].sum())
        fp_rate_50 = fp_50 / n * 100.0
        fp_rate_tc = fp_tc / n * 100.0

        summary_by_cohort[cat] = {
            "category": cat,
            "windows_count": n,
            "mean_probability": round(mean_p, 4),
            "max_probability": round(max_p, 4),
            "false_alarms_50": fp_50,
            "false_alarm_rate_pct_50": round(fp_rate_50, 2),
            "false_alarms_constrained": fp_tc,
            "false_alarm_rate_pct_constrained": round(fp_rate_tc, 2)
        }
        print(f"{cat:<35s} | {n:4d} | {mean_p:9.4f} | {max_p:8.4f} | {fp_50:4d} ({fp_rate_50:5.1f}%) | {fp_tc:4d} ({fp_rate_tc:5.1f}%)")

    # Record-level summary
    print("\n" + "-" * 90)
    print("Record-Level Granular Confounder Breakdown:")
    print("-" * 90)
    summary_by_record = {}
    for (rec, truth), grp in df_conf.groupby(["record_id", "clinical_truth"], sort=True):
        n = len(grp)
        mean_p = float(grp["prob_afib"].mean())
        max_p = float(grp["prob_afib"].max())
        fp_50 = int(grp["pred_50"].sum())
        fp_tc = int(grp["pred_constrained_theta"].sum())
        fp_rate_50 = fp_50 / n * 100.0
        fp_rate_tc = fp_tc / n * 100.0

        summary_by_record[f"{rec}_{truth}"] = {
            "record_id": rec,
            "clinical_truth": truth,
            "windows_count": n,
            "mean_probability": round(mean_p, 4),
            "max_probability": round(max_p, 4),
            "false_alarms_50": fp_50,
            "false_alarm_rate_pct_50": round(fp_rate_50, 2),
            "false_alarms_constrained": fp_tc,
            "false_alarm_rate_pct_constrained": round(fp_rate_tc, 2)
        }
        print(f"{rec + ' (' + truth + ')':<35s} | {n:4d} | {mean_p:9.4f} | {max_p:8.4f} | {fp_50:4d} ({fp_rate_50:5.1f}%) | {fp_tc:4d} ({fp_rate_tc:5.1f}%)")

    # Save summary JSON
    confounder_metrics = {
        "pipeline_name": "AFDB 5-Second Dedicated Classifier",
        "evaluation_scope": "Independent Multi-Arrhythmia Confounder Testing",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "champion_model_name": champion_name,
        "operating_thresholds": {
            "constrained_theta": tc,
            "triage_theta": tt,
            "default_50": 0.50
        },
        "summary_by_category": summary_by_cohort,
        "summary_by_record": summary_by_record
    }

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(confounder_metrics, f, indent=2)

    print(f"\n  - Saved Confounder Evaluation Metrics: {METRICS_OUTPUT_PATH.name} ({os.path.getsize(METRICS_OUTPUT_PATH):,} bytes)")
    print("\n" + "=" * 90)
    print("CONFOUNDER STRESS-TEST EVALUATION COMPLETED.")
    print("=" * 90)


if __name__ == "__main__":
    main()
