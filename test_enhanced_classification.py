#!/usr/bin/env python3
"""
Standalone Enhanced Classification & Temporal Consensus Evaluation Script.
Evaluates Live ADS1292R Hardware (Option B: Live hardware with zero web UI overhead).

Features evaluated:
  - Method A: Current production baseline (peak detection on raw window)
  - Method B: Filtered peak detection on diag_window + HR consistency check
  - Method C: Filtered peak detection + 10-second temporal consensus (3 overlapping windows)
  - Benchmark Mode: Quantitative evaluation on the multi-source dataset
"""

import sys
import os
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import deque, Counter
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch

# Ensure deploy modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "deploy"))

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime is required. Install via: pip install onnxruntime")
    sys.exit(1)

from deploy.ads1292r_driver import (
    FS,
    WINDOW_SAMPLES,
    STEP_SAMPLES,
    create_adc,
    research_grade_filter,
    detect_r_peaks,
)

# Model paths
MODELS_DIR = PROJECT_ROOT / "deploy" / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = PROJECT_ROOT / "training" / "final_models" / "export_candidates"

M1_PATH = MODELS_DIR / "shockable_classifier_candidate.onnx"
M2_PATH = MODELS_DIR / "arrhythmia_multiclass_4class_candidate.onnx"
M3_PATH = MODELS_DIR / "vf_vt_subtype_classifier_candidate.onnx"

M1_FEATURES = [
    "vf_band_power_ratio", "dominant_freq", "spectral_entropy", "spectral_peak_purity",
    "zero_crossings", "hjorth_mobility", "hjorth_complexity", "lz_complexity",
    "calibrated_ptp", "calibrated_std",
]

M2_FEATURES = [
    "mean_rr", "rr_cv", "qrs_width", "peak_count", "dominant_freq",
    "vf_band_power_ratio", "spectral_entropy", "zero_crossings", "hjorth_mobility",
    "hjorth_complexity", "calibrated_ptp", "calibrated_std",
]

M3_FEATURES = [
    "lz_complexity", "ac_decay_time", "ac_zero_crossing_lag",
    "ac_max_peak_ratio", "sample_entropy", "hjorth_mobility",
]

M2_CLASSES = ["NSR", "TACHY", "BRADY_ASY", "VENTRICULAR"]
M3_CLASSES = ["VT", "VF"]


# ======================================================================
# SIGNAL QUALITY LAYER
# ======================================================================

@dataclass
class SignalQualityReport:
    status: str
    is_valid_for_ml: bool
    n_samples: int
    mean_mv: float
    std_mv: float
    ptp_mv: float
    detail: str


def evaluate_signal_quality(window: np.ndarray) -> SignalQualityReport:
    n_samples = len(window)
    if n_samples < WINDOW_SAMPLES:
        return SignalQualityReport("WARMING_UP", False, n_samples, 0.0, 0.0, 0.0, "Buffering")

    if np.any(~np.isfinite(window)):
        return SignalQualityReport("BAD_SIGNAL_NONFINITE", False, n_samples, 0.0, 0.0, 0.0, "Non-finite samples")

    # 1. Hardware ADC rail clipping (+/- 403.3 mV full-scale on ADS1292R)
    n_rail = int(np.sum(np.abs(window) >= 380.0))
    if (n_rail / n_samples) * 100.0 > 5.0:
        return SignalQualityReport("BAD_SIGNAL_SATURATED", False, n_samples, float(np.mean(window)),
                                   float(np.std(window)), float(np.ptp(window)), "Hardware ADC rail saturated")

    # 2. AC dynamic range (de-meaned)
    ac_sig = window - np.median(window)
    ac_std = float(np.std(ac_sig))
    ac_ptp = float(np.ptp(ac_sig))

    if ac_std < 0.015 or ac_ptp < 0.050:
        return SignalQualityReport("BAD_SIGNAL_FLAT", False, n_samples, float(np.mean(window)),
                                   ac_std, ac_ptp, "Signal flatline or disconnected")

    n_ac_sat = int(np.sum(np.abs(ac_sig) > 15.0))
    if (n_ac_sat / n_samples) * 100.0 > 10.0 or ac_ptp > 35.0:
        return SignalQualityReport("BAD_SIGNAL_SATURATED", False, n_samples, float(np.mean(window)),
                                   ac_std, ac_ptp, "AC voltage saturated (>15mV)")

    return SignalQualityReport("SIGNAL_OK", True, n_samples, float(np.mean(window)),
                               ac_std, ac_ptp, "Signal quality OK")


# ======================================================================
# FEATURE EXTRACTION ENGINE
# ======================================================================

def extract_spectral_and_complexity(diag_window: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """Computes spectral, complexity, and autocorrelation features."""
    ptp_val = float(np.ptp(diag_window))
    std_val = float(np.std(diag_window))
    zero_cross = int(np.sum(np.diff(diag_window > 0) != 0))

    # Welch PSD
    nperseg = min(len(diag_window), int(fs * 2))
    freqs, psd = welch(diag_window, fs=fs, nperseg=nperseg)
    valid_mask = (freqs >= 0.5) & (freqs <= 30.0)
    freqs_band = freqs[valid_mask]
    psd_band = psd[valid_mask]

    total_power = np.sum(psd_band) + 1e-12
    vf_mask = (freqs_band >= 3.0) & (freqs_band <= 9.0)
    vf_ratio = float(np.sum(psd_band[vf_mask]) / total_power)

    dom_idx = int(np.argmax(psd_band))
    dom_freq = float(freqs_band[dom_idx])

    psd_norm = psd_band / total_power
    psd_norm = psd_norm[psd_norm > 0]
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm) + 1e-12))
    spec_purity = float(psd_band[dom_idx] / (np.median(psd_band) + 1e-12))

    # Hjorth parameters
    diff1 = np.diff(diag_window)
    diff2 = np.diff(diff1)
    var_zero = std_val ** 2 + 1e-12
    var_d1 = float(np.var(diff1)) + 1e-12
    var_d2 = float(np.var(diff2)) + 1e-12
    mobility = float(np.sqrt(var_d1 / var_zero))
    complexity = float(np.sqrt(var_d2 / var_d1) / mobility)

    # Lempel-Ziv complexity
    step = max(1, int(round(fs / 90.0)))
    sig_down = diag_window[::step]
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

    # Autocorrelation features (for Tier 3)
    x_centered = diag_window - np.mean(diag_window)
    var_ac = float(np.var(x_centered))
    if var_ac < 1e-12:
        ac_max_peak, decay_time, zero_lag = 0.0, 0.0, 0.0
    else:
        n = len(x_centered)
        r = np.correlate(x_centered, x_centered, mode="full")[n - 1:]
        r_norm = r / r[0]
        min_lag = int(0.2 * fs)
        max_lag = min(int(1.0 * fs), len(r_norm) - 1)
        cardiac_r = r_norm[min_lag:max_lag]
        ac_max_peak = float(np.max(cardiac_r)) if len(cardiac_r) > 0 else 0.0
        decay_idx = np.where(r_norm < (1.0 / np.e))[0]
        decay_time = float(decay_idx[0] / fs) if len(decay_idx) > 0 else float(max_lag / fs)
        zero_idx = np.where(r_norm < 0.0)[0]
        zero_lag = float(zero_idx[0] / fs) if len(zero_idx) > 0 else float(max_lag / fs)

    # Sample entropy
    x_down = diag_window[::8].astype(np.float32)
    std_down = float(np.std(x_down))
    if std_down > 1e-8:
        r_thresh = 0.2 * std_down
        X2 = np.lib.stride_tricks.sliding_window_view(x_down, 2)
        X3 = np.lib.stride_tricks.sliding_window_view(x_down, 3)
        d2 = np.max(np.abs(X2[:, None, :] - X2[None, :, :]), axis=2)
        d3 = np.max(np.abs(X3[:, None, :] - X3[None, :, :]), axis=2)
        np.fill_diagonal(d2, np.inf)
        np.fill_diagonal(d3, np.inf)
        B = np.sum(d2 < r_thresh)
        A = np.sum(d3 < r_thresh)
        samp_en = float(-math.log((A + 1e-12) / (B + 1e-12))) if B > 0 else 0.0
    else:
        samp_en = 0.0

    return {
        "calibrated_ptp": ptp_val,
        "calibrated_std": std_val,
        "zero_crossings": zero_cross,
        "vf_band_power_ratio": vf_ratio,
        "dominant_freq": dom_freq,
        "spectral_entropy": spec_entropy,
        "spectral_peak_purity": spec_purity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "lz_complexity": lz_norm,
        "ac_max_peak_ratio": ac_max_peak,
        "ac_decay_time": decay_time,
        "ac_zero_crossing_lag": zero_lag,
        "sample_entropy": samp_en,
    }


def compute_peak_features(peaks: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """Computes mean_rr, rr_cv, qrs_width, and heart rate metrics from detected peaks."""
    peak_count = len(peaks)
    if peak_count >= 2:
        rr_intervals = np.diff(peaks) / float(fs)
        mean_rr = float(np.mean(rr_intervals))
        median_rr = float(np.median(rr_intervals))
        rr_cv = float(np.std(rr_intervals) / (mean_rr + 1e-12))
        qrs_width = 0.080
        hr_bpm = 60.0 / mean_rr
        hr_median_bpm = 60.0 / median_rr
    else:
        mean_rr = 5.0
        median_rr = 5.0
        rr_cv = 0.0
        qrs_width = 0.0
        hr_bpm = 0.0
        hr_median_bpm = 0.0

    return {
        "peak_count": peak_count,
        "mean_rr": mean_rr,
        "median_rr": median_rr,
        "rr_cv": rr_cv,
        "qrs_width": qrs_width,
        "heart_rate_bpm": hr_bpm,
        "heart_rate_median_bpm": hr_median_bpm,
    }


def determine_hr_category(peak_count: int, mean_rr: float, hr_bpm: float) -> str:
    """
    Physiological reference categorization based on clinical heart rate criteria.
    Used for agreement analysis (does NOT override Model 2).
    """
    if peak_count <= 1 or mean_rr >= 3.0:
        return "BRADY_ASY"  # Asystole / extreme pause
    if hr_bpm < 60.0:
        return "BRADY_ASY"  # Bradycardia
    elif hr_bpm <= 100.0:
        return "NSR"        # Normal Sinus Rhythm
    else:
        return "TACHY"      # Tachycardia


# ======================================================================
# ONNX CASCADE RUNTIME
# ======================================================================

class InferenceEngine:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess_m1 = ort.InferenceSession(str(M1_PATH), opts, providers=["CPUExecutionProvider"])
        self.sess_m2 = ort.InferenceSession(str(M2_PATH), opts, providers=["CPUExecutionProvider"])
        self.sess_m3 = ort.InferenceSession(str(M3_PATH), opts, providers=["CPUExecutionProvider"])

        self.in_m1 = self.sess_m1.get_inputs()[0].name
        self.in_m2 = self.sess_m2.get_inputs()[0].name
        self.in_m3 = self.sess_m3.get_inputs()[0].name

    def run_cascade(self, features: Dict[str, float]) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Model 1: Shockable Screener
        v1 = np.array([[features[k] for k in M1_FEATURES]], dtype=np.float32)
        out1 = self.sess_m1.run(None, {self.in_m1: v1})
        p1 = out1[1][0] if isinstance(out1[1], (list, np.ndarray)) else out1[0][0]
        p_shock = float(p1[1]) if not isinstance(p1, dict) else float(p1.get(1, 0.0))
        is_shock = p_shock >= 0.50

        # Model 2: 4-Class Multi-Arrhythmia
        v2 = np.array([[features[k] for k in M2_FEATURES]], dtype=np.float32)
        out2 = self.sess_m2.run(None, {self.in_m2: v2})
        p2 = out2[1][0] if isinstance(out2[1], (list, np.ndarray)) else out2[0][0]
        if isinstance(p2, dict):
            p2_list = [float(p2.get(i, 0.0)) for i in range(len(M2_CLASSES))]
        else:
            p2_list = [float(p) for p in p2]
        pred_idx2 = int(np.argmax(p2_list))
        m2_class = M2_CLASSES[pred_idx2]
        m2_conf = float(p2_list[pred_idx2])

        # Model 3: Ventricular Specialist
        m3_sub = "--"
        if is_shock or m2_class == "VENTRICULAR":
            v3 = np.array([[features[k] for k in M3_FEATURES]], dtype=np.float32)
            out3 = self.sess_m3.run(None, {self.in_m3: v3})
            p3 = out3[1][0] if isinstance(out3[1], (list, np.ndarray)) else out3[0][0]
            p_vf = float(p3[1]) if not isinstance(p3, dict) else float(p3.get(1, 0.0))
            m3_sub = "VF" if p_vf >= 0.50 else "VT"

        # Final classification
        if is_shock:
            final_class = f"SHOCKABLE ({m3_sub})"
        elif m2_class == "VENTRICULAR":
            final_class = f"VENTRICULAR ({m3_sub})"
        else:
            final_class = m2_class

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "is_shockable": is_shock,
            "m1_prob_shock": p_shock,
            "m2_class": m2_class,
            "m2_conf": m2_conf,
            "m2_probs": {M2_CLASSES[i]: round(p2_list[i], 3) for i in range(len(M2_CLASSES))},
            "m3_sub": m3_sub,
            "final_class": final_class,
            "latency_ms": latency_ms,
        }


# ======================================================================
# MULTI-METHOD WINDOW EVALUATOR
# ======================================================================

def evaluate_window_methods(raw_window: np.ndarray, engine: InferenceEngine) -> Dict[str, Any]:
    """Runs Method A (raw peaks) and Method B (filtered peaks) on a 5-second window."""
    # 1. Diagnostic filtering
    diag_window = research_grade_filter(raw_window, fs=FS)

    # 2. Base spectral/complexity features (identical for both methods)
    base_feats = extract_spectral_and_complexity(diag_window, fs=FS)

    # Method A: Peaks on raw window (Production baseline)
    peaks_a = detect_r_peaks(raw_window, fs=FS)
    peak_feats_a = compute_peak_features(peaks_a, fs=FS)
    feats_a = {**base_feats, **peak_feats_a}
    res_a = engine.run_cascade(feats_a)

    # Method B: Peaks on filtered diagnostic window
    peaks_b = detect_r_peaks(diag_window, fs=FS)
    peak_feats_b = compute_peak_features(peaks_b, fs=FS)
    feats_b = {**base_feats, **peak_feats_b}
    res_b = engine.run_cascade(feats_b)

    # HR Consistency evaluation on Method B
    hr_cat_b = determine_hr_category(
        peak_feats_b["peak_count"],
        peak_feats_b["mean_rr"],
        peak_feats_b["heart_rate_bpm"]
    )
    hr_agree_b = (res_b["m2_class"] == hr_cat_b)

    return {
        "diag_window": diag_window,
        "method_a": {
            **res_a,
            "peaks": len(peaks_a),
            "hr_bpm": peak_feats_a["heart_rate_bpm"],
            "mean_rr": peak_feats_a["mean_rr"],
        },
        "method_b": {
            **res_b,
            "peaks": len(peaks_b),
            "hr_bpm": peak_feats_b["heart_rate_bpm"],
            "median_hr_bpm": peak_feats_b["heart_rate_median_bpm"],
            "mean_rr": peak_feats_b["mean_rr"],
            "median_rr": peak_feats_b["median_rr"],
            "hr_category": hr_cat_b,
            "hr_agreement": hr_agree_b,
        },
    }


# ======================================================================
# LIVE EVALUATION RUNTIME (OPTION B: LIVE ADS1292R, NO WEB OVERHEAD)
# ======================================================================

def run_live_evaluation(mock: bool = False, interval_sec: float = 10.0):
    print("=" * 80)
    print("  ECG ARRHYTHMIA EDGE AI — ENHANCED LIVE CLASSIFICATION EVALUATION")
    print(f"  Hardware: ADS1292R (24-bit SPI, 360 Hz) {'[MOCK SIMULATOR]' if mock else '[PHYSICAL SPI]'}")
    print(f"  Mode    : OPTION B (Headless Inference, Web UI Chart Polling Bypassed)")
    print(f"  Cadence : Evaluation every {interval_sec:.1f} seconds")
    print("=" * 80)

    adc = create_adc(mock=mock, fs=FS)
    engine = InferenceEngine()

    # 10-second history buffer (3600 samples) to allow multi-window temporal consensus
    BUFFER_10S_SAMPLES = int(FS * 10.0)  # 3600 samples
    history_buffer = np.zeros(BUFFER_10S_SAMPLES, dtype=np.float32)
    total_acquired = 0

    # Session metrics
    session_stats = {
        "intervals_evaluated": 0,
        "valid_windows": 0,
        "rejected_windows": 0,
        "m_a_vs_m_b_agreed": 0,
        "m_b_vs_hr_agreed": 0,
        "consensus_stable_100": 0,
        "consensus_split": 0,
        "class_counts_a": Counter(),
        "class_counts_b": Counter(),
        "class_counts_c": Counter(),
    }

    t_start = time.perf_counter()
    sample_count = 0
    last_infer_time = t_start
    last_rate_time = t_start
    measured_sps = 360.0

    print("\n[SYSTEM] Continuous acquisition started. Warming up 10-second buffer...\n")

    try:
        while True:
            t_now = time.perf_counter()
            ch1_mv, _ = adc.read_sample()
            sample_count += 1
            total_acquired += 1

            # Roll 10-second buffer
            history_buffer = np.roll(history_buffer, -1)
            history_buffer[-1] = ch1_mv

            # Update sample rate estimate every 2 seconds
            if t_now - last_rate_time >= 2.0:
                measured_sps = (sample_count - (total_acquired - sample_count)) / (t_now - last_rate_time) if (t_now - last_rate_time) > 0 else 360.0
                last_rate_time = t_now
                sample_count = 0

            # Trigger evaluation every interval_sec
            if total_acquired >= BUFFER_10S_SAMPLES and (t_now - last_infer_time >= interval_sec):
                last_infer_time = t_now
                session_stats["intervals_evaluated"] += 1
                timestamp_str = time.strftime("%H:%M:%S")

                # Most recent 5-second window (last 1800 samples)
                w_curr_raw = history_buffer[-WINDOW_SAMPLES:].copy()

                # Signal Quality Check
                quality = evaluate_signal_quality(w_curr_raw)
                if not quality.is_valid_for_ml:
                    session_stats["rejected_windows"] += 1
                    print(f"[{timestamp_str}] [SIGNAL REJECTED] Status: {quality.status:<20} | Reason: {quality.detail}")
                    continue

                session_stats["valid_windows"] += 1

                # 1. Run Method A & Method B on the primary 5-second window
                primary_eval = evaluate_window_methods(w_curr_raw, engine)
                res_a = primary_eval["method_a"]
                res_b = primary_eval["method_b"]

                # 2. Run Method C: 10-Second Temporal Consensus (3 overlapping sub-windows)
                # Sub-window 1: [0:1800]   (-10s to -5s)
                # Sub-window 2: [900:2700]  (-7.5s to -2.5s)
                # Sub-window 3: [1800:3600] (-5.0s to 0s)
                w1 = history_buffer[0:1800].copy()
                w2 = history_buffer[900:2700].copy()
                w3 = history_buffer[1800:3600].copy()

                eval1 = evaluate_window_methods(w1, engine)["method_b"]
                eval2 = evaluate_window_methods(w2, engine)["method_b"]
                eval3 = res_b  # w3 is identical to w_curr_raw

                window_predictions = [eval1["m2_class"], eval2["m2_class"], eval3["m2_class"]]
                pred_counts = Counter(window_predictions)
                consensus_class, top_count = pred_counts.most_common(1)[0]
                stability_pct = (top_count / 3.0) * 100.0

                hrs_10s = [eval1["hr_bpm"], eval2["hr_bpm"], eval3["hr_bpm"]]
                median_hr_10s = float(np.median([h for h in hrs_10s if h > 0])) if any(h > 0 for h in hrs_10s) else 0.0

                # Update session statistics
                if res_a["m2_class"] == res_b["m2_class"]:
                    session_stats["m_a_vs_m_b_agreed"] += 1
                if res_b["hr_agreement"]:
                    session_stats["m_b_vs_hr_agreed"] += 1
                if stability_pct == 100.0:
                    session_stats["consensus_stable_100"] += 1
                else:
                    session_stats["consensus_split"] += 1

                session_stats["class_counts_a"][res_a["m2_class"]] += 1
                session_stats["class_counts_b"][res_b["m2_class"]] += 1
                session_stats["class_counts_c"][consensus_class] += 1

                # Terminal Report
                hr_agree_str = "AGREE" if res_b["hr_agreement"] else f"DISAGREE (HR indicates {res_b['hr_category']})"
                stab_str = "100% UNANIMOUS" if stability_pct == 100.0 else f"{stability_pct:.0f}% SPLIT {window_predictions}"

                print("-" * 80)
                print(f"  [{timestamp_str}] 10s EVALUATION #{session_stats['valid_windows']:03d} | Measured SPS: {measured_sps:.1f} | Signal: {quality.status}")
                print(f"  Quality Metrics    : PTP={quality.ptp_mv:.2f} mV | AC Std={quality.std_mv:.2f} mV | Samples={WINDOW_SAMPLES}")
                print("  " + "-" * 76)
                print(f"  Method A (Raw Pk)  : Class: {res_a['m2_class']:<12} | Conf: {res_a['m2_conf']*100:5.1f}% | Peaks: {res_a['peaks']} | HR: {res_a['hr_bpm']:5.1f} BPM | Latency: {res_a['latency_ms']:.2f}ms")
                print(f"  Method B (Filt Pk) : Class: {res_b['m2_class']:<12} | Conf: {res_b['m2_conf']*100:5.1f}% | Peaks: {res_b['peaks']} | HR: {res_b['hr_bpm']:5.1f} BPM | Latency: {res_b['latency_ms']:.2f}ms")
                print(f"  HR Consistency     : {hr_agree_str}")
                print(f"  Method C (Consensus: Class: {consensus_class:<12} | Stability: {stab_str:<26} | 10s Med HR: {median_hr_10s:5.1f} BPM")
                print("-" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("  SESSION EVALUATION SUMMARY (Ctrl+C Caught)")
        print("=" * 80)
        n_eval = session_stats["intervals_evaluated"]
        n_valid = session_stats["valid_windows"]
        n_rej = session_stats["rejected_windows"]

        print(f"  Total Intervals Run : {n_eval}")
        print(f"  Valid Windows       : {n_valid} ({(n_valid/max(1,n_eval))*100:.1f}%)")
        print(f"  Rejected (Bad Signal: {n_rej} ({(n_rej/max(1,n_eval))*100:.1f}%)")
        print("  " + "-" * 76)
        print(f"  Method A vs B Match : {session_stats['m_a_vs_m_b_agreed']}/{max(1,n_valid)} ({(session_stats['m_a_vs_m_b_agreed']/max(1,n_valid))*100:.1f}%)")
        print(f"  Method B vs HR Match: {session_stats['m_b_vs_hr_agreed']}/{max(1,n_valid)} ({(session_stats['m_b_vs_hr_agreed']/max(1,n_valid))*100:.1f}%)")
        print(f"  10s Unanimous Cons. : {session_stats['consensus_stable_100']}/{max(1,n_valid)} ({(session_stats['consensus_stable_100']/max(1,n_valid))*100:.1f}%)")
        print("  " + "-" * 76)
        print("  Predicted Distribution Across Methods:")
        print(f"    Method A (Raw)    : {dict(session_stats['class_counts_a'])}")
        print(f"    Method B (Filt)   : {dict(session_stats['class_counts_b'])}")
        print(f"    Method C (Consens): {dict(session_stats['class_counts_c'])}")
        print("=" * 80)


# ======================================================================
# QUANTITATIVE BENCHMARK EVALUATION ON VALIDATED DATASET
# ======================================================================

def run_benchmark_evaluation():
    csv_path = PROJECT_ROOT / "training" / "final_models" / "artifacts" / "model2_multiclass_4class_dataset.csv"
    if not csv_path.exists():
        print(f"[ERROR] Benchmark dataset not found at {csv_path}")
        return

    import pandas as pd
    print("=" * 80)
    print("  RUNNING QUANTITATIVE BENCHMARK ON MODEL 2 (4-CLASS DATASET)")
    print(f"  Dataset: {csv_path.name}")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    engine = InferenceEngine()

    in_name = engine.in_m2
    X = df[M2_FEATURES].values.astype(np.float32)

    t0 = time.perf_counter()
    out = engine.sess_m2.run(None, {in_name: X})
    latency_total = (time.perf_counter() - t0) * 1000.0
    latency_per_win = latency_total / len(df)

    probs = out[1] if isinstance(out[1], np.ndarray) else np.array([[row[i] for i in range(4)] for row in out[1]])
    preds = np.argmax(probs, axis=1)
    df["m2_pred"] = [M2_CLASSES[p] for p in preds]

    # HR Consistency
    def row_hr_cat(r):
        return determine_hr_category(int(r["peak_count"]), float(r["mean_rr"]), 60.0 / float(r["mean_rr"]))

    df["hr_cat"] = df.apply(row_hr_cat, axis=1)
    df["hr_match"] = df["m2_pred"] == df["hr_cat"]

    # Metrics
    y_true = df["model2_4class"].values
    y_pred = df["m2_pred"].values

    print(f"\nTotal Evaluated Windows: {len(df):,}")
    print(f"Mean Inference Latency : {latency_per_win:.3f} ms / window\n")

    print("Confusion Matrix (Rows: Ground Truth, Cols: Model 2 Prediction):")
    ct = pd.crosstab(df["model2_4class"], df["m2_pred"], margins=True)
    print(ct.to_string())

    print("\nPer-Class Performance Metrics:")
    print("-" * 65)
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)

    for c in M2_CLASSES:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        supp = np.sum(y_true == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{c:<14} | {prec*100:8.2f}% | {rec*100:8.2f}% | {f1*100:8.2f}% | {supp:8d}")
    print("-" * 65)

    # NSR & BRADY_ASY Specific Cross-Agreement Analysis
    print("\nNSR & BRADY_ASY Agreement with Physiological Heart Rate Check:")
    print("-" * 65)
    for c in ["NSR", "BRADY_ASY"]:
        sub = df[df["m2_pred"] == c]
        match_count = (sub["hr_cat"] == c).sum()
        total_c = len(sub)
        pct = (match_count / max(1, total_c)) * 100.0
        print(f"When Model 2 predicts {c:<9}: HR check agrees on {match_count}/{total_c} ({pct:.1f}%)")
        mismatches = sub[sub["hr_cat"] != c]["hr_cat"].value_counts().to_dict()
        print(f"   Mismatches categorized by HR check as: {mismatches}")
    print("-" * 65)
    print("=" * 80 + "\n")


# ======================================================================
# CLI ENTRYPOINT
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced ECG Classification & Temporal Consensus Evaluation")
    parser.add_argument("--mock", action="store_true", help="Run offline simulation mode without physical ADS1292R SPI")
    parser.add_argument("--interval", type=float, default=10.0, help="Inference interval in seconds (default: 10.0)")
    parser.add_argument("--benchmark", action="store_true", help="Run quantitative evaluation on the 4-class multi-source benchmark dataset")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_evaluation()
    else:
        run_live_evaluation(mock=args.mock, interval_sec=args.interval)
