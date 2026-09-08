#!/usr/bin/env python3
"""
Unified ECG Arrhythmia Edge AI Application
Target: Raspberry Pi 4B+ (with ADS1292R SPI Analog Front-End)

CRITICAL DESIGN:
  - EXACTLY ONE ADS1292R instance.
  - EXACTLY ONE acquisition thread.
  - The acquisition thread feeds:
      1. Rolling 6-second display buffer for browser visualization (RAW + FILTERED).
      2. Rolling 5-second inference buffer (1,800 samples @ 360 Hz).
  - Conservative Signal Quality Diagnostic Layer (Flatline, Non-finite, Saturation).
  - 3-Model Hierarchical ML Cascade:
      Model 1: Shockable Screener (Candidate ONNX)
      Model 2: 4-Class Rhythm Classifier (Candidate ONNX)
      Model 3: VT vs VF Ventricular Subtype Specialist (Candidate ONNX)
  - Interactive Web Dashboard at http://<pi-ip>:5000.
"""

import argparse
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch, fftconvolve
import onnxruntime as ort
from flask import Flask, jsonify, Response

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from deploy.ads1292r_driver import (
    FS,
    WINDOW_SAMPLES,
    STEP_SAMPLES,
    create_adc,
    research_grade_filter,
    detect_r_peaks,
)

# Model paths
MODELS_DIR = Path(__file__).resolve().parent / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = ROOT_DIR / "training" / "final_models" / "export_candidates"

M1_PATH = MODELS_DIR / "shockable_classifier_candidate.onnx"
M2_PATH = MODELS_DIR / "arrhythmia_multiclass_4class_candidate.onnx"
M3_PATH = MODELS_DIR / "vf_vt_subtype_classifier_candidate.onnx"

# Feature definitions matching training EXACTLY
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

# Visualization config
DISPLAY_SECONDS = 6.0
DISPLAY_BUFFER_LEN = int(FS * DISPLAY_SECONDS)
CHART_REFILTER_EVERY_N = 90  # ~0.25 sec


# ======================================================================
# SIGNAL QUALITY DIAGNOSTIC LAYER
# ======================================================================

@dataclass
class SignalQualityReport:
    status: str             # "SIGNAL_OK", "BAD_SIGNAL_FLAT", "BAD_SIGNAL_NONFINITE", "BAD_SIGNAL_SATURATED", "WARMING_UP"
    is_valid_for_ml: bool
    n_samples: int
    mean_mv: float
    std_mv: float
    ptp_mv: float
    min_mv: float
    max_mv: float
    pct_nonfinite: float
    pct_saturated: float
    detail: str


def evaluate_signal_quality(window: np.ndarray) -> SignalQualityReport:
    n_samples = len(window)
    if n_samples < WINDOW_SAMPLES:
        return SignalQualityReport(
            status="WARMING_UP",
            is_valid_for_ml=False,
            n_samples=n_samples,
            mean_mv=0.0, std_mv=0.0, ptp_mv=0.0, min_mv=0.0, max_mv=0.0,
            pct_nonfinite=0.0, pct_saturated=0.0,
            detail=f"Buffering samples ({n_samples}/{WINDOW_SAMPLES})"
        )

    n_nonfinite = int(np.sum(~np.isfinite(window)))
    pct_nonfinite = (n_nonfinite / n_samples) * 100.0

    if pct_nonfinite > 0.0:
        return SignalQualityReport(
            status="BAD_SIGNAL_NONFINITE",
            is_valid_for_ml=False,
            n_samples=n_samples,
            mean_mv=0.0, std_mv=0.0, ptp_mv=0.0, min_mv=0.0, max_mv=0.0,
            pct_nonfinite=pct_nonfinite, pct_saturated=0.0,
            detail="NaN or Infinite samples detected in window"
        )

    min_val = float(np.min(window))
    max_val = float(np.max(window))
    mean_val = float(np.mean(window))
    std_val = float(np.std(window))
    ptp_val = float(np.ptp(window))

    # Flatline detection: variance near zero or minimal movement
    if std_val < 0.015 or ptp_val < 0.050:
        return SignalQualityReport(
            status="BAD_SIGNAL_FLAT",
            is_valid_for_ml=False,
            n_samples=n_samples,
            mean_mv=mean_val, std_mv=std_val, ptp_mv=ptp_val, min_mv=min_val, max_mv=max_val,
            pct_nonfinite=0.0, pct_saturated=0.0,
            detail="Signal is flatline or disconnected"
        )

    # Extreme saturation detection (|V| > 10.0 mV indicates rail or detached electrode)
    n_sat = int(np.sum(np.abs(window) > 10.0))
    pct_sat = (n_sat / n_samples) * 100.0
    if pct_sat > 5.0:
        return SignalQualityReport(
            status="BAD_SIGNAL_SATURATED",
            is_valid_for_ml=False,
            n_samples=n_samples,
            mean_mv=mean_val, std_mv=std_val, ptp_mv=ptp_val, min_mv=min_val, max_mv=max_val,
            pct_nonfinite=0.0, pct_saturated=pct_sat,
            detail=f"Voltage saturated (|V| > 10mV) on {pct_sat:.1f}% of window"
        )

    return SignalQualityReport(
        status="SIGNAL_OK",
        is_valid_for_ml=True,
        n_samples=n_samples,
        mean_mv=mean_val, std_mv=std_val, ptp_mv=ptp_val, min_mv=min_val, max_mv=max_val,
        pct_nonfinite=0.0, pct_saturated=pct_sat,
        detail="Signal quality is acceptable for ML triage"
    )


# ======================================================================
# FEATURE EXTRACTION (EXACT TRAINING IMPLEMENTATION)
# ======================================================================

def extract_features(diag_window: np.ndarray, raw_window: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """Extracts all 14 unified temporal, spectral, and complexity features."""
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

    # Lempel-Ziv complexity (downsampled to 90 Hz, N=450)
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

    # Path B: Pan-Tompkins peak timing on raw window
    peaks = detect_r_peaks(raw_window, fs=fs)
    peak_count = len(peaks)
    if peak_count >= 2:
        rr_intervals = np.diff(peaks) / fs
        mean_rr = float(np.mean(rr_intervals))
        rr_cv = float(np.std(rr_intervals) / (mean_rr + 1e-12))
        qrs_width = 0.080
    else:
        mean_rr = 5.0
        rr_cv = 0.0
        qrs_width = 0.0

    # Autocorrelation features (Domain-Stable Model 3)
    x_centered = diag_window - np.mean(diag_window)
    var = float(np.var(x_centered))
    if var < 1e-12:
        ac_max_peak, decay_time, zero_lag = 0.0, 0.0, 0.0
    else:
        n = len(x_centered)
        r = fftconvolve(x_centered, x_centered[::-1], mode="full")[n - 1:]
        r_norm = r / r[0]
        min_lag = int(0.2 * fs)
        max_lag = min(int(1.0 * fs), len(r_norm) - 1)
        cardiac_r = r_norm[min_lag:max_lag]
        ac_max_peak = float(np.max(cardiac_r)) if len(cardiac_r) > 0 else 0.0
        decay_idx = np.where(r_norm < (1.0 / np.e))[0]
        decay_time = float(decay_idx[0] / fs) if len(decay_idx) > 0 else float(max_lag / fs)
        zero_idx = np.where(r_norm < 0.0)[0]
        zero_lag = float(zero_idx[0] / fs) if len(zero_idx) > 0 else float(max_lag / fs)

    # Sample entropy (downsampled to 45 Hz)
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
        "peak_count": peak_count,
        "mean_rr": mean_rr,
        "rr_cv": rr_cv,
        "qrs_width": qrs_width,
        "ac_max_peak_ratio": ac_max_peak,
        "ac_decay_time": decay_time,
        "ac_zero_crossing_lag": zero_lag,
        "sample_entropy": samp_en,
    }


# ======================================================================
# 3-MODEL HIERARCHICAL CLASSIFIER CASCADE
# ======================================================================

class HierarchicalEdgeCascade:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print("[ML] Loading Tier 1: Shockable Classifier...")
        self.sess_m1 = ort.InferenceSession(str(M1_PATH), opts, providers=["CPUExecutionProvider"])
        self.in_m1 = self.sess_m1.get_inputs()[0].name

        print("[ML] Loading Tier 2: 4-Class Arrhythmia Classifier...")
        self.sess_m2 = ort.InferenceSession(str(M2_PATH), opts, providers=["CPUExecutionProvider"])
        self.in_m2 = self.sess_m2.get_inputs()[0].name

        print("[ML] Loading Tier 3: VT vs VF Subtype Specialist...")
        self.sess_m3 = ort.InferenceSession(str(M3_PATH), opts, providers=["CPUExecutionProvider"])
        self.in_m3 = self.sess_m3.get_inputs()[0].name
        print("[ML] All 3 candidate models successfully initialized.\n")

    def predict(self, feature_map: Dict[str, float]) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Tier 1: Shockable Screener
        v1 = np.array([[feature_map[k] for k in M1_FEATURES]], dtype=np.float32)
        out1 = self.sess_m1.run(None, {self.in_m1: v1})
        probs1 = out1[1][0] if isinstance(out1[1], (list, np.ndarray)) else out1[0][0]
        if isinstance(probs1, dict):
            p_nonshock, p_shock = float(probs1.get(0, 0.5)), float(probs1.get(1, 0.5))
        else:
            p_nonshock, p_shock = float(probs1[0]), float(probs1[1])
        is_shockable = p_shock >= 0.50
        m1_result = {
            "is_shockable": is_shockable,
            "label": "SHOCKABLE" if is_shockable else "NON_SHOCKABLE",
            "p_shockable": p_shock,
            "p_non_shockable": p_nonshock,
            "confidence": max(p_shock, p_nonshock),
        }

        # Tier 2: 4-Class Rhythm Classifier
        v2 = np.array([[feature_map[k] for k in M2_FEATURES]], dtype=np.float32)
        out2 = self.sess_m2.run(None, {self.in_m2: v2})
        probs2 = out2[1][0] if isinstance(out2[1], (list, np.ndarray)) else out2[0][0]
        if isinstance(probs2, dict):
            p2_list = [float(probs2.get(i, 0.0)) for i in range(len(M2_CLASSES))]
        else:
            p2_list = [float(p) for p in probs2]
        pred_idx2 = int(np.argmax(p2_list))
        m2_class = M2_CLASSES[pred_idx2]
        m2_result = {
            "predicted_class": m2_class,
            "confidence": float(p2_list[pred_idx2]),
            "probabilities": {M2_CLASSES[i]: float(p2_list[i]) for i in range(len(M2_CLASSES))},
        }

        # Tier 3: Ventricular Specialist (Triggered if Shockable OR Class == "VENTRICULAR")
        m3_result = None
        if is_shockable or m2_class == "VENTRICULAR":
            v3 = np.array([[feature_map[k] for k in M3_FEATURES]], dtype=np.float32)
            out3 = self.sess_m3.run(None, {self.in_m3: v3})
            probs3 = out3[1][0] if isinstance(out3[1], (list, np.ndarray)) else out3[0][0]
            if isinstance(probs3, dict):
                p_vt, p_vf = float(probs3.get(0, 0.5)), float(probs3.get(1, 0.5))
            else:
                p_vt, p_vf = float(probs3[0]), float(probs3[1])
            subtype = "VF" if p_vf >= 0.50 else "VT"
            m3_result = {
                "subtype": subtype,
                "confidence": max(p_vt, p_vf),
                "p_vt": p_vt,
                "p_vf": p_vf,
            }

        # Synthesize Final Research Classification
        if is_shockable:
            sub = m3_result["subtype"] if m3_result else "UNKNOWN"
            final_class = f"SHOCKABLE ({sub})"
            triage_level = "EMERGENCY_SHOCK_ADVISED"
            confidence = m1_result["confidence"]
        elif m2_class == "VENTRICULAR":
            sub = m3_result["subtype"] if m3_result else "VT"
            final_class = f"VENTRICULAR ({sub})"
            triage_level = "URGENT_VENTRICULAR"
            confidence = m2_result["confidence"]
        elif m2_class in ("TACHY", "BRADY_ASY"):
            final_class = m2_class
            triage_level = "MONITORING_REQUIRED"
            confidence = m2_result["confidence"]
        else:
            final_class = "NORMAL_SINUS_RHYTHM"
            triage_level = "ROUTINE"
            confidence = m2_result["confidence"]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "final_classification": final_class,
            "triage_level": triage_level,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model1": m1_result,
            "model2": m2_result,
            "model3": m3_result,
        }


# ======================================================================
# FLASK APPLICATION & SHARED STATE
# ======================================================================

app = Flask(__name__)
state_lock = threading.Lock()

# Waveform history
display_raw = deque([0.0] * DISPLAY_BUFFER_LEN, maxlen=DISPLAY_BUFFER_LEN)
display_filtered = deque([0.0] * DISPLAY_BUFFER_LEN, maxlen=DISPLAY_BUFFER_LEN)
chart_new_samples = 0

# Diagnostics & Triage state
system_state = {
    "signal_quality": "WARMING_UP",
    "signal_details": "System initializing...",
    "hardware_sps": 0.0,
    "total_samples": 0,
    "model1": {"label": "--", "confidence": 0.0, "is_shockable": False},
    "model2": {"rhythm": "--", "confidence": 0.0},
    "model3": {"subtype": "--", "confidence": 0.0, "active": False},
    "final_decision": "-- WARMING UP --",
    "triage_level": "WARMING_UP",
    "latency_ms": 0.0,
    "timestamp": "",
}


def acquisition_and_inference_worker(mock: bool = False):
    global chart_new_samples, system_state

    print("[SYSTEM] Starting acquisition & inference thread...")
    adc = create_adc(mock=mock, fs=FS)
    cascade = HierarchicalEdgeCascade()

    infer_buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    filled_samples = 0
    samples_since_infer = 0

    t_start = time.perf_counter()
    sample_count = 0
    last_rate_calc = t_start

    try:
        while True:
            t_now = time.perf_counter()
            ch1_mv, _ = adc.read_sample()
            sample_count += 1

            # Update rolling display buffer
            with state_lock:
                display_raw.append(ch1_mv)
                chart_new_samples += 1
                if chart_new_samples >= CHART_REFILTER_EVERY_N:
                    chart_new_samples = 0
                    arr = np.array(display_raw)
                    try:
                        filt = research_grade_filter(arr, fs=FS)
                        display_filtered.clear()
                        display_filtered.extend(filt.tolist())
                    except Exception:
                        pass

            # Update rolling inference buffer
            infer_buffer = np.roll(infer_buffer, -1)
            infer_buffer[-1] = ch1_mv
            filled_samples = min(filled_samples + 1, WINDOW_SAMPLES)
            samples_since_infer += 1

            # Measure actual hardware sampling rate every 2 seconds
            if t_now - last_rate_calc >= 2.0:
                dt = t_now - last_rate_calc
                current_sps = (sample_count - system_state.get("total_samples", 0)) / dt
                last_rate_calc = t_now
                with state_lock:
                    system_state["hardware_sps"] = round(current_sps, 1)
                    system_state["total_samples"] = sample_count

            # Trigger inference step every 2.5 seconds (900 samples)
            if filled_samples == WINDOW_SAMPLES and samples_since_infer >= STEP_SAMPLES:
                samples_since_infer = 0
                window_raw = infer_buffer.copy()

                # 1. Evaluate Signal Quality
                quality = evaluate_signal_quality(window_raw)

                timestamp_str = time.strftime("%H:%M:%S")

                if not quality.is_valid_for_ml:
                    with state_lock:
                        system_state["signal_quality"] = quality.status
                        system_state["signal_details"] = quality.detail
                        system_state["final_decision"] = f"BAD SIGNAL ({quality.status})"
                        system_state["triage_level"] = "BAD_SIGNAL"
                        system_state["model1"] = {"label": "NOT RUN", "confidence": 0.0, "is_shockable": False}
                        system_state["model2"] = {"rhythm": "NOT RUN", "confidence": 0.0}
                        system_state["model3"] = {"subtype": "NOT RUN", "confidence": 0.0, "active": False}
                        system_state["timestamp"] = timestamp_str
                    print(f"[{timestamp_str}] [SIGNAL DIAGNOSTIC] {quality.status}: {quality.detail} -> ML NOT RUN")
                    continue

                # 2. Filter window and extract features
                diag_window = research_grade_filter(window_raw, fs=FS)
                feats = extract_features(diag_window, window_raw, fs=FS)

                # 3. Execute 3-Model Cascade
                prediction = cascade.predict(feats)

                # 4. Update shared state
                with state_lock:
                    system_state["signal_quality"] = "GOOD"
                    system_state["signal_details"] = f"OK (PTP={quality.ptp_mv:.2f}mV, Std={quality.std_mv:.2f}mV)"
                    system_state["final_decision"] = prediction["final_classification"]
                    system_state["triage_level"] = prediction["triage_level"]
                    system_state["latency_ms"] = round(prediction["latency_ms"], 2)
                    system_state["timestamp"] = timestamp_str

                    # Model 1
                    system_state["model1"] = {
                        "label": "YES (SHOCKABLE)" if prediction["model1"]["is_shockable"] else "NO (NON-SHOCKABLE)",
                        "confidence": round(prediction["model1"]["confidence"] * 100, 1),
                        "is_shockable": prediction["model1"]["is_shockable"],
                    }

                    # Model 2
                    system_state["model2"] = {
                        "rhythm": prediction["model2"]["predicted_class"],
                        "confidence": round(prediction["model2"]["confidence"] * 100, 1),
                    }

                    # Model 3
                    if prediction["model3"]:
                        system_state["model3"] = {
                            "subtype": prediction["model3"]["subtype"],
                            "confidence": round(prediction["model3"]["confidence"] * 100, 1),
                            "active": True,
                        }
                    else:
                        system_state["model3"] = {
                            "subtype": "--",
                            "confidence": 0.0,
                            "active": False,
                        }

                print(f"[{timestamp_str}] {prediction['final_classification']:<22} | "
                      f"Triage: {prediction['triage_level']:<22} | "
                      f"M1: {system_state['model1']['label']:<18} ({system_state['model1']['confidence']}%) | "
                      f"M2: {system_state['model2']['rhythm']:<12} ({system_state['model2']['confidence']}%) | "
                      f"M3: {system_state['model3']['subtype']:<4} | Latency: {prediction['latency_ms']:.2f}ms")

    except Exception as e:
        print(f"[FATAL] Acquisition loop crashed: {e}")
    finally:
        adc.close()


@app.route("/data")
def get_data():
    with state_lock:
        r = list(display_raw)
        f = list(display_filtered) if display_filtered else r
        st = dict(system_state)
    return jsonify({"raw": r, "filtered": f, "fs": FS, "state": st})


@app.route("/")
def dashboard():
    html = """<!DOCTYPE html>
<html>
<head>
  <title>ECG Arrhythmia Edge AI — Live Dashboard</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; }
    body {
      background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, monospace;
      margin: 0; padding: 16px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid #30363d; padding-bottom: 12px; margin-bottom: 16px;
    }
    h1 { margin: 0; font-size: 20px; color: #58a6ff; }
    .badge {
      display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;
    }
    .badge-info { background: #1f6feb; color: #fff; }
    .badge-good { background: #238636; color: #fff; }
    .badge-bad { background: #da3633; color: #fff; }
    .badge-warn { background: #9e6a03; color: #fff; }

    /* Canvas Chart */
    .chart-container {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 16px;
    }
    .legend { font-size: 13px; margin-bottom: 8px; display: flex; gap: 16px; }
    .legend-raw { color: #58a6ff; font-weight: bold; }
    .legend-filt { color: #3fb950; font-weight: bold; }
    canvas { background: #010409; border: 1px solid #21262d; border-radius: 4px; display: block; width: 100%; height: auto; }

    /* Grid Layout */
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 16px; }
    .card {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px;
      display: flex; flex-direction: column; justify-content: space-between;
    }
    .card h3 { margin: 0 0 8px 0; font-size: 14px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
    .value { font-size: 20px; font-weight: bold; margin-bottom: 4px; }
    .subtext { font-size: 12px; color: #8b949e; }

    /* Triage Banner */
    .triage-banner {
      padding: 16px; border-radius: 8px; text-align: center; font-size: 22px; font-weight: bold;
      margin-bottom: 16px; border: 1px solid transparent; transition: all 0.3s;
    }
    .triage-EMERGENCY_SHOCK_ADVISED { background: #490202; color: #ff7b72; border-color: #f85149; }
    .triage-URGENT_VENTRICULAR { background: #3d1a00; color: #f0883e; border-color: #d29922; }
    .triage-MONITORING_REQUIRED { background: #2e2600; color: #e3b341; border-color: #bb8009; }
    .triage-ROUTINE { background: #0e2a18; color: #56d364; border-color: #2ea043; }
    .triage-BAD_SIGNAL { background: #21262d; color: #8b949e; border-color: #30363d; }
    .triage-WARMING_UP { background: #161b22; color: #8b949e; border-color: #30363d; }

    /* Disclaimer */
    .disclaimer {
      background: #161b22; border: 1px solid #8b949e; border-left: 4px solid #f85149;
      padding: 12px 16px; border-radius: 6px; font-size: 12px; color: #8b949e; line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>ECG Arrhythmia Edge AI — Real-Time Monitor</h1>
        <div style="font-size: 12px; color: #8b949e; margin-top: 4px;">
          Hardware: <b>ADS1292R (24-bit SPI)</b> | Channel: <b>CH1 (MLII)</b> | Sampling: <b>360 Hz</b>
        </div>
      </div>
      <div>
        <span id="rate-badge" class="badge badge-info">-- SPS</span>
        <span id="quality-badge" class="badge badge-warn">WARMING UP</span>
      </div>
    </header>

    <!-- Final Triage Banner -->
    <div id="triage-banner" class="triage-banner triage-WARMING_UP">
      <div id="triage-title">-- WARMING UP --</div>
      <div id="triage-sub" style="font-size: 13px; font-weight: normal; margin-top: 4px;">Acquiring 5-second baseline buffer...</div>
    </div>

    <!-- Live Waveform Chart -->
    <div class="chart-container">
      <div class="legend">
        <span class="legend-raw">― Raw Channel 1 (mV)</span>
        <span class="legend-filt">― Research-Grade Filtered (mV)</span>
        <span style="color: #8b949e; margin-left: auto;" id="latency-info">Inference Latency: -- ms</span>
      </div>
      <canvas id="chart" width="1160" height="420"></canvas>
    </div>

    <!-- 3-Model Cascade Cards -->
    <div class="grid">
      <!-- Model 1 -->
      <div class="card">
        <h3>Model 1: Shockable Screener</h3>
        <div id="m1-val" class="value">--</div>
        <div id="m1-conf" class="subtext">Confidence: --%</div>
        <div style="font-size: 11px; color: #8b949e; margin-top: 8px;">10 QRS-independent chaos & spectral features</div>
      </div>

      <!-- Model 2 -->
      <div class="card">
        <h3>Model 2: 4-Class Rhythm</h3>
        <div id="m2-val" class="value">--</div>
        <div id="m2-conf" class="subtext">Confidence: --%</div>
        <div style="font-size: 11px; color: #8b949e; margin-top: 8px;">NSR, Tachy, Brady/Asystole, Ventricular</div>
      </div>

      <!-- Model 3 -->
      <div class="card">
        <h3>Model 3: VT vs VF Subtype</h3>
        <div id="m3-val" class="value">--</div>
        <div id="m3-conf" class="subtext">Status: Standby</div>
        <div style="font-size: 11px; color: #8b949e; margin-top: 8px;">6 Domain-Stable Complexity Features</div>
      </div>

      <!-- Signal Quality Metrics -->
      <div class="card">
        <h3>Signal Quality Diagnostic</h3>
        <div id="sq-val" class="value" style="font-size: 18px;">WARMING UP</div>
        <div id="sq-detail" class="subtext">Buffering...</div>
        <div style="font-size: 11px; color: #8b949e; margin-top: 8px;">Continuous flatline & rail protection</div>
      </div>
    </div>

    <!-- Safety Disclaimer -->
    <div class="disclaimer">
      <b>RESEARCH AND EDUCATIONAL PROTOTYPE ONLY:</b> This system is not a certified medical device and is not approved for clinical diagnostic use, autonomous defibrillation decisions, or automated therapy delivery. All classifications are software outputs for algorithmic research and edge computing demonstration.
    </div>
  </div>

  <script>
    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');

    function drawTrace(values, color, yOffset, yScale) {
      if (!values || values.length === 0) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const w = canvas.width;
      const step = w / values.length;
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      for (let i = 0; i < values.length; i++) {
        const x = i * step;
        const y = yOffset - (values[i] - mean) * yScale;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    async function poll() {
      try {
        const res = await fetch('/data');
        const d = await res.json();
        const st = d.state;

        // Redraw canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawTrace(d.raw, '#58a6ff', 120, 50);
        drawTrace(d.filtered, '#3fb950', 300, 50);

        // Hardware rate
        document.getElementById('rate-badge').textContent = `${st.hardware_sps} SPS`;

        // Signal quality badge
        const qb = document.getElementById('quality-badge');
        qb.textContent = st.signal_quality;
        qb.className = `badge ${st.signal_quality === 'GOOD' ? 'badge-good' : st.signal_quality.startsWith('BAD') ? 'badge-bad' : 'badge-warn'}`;

        // Triage banner
        const tb = document.getElementById('triage-banner');
        tb.className = `triage-banner triage-${st.triage_level}`;
        document.getElementById('triage-title').textContent = st.final_decision;
        document.getElementById('triage-sub').textContent = `Triage Level: ${st.triage_level} | Updated: ${st.timestamp || '--'}`;

        // Model 1
        const m1 = st.model1;
        document.getElementById('m1-val').textContent = m1.label;
        document.getElementById('m1-val').style.color = m1.is_shockable ? '#ff7b72' : '#3fb950';
        document.getElementById('m1-conf').textContent = `Confidence: ${m1.confidence}%`;

        // Model 2
        const m2 = st.model2;
        document.getElementById('m2-val').textContent = m2.rhythm;
        document.getElementById('m2-conf').textContent = `Confidence: ${m2.confidence}%`;

        // Model 3
        const m3 = st.model3;
        if (m3.active) {
          document.getElementById('m3-val').textContent = m3.subtype;
          document.getElementById('m3-val').style.color = m3.subtype === 'VF' ? '#ff7b72' : '#f0883e';
          document.getElementById('m3-conf').textContent = `Confidence: ${m3.confidence}% (Active)`;
        } else {
          document.getElementById('m3-val').textContent = '--';
          document.getElementById('m3-val').style.color = '#8b949e';
          document.getElementById('m3-conf').textContent = 'Standby (Non-Ventricular)';
        }

        // Diagnostic box
        document.getElementById('sq-val').textContent = st.signal_quality;
        document.getElementById('sq-detail').textContent = st.signal_details;
        document.getElementById('latency-info').textContent = `Inference Latency: ${st.latency_ms} ms`;

      } catch (err) {
        console.error(err);
      }
      setTimeout(poll, 120);
    }
    poll();
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified ECG Arrhythmia Edge AI Monitor")
    parser.add_argument("--mock", action="store_true", help="Run in offline simulation mode without physical SPI")
    parser.add_argument("--port", type=int, default=5000, help="Web dashboard port (default: 5000)")
    args = parser.parse_args()

    # Launch background acquisition thread
    worker_t = threading.Thread(target=acquisition_and_inference_worker, kwargs={"mock": args.mock}, daemon=True)
    worker_t.start()

    time.sleep(1.0)
    print(f"\n===========================================================================")
    print(f"  ECG ARRHYTHMIA EDGE AI DASHBOARD READY")
    print(f"  Open in browser: http://0.0.0.0:{args.port}")
    print(f"  (Or from your laptop: http://<raspberry-pi-ip>:{args.port})")
    print(f"===========================================================================\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
