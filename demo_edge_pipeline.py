"""Hierarchical 3-Model ECG Arrhythmia Edge AI Cascade Demonstration.

This script implements the recommended 3-tier hierarchical edge triage architecture:
  - Tier 1: Shockable Rhythm Screener (Model 1 - 10 features)
  - Tier 2: General Arrhythmia 4-Class Classifier (Model 2 - 12 features)
  - Tier 3: VT vs. VF/VFL Subtype Specialist (Model 3 - 6 domain-stable features)
  
Includes clinical uncertainty flagging, artifact safeguards, latency profiling,
and automated demonstration on verified clinical recordings from MIT-BIH, VFDB, and C2015.
"""

import argparse
import io
import math
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, fftconvolve, resample_poly, welch
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent
EXPORT_DIR = PROJECT_ROOT / "training" / "final_models" / "export_candidates"
TARGET_FS = 360
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)

# Feature contracts
MODEL1_FEATURES = [
    "vf_band_power_ratio", "dominant_freq", "spectral_entropy", "spectral_peak_purity",
    "zero_crossings", "hjorth_mobility", "hjorth_complexity", "lz_complexity",
    "calibrated_ptp", "calibrated_std",
]

MODEL2_FEATURES = [
    "mean_rr", "rr_cv", "qrs_width", "peak_count", "dominant_freq",
    "vf_band_power_ratio", "spectral_entropy", "zero_crossings", "hjorth_mobility",
    "hjorth_complexity", "calibrated_ptp", "calibrated_std",
]

MODEL3_FEATURES = [
    "lz_complexity", "ac_decay_time", "ac_zero_crossing_lag",
    "ac_max_peak_ratio", "sample_entropy", "hjorth_mobility",
]

MODEL2_CLASSES = ["NSR", "TACHY", "BRADY_ASY", "VENTRICULAR"]
MODEL3_CLASSES = ["VT", "VF_VFL"]


def butter_bandpass(sig: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: int = TARGET_FS) -> np.ndarray:
    """Zero-phase 2nd order Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(2, [max(0.001, lowcut / nyq), min(0.999, highcut / nyq)], btype="band")
    return filtfilt(b, a, sig)


def extract_cascade_features(sig: np.ndarray, fs: int = TARGET_FS) -> Dict[str, float]:
    """Extract the unified feature dictionary for all 3 models in a single optimized pass."""
    sig_filt = butter_bandpass(sig, fs=fs)
    ptp_val = float(np.ptp(sig_filt))
    std_val = float(np.std(sig_filt))
    zero_cross = int(np.sum(np.diff(sig_filt > 0) != 0))

    # Spectral analysis
    nperseg = min(len(sig_filt), int(fs * 2))
    freqs, psd = welch(sig_filt, fs=fs, nperseg=nperseg)
    valid_mask = (freqs >= 0.5) & (freqs <= 30.0)
    freqs_band = freqs[valid_mask]
    psd_band = psd[valid_mask]

    total_power = float(np.sum(psd_band)) + 1e-12
    vf_mask = (freqs_band >= 3.0) & (freqs_band <= 9.0)
    vf_power = float(np.sum(psd_band[vf_mask]))
    vf_ratio = float(vf_power / total_power)

    dom_idx = int(np.argmax(psd_band))
    dom_freq = float(freqs_band[dom_idx])

    psd_norm = psd_band / total_power
    psd_norm = psd_norm[psd_norm > 0]
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm) + 1e-12))
    spec_purity = float(psd_band[dom_idx] / (np.median(psd_band) + 1e-12))

    # Hjorth parameters
    diff1 = np.diff(sig_filt)
    diff2 = np.diff(diff1)
    var_zero = std_val**2 + 1e-12
    var_d1 = float(np.var(diff1)) + 1e-12
    var_d2 = float(np.var(diff2)) + 1e-12
    mobility = float(np.sqrt(var_d1 / var_zero))
    complexity = float(np.sqrt(var_d2 / var_d1) / (mobility + 1e-12))

    # Lempel-Ziv complexity (Strategy B: downsampled to 90 Hz, N=450)
    step = max(1, int(round(fs / 90.0)))
    sig_down = sig_filt[::step]
    binary_seq = (sig_down > np.median(sig_down)).astype(np.int8)
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

    # Pan-Tompkins peak detection
    diff_sq = diff1**2
    int_len = int(fs * 0.150)
    integrated = np.convolve(diff_sq, np.ones(int_len) / int_len, mode="same")
    threshold = 0.3 * np.max(integrated) if np.max(integrated) > 0 else 1.0
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
    else:
        mean_rr = 5.0
        rr_cv = 0.0
        qrs_width = 0.0

    # Autocorrelation features (Domain-Stable Model 3)
    x_centered = sig_filt - np.mean(sig_filt)
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
    x_down = sig_filt[::8].astype(np.float32)
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


@dataclass
class TriageDecision:
    triage_level: str          # CRITICAL_EMERGENCY, URGENT_MONITORING, ROUTINE_NORMAL
    primary_rhythm: str        # Formatted clinical name
    is_shockable: bool         # Shock advice flag
    confidence: float          # Confidence in primary diagnosis
    defibrillator_advised: bool
    recommended_action: str

    tier1_result: Dict[str, Any]
    tier2_result: Dict[str, Any]
    tier3_result: Optional[Dict[str, Any]]
    
    latencies_ms: Dict[str, float]
    safety_flags: List[str]


class HierarchicalEdgeClassifier:
    """3-tier edge inference cascade with domain-shift guardrails."""

    def __init__(self, export_dir: Path = EXPORT_DIR):
        self.export_dir = Path(export_dir)
        m1_path = self.export_dir / "shockable_classifier_candidate.onnx"
        m2_path = self.export_dir / "arrhythmia_multiclass_4class_candidate.onnx"
        m3_path = self.export_dir / "vf_vt_subtype_classifier_candidate.onnx"

        for p in [m1_path, m2_path, m3_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing candidate model: {p}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess_m1 = ort.InferenceSession(str(m1_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess_m2 = ort.InferenceSession(str(m2_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess_m3 = ort.InferenceSession(str(m3_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def predict_window(self, sig: np.ndarray, fs: int = TARGET_FS) -> TriageDecision:
        latencies = {}
        t_start = time.perf_counter()

        # Step 1: Feature Extraction
        t0 = time.perf_counter()
        feats = extract_cascade_features(sig, fs=fs)
        latencies["feature_extraction"] = (time.perf_counter() - t0) * 1000.0

        # Assemble feature vectors
        vec_m1 = np.array([[feats[f] for f in MODEL1_FEATURES]], dtype=np.float32)
        vec_m2 = np.array([[feats[f] for f in MODEL2_FEATURES]], dtype=np.float32)
        vec_m3 = np.array([[feats[f] for f in MODEL3_FEATURES]], dtype=np.float32)

        # ------------------------------------------------------------------
        # Tier 1: Shockable Screener
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        out_m1 = self.sess_m1.run(None, {self.sess_m1.get_inputs()[0].name: vec_m1})
        latencies["tier1_shockable"] = (time.perf_counter() - t0) * 1000.0

        label_m1 = int(out_m1[0][0])
        prob_dict_m1 = out_m1[1][0] if isinstance(out_m1[1], list) else out_m1[1]
        p_shockable = float(prob_dict_m1.get(1, 0.0))
        is_shockable = bool(label_m1 == 1 or p_shockable >= 0.50)

        t1_diag = {
            "label": "SHOCKABLE" if is_shockable else "NON_SHOCKABLE",
            "prob_shockable": p_shockable,
            "prob_non_shockable": 1.0 - p_shockable,
        }

        # ------------------------------------------------------------------
        # Tier 2: Multi-Arrhythmia 4-Class Classifier
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        out_m2 = self.sess_m2.run(None, {self.sess_m2.get_inputs()[0].name: vec_m2})
        latencies["tier2_multiclass"] = (time.perf_counter() - t0) * 1000.0

        label_idx_m2 = int(out_m2[0][0])
        pred_class_m2 = MODEL2_CLASSES[label_idx_m2]
        prob_dict_m2 = out_m2[1][0] if isinstance(out_m2[1], list) else out_m2[1]
        probs_m2 = {MODEL2_CLASSES[i]: float(prob_dict_m2.get(i, 0.0)) for i in range(len(MODEL2_CLASSES))}

        t2_diag = {
            "predicted_class": pred_class_m2,
            "class_probabilities": probs_m2,
            "confidence": probs_m2[pred_class_m2],
        }

        # ------------------------------------------------------------------
        # Tier 3: VT vs. VF Subtype Specialist (Triggered if Shockable or Ventricular)
        # ------------------------------------------------------------------
        t3_diag = None
        safety_flags = []
        is_ventricular = (is_shockable or pred_class_m2 == "VENTRICULAR")

        if is_ventricular:
            t0 = time.perf_counter()
            out_m3 = self.sess_m3.run(None, {self.sess_m3.get_inputs()[0].name: vec_m3})
            latencies["tier3_subtype"] = (time.perf_counter() - t0) * 1000.0

            label_idx_m3 = int(out_m3[0][0])
            pred_class_m3 = MODEL3_CLASSES[label_idx_m3]
            prob_dict_m3 = out_m3[1][0] if isinstance(out_m3[1], list) else out_m3[1]
            p_vf = float(prob_dict_m3.get(1, 0.0))
            p_vt = float(prob_dict_m3.get(0, 0.0))

            # Domain-Shift & Equivocal Subtype Guardrails
            if 0.40 <= p_vf <= 0.60:
                safety_flags.append("EQUIVOCAL_VENTRICULAR_SUBTYPE: Borderline morphological complexity between rapid VT and fine VF.")
            
            # Anomaly / high-frequency artifact safeguard
            if feats["lz_complexity"] > 0.85 and feats["zero_crossings"] > 75:
                safety_flags.append("HIGH_COMPLEXITY_WARNING: Elevated high-frequency crossings detected; check lead impedance.")

            t3_diag = {
                "subtype": pred_class_m3,
                "prob_vt": p_vt,
                "prob_vf_vfl": p_vf,
                "lz_complexity": feats["lz_complexity"],
                "sample_entropy": feats["sample_entropy"],
                "ac_decay_time": feats["ac_decay_time"],
            }
        else:
            latencies["tier3_subtype"] = 0.0

        latencies["total_cycle"] = (time.perf_counter() - t_start) * 1000.0

        # ------------------------------------------------------------------
        # Integrated Clinical Triage Synthesis
        # ------------------------------------------------------------------
        if is_ventricular and t3_diag is not None:
            if t3_diag["subtype"] == "VF_VFL":
                triage_level = "CRITICAL_EMERGENCY"
                primary_rhythm = "Ventricular Fibrillation / Flutter (VF/VFL)"
                defib_advised = True
                action = "CRITICAL: Immediate Shock Recommended! Initiate CPR and charge defibrillator."
                conf = t3_diag["prob_vf_vfl"]
            else:
                triage_level = "CRITICAL_EMERGENCY" if is_shockable else "URGENT_MONITORING"
                primary_rhythm = "Ventricular Tachycardia (VT)"
                defib_advised = is_shockable
                action = "CRITICAL: Sustained Ventricular Tachycardia! Prepare synchronized cardioversion / evaluate hemodynamics." if is_shockable else "URGENT: Ventricular rhythm detected. Notify cardiology."
                conf = t3_diag["prob_vt"]
        elif pred_class_m2 == "TACHY":
            triage_level = "URGENT_MONITORING"
            primary_rhythm = "Supraventricular Tachycardia (SVTA / Rapid Rhythm)"
            defib_advised = False
            action = "URGENT: Rapid narrow-complex rhythm detected. Assess heart rate and patient symptoms."
            conf = probs_m2["TACHY"]
        elif pred_class_m2 == "BRADY_ASY":
            is_asy = (feats["calibrated_std"] < 0.05)
            triage_level = "CRITICAL_EMERGENCY" if is_asy else "URGENT_MONITORING"
            primary_rhythm = "Asystole / Severe Bradycardia" if is_asy else "Bradycardia / Conduction Pause"
            defib_advised = False
            action = "CRITICAL: Possible Asystole / No pulse! Check responsiveness and start CPR if pulseless." if is_asy else "URGENT: Marked bradycardia detected. Check vitals."
            conf = probs_m2["BRADY_ASY"]
        else:
            triage_level = "ROUTINE_NORMAL"
            primary_rhythm = "Normal Sinus Rhythm (NSR)"
            defib_advised = False
            action = "STABLE: Rhythm within normal physiological parameters. Continue routine monitoring."
            conf = probs_m2["NSR"]

        return TriageDecision(
            triage_level=triage_level,
            primary_rhythm=primary_rhythm,
            is_shockable=defib_advised,
            confidence=conf,
            defibrillator_advised=defib_advised,
            recommended_action=action,
            tier1_result=t1_diag,
            tier2_result=t2_diag,
            tier3_result=t3_diag,
            latencies_ms=latencies,
            safety_flags=safety_flags,
        )


def render_dashboard(record_id: str, source: str, true_label: str, dec: TriageDecision):
    """Render rich terminal dashboard for the decision cascade."""
    color_map = {
        "CRITICAL_EMERGENCY": "\033[91m",  # Red
        "URGENT_MONITORING":  "\033[93m",  # Yellow
        "ROUTINE_NORMAL":    "\033[92m",  # Green
    }
    reset = "\033[0m"
    bold = "\033[1m"
    c = color_map.get(dec.triage_level, "")

    print(f"\n{'='*84}")
    print(f"{bold}PATIENT RECORD: {record_id:<14} | SOURCE: {source:<12} | TRUE: {true_label}{reset}")
    print(f"{'='*84}")

    # Step-by-Step Cascade Execution
    print(f"{bold}--- HIERARCHICAL 3-TIER CASCADE EXECUTION ---{reset}")
    
    # Tier 1
    t1 = dec.tier1_result
    t1_bar = "#" * int(t1["prob_shockable"] * 20) + "-" * (20 - int(t1["prob_shockable"] * 20))
    print(f"  [Tier 1: Shockable Screener]  Result: {bold}{t1['label']:<14}{reset} | P(Shock): {t1['prob_shockable']:.1%} [{t1_bar}] ({dec.latencies_ms['tier1_shockable']:.2f} ms)")

    # Tier 2
    t2 = dec.tier2_result
    p_str = ", ".join([f"{k}: {v:.1%}" for k, v in t2["class_probabilities"].items()])
    print(f"  [Tier 2: Arrhythmia 4-Class]  Result: {bold}{t2['predicted_class']:<14}{reset} | Confidence: {t2['confidence']:.1%} ({dec.latencies_ms['tier2_multiclass']:.2f} ms)")
    print(f"                                 Distribution: [{p_str}]")

    # Tier 3
    if dec.tier3_result:
        t3 = dec.tier3_result
        t3_bar = "#" * int(t3["prob_vf_vfl"] * 20) + "-" * (20 - int(t3["prob_vf_vfl"] * 20))
        print(f"  [Tier 3: Ventricular Subtype] Result: {bold}{t3['subtype']:<14}{reset} | P(VF): {t3['prob_vf_vfl']:.1%} [{t3_bar}] ({dec.latencies_ms['tier3_subtype']:.2f} ms)")
        print(f"                                 Biomarkers: LZ Complexity={t3['lz_complexity']:.3f}, Sample Entropy={t3['sample_entropy']:.3f}, AC Decay={t3['ac_decay_time']:.3f}s")
    else:
        print(f"  [Tier 3: Ventricular Subtype] Bypassed (Rhythm classified as non-ventricular by Tier 1 & 2)")

    # Final Clinical Synthesis
    print(f"\n{bold}--- INTEGRATED CLINICAL TRIAGE SYNTHESIS ---{reset}")
    print(f"  Triage Status       : {c}{bold}{dec.triage_level}{reset}")
    print(f"  Primary Diagnosis   : {bold}{dec.primary_rhythm}{reset}")
    print(f"  Diagnostic Certainty: {dec.confidence:.1%}")
    print(f"  Shock Advised (AED) : {bold}{'YES (SHOCKABLE)' if dec.defibrillator_advised else 'NO (NON-SHOCKABLE)'}{reset}")
    print(f"  Recommended Action  : {c}{dec.recommended_action}{reset}")

    if dec.safety_flags:
        print(f"\n{bold}--- SAFETY & DOMAIN-SHIFT WARNINGS ---{reset}")
        for flag in dec.safety_flags:
            print(f"  ! {flag}")

    # Latency Breakdown
    l = dec.latencies_ms
    print(f"\n{bold}--- EDGE RUNTIME PROFILE ---{reset}")
    print(f"  Features: {l['feature_extraction']:.2f} ms | T1: {l['tier1_shockable']:.2f} ms | T2: {l['tier2_multiclass']:.2f} ms | T3: {l['tier3_subtype']:.2f} ms | Total: {l['total_cycle']:.2f} ms")
    print(f"{'='*84}")


def load_clinical_recordings() -> List[Tuple[str, str, str, np.ndarray]]:
    """Extract verified sample event recordings across MIT-BIH, VFDB, and Challenge 2015."""
    samples = []

    # 1. MIT-BIH Record 100 (Verified Normal Sinus Rhythm)
    try:
        with zipfile.ZipFile(PROJECT_ROOT / "MITBIH.zip") as z:
            with z.open("100.csv") as f:
                df = pd.read_csv(f)
            col = [c for c in df.columns if "MLII" in c][0]
            sig = ((df[col].values[:WINDOW_SAMPLES] - 1024.0) / 200.0).astype(np.float32)
            samples.append(("mitbih_100", "MIT-BIH", "Normal Sinus Rhythm", sig))
    except Exception as e:
        print(f"Could not load MIT-BIH 100: {e}")

    # 2. MIT-BIH Record 205 (Verified Sustained Ventricular Tachycardia)
    try:
        with zipfile.ZipFile(PROJECT_ROOT / "MITBIH.zip") as z:
            with z.open("205.csv") as f:
                df = pd.read_csv(f)
            col = [c for c in df.columns if "MLII" in c][0]
            # Event interval around sample 107,561
            sig = ((df[col].values[107561:107561 + WINDOW_SAMPLES] - 1024.0) / 200.0).astype(np.float32)
            samples.append(("mitbih_205", "MIT-BIH", "Ventricular Tachycardia", sig))
    except Exception as e:
        print(f"Could not load MIT-BIH 205: {e}")

    # 3. MIT-BIH Record 207 (Verified Ventricular Flutter)
    try:
        with zipfile.ZipFile(PROJECT_ROOT / "MITBIH.zip") as z:
            with z.open("207.csv") as f:
                df = pd.read_csv(f)
            col = [c for c in df.columns if "MLII" in c][0]
            # Event interval at sample 14,700
            sig = ((df[col].values[14700:14700 + WINDOW_SAMPLES] - 1024.0) / 200.0).astype(np.float32)
            samples.append(("mitbih_207", "MIT-BIH", "Ventricular Flutter", sig))
    except Exception as e:
        print(f"Could not load MIT-BIH 207: {e}")

    # 4. VFDB Record 422 (Verified Ventricular Fibrillation)
    try:
        import wfdb
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(PROJECT_ROOT / "VFDB.zip") as z:
                z.extractall(tmpdir)
            # Event interval at sample 383,788
            rec_obj = wfdb.rdrecord(os.path.join(tmpdir, "422"), sampfrom=383788, sampto=383788 + 1250)
            sig_raw = rec_obj.p_signal[:, 0].astype(np.float32)
            sig_360 = resample_poly(sig_raw, TARGET_FS, int(rec_obj.fs)).astype(np.float32)[:WINDOW_SAMPLES]
            samples.append(("vfdb_422", "VFDB", "Ventricular Fibrillation", sig_360))
    except Exception as e:
        print(f"Could not load VFDB 422: {e}")

    # 5. Challenge 2015 b124s (Verified ICU Bradycardia True Alarm)
    try:
        with zipfile.ZipFile(PROJECT_ROOT / "mitbih-vtvtf.zip") as z_out:
            info = next(i for i in z_out.infolist() if i.filename.endswith("training.zip"))
            with zipfile.ZipFile(io.BytesIO(z_out.read(info))) as z_in:
                mat = loadmat(io.BytesIO(z_in.read("training/b124s.mat")))["val"]
                sig_raw = mat[0, -1250:].astype(np.float32) / 200.0
                sig_360 = resample_poly(sig_raw, TARGET_FS, 250).astype(np.float32)[:WINDOW_SAMPLES]
                samples.append(("c2015_b124s", "C2015 (ICU)", "Bradycardia", sig_360))
    except Exception as e:
        print(f"Could not load C2015 b124s: {e}")

    # 6. Challenge 2015 t106s (Verified ICU Supraventricular Tachycardia True Alarm)
    try:
        with zipfile.ZipFile(PROJECT_ROOT / "mitbih-vtvtf.zip") as z_out:
            info = next(i for i in z_out.infolist() if i.filename.endswith("training.zip"))
            with zipfile.ZipFile(io.BytesIO(z_out.read(info))) as z_in:
                mat = loadmat(io.BytesIO(z_in.read("training/t106s.mat")))["val"]
                sig_raw = mat[0, -1250:].astype(np.float32) / 200.0
                sig_360 = resample_poly(sig_raw, TARGET_FS, 250).astype(np.float32)[:WINDOW_SAMPLES]
                samples.append(("c2015_t106s", "C2015 (ICU)", "Supraventricular Tachycardia", sig_360))
    except Exception as e:
        print(f"Could not load C2015 t106s: {e}")

    return samples


def run_demonstration():
    print("\n" + "="*84)
    print("      3-MODEL HIERARCHICAL ARRHYTHMIA EDGE AI CASCADE CLINICAL DEMO")
    print("="*84)

    classifier = HierarchicalEdgeClassifier()
    clinical_samples = load_clinical_recordings()

    summary_records = []

    for rec_id, source, true_rhythm, sig in clinical_samples:
        decision = classifier.predict_window(sig, fs=TARGET_FS)
        render_dashboard(rec_id, source, true_rhythm, decision)

        summary_records.append({
            "Record": rec_id,
            "Source": source,
            "True Rhythm": true_rhythm,
            "Predicted Rhythm": decision.primary_rhythm,
            "Triage Level": decision.triage_level,
            "Shock Advised": "YES" if decision.defibrillator_advised else "NO",
            "Latency": f"{decision.latencies_ms['total_cycle']:.2f} ms",
        })

    # Summary Table
    print("\n" + "="*84)
    print("            CASCADE CLINICAL DEMONSTRATION PERFORMANCE SUMMARY")
    print("="*84)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))
    print("="*84 + "\n")


def run_benchmark(n_iterations: int = 200):
    print(f"\nRunning edge throughput and latency benchmark ({n_iterations} iterations)...\n")
    classifier = HierarchicalEdgeClassifier()
    dummy_sig = np.random.randn(WINDOW_SAMPLES).astype(np.float32)

    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = classifier.predict_window(dummy_sig, fs=TARGET_FS)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    print("Benchmark Results:")
    print(f"  Mean Latency       : {np.mean(times):.2f} ms")
    print(f"  Median Latency     : {np.median(times):.2f} ms")
    print(f"  95th Percentile    : {np.percentile(times, 95):.2f} ms")
    print(f"  Throughput         : {1000.0 / np.mean(times):.1f} windows/sec ({1000.0 / np.mean(times) * 5.0:.1f}x real-time speedup)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-Model ECG Arrhythmia Edge AI Cascade Demo")
    parser.add_argument("--benchmark", action="store_true", help="Run edge throughput and latency profiling")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    else:
        run_demonstration()
