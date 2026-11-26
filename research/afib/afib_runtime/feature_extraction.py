"""
AFib 5-Second Window Feature Extraction Runtime Module
======================================================
Target Hardware: Raspberry Pi ARM64 + ADS1292R (360 Hz)
Runtime Feature Extractor for Patient-Adaptive AFib Model (Config C)

Extracts 21 physically justified features:
  - Ventricular timing dynamics: n_beats, hr_mean_bpm, hr_std_bpm, rr_cv, rr_rmssd, rr_mad,
                                rr_pnn50, rr_sample_dispersion, rr_turning_point_ratio, alternating_sign_index
  - Fibrillatory & spectral bands: f_band_power_ratio (3-9 Hz), f_band_flatness, f_band_peak_purity,
                                  p_band_power_ratio (1-3 Hz), qrs_power_ratio (10-25 Hz)
  - Morphology & complexity: hjorth_mobility, hjorth_complexity, ac_std_mv, ac_ptp_mv,
                            zero_crossings, lz_complexity
"""

from typing import Dict, List
import numpy as np
import scipy.signal as signal

FS_TARGET = 360
WINDOW_SEC = 5.0
WINDOW_SAMPLES_360 = int(WINDOW_SEC * FS_TARGET)  # 1800


def butter_bandpass_filter(
    sig: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = FS_TARGET,
    order: int = 2
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter matching edge runtime."""
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
