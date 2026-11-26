"""Feature extraction ported 1:1 from the training code.

Sources (verified against these files by tests/test_feature_parity.py):
  M1 + M2 features : training/final_models/artifacts/build_multisource_datasets.py
                     -> extract_features_single_window()
  M3 features      : training/model3_phase5/extract_physiological_features.py
                     -> compute_autocorrelation_features(), compute_complexity_features_strategy_b()

NOTE the two training code paths use DIFFERENT Lempel-Ziv implementations
(LZ78-style word parsing for M1/M2, LZ76 for M3). Each model gets the
variant it was trained with.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal
from scipy.signal import welch

FS = 360

M1_FEATURES = ["vf_band_power_ratio", "dominant_freq", "spectral_entropy", "spectral_peak_purity",
               "zero_crossings", "hjorth_mobility", "hjorth_complexity", "lz_complexity",
               "calibrated_ptp", "calibrated_std"]
M2_FEATURES = ["mean_rr", "rr_cv", "qrs_width", "peak_count", "dominant_freq",
               "vf_band_power_ratio", "spectral_entropy", "zero_crossings", "hjorth_mobility",
               "hjorth_complexity", "calibrated_ptp", "calibrated_std"]
M3_FEATURES = ["lz_complexity", "ac_decay_time", "ac_zero_crossing_lag",
               "ac_max_peak_ratio", "sample_entropy", "hjorth_mobility"]


def features_m1_m2(sig: np.ndarray, fs: int = FS) -> dict:
    sig = np.asarray(sig, dtype=np.float64)
    ptp_val = float(np.ptp(sig))
    std_val = float(np.std(sig))
    zero_cross = int(np.sum(np.diff(sig > 0) != 0))

    nperseg = min(len(sig), int(fs * 2))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    valid = (freqs >= 0.5) & (freqs <= 30.0)
    fb, pb = freqs[valid], psd[valid]
    total = np.sum(pb) + 1e-12
    vf_ratio = float(np.sum(pb[(fb >= 3.0) & (fb <= 9.0)]) / total)
    dom_idx = np.argmax(pb)
    dom_freq = float(fb[dom_idx])
    pn = pb / total
    pn = pn[pn > 0]
    spec_entropy = float(-np.sum(pn * np.log2(pn)) / np.log2(len(pn) + 1e-12))
    spec_purity = float(pb[dom_idx] / (np.median(pb) + 1e-12))

    diff1 = np.diff(sig)
    diff2 = np.diff(diff1)
    var_zero = std_val ** 2 + 1e-12
    var_d1 = float(np.var(diff1)) + 1e-12
    var_d2 = float(np.var(diff2)) + 1e-12
    mobility = float(np.sqrt(var_d1 / var_zero))
    complexity = float(np.sqrt(var_d2 / var_d1) / mobility)

    # Lempel-Ziv (word-parsing variant used for M1/M2), 90 Hz binary sequence
    step = max(1, int(round(fs / 90.0)))
    sd = sig[::step]
    binary = (sd > np.median(sd)).astype(np.int8)
    words, w, lz_count = set(), "", 0
    for b in binary:
        w += str(b)
        if w not in words:
            words.add(w)
            lz_count += 1
            w = ""
    if w:
        lz_count += 1
    n_seq = len(binary)
    lz_norm = float(lz_count * np.log2(n_seq) / n_seq) if n_seq > 0 else 0.0

    # Training peak detector (runs on the already band-passed window)
    diff_sq = diff1 ** 2
    int_len = int(fs * 0.150)
    integrated = np.convolve(diff_sq, np.ones(int_len) / int_len, mode="same")
    threshold = 0.3 * np.max(integrated) if np.max(integrated) > 0 else 1.0
    refractory = int(fs * 0.300)
    peaks, last_p = [], -refractory
    for p in range(1, len(integrated) - 1):
        if integrated[p] > integrated[p - 1] and integrated[p] > integrated[p + 1] and integrated[p] > threshold:
            if p - last_p >= refractory:
                peaks.append(p)
                last_p = p
    peak_count = len(peaks)
    if peak_count >= 2:
        rr = np.diff(peaks) / fs
        mean_rr = float(np.mean(rr))
        rr_cv = float(np.std(rr) / (mean_rr + 1e-12))
        qrs_width = 0.080
    else:
        mean_rr, rr_cv, qrs_width = 5.0, 0.0, 0.0

    return {"calibrated_ptp": ptp_val, "calibrated_std": std_val, "zero_crossings": zero_cross,
            "vf_band_power_ratio": vf_ratio, "dominant_freq": dom_freq,
            "spectral_entropy": spec_entropy, "spectral_peak_purity": spec_purity,
            "hjorth_mobility": mobility, "hjorth_complexity": complexity, "lz_complexity": lz_norm,
            "peak_count": peak_count, "mean_rr": mean_rr, "rr_cv": rr_cv, "qrs_width": qrs_width}


def _lz76(binary: bytes) -> float:
    n = len(binary)
    i, k, l, c = 0, 1, 1, 1
    while l + k <= n:
        if binary[i + k - 1] == binary[l + k - 1]:
            k += 1
        else:
            if k > 1:
                i += 1
                if i == l:
                    c += 1
                    l += k
                    i = 0
                    k = 1
                else:
                    k = 1
            else:
                c += 1
                l += 1
                i = 0
                k = 1
    b = n / (math.log2(n) if n > 1 else 1)
    return float(c / b)


def features_m3(x: np.ndarray, fs: int = FS) -> dict:
    x = np.asarray(x, dtype=np.float64)
    # --- autocorrelation (training: compute_autocorrelation_features)
    xc = x - np.mean(x)
    if float(np.var(xc)) < 1e-12:
        ac_max, decay_time, zero_lag = 0.0, 0.0, 0.0
    else:
        n = len(xc)
        r = signal.fftconvolve(xc, xc[::-1], mode="full")[n - 1:]
        rn = r / r[0]
        min_lag = int(0.2 * fs)
        max_lag = min(int(1.0 * fs), len(rn) - 1)
        cardiac = rn[min_lag:max_lag]
        ac_max = float(np.max(cardiac)) if len(cardiac) > 0 else 0.0
        di = np.where(rn < (1.0 / np.e))[0]
        decay_time = float(di[0] / fs) if len(di) > 0 else float(max_lag / fs)
        zi = np.where(rn < 0.0)[0]
        zero_lag = float(zi[0] / fs) if len(zi) > 0 else float(max_lag / fs)

    # --- complexity (training: compute_complexity_features_strategy_b)
    var_0 = float(np.var(xc))
    var_1 = float(np.var(np.diff(xc)))
    mobility = float(np.sqrt(var_1 / (var_0 + 1e-12)))
    lz = _lz76((x[::4] > np.median(x[::4])).tobytes())
    xd = x[::8].astype(np.float32)
    sd = float(np.std(xd))
    if sd > 1e-8:
        r_th = 0.2 * sd
        X2 = np.lib.stride_tricks.sliding_window_view(xd, 2)
        X3 = np.lib.stride_tricks.sliding_window_view(xd, 3)
        d2 = np.max(np.abs(X2[:, None, :] - X2[None, :, :]), axis=2)
        d3 = np.max(np.abs(X3[:, None, :] - X3[None, :, :]), axis=2)
        np.fill_diagonal(d2, np.inf)
        np.fill_diagonal(d3, np.inf)
        B, A = np.sum(d2 < r_th), np.sum(d3 < r_th)
        samp_en = float(-np.log(A / B)) if A > 0 and B > 0 else 0.0
    else:
        samp_en = 0.0
    return {"lz_complexity": lz, "ac_decay_time": decay_time, "ac_zero_crossing_lag": zero_lag,
            "ac_max_peak_ratio": ac_max, "sample_entropy": samp_en, "hjorth_mobility": mobility}
