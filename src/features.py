"""The fixed 10-feature contract shared by all inference models."""

import numpy as np
from scipy.signal import welch

from .preprocessing import TARGET_FS, pan_tompkins_detect, validate_window

FEATURE_NAMES = [
    "mean",
    "std",
    "ptp",
    "zero_crossings",
    "peak_count",
    "mean_rr",
    "rr_cv",
    "qrs_width",
    "dominant_freq",
    "vf_band_power_ratio",
]


def estimate_qrs_width(signal: np.ndarray, peaks: np.ndarray, fs: int) -> float:
    if len(peaks) == 0:
        return 0.0
    search_samples = int(0.1 * fs)
    baseline = np.median(signal)
    widths = []
    for peak in peaks:
        start = max(0, int(peak) - search_samples)
        end = min(signal.size, int(peak) + search_samples)
        segment = signal[start:end]
        if segment.size < 3:
            continue
        relative_peak = int(peak) - start
        peak_value = signal[int(peak)] - baseline
        if abs(peak_value) < 1e-9:
            continue
        half_max = baseline + peak_value / 2
        positive = peak_value > 0
        left = relative_peak
        while left > 0 and ((segment[left] > half_max) if positive else (segment[left] < half_max)):
            left -= 1
        right = relative_peak
        while right < segment.size - 1 and ((segment[right] > half_max) if positive else (segment[right] < half_max)):
            right += 1
        if right - left > 0:
            widths.append((right - left) / fs)
    return float(np.mean(widths)) if widths else 0.0


def extract_features(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Extract features in exactly ``FEATURE_NAMES`` order as shape (1, 10)."""
    signal = validate_window(signal, fs)
    peaks = pan_tompkins_detect(signal, fs)
    if len(peaks) >= 3:
        intervals = np.diff(peaks) / fs
        mean_rr = float(np.mean(intervals))
        rr_cv = float(np.std(intervals) / (mean_rr + 1e-8))
    else:
        mean_rr = 0.0
        rr_cv = 0.0

    frequencies, power = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    vf_band = (frequencies >= 3) & (frequencies <= 9)
    features = [
        np.mean(signal),
        np.std(signal),
        np.ptp(signal),
        np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0),
        len(peaks),
        mean_rr,
        rr_cv,
        estimate_qrs_width(signal, peaks, fs),
        frequencies[np.argmax(power)],
        np.sum(power[vf_band]) / (np.sum(power) + 1e-8),
    ]
    vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
    if vector.shape != (1, 10):
        raise RuntimeError(f"Feature contract violated: got {vector.shape}")
    return vector