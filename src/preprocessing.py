"""Signal validation and Pan-Tompkins QRS detection for the 10-feature pipeline."""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

TARGET_FS = 360
WINDOW_SECONDS = 5
WINDOW_SAMPLES = TARGET_FS * WINDOW_SECONDS


def validate_window(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Return one finite 5-second ECG window as a float64 array."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    expected = fs * WINDOW_SECONDS
    if signal.size != expected:
        raise ValueError(f"Expected {expected} samples at {fs} Hz, got {signal.size}")
    if not np.isfinite(signal).all():
        raise ValueError("ECG window contains NaN or infinite values")
    return signal


def pan_tompkins_detect(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Detect QRS peaks using the notebook's final Pan-Tompkins pipeline."""
    signal = validate_window(signal, fs)
    nyquist = fs / 2
    coefficients_b, coefficients_a = butter(
        1, [5 / nyquist, 15 / nyquist], btype="band"
    )
    filtered = filtfilt(coefficients_b, coefficients_a, signal)
    squared = np.diff(filtered) ** 2
    integration_size = int(0.15 * fs)
    integrated = np.convolve(
        squared, np.ones(integration_size) / integration_size, mode="same"
    )

    minimum_distance = int(0.3 * fs)
    threshold = 0.3 * np.max(integrated) if np.max(integrated) > 0 else 0
    peaks, _ = find_peaks(
        integrated, distance=minimum_distance, height=threshold
    )

    corrected_peaks = []
    search_window = int(0.075 * fs)
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(signal.size, peak + search_window)
        if end > start:
            corrected_peaks.append(start + int(np.argmax(signal[start:end])))
    return np.asarray(corrected_peaks, dtype=np.int64)