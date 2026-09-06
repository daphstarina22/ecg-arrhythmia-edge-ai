"""Signal preprocessing and physical calibration utilities for ECG Edge AI."""

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

TARGET_FS = 360
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)


def butter_bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = TARGET_FS,
    order: int = 2,
) -> np.ndarray:
    """Applies a zero-phase forward-backward Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = max(0.001, lowcut / nyquist)
    high = min(0.999, highcut / nyquist)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float = TARGET_FS) -> np.ndarray:
    """Resamples signal to target_fs using polyphase filtering."""
    if int(orig_fs) == int(target_fs):
        return signal.astype(np.float32)
    return resample_poly(signal, int(target_fs), int(orig_fs)).astype(np.float32)


def calibrate_digital_signal(
    raw_signal: np.ndarray,
    baseline: float,
    gain: float,
    adc_zero: float = 0.0,
) -> np.ndarray:
    """Converts raw digital ADC integers into physical millivolts (mV).

    Formula: physical_mV = (raw - adc_zero) / gain + baseline
    """
    if gain <= 0:
        raise ValueError(f"Invalid ADC gain: {gain}")
    return ((raw_signal.astype(np.float64) - adc_zero) / gain + baseline).astype(np.float32)


def extract_windows(
    signal: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    step_samples: int = int(WINDOW_SAMPLES // 2),
) -> list[np.ndarray]:
    """Slices a 1D continuous ECG signal into overlapping fixed-size windows."""
    windows = []
    n_samples = len(signal)
    if n_samples < window_samples:
        return windows
    for start in range(0, n_samples - window_samples + 1, step_samples):
        windows.append(signal[start : start + window_samples])
    return windows
