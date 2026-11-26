"""Raw 500 SPS window -> 360 Hz conditioned window that matches model training.

Training (training/final_models/artifacts/build_multisource_datasets.py):
signals were resampled to 360 Hz, then filtered with a 2nd-order Butterworth
0.5-40 Hz zero-phase band-pass, then cut into 5 s windows. We reproduce that,
and add a 50 Hz notch because mains pickup on a bare ADS1292R front end is
not attenuated enough by the 40 Hz roll-off alone.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

from . import config as C


def resample_to_model_rate(raw_mv: np.ndarray) -> np.ndarray:
    """500 Hz -> 360 Hz (polyphase 18/25). 2500 samples in -> 1800 out.

    The median is removed first and padtype='line' is used: the electrode DC
    offset (tens of mV) would otherwise create a step at both window edges
    (zero-padding) that rings through the anti-alias FIR for ~0.5 s.
    """
    x = np.asarray(raw_mv, dtype=np.float64)
    x = x - np.median(x)
    y = resample_poly(x, C.RESAMPLE_UP, C.RESAMPLE_DOWN, padtype="line")
    if len(y) != C.MODEL_SAMPLES:
        raise ValueError(f"resampled length {len(y)} != {C.MODEL_SAMPLES}")
    return y


def condition(x360: np.ndarray) -> np.ndarray:
    """50 Hz notch + 0.5-40 Hz order-2 zero-phase band-pass (training filter)."""
    b, a = iirnotch(C.MAINS_HZ, C.NOTCH_Q, fs=C.MODEL_FS)
    y = filtfilt(b, a, x360)
    nyq = 0.5 * C.MODEL_FS
    b, a = butter(C.BAND_ORDER, [C.BAND_HZ[0] / nyq, C.BAND_HZ[1] / nyq], btype="band")
    return filtfilt(b, a, y)


def preprocess(raw_mv: np.ndarray) -> np.ndarray:
    return condition(resample_to_model_rate(raw_mv))
