import numpy as np

from ecg_edge import config as C
from ecg_edge import preprocessing as P
from ecg_edge.features import features_m1_m2
from ecg_edge.simulator import synth_ecg


def test_lengths_500_to_360():
    assert C.RAW_SAMPLES == 2500 and C.MODEL_SAMPLES == 1800
    assert len(P.resample_to_model_rate(np.zeros(2500))) == 1800


def test_frequency_is_preserved_not_rescaled():
    """10 Hz at 500 SPS must be 10 Hz at 360 Hz (old code read 500 SPS as 360 Hz -> 13.9 Hz)."""
    t = np.arange(2500) / 500
    y = P.resample_to_model_rate(np.sin(2 * np.pi * 10 * t))
    f = np.fft.rfftfreq(1800, 1 / 360)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(y)))] - 10) < 0.3


def test_heart_rate_survives_pipeline():
    f = features_m1_m2(P.preprocess(synth_ecg("NSR", seed=1)[:2500]))   # 72 bpm = 0.833 s
    assert 0.75 < f["mean_rr"] < 0.92


def test_dc_offset_does_not_create_edge_transient():
    x = synth_ecg("NSR", realistic_artifacts=False)[:2500]
    assert np.max(np.abs(P.resample_to_model_rate(x) - P.resample_to_model_rate(x + 200.0))) < 1e-6


def test_mains_removed():
    t = np.arange(2500) / 500
    y = P.preprocess(0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 10 * t))
    spec = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(1800, 1 / 360)
    assert spec[np.argmin(abs(f - 50))] < 0.02 * spec[np.argmin(abs(f - 10))]
