"""Self-contained synthetic ECG generator used by --mock and the tests.

This is a *plumbing* simulator (it lets the HMI/pipeline run without
hardware). It is NOT evidence of clinical or model accuracy.
"""
from __future__ import annotations

import numpy as np

KINDS = ("NSR", "TACHY", "BRADY", "VF", "VT", "FLAT", "LEADOFF")

# (centre as fraction of RR before the R wave, width s, amplitude mV)
_WAVES = ((-0.20, 0.025, 0.12),   # P
          (-0.035, 0.010, -0.12),  # Q
          (0.0, 0.012, 1.20),      # R
          (0.035, 0.012, -0.25),   # S
          (0.30, 0.050, 0.30))     # T


def _beats(fs: float, dur: float, hr_bpm: float, rng, jitter=0.02) -> np.ndarray:
    n = int(round(fs * dur))
    t = np.arange(n) / fs
    x = np.zeros(n)
    rr = 60.0 / hr_bpm
    r_time = rng.uniform(0.2, 0.6) * rr
    while r_time < dur + 1.0:
        for c, w, a in _WAVES:
            centre = r_time + (c * rr if abs(c) > 0.1 else c)
            x += a * np.exp(-0.5 * ((t - centre) / w) ** 2)
        r_time += rr * (1.0 + rng.normal(0, jitter))
    return x


def synth_ecg(kind: str, fs: float = 500.0, dur: float = 10.0, seed: int = 0,
              realistic_artifacts: bool = True) -> np.ndarray:
    """Return a mV waveform of `dur` seconds at `fs` Hz."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    rng = np.random.default_rng(seed)
    n = int(round(fs * dur))
    t = np.arange(n) / fs
    if kind in ("NSR", "TACHY", "BRADY"):
        x = _beats(fs, dur, {"NSR": 72.0, "TACHY": 150.0, "BRADY": 40.0}[kind], rng)
    elif kind == "VF":
        f = 5.0 + 1.2 * np.sin(2 * np.pi * 0.3 * t) + np.cumsum(rng.normal(0, 0.02, n))
        x = 0.6 * np.sin(2 * np.pi * np.cumsum(f) / fs) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    elif kind == "VT":
        x = 1.0 * np.sin(2 * np.pi * 3.3 * t) + 0.3 * np.sin(2 * np.pi * 6.6 * t)
    elif kind == "FLAT":
        return np.full(n, 12.0) + rng.normal(0, 0.001, n)
    else:  # LEADOFF: input floating to the ADC rail
        return np.full(n, 395.0) + rng.normal(0, 0.5, n)
    x = x + rng.normal(0, 0.01, n)
    if realistic_artifacts:
        x = x + 25.0                                         # electrode DC offset
        x = x + 0.25 * np.sin(2 * np.pi * 0.25 * t)          # respiration baseline wander
        x = x + 0.08 * np.sin(2 * np.pi * 50.0 * t + 1.0)    # mains pickup
    return x
