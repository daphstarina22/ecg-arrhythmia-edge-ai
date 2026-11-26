"""Signal-quality gate. Rejects windows that must not reach the classifiers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import welch

from . import config as C


@dataclass
class QualityReport:
    ok: bool
    status: str                 # SIGNAL_OK | BAD_NONFINITE | BAD_SATURATED | BAD_FLAT | BAD_ARTIFACT
    detail: str
    warnings: list = field(default_factory=list)
    ptp_mv: float = 0.0
    std_mv: float = 0.0


def evaluate(raw_mv: np.ndarray, conditioned: np.ndarray | None = None) -> QualityReport:
    raw = np.asarray(raw_mv, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        return QualityReport(False, "BAD_NONFINITE", "NaN/Inf samples in capture")

    rail = float(np.mean(np.abs(raw) >= C.ADC_RAIL_MV))
    if rail > C.RAIL_FRAC_MAX:
        return QualityReport(False, "BAD_SATURATED",
                             f"ADC at rail on {rail*100:.0f}% of samples - electrodes off / cable?")

    ac = raw - np.median(raw)
    std, ptp = float(np.std(ac)), float(np.ptp(ac))
    if std < C.FLAT_STD_MV or ptp < C.FLAT_PTP_MV:
        return QualityReport(False, "BAD_FLAT", "flat line - electrodes not connected?", ptp_mv=ptp, std_mv=std)

    warnings = []
    f, p = welch(ac, fs=C.ADC_SPS, nperseg=min(len(ac), 512))
    ecg_p = p[(f >= 0.5) & (f <= 40)].sum() + 1e-12
    mains_p = p[(f >= C.MAINS_HZ - 2) & (f <= C.MAINS_HZ + 2)].sum()
    if mains_p / ecg_p > C.MAINS_RATIO_WARN:
        warnings.append(f"strong 50 Hz pickup (x{mains_p/ecg_p:.1f} of ECG band) - check electrode contact")

    if conditioned is not None:
        cptp = float(np.ptp(conditioned))
        if cptp > C.MAX_PLAUSIBLE_PTP_MV:
            return QualityReport(False, "BAD_ARTIFACT",
                                 f"{cptp:.1f} mV peak-to-peak after filtering - motion/contact artifact",
                                 warnings, ptp_mv=cptp, std_mv=std)
        ptp = cptp
    return QualityReport(True, "SIGNAL_OK", "signal acceptable", warnings, ptp_mv=ptp, std_mv=std)
