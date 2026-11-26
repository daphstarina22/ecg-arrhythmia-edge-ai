"""One Start -> one 5 s capture -> one classification -> back to idle."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config as C
from . import decision as D
from . import preprocessing, quality
from .acquisition import AcquisitionError, Cancelled, CaptureData, Progress, capture_window
from .models import Cascade, ModelOutput


@dataclass
class CycleResult:
    decision: D.Decision
    filtered: Optional[np.ndarray] = None       # 1800 samples @ 360 Hz (what the models saw)
    fs: int = C.MODEL_FS
    m1: Optional[ModelOutput] = None
    m2: Optional[ModelOutput] = None
    m3: Optional[ModelOutput] = None
    quality: Optional[quality.QualityReport] = None
    capture: Optional[CaptureData] = None
    latency_ms: float = 0.0
    warnings: list = field(default_factory=list)


def process_window(raw_mv: np.ndarray, cascade: Cascade, capture: CaptureData | None = None) -> CycleResult:
    """Offline-testable core: raw 2500 @ 500 Hz -> CycleResult. Runs each model at most once."""
    t0 = time.perf_counter()
    raw_mv = np.asarray(raw_mv, dtype=np.float64)
    if len(raw_mv) != C.RAW_SAMPLES:
        raise ValueError(f"need exactly {C.RAW_SAMPLES} samples, got {len(raw_mv)}")
    pre = quality.evaluate(raw_mv)                     # raw checks (rail / flat / NaN)
    if not pre.ok:
        return CycleResult(D.poor_signal(f"{pre.status}: {pre.detail}"), quality=pre, capture=capture)
    filtered = preprocessing.preprocess(raw_mv)
    q = quality.evaluate(raw_mv, filtered)             # + artifact check on conditioned signal
    if not q.ok:
        return CycleResult(D.poor_signal(f"{q.status}: {q.detail}"), filtered, quality=q, capture=capture)
    m1, m2, m3, _ = cascade.classify(filtered)
    dec = D.decide(m1, m2, m3)
    warns = list(q.warnings) + (capture.warnings if capture else [])
    return CycleResult(dec, filtered, m1=m1, m2=m2, m3=m3, quality=q, capture=capture,
                       latency_ms=(time.perf_counter() - t0) * 1000, warnings=warns)


def run_cycle(adc, cascade: Cascade, progress: Progress | None = None, cancel=None) -> CycleResult:
    """Acquire exactly 5 s, process, classify once. ADC is stopped on every exit path."""
    adc.start()
    try:
        cap = capture_window(adc, cancel=cancel, progress=progress)
    except AcquisitionError as exc:
        return CycleResult(D.error(str(exc)))
    finally:
        adc.stop()                                     # idle again, no matter what
    return process_window(cap.raw_mv, cascade, cap)
