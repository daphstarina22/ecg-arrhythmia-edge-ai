"""One Start = one 5 s capture + one inference + back to idle."""
import sys
import time

import numpy as np
import pytest

from ecg_edge import config as C
from ecg_edge.acquisition import AcquisitionError, MockADS1292R, create_adc
from ecg_edge.pipeline import process_window, run_cycle


@pytest.mark.parametrize("kind,expected", [
    ("TACHY", "TACHY — NON-SHOCKABLE"), ("BRADY", "BRADY_ASY — NON-SHOCKABLE"),
    ("VF", "VF — SHOCKABLE"), ("VT", "VT — SHOCKABLE"),
    ("FLAT", "NO RESULT — POOR SIGNAL"), ("LEADOFF", "NO RESULT — POOR SIGNAL")])
def test_mock_rhythms_end_to_end(cascade, kind, expected):
    """Plumbing check on SYNTHETIC rhythms (not a model-accuracy claim)."""
    assert run_cycle(MockADS1292R(kind, realtime=False), cascade).decision.headline == expected


def test_exactly_one_acquisition_and_one_inference_per_start(cascade):
    adc = MockADS1292R("VF", realtime=False)
    before = dict(cascade.calls)
    r = run_cycle(adc, cascade)
    assert adc.start_calls == 1 and adc.stop_calls == 1 and not adc.running
    assert adc.samples_read == C.RAW_SAMPLES + C.SETTLE_DISCARD_SAMPLES
    assert len(r.filtered) == C.MODEL_SAMPLES
    used = {k: cascade.calls[k] - before[k] for k in before}
    assert used["m1"] == 1 and used["m2"] == 1 and used["m3"] <= 1


def test_m3_only_runs_when_ventricular_suspected(cascade):
    before = cascade.calls["m3"]
    run_cycle(MockADS1292R("BRADY", realtime=False), cascade)
    assert cascade.calls["m3"] == before


def test_nothing_runs_between_cycles(cascade):
    adc = MockADS1292R("TACHY", realtime=False)
    run_cycle(adc, cascade)
    n, calls = adc.samples_read, dict(cascade.calls)
    time.sleep(0.2)
    assert adc.samples_read == n and cascade.calls == calls and not adc.running
    run_cycle(adc, cascade)
    assert adc.start_calls == 2 and cascade.calls["m1"] == calls["m1"] + 1


def test_adc_stopped_even_on_hardware_error(cascade):
    class Broken(MockADS1292R):
        def read_sample(self, timeout=0.25): raise AcquisitionError("DRDY timeout: test")
    adc = Broken("NSR", realtime=False)
    r = run_cycle(adc, cascade)
    assert r.decision.status == "ERROR" and "DRDY" in r.decision.details and not adc.running


def test_wrong_length_rejected(cascade):
    with pytest.raises(ValueError):
        process_window(np.zeros(1800), cascade)


def test_mock_never_chosen_implicitly(monkeypatch):
    monkeypatch.setitem(sys.modules, "spidev", None)
    with pytest.raises(AcquisitionError):
        create_adc(mock=None)
