"""Driver behaviour against fake SPI/GPIO + capture timing logic."""
import sys
import threading
import types

import pytest

from ecg_edge import acquisition as A
from ecg_edge import config as C


class FakeSPI:
    max_speed_hz = 0

    def __init__(self):
        self.log, self.mode, self.id = [], None, 0x73
        self.frame = [0] * 9

    def open(self, b, d): pass
    def close(self): pass

    def xfer2(self, data):
        self.log.append(tuple(data))
        if len(data) == 3 and data[0] == 0x20:
            return [0, 0, self.id]
        if len(data) == 9:
            return list(self.frame)
        return [0] * len(data)


class FakeGPIO:
    BCM, IN, OUT, HIGH, LOW, PUD_UP = 1, 2, 3, 1, 0, 4

    def __init__(self): self.pins, self.drdy = {}, 0
    def setmode(self, m): pass
    def setup(self, pin, mode, **kw):
        if mode == self.OUT: self.pins[pin] = kw.get("initial", 0)
    def output(self, pin, v): self.pins[pin] = v
    def input(self, pin): return self.drdy
    def cleanup(self): pass


@pytest.fixture
def hw(monkeypatch):
    spi, gpio = FakeSPI(), FakeGPIO()
    sp = types.ModuleType("spidev"); sp.SpiDev = lambda: spi
    rpi = types.ModuleType("RPi"); rpi.GPIO = gpio
    monkeypatch.setitem(sys.modules, "spidev", sp)
    monkeypatch.setitem(sys.modules, "RPi", rpi)
    monkeypatch.setitem(sys.modules, "RPi.GPIO", gpio)
    return spi, gpio


def test_register_config_unchanged_and_idle_after_init(hw):
    spi, gpio = hw
    A.ADS1292R()
    writes = [(d[0] & 0x1F, d[2]) for d in spi.log if len(d) == 3 and (d[0] & 0xE0) == 0x40]
    assert writes == [(0x01, 0x02), (0x02, 0xA0), (0x04, 0x00), (0x05, 0x00), (0x06, 0x2C)]
    assert gpio.pins[C.START_PIN] == gpio.LOW            # not converting after init
    assert (0x10,) not in spi.log and (0x08,) not in spi.log


def test_bad_chip_id_raises(hw):
    hw[0].id = 0x00
    with pytest.raises(A.AcquisitionError, match="not responding"):
        A.ADS1292R()


def test_start_stop_sequence(hw):
    spi, gpio = hw
    adc = A.ADS1292R(); spi.log.clear()
    adc.start()
    assert spi.log == [(0x08,), (0x10,)] and gpio.pins[C.START_PIN] == gpio.HIGH
    spi.log.clear(); adc.stop()
    assert spi.log == [(0x11,), (0x0A,)] and gpio.pins[C.START_PIN] == gpio.LOW


def test_read_decodes_24bit_signed_and_scales(hw):
    spi, gpio = hw
    adc = A.ADS1292R(); gpio.drdy = 0
    spi.frame = [0, 0, 0, 0x7F, 0xFF, 0xFF, 0, 0, 0]
    assert adc.read_sample() == pytest.approx(C.VREF / C.ADC_GAIN * 1000, rel=1e-6)
    spi.frame = [0, 0, 0, 0x80, 0x00, 0x00, 0, 0, 0]
    assert adc.read_sample() < -403
    spi.frame = [0, 0, 0, 0xFF, 0xFF, 0xFF, 0, 0, 0]
    assert -1e-3 < adc.read_sample() < 0


def test_drdy_timeout_raises_instead_of_hanging(hw):
    spi, gpio = hw
    adc = A.ADS1292R(); gpio.drdy = 1
    with pytest.raises(A.AcquisitionError, match="DRDY timeout"):
        adc.read_sample(timeout=0.05)


class FakeAdc:
    is_realtime = True
    def __init__(self): self.n = 0
    def read_sample(self): self.n += 1; return 0.1 * (self.n % 7)


def _clock(total_s):
    it = iter([0.0, total_s])
    return lambda: next(it)


def test_capture_reads_exactly_5s_worth():
    adc = FakeAdc()
    cap = A.capture_window(adc, clock=_clock(4.998))
    assert len(cap.raw_mv) == C.RAW_SAMPLES == 2500
    assert adc.n == 2500 + C.SETTLE_DISCARD_SAMPLES
    assert cap.timing_error_frac < 1e-3 and not cap.warnings


def test_capture_rejects_lost_samples():
    with pytest.raises(A.AcquisitionError, match="timing fault"):
        A.capture_window(FakeAdc(), clock=_clock(5.4))


def test_capture_warns_on_small_drift():
    cap = A.capture_window(FakeAdc(), clock=_clock(5.04))
    assert cap.warnings and "timing" in cap.warnings[0]


def test_capture_cancel():
    ev = threading.Event(); ev.set()
    with pytest.raises(A.Cancelled):
        A.capture_window(FakeAdc(), cancel=ev, clock=_clock(5.0))
