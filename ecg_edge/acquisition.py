"""ADS1292R acquisition (on demand only) + a mock with the same interface.

Hardware configuration (SPI mode, pins, register values, scaling) is carried
over unchanged from the validated driver. What changed:
  * conversions run ONLY between start() and stop() (no background acquisition);
  * read_sample() has a DRDY timeout instead of spinning forever;
  * capture_window() does nothing but read samples (no filtering / inference /
    array rolling inside the loop) and measures timing so lost samples are
    detected instead of silently distorting the time axis;
  * the mock is never selected implicitly.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import config as C
from .simulator import synth_ecg

# ADS1292R registers / commands
REG_ID, REG_CONFIG1, REG_CONFIG2, REG_LOFF = 0x00, 0x01, 0x02, 0x03
REG_CH1SET, REG_CH2SET, REG_RLDSENS = 0x04, 0x05, 0x06
CMD_RESET, CMD_START, CMD_STOP, CMD_RDATAC, CMD_SDATAC = 0x06, 0x08, 0x0A, 0x10, 0x11


class AcquisitionError(RuntimeError):
    """Hardware / timing failure. Carries a short, HMI-displayable message."""


class Cancelled(Exception):
    pass


# ------------------------------------------------------------------ hardware
class ADS1292R:
    is_realtime = True
    simulated = False

    def __init__(self):
        try:
            import spidev
            import RPi.GPIO as GPIO
        except ImportError as exc:
            raise AcquisitionError(
                f"spidev / RPi.GPIO not available ({exc}). Run on the Raspberry Pi, "
                "or start with --mock for a simulated signal.") from exc
        self.GPIO = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(C.DRDY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(C.START_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(C.PWDN_PIN, GPIO.OUT, initial=GPIO.HIGH)
        self.spi = spidev.SpiDev()
        self.spi.open(C.SPI_BUS, C.SPI_DEVICE)
        self.spi.max_speed_hz = C.SPI_HZ
        self.spi.mode = 0b01          # CPOL=0, CPHA=1 (ADS1292R requirement)
        self.running = False
        self._configure()

    # -- low level
    def _cmd(self, cmd):
        self.spi.xfer2([cmd])
        time.sleep(0.001)

    def _read_reg(self, addr):
        return self.spi.xfer2([0x20 | addr, 0x00, 0x00])[2]

    def _write_reg(self, addr, val):
        self.spi.xfer2([0x40 | addr, 0x00, val])
        time.sleep(0.001)

    def _configure(self):
        G = self.GPIO
        G.output(C.PWDN_PIN, G.LOW); time.sleep(0.02)       # hardware reset pulse
        G.output(C.PWDN_PIN, G.HIGH); time.sleep(0.20)
        self._cmd(CMD_SDATAC); time.sleep(0.02)
        self._cmd(CMD_RESET); time.sleep(0.10)
        self._cmd(CMD_SDATAC); time.sleep(0.02)             # chip resets into RDATAC
        chip_id = 0x00
        for _ in range(5):
            chip_id = self._read_reg(REG_ID)
            if self._id_ok(chip_id):
                break
            time.sleep(0.05)
        if not self._id_ok(chip_id):
            hint = {0x00: "chip not responding: check 3.3 V/GND, SPI wiring, PWDN/RESET",
                    0xFF: "MISO floating high: check MISO wiring"}.get(chip_id, "unexpected ID")
            raise AcquisitionError(f"ADS1292R ID 0x{chip_id:02X} ({hint}). "
                                   "Run: python3 tools/inspect_ads1292r.py")
        self._write_reg(REG_CONFIG1, 0x02)   # 500 SPS, continuous conversion
        self._write_reg(REG_CONFIG2, 0xA0)   # internal 2.42 V reference on
        self._write_reg(REG_CH1SET, 0x00)    # gain 6, normal electrode input
        self._write_reg(REG_CH2SET, 0x00)
        self._write_reg(REG_RLDSENS, 0x2C)   # RLD from CH1P/CH1N/CH2P
        time.sleep(0.01)
        # chip is left IDLE here (SDATAC, START low): no conversions until start()

    @staticmethod
    def _id_ok(i):
        return (i & 0x1F) in (0x12, 0x13) or i in (0x53, 0x73, 0x93, 0x52, 0x72)

    # -- run control
    def start(self):
        self._cmd(CMD_START)
        self._cmd(CMD_RDATAC)
        self.GPIO.output(C.START_PIN, self.GPIO.HIGH)
        self.running = True

    def stop(self):
        if not self.running:
            return
        self.GPIO.output(C.START_PIN, self.GPIO.LOW)
        self._cmd(CMD_SDATAC)
        self._cmd(CMD_STOP)
        self.running = False

    def read_sample(self, timeout: float = C.DRDY_TIMEOUT_SEC) -> float:
        """Block until DRDY falls, return CH1 in mV. Raises on timeout."""
        deadline = time.perf_counter() + timeout
        gpio_in, high, drdy = self.GPIO.input, self.GPIO.HIGH, C.DRDY_PIN
        while gpio_in(drdy) == high:
            if time.perf_counter() > deadline:
                raise AcquisitionError("DRDY timeout: ADS1292R is not producing samples "
                                       "(check START/DRDY wiring and power).")
        f = self.spi.xfer2([0x00] * 9)            # 3 status + 3 CH1 + 3 CH2
        raw = (f[3] << 16) | (f[4] << 8) | f[5]
        if raw & 0x800000:
            raw -= 1 << 24
        return (raw / (2 ** 23 - 1)) * (C.VREF / C.ADC_GAIN) * 1000.0

    def close(self):
        try:
            self.stop()
            self.spi.close()
            self.GPIO.cleanup()
        except Exception as exc:  # pragma: no cover
            print(f"[ADS1292R] warning on close: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------- mock
class MockADS1292R:
    """Synthetic 500 SPS source with the same interface as ADS1292R."""
    simulated = True

    def __init__(self, kind: str = "NSR", realtime: bool = True, seed: int = 0):
        self.kind, self.realtime, self.seed = kind, realtime, seed
        self.is_realtime = realtime
        self.running = False
        self.samples_read = 0          # test hooks
        self.start_calls = 0
        self.stop_calls = 0
        self._cycle = 0
        self._buf = None
        self._i = 0

    def start(self):
        self.start_calls += 1
        self.running = True
        self._buf = synth_ecg(self.kind, C.ADC_SPS, C.CAPTURE_SEC + 5.0, seed=self.seed + self._cycle)
        self._cycle += 1
        self._i = 0
        self._t_next = time.perf_counter()

    def stop(self):
        if self.running:
            self.stop_calls += 1
        self.running = False

    def read_sample(self, timeout: float = C.DRDY_TIMEOUT_SEC) -> float:
        if not self.running:
            raise AcquisitionError("mock read while stopped")
        if self.realtime:
            self._t_next += 1.0 / C.ADC_SPS
            while time.perf_counter() < self._t_next:
                time.sleep(0.0002)
        v = float(self._buf[self._i]); self._i += 1
        self.samples_read += 1
        return v

    def close(self):
        self.stop()


def create_adc(mock: Optional[str] = None, realtime: bool = True):
    """Real hardware unless `mock` (a rhythm name) is explicitly given."""
    return MockADS1292R(kind=mock, realtime=realtime) if mock else ADS1292R()


# ------------------------------------------------------------------- capture
@dataclass
class CaptureData:
    raw_mv: np.ndarray          # exactly RAW_SAMPLES values @ ADC_SPS
    elapsed_s: float            # first->last sample wall time (+1 period)
    timing_error_frac: float    # |elapsed - 5 s| / 5 s   (0.0 when not real-time)
    warnings: list


class Progress:
    """Plain counter the HMI thread reads (no locking/queue work in the sample loop)."""
    def __init__(self):
        self.n = 0
        self.total = C.RAW_SAMPLES


def capture_window(adc, cancel: Optional[threading.Event] = None,
                   progress: Optional[Progress] = None,
                   clock: Callable[[], float] = time.perf_counter) -> CaptureData:
    """Read exactly RAW_SAMPLES samples (after a short settle discard).

    adc.start() must already have been called. The loop does nothing except
    read + store so it cannot starve the DRDY polling.
    """
    old_switch = sys.getswitchinterval()
    sys.setswitchinterval(0.0005)          # let this thread get the GIL back quickly
    try:
        for _ in range(C.SETTLE_DISCARD_SAMPLES):
            adc.read_sample()
        n = C.RAW_SAMPLES
        buf = np.empty(n, dtype=np.float64)
        read = adc.read_sample
        t_first = clock()
        for i in range(n):
            buf[i] = read()
            if progress is not None:
                progress.n = i + 1
            if cancel is not None and cancel.is_set():
                raise Cancelled()
        t_last = clock()
    finally:
        sys.setswitchinterval(old_switch)

    warnings, err = [], 0.0
    elapsed = (t_last - t_first) + 1.0 / C.ADC_SPS
    if getattr(adc, "is_realtime", True):
        err = abs(elapsed - C.CAPTURE_SEC) / C.CAPTURE_SEC
        lost = max(0, int(round((elapsed - C.CAPTURE_SEC) * C.ADC_SPS)))
        if err > C.TIMING_REJECT_FRAC:
            raise AcquisitionError(
                f"Acquisition timing fault: 5 s capture took {elapsed:.3f} s "
                f"(~{lost} samples lost). Retry; close other CPU-heavy programs.")
        if err > C.TIMING_WARN_FRAC:
            warnings.append(f"timing off by {err*100:.2f}% (~{lost} samples)")
    return CaptureData(buf, elapsed, err, warnings)
