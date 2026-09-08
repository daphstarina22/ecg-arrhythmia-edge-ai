"""
ADS1292R Hardware Driver & Diagnostic Signal Preprocessing
Target: Raspberry Pi 4B+
Hardware: Texas Instruments ADS1292R 24-bit ECG Analog Front-End (SPI)

PRESERVES EXACT TESTED HARDWARE IMPLEMENTATION from ecg_pipeline_pi.py:
  - SPI Bus 0, Device 0 (CE0), 1 MHz clock, Mode 0b01 (CPOL=0, CPHA=1)
  - GPIO BCM: DRDY=17, START=27, PWDN=22
  - Hardware reset via PWDN pulse
  - Chip ID validation (0x13 / 0x73)
  - Registers: CONFIG1=0x02, CONFIG2=0xA0, CH1SET=0x00, CH2SET=0x00, RLDSENS=0x2C
  - DRDY falling edge synchronization
  - 24-bit signed integer decoding & physical mV scaling (VREF=2.42V, GAIN=6)
  - Research-grade diagnostic filter (median baseline removal + 50Hz notch + 0.05-100Hz bandpass)
  - Pan-Tompkins QRS detection
"""

import time
import math
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt

# ======================================================================
# HARDWARE CONFIGURATION (EXACT MATCH TO ecg_pipeline_pi.py)
# ======================================================================
FS = 360                           # Nominal sampling frequency (Hz)
POWERLINE_HZ = 50.0                # Powerline notch frequency (India mains)
DIAG_BAND = (0.05, 100.0)          # AAMI EC57 diagnostic bandpass
PT_BAND = (5.0, 15.0)              # Pan-Tompkins detection bandpass

WINDOW_SEC = 5.0                   # 5.0 second analysis window
WINDOW_SAMPLES = int(FS * WINDOW_SEC)  # 1800 samples
STEP_SEC = 2.5                     # 2.5 second hop (50% overlap)
STEP_SAMPLES = int(FS * STEP_SEC)      # 900 samples

# Raspberry Pi BCM GPIO pin assignments
DRDY_PIN = 17   # Data-ready interrupt (active-low)
START_PIN = 27  # Conversion start pin
PWDN_PIN = 22   # Power-down / hardware reset pin

# ADS1292R Register Addresses
REG_ID = 0x00
REG_CONFIG1 = 0x01
REG_CONFIG2 = 0x02
REG_LOFF = 0x03
REG_CH1SET = 0x04
REG_CH2SET = 0x05
REG_RLDSENS = 0x06

# ADS1292R SPI Commands
CMD_RESET = 0x06
CMD_START = 0x08
CMD_RDATAC = 0x10
CMD_SDATAC = 0x11

ADC_GAIN = 6.0
VREF = 2.42


class ADS1292R:
    """
    Physical ADS1292R hardware driver.
    Directly ported from tested ecg_pipeline_pi.py.
    """

    def __init__(self, spi_bus=0, spi_device=0, spi_speed_hz=1_000_000, fs=FS):
        import spidev
        import RPi.GPIO as GPIO

        self.GPIO = GPIO
        self.fs = fs
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(DRDY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(START_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(PWDN_PIN, GPIO.OUT, initial=GPIO.HIGH)

        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = spi_speed_hz
        self.spi.mode = 0b01  # CPOL=0, CPHA=1 -- ADS1292R requirement

        self._init_chip()

    def _cmd(self, cmd: int):
        self.spi.xfer2([cmd])
        time.sleep(0.001)

    def _read_reg(self, addr: int) -> int:
        resp = self.spi.xfer2([0x20 | addr, 0x00, 0x00])
        return resp[2]

    def _write_reg(self, addr: int, val: int):
        self.spi.xfer2([0x40 | addr, 0x00, val])
        time.sleep(0.001)

    def _init_chip(self):
        # Hardware reset via PWDN
        self.GPIO.output(PWDN_PIN, self.GPIO.LOW)
        time.sleep(0.01)
        self.GPIO.output(PWDN_PIN, self.GPIO.HIGH)
        time.sleep(0.15)

        self._cmd(CMD_RESET)
        time.sleep(0.1)
        self._cmd(CMD_SDATAC)
        time.sleep(0.01)

        # Chip-ID validation (fails loudly on wiring/SPI failure)
        chip_id = self._read_reg(REG_ID)
        if (chip_id & 0x1F) != 0x13 and chip_id != 0x73:
            raise RuntimeError(
                f"ADS1292R ID mismatch (got 0x{chip_id:02X}, expected 0x13 or 0x73) -- check wiring/SPI!"
            )
        print(f"[ADS1292R] Hardware Chip ID validated: 0x{chip_id:02X}")

        self._write_reg(REG_CONFIG1, 0x02)    # 500 SPS, continuous conversion
        self._write_reg(REG_CONFIG2, 0xA0)    # Internal 2.42V reference ON
        self._write_reg(REG_CH1SET, 0x00)     # Gain=6, normal electrode input
        self._write_reg(REG_CH2SET, 0x00)     # Gain=6, normal electrode input
        self._write_reg(REG_RLDSENS, 0x2C)    # RLD sensed from CH1P/CH1N/CH2P
        time.sleep(0.01)

        self._cmd(CMD_START)
        time.sleep(0.01)
        self._cmd(CMD_RDATAC)
        time.sleep(0.01)

        self.GPIO.output(START_PIN, self.GPIO.HIGH)
        print("[ADS1292R] Acquisition started (RDATAC active, START=HIGH).")

    def read_sample(self) -> tuple[float, float]:
        """Blocks until DRDY falling edge, reads 9-byte frame, returns (ch1_mv, ch2_mv)."""
        while self.GPIO.input(DRDY_PIN) == self.GPIO.HIGH:
            pass  # Busy-wait for DRDY falling edge

        frame = self.spi.xfer2([0x00] * 9)  # 3 status bytes + 3 bytes CH1 + 3 bytes CH2

        ch1_raw = self._to_signed24(frame[3:6])
        ch2_raw = self._to_signed24(frame[6:9])

        # Raw ADC counts -> physical millivolts (mV)
        ch1_mv = (ch1_raw / (2 ** 23 - 1)) * (VREF / ADC_GAIN) * 1000.0
        ch2_mv = (ch2_raw / (2 ** 23 - 1)) * (VREF / ADC_GAIN) * 1000.0
        return ch1_mv, ch2_mv

    @staticmethod
    def _to_signed24(bytes3: list[int]) -> int:
        val = (bytes3[0] << 16) | (bytes3[1] << 8) | bytes3[2]
        if val & 0x800000:
            val -= 1 << 24
        return val

    def close(self):
        try:
            self._cmd(CMD_SDATAC)
            self.spi.close()
            self.GPIO.cleanup()
            print("[ADS1292R] Hardware connection closed cleanly.")
        except Exception as e:
            print(f"[ADS1292R] Warning on close: {e}")


class MockADS1292R:
    """
    Software simulator for development and testing when physical SPI is absent.
    Preserves the exact same read_sample() interface and timing characteristics.
    """

    def __init__(self, fs: int = FS, noise_std: float = 0.02):
        self.fs = fs
        self.noise_std = noise_std
        self.sample_idx = 0
        self.period = 1.0 / fs
        self.last_time = time.perf_counter()
        print(f"[MockADS1292R] Initialized offline simulation @ {fs} Hz.")

    def read_sample(self) -> tuple[float, float]:
        # High precision pacing to match exact sampling period across all platforms
        while time.perf_counter() - self.last_time < self.period:
            pass
        self.last_time = time.perf_counter()

        t = self.sample_idx / self.fs
        self.sample_idx += 1

        # Synthesize 72 BPM Normal Sinus Rhythm (P-QRS-T)
        beat_phase = (t % 0.833) / 0.833  # 0 to 1 per beat
        ecg = 0.0

        # P wave
        if 0.10 <= beat_phase < 0.20:
            ecg += 0.15 * math.sin(math.pi * (beat_phase - 0.10) / 0.10)
        # Q wave
        elif 0.28 <= beat_phase < 0.30:
            ecg -= 0.15 * math.sin(math.pi * (beat_phase - 0.28) / 0.02)
        # R wave spike
        elif 0.30 <= beat_phase < 0.35:
            ecg += 1.40 * math.sin(math.pi * (beat_phase - 0.30) / 0.05)
        # S wave
        elif 0.35 <= beat_phase < 0.38:
            ecg -= 0.25 * math.sin(math.pi * (beat_phase - 0.35) / 0.03)
        # T wave
        elif 0.50 <= beat_phase < 0.68:
            ecg += 0.30 * math.sin(math.pi * (beat_phase - 0.50) / 0.18)

        # Baseline wander + small noise
        ecg += 0.05 * math.sin(2 * math.pi * 0.2 * t)
        ecg += float(np.random.normal(0, self.noise_std))

        return float(ecg), 0.0

    def close(self):
        print("[MockADS1292R] Closed simulation.")


def create_adc(mock: bool = False, fs: int = FS):
    """Factory helper: returns physical ADS1292R if available, else MockADS1292R."""
    if mock:
        return MockADS1292R(fs=fs)
    try:
        import spidev
        import RPi.GPIO
        return ADS1292R(fs=fs)
    except (ImportError, RuntimeError) as err:
        print(f"[NOTE] Physical ADS1292R not accessible ({err}). Falling back to MockADS1292R.")
        return MockADS1292R(fs=fs)


# ======================================================================
# PATH A: RESEARCH-GRADE / DIAGNOSTIC FILTERING (from ecg_pipeline_pi.py)
# ======================================================================

def remove_baseline_wander(sig: np.ndarray, fs: float = FS, kernel_sec: tuple[float, float] = (0.2, 0.6)) -> np.ndarray:
    """Two-stage median filter baseline removal (avoids ST-segment phase distortion)."""
    k1 = int(kernel_sec[0] * fs)
    k1 += 1 - (k1 % 2)
    k2 = int(kernel_sec[1] * fs)
    k2 += 1 - (k2 % 2)
    baseline = medfilt(sig, kernel_size=k1)
    baseline = medfilt(baseline, kernel_size=k2)
    return sig - baseline


def diagnostic_bandpass(sig: np.ndarray, fs: float = FS, band: tuple[float, float] = DIAG_BAND, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(0.001, band[0] / nyq)
    high = min(0.999, band[1] / nyq)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)


def notch_filter(sig: np.ndarray, fs: float = FS, freq: float = POWERLINE_HZ, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, sig)


def research_grade_filter(raw_sig: np.ndarray, fs: float = FS) -> np.ndarray:
    """Combines median baseline removal, 50Hz notch filter, and 0.05-100Hz diagnostic bandpass."""
    x = remove_baseline_wander(raw_sig, fs)
    x = notch_filter(x, fs)
    x = diagnostic_bandpass(x, fs)
    return x


# ======================================================================
# PATH B: PAN-TOMPKINS PEAK TIMING (from ecg_pipeline_pi.py)
# ======================================================================

def pt_bandpass(sig: np.ndarray, fs: float = FS, band: tuple[float, float] = PT_BAND, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [band[0] / nyq, band[1] / nyq], btype='band')
    return filtfilt(b, a, sig)


def derivative_filter(sig: np.ndarray) -> np.ndarray:
    kernel = np.array([1, 2, 0, -2, -1]) * (1.0 / 8.0)
    return np.convolve(sig, kernel, mode='same')


def moving_window_integration(sig: np.ndarray, fs: float = FS, window_ms: float = 150.0) -> np.ndarray:
    w = int(window_ms / 1000.0 * fs)
    return np.convolve(sig, np.ones(w) / w, mode='same')


def detect_r_peaks(raw_sig: np.ndarray, fs: float = FS, refractory_ms: float = 200.0) -> np.ndarray:
    """Pan-Tompkins QRS detector returning array of sample indices."""
    bp = pt_bandpass(raw_sig, fs)
    deriv = derivative_filter(bp)
    sq = deriv ** 2
    integrated = moving_window_integration(sq, fs)

    threshold = 0.3 * np.max(integrated) if np.max(integrated) > 0 else 0.0
    refractory = int(refractory_ms / 1000.0 * fs)

    peaks = []
    i = 0
    while i < len(integrated):
        if integrated[i] > threshold:
            window_end = min(i + refractory, len(integrated))
            local_max = i + int(np.argmax(integrated[i:window_end]))
            peaks.append(local_max)
            i = local_max + refractory
        else:
            i += 1
    return np.array(peaks, dtype=int)
