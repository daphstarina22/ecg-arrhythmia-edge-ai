#!/usr/bin/env python3
"""
ADS1292R Live ECG Streamer & Classifier for Raspberry Pi.

Interfaces with the Texas Instruments ADS1292R 24-bit ECG Analog Front-End (AFE)
over SPI, acquires samples at 500 SPS, resamples to 360 Hz (1,800 samples per 5-second
window) using scipy polyphase rational resampling, and runs the 3-model hierarchical
edge AI classifier.

Usage on Raspberry Pi:
    python3 deploy/ads1292r_stream.py
    python3 deploy/ads1292r_stream.py --mock   # Simulate ADS1292R stream for offline testing
"""

import argparse
import collections
import sys
import time
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy.edge_runtime import HierarchicalEdgeClassifier

# ADS1292R SPI & AFE Parameters
VREF = 2.42         # Internal voltage reference (Volts)
PGA_GAIN = 6.0      # Programmable gain amplifier setting
SAMPLES_500HZ = 2500  # 5 seconds at 500 SPS
TARGET_FS = 360     # Pipeline input sampling rate (Hz)


def decode_24bit_mv(b0: int, b1: int, b2: int, vref: float = VREF, gain: float = PGA_GAIN) -> float:
    """Converts 3 raw SPI data bytes (24-bit 2's complement) to millivolts (mV)."""
    val = (b0 << 16) | (b1 << 8) | b2
    if val & 0x800000:
        val -= 0x1000000
    return (val / 8388607.0) * (vref / gain) * 1000.0


def run_live_stream(spi_bus: int = 0, spi_device: int = 0):
    """Acquires live ECG data from ADS1292R over SPI on Raspberry Pi."""
    try:
        import spidev
    except ImportError:
        print("[ERROR] 'spidev' library is not installed.")
        print("Install it on Raspberry Pi via: pip install spidev")
        print("Or run in simulation mode with: python3 deploy/ads1292r_stream.py --mock")
        sys.exit(1)

    spi = spidev.SpiDev()
    spi.open(spi_bus, spi_device)
    spi.max_speed_hz = 1000000  # 1 MHz SPI clock
    spi.mode = 0b01            # SPI Mode 1 (CPOL=0, CPHA=1)

    print("===========================================================================")
    print("  ADS1292R LIVE ECG STREAMER & 3-MODEL HIERARCHICAL CLASSIFIER")
    print("===========================================================================")
    print(f"  Sampling Rate : 500 SPS -> Resampled to 360 Hz (1,800 samples / 5s)")
    print(f"  SPI Interface : Bus {spi_bus}, Device {spi_device}")
    print("  Classification: Every 5-second window")
    print("===========================================================================\n")

    classifier = HierarchicalEdgeClassifier()
    buffer_500hz = collections.deque(maxlen=SAMPLES_500HZ)

    print("[INFO] Listening for ECG samples... Press Ctrl+C to stop.\n")

    try:
        while True:
            # Read 9 bytes: 3 bytes STATUS + 3 bytes CH1 (ECG) + 3 bytes CH2
            raw = spi.readbytes(9)
            
            # Channel 1 ECG voltage (bytes 3, 4, 5)
            ecg_mv = decode_24bit_mv(raw[3], raw[4], raw[5])
            buffer_500hz.append(ecg_mv)

            # When 5 full seconds have been accumulated
            if len(buffer_500hz) == SAMPLES_500HZ:
                raw_array = np.array(buffer_500hz, dtype=np.float32)

                # Polyphase rational resample: 500 Hz * (18 / 25) = 360 Hz (1,800 samples)
                window_360hz = resample_poly(raw_array, up=18, down=25).astype(np.float32)

                # Run inference
                t0 = time.perf_counter()
                decision = classifier.predict_window(window_360hz)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Rhythm: {decision.primary_rhythm.upper():<16} | "
                      f"Triage: {decision.triage_level:<18} | "
                      f"Conf: {decision.confidence*100:5.1f}% | "
                      f"Latency: {latency_ms:4.1f} ms")

                if decision.defibrillator_advised:
                    print("  >>> [CRITICAL ALARM] SHOCK ADVISED - IMMEDIATE INTERVENTION REQUIRED! <<<\n")

                # Slide window forward by 1 second (500 samples)
                for _ in range(500):
                    if buffer_500hz:
                        buffer_500hz.popleft()

    except KeyboardInterrupt:
        print("\n[INFO] Stopping ADS1292R acquisition.")
    finally:
        spi.close()


def run_mock_stream():
    """Simulates an ADS1292R 500 SPS stream for testing."""
    print("===========================================================================")
    print("  SIMULATING ADS1292R STREAM (500 SPS -> 360 Hz Polyphase Resampling)")
    print("===========================================================================")

    classifier = HierarchicalEdgeClassifier()
    
    t_500 = np.linspace(0, 5, SAMPLES_500HZ, endpoint=False)
    mock_500hz = 0.1 * np.sin(2 * np.pi * 1.2 * t_500)
    for beat_time in np.arange(0.4, 5.0, 0.83):
        spike = np.exp(-((t_500 - beat_time) ** 2) / (2 * (0.02 ** 2))) * 1.2
        mock_500hz += spike

    print("[1/2] Simulating 2,500 raw samples at 500 SPS...")
    print(f"      Signal amplitude range: [{mock_500hz.min():.2f} mV, {mock_500hz.max():.2f} mV]")

    print("[2/2] Applying polyphase resampling (500 Hz -> 360 Hz, up=18, down=25)...")
    t0 = time.perf_counter()
    window_360hz = resample_poly(mock_500hz, up=18, down=25).astype(np.float32)
    resample_ms = (time.perf_counter() - t0) * 1000.0

    print(f"      Resampling complete in {resample_ms:.2f} ms! Resampled shape: {window_360hz.shape}")

    print("\n[INFERENCE] Executing 3-model hierarchical classification...")
    t1 = time.perf_counter()
    decision = classifier.predict_window(window_360hz)
    infer_ms = (time.perf_counter() - t1) * 1000.0

    print(f"  Classification Result : {decision.primary_rhythm.upper()}")
    print(f"  Emergency Priority    : {decision.triage_level}")
    print(f"  Confidence            : {decision.confidence * 100:.1f}%")
    print(f"  Defibrillator Advised : {decision.defibrillator_advised}")
    print(f"  Inference Latency     : {infer_ms:.2f} ms")
    print("\n[SUCCESS] Mock ADS1292R pipeline verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADS1292R Live ECG Streamer & Classifier")
    parser.add_argument("--mock", action="store_true", help="Run offline simulation test without physical SPI")
    parser.add_argument("--bus", type=int, default=0, help="SPI bus number (default: 0)")
    parser.add_argument("--device", type=int, default=0, help="SPI device/CE number (default: 0)")
    args = parser.parse_args()

    if args.mock:
        run_mock_stream()
    else:
        try:
            import spidev
            run_live_stream(args.bus, args.device)
        except ImportError:
            print("[NOTE] 'spidev' not detected. Running mock simulation test...")
            run_mock_stream()
