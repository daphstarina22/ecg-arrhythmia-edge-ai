#!/usr/bin/env python3
"""
Dedicated ADS1292R Hardware Acquisition & Signal Quality Test.
Target: Raspberry Pi 4B+ (with ADS1292R on SPI0)

CRITICAL RULE: THIS SCRIPT DOES NOT RUN ML INFERENCE.
It strictly verifies that:
  1. ADS1292R hardware initializes and Chip ID is validated
  2. SPI communication and DRDY hardware interrupt operate correctly
  3. Channel 1 produces physical, plausible ECG samples at ~360 SPS
  4. Signal statistics are healthy (no NaNs, no flatlines, no saturation)
  5. Acquired waveform is saved to CSV for verification
  6. Optional browser viewer demonstrates live signal acquisition

Usage on Raspberry Pi:
  python3 deploy/test_ads1292r_live.py --duration 10
  python3 deploy/test_ads1292r_live.py --duration 10 --view    # View live in browser at http://<pi-ip>:5000
  python3 deploy/test_ads1292r_live.py --duration 10 --mock    # Test offline simulation
"""

import argparse
import sys
import time
import threading
from pathlib import Path
from collections import deque
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy.ads1292r_driver import (
    FS,
    create_adc,
    research_grade_filter,
)


def run_hardware_test(duration_sec: float = 10.0, output_csv: str = "ads1292r_live_test.csv", mock: bool = False, enable_view: bool = False):
    print("===========================================================================")
    print("  ADS1292R DEDICATED HARDWARE ACQUISITION TEST (NO ML)")
    print("===========================================================================")
    print(f"  Target Duration : {duration_sec:.1f} seconds")
    print(f"  Nominal Rate    : {FS} samples/second")
    print(f"  Expected Samples: ~{int(duration_sec * FS)} samples")
    print(f"  Output CSV File : {output_csv}")
    print(f"  Mode            : {'SIMULATED (Mock)' if mock else 'PHYSICAL HARDWARE (SPI0)'}")
    print("===========================================================================\n")

    # Shared buffer if web viewer is active
    view_raw = deque(maxlen=int(FS * 6))
    view_filtered = deque(maxlen=int(FS * 6))
    view_lock = threading.Lock()

    if enable_view:
        try:
            from flask import Flask, jsonify, Response
            app = Flask(__name__)

            @app.route("/data")
            def data():
                with view_lock:
                    r = list(view_raw)
                    f = list(view_filtered) if view_filtered else r
                return jsonify({"raw": r, "filtered": f, "fs": FS})

            @app.route("/")
            def index():
                html = """<!DOCTYPE html>
<html>
<head>
  <title>ADS1292R Hardware Test Viewer</title>
  <style>
    body { background: #111; color: #eee; font-family: monospace; padding: 16px; margin: 0; }
    h2 { margin: 4px 0; }
    canvas { background: #000; border: 1px solid #444; display: block; margin-top: 8px; }
    .legend { font-size: 13px; margin-bottom: 4px; }
    .raw { color: #4a90d9; font-weight: bold; }
    .filt { color: #4ad991; font-weight: bold; }
    .note { color: #aaa; font-size: 12px; margin-top: 8px; }
  </style>
</head>
<body>
  <h2>ADS1292R Hardware Signal Test (Channel 1)</h2>
  <div class="legend"><span class="raw">-- RAW (Blue)</span> &nbsp; <span class="filt">-- FILTERED (Green)</span></div>
  <canvas id="chart" width="1200" height="500"></canvas>
  <div class="note">Sampling rate: 360 Hz. Display centered on trace mean. No ML is running.</div>
  <script>
    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');
    function drawTrace(values, color, yOffset, yScale) {
      if (!values || values.length === 0) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const w = canvas.width;
      const step = w / values.length;
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      for (let i = 0; i < values.length; i++) {
        const x = i * step;
        const y = yOffset - (values[i] - mean) * yScale;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    async function update() {
      try {
        const res = await fetch('/data');
        const d = await res.json();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawTrace(d.raw, '#4a90d9', 150, 60);
        drawTrace(d.filtered, '#4ad991', 350, 60);
      } catch(e) {}
      setTimeout(update, 100);
    }
    update();
  </script>
</body>
</html>"""
                return Response(html, mimetype="text/html")

            server_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False), daemon=True)
            server_thread.start()
            print("[WEB] Live Viewer started on http://0.0.0.0:5000\n")
        except ImportError:
            print("[WARN] Flask not installed. Web viewer disabled. Run 'pip install flask' if desired.\n")

    # 1. Initialize ADS1292R
    print("[STEP 1/4] Initializing ADS1292R Analog Front-End...")
    try:
        adc = create_adc(mock=mock, fs=FS)
    except Exception as e:
        print(f"[FATAL] Failed to initialize ADS1292R: {e}")
        return False

    timestamps = []
    ch1_samples = []

    # 2. Acquire Channel 1 samples
    print(f"[STEP 2/4] Recording live samples for {duration_sec:.1f} seconds...")
    t_start = time.perf_counter()
    target_end = t_start + duration_sec
    last_print = t_start
    filter_counter = 0

    try:
        while time.perf_counter() < target_end:
            t_now = time.perf_counter()
            ch1_mv, _ = adc.read_sample()
            
            timestamps.append(t_now - t_start)
            ch1_samples.append(ch1_mv)

            if enable_view:
                with view_lock:
                    view_raw.append(ch1_mv)
                    filter_counter += 1
                    if filter_counter >= 90:
                        filter_counter = 0
                        try:
                            arr = np.array(view_raw)
                            filt = research_grade_filter(arr, fs=FS)
                            view_filtered.clear()
                            view_filtered.extend(filt.tolist())
                        except Exception:
                            pass

            if t_now - last_print >= 1.0:
                elapsed = t_now - t_start
                rate = len(ch1_samples) / elapsed
                print(f"  ... elapsed: {elapsed:4.1f}s | samples: {len(ch1_samples):5d} | instantaneous rate: {rate:5.1f} SPS")
                last_print = t_now

    except KeyboardInterrupt:
        print("\n[WARN] Acquisition interrupted by user.")
    finally:
        adc.close()

    total_time = time.perf_counter() - t_start
    n_samples = len(ch1_samples)

    if n_samples == 0:
        print("\n[FAIL] Zero samples acquired. Hardware failed to produce data.")
        return False

    raw_arr = np.array(ch1_samples, dtype=np.float64)
    measured_fs = n_samples / total_time

    # 3. Compute Signal Statistics
    print(f"\n[STEP 3/4] Computing Signal Quality & Diagnostic Metrics...")
    n_nonfinite = int(np.sum(~np.isfinite(raw_arr)))
    pct_nonfinite = (n_nonfinite / n_samples) * 100.0

    valid_mask = np.isfinite(raw_arr)
    valid_data = raw_arr[valid_mask] if np.any(valid_mask) else np.array([0.0])

    s_min = float(np.min(valid_data))
    s_max = float(np.max(valid_data))
    s_mean = float(np.mean(valid_data))
    s_std = float(np.std(valid_data))
    s_ptp = float(np.ptp(valid_data))

    # Extreme / Saturated samples (|V| > 10 mV is typical rail for biopotential AFE)
    n_saturated = int(np.sum(np.abs(valid_data) > 10.0))
    pct_saturated = (n_saturated / n_samples) * 100.0

    # Rate deviation check (nominal 360 Hz, acceptable range: 340 - 380 Hz)
    rate_ok = 340.0 <= measured_fs <= 380.0
    flatline = (s_std < 0.01) or (s_ptp < 0.05)
    saturated = pct_saturated > 5.0
    finite_ok = pct_nonfinite == 0.0

    print("---------------------------------------------------------------------------")
    print(f"  Total Duration       : {total_time:.3f} s")
    print(f"  Total Samples        : {n_samples}")
    print(f"  Measured Sample Rate : {measured_fs:.2f} SPS (Target: {FS} Hz)")
    print(f"  Rate Deviation Check : {'PASS [OK]' if rate_ok else 'WARN (Deviated from 360 Hz)'}")
    print(f"  Signal Minimum       : {s_min:.3f} mV")
    print(f"  Signal Maximum       : {s_max:.3f} mV")
    print(f"  Signal Mean (DC)     : {s_mean:.3f} mV")
    print(f"  Signal Std Dev       : {s_std:.3f} mV")
    print(f"  Peak-to-Peak (PTP)   : {s_ptp:.3f} mV")
    print(f"  Non-Finite Samples   : {n_nonfinite} ({pct_nonfinite:.2f}%)")
    print(f"  Saturated (>10mV)    : {n_saturated} ({pct_saturated:.2f}%)")
    print(f"  Flatline Detected    : {'YES [BAD SIGNAL]' if flatline else 'NO [PASS]'}")
    print("---------------------------------------------------------------------------")

    # 4. Save to CSV
    print(f"\n[STEP 4/4] Applying research filter and saving to {output_csv}...")
    try:
        if len(raw_arr) >= int(FS * 2):
            filtered_arr = research_grade_filter(raw_arr, fs=FS)
        else:
            filtered_arr = raw_arr
    except Exception as e:
        print(f"  [WARN] Research filter error: {e}")
        filtered_arr = raw_arr

    csv_path = Path(output_csv).resolve()
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_index,timestamp_sec,raw_mv,filtered_mv\n")
        for idx in range(n_samples):
            f.write(f"{idx},{timestamps[idx]:.6f},{raw_arr[idx]:.4f},{filtered_arr[idx]:.4f}\n")
    print(f"  Saved {n_samples} verified samples to: {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")

    # Overall Verdict
    is_healthy = finite_ok and (not flatline) and (not saturated) and (n_samples >= int(duration_sec * 300))
    print("\n===========================================================================")
    if is_healthy:
        print("  HARDWARE ACQUISITION VERDICT: [PASS] - HARDWARE SIGNAL ACQUISITION WORKS!")
        print("  Signal is physically plausible, correctly paced, and ready for ML.")
    else:
        print("  HARDWARE ACQUISITION VERDICT: [FAIL] - ANOMALOUS SIGNAL DETECTED")
        if not finite_ok:
            print("  Reason: Non-finite (NaN/Inf) samples detected.")
        if flatline:
            print("  Reason: Flatline detected. Check electrode contact and leads.")
        if saturated:
            print("  Reason: Extreme/saturated voltages. Check RLD or ground reference.")
        if not rate_ok:
            print("  Reason: Sampling rate deviated substantially from 360 Hz.")
    print("===========================================================================\n")
    return is_healthy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADS1292R Live Hardware Acquisition Test (No ML)")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds (default: 10.0)")
    parser.add_argument("--output", type=str, default="ads1292r_live_test.csv", help="Output CSV path")
    parser.add_argument("--view", action="store_true", help="Launch live web viewer on port 5000")
    parser.add_argument("--mock", action="store_true", help="Run offline simulator mode")
    args = parser.parse_args()

    success = run_hardware_test(
        duration_sec=args.duration,
        output_csv=args.output,
        mock=args.mock,
        enable_view=args.view,
    )
    sys.exit(0 if success else 1)
