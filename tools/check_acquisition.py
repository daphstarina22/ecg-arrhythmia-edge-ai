#!/usr/bin/env python3
"""Hardware acquisition check - RUN THIS ON THE PI before trusting the HMI.

Performs N real 5 s captures (no ML) and reports: wall-clock duration per
capture, lost-sample estimate, sample std (noise), and DC offset. Optionally
saves each capture as .npy so it can be replayed with tools/replay_window.py.

    python3 tools/check_acquisition.py -n 5 --save captures/
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_edge import config as C                                   # noqa: E402
from ecg_edge.acquisition import AcquisitionError, ADS1292R, capture_window  # noqa: E402
from ecg_edge import quality                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--save", type=Path)
    a = ap.parse_args()
    try:
        adc = ADS1292R()
    except AcquisitionError as e:
        print("INIT FAILED:", e); return 1
    print("ADS1292R init OK (chip ID verified). Idle - no conversions yet.\n")
    bad = 0
    try:
        for i in range(a.n):
            adc.start()
            try:
                t0 = time.perf_counter()
                cap = capture_window(adc)
                wall = time.perf_counter() - t0
            except AcquisitionError as e:
                print(f"[{i+1}] FAIL: {e}"); bad += 1; continue
            finally:
                adc.stop()
            q = quality.evaluate(cap.raw_mv)
            lost = max(0, round((cap.elapsed_s - C.CAPTURE_SEC) * C.ADC_SPS))
            print(f"[{i+1}] {len(cap.raw_mv)} samples | capture {cap.elapsed_s:.4f} s | wall {wall:.3f} s | "
                  f"est. lost {lost} | std {np.std(cap.raw_mv):.4f} mV | mean {np.mean(cap.raw_mv):.2f} mV | "
                  f"quality {q.status} {q.warnings}")
            if a.save:
                a.save.mkdir(parents=True, exist_ok=True)
                np.save(a.save / f"capture_{time.strftime('%Y%m%d_%H%M%S')}_{i+1}.npy", cap.raw_mv)
            time.sleep(0.5)
    finally:
        adc.close()
    print("\nPASS" if bad == 0 else f"\n{bad} capture(s) failed")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
