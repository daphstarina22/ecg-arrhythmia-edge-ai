#!/usr/bin/env python3
"""Entry point.   Pi (real ADS1292R):  python3 run_hmi.py
                  Simulated signal:    python3 run_hmi.py --mock NSR     (NSR TACHY BRADY VF VT FLAT LEADOFF)
"""
import argparse
import sys
import tkinter as tk
from tkinter import messagebox

from ecg_edge.acquisition import AcquisitionError, create_adc
from ecg_edge.simulator import KINDS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", choices=KINDS, help="use a SIMULATED signal instead of the ADS1292R")
    ap.add_argument("--no-voice", action="store_true", help="disable spoken prompts")
    args = ap.parse_args()

    root = tk.Tk()
    try:
        from ecg_edge.models import Cascade
        from ecg_edge.voice import VoiceWorker
        from ecg_edge.hmi import ECGApp
        cascade = Cascade()
        adc = create_adc(mock=args.mock)
    except (AcquisitionError, FileNotFoundError) as exc:
        print(f"[STARTUP ERROR] {exc}", file=sys.stderr)
        messagebox.showerror("ECG Edge AI - startup error", str(exc))
        return 1
    ECGApp(root, adc, cascade, VoiceWorker(enabled=not args.no_voice))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
