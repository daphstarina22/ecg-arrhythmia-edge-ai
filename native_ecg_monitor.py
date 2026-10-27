#!/usr/bin/env python3
"""
Root entrypoint for ECG Edge AI Native HMI Monitor.
Runs deploy/native_ecg_monitor.py.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "deploy" / "native_ecg_monitor.py"
    runpy.run_path(str(target), run_name="__main__")
