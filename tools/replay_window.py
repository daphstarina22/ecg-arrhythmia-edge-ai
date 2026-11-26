#!/usr/bin/env python3
"""Run a saved 5 s capture (2500 samples @ 500 Hz, mV; .npy or one-column .csv)
through the exact production pipeline and print the decision + raw model outputs.

    python3 tools/replay_window.py captures/capture_xxx.npy
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_edge.models import Cascade          # noqa: E402
from ecg_edge.pipeline import process_window  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print(__doc__); return 1
    p = Path(sys.argv[1])
    raw = np.load(p) if p.suffix == ".npy" else np.loadtxt(p, delimiter=",", ndmin=1)
    r = process_window(raw.ravel(), Cascade())
    d = r.decision
    print(f"{d.headline}\n  {d.details}")
    for n in d.notes: print("  note:", n)
    if r.quality: print(f"  quality: {r.quality.status} {r.quality.warnings}")
    for name, m in (("M1", r.m1), ("M2", r.m2), ("M3", r.m3)):
        if m: print(f"  {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.probs.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
