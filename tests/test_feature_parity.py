"""Our feature code must equal the ACTUAL training functions (extracted by AST,
so the heavy training-script imports like wfdb are not needed)."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.signal as signal
from scipy.signal import butter, filtfilt, welch

from ecg_edge import features as F
from ecg_edge import preprocessing as P
from ecg_edge.simulator import synth_ecg

TRAIN = Path(__file__).resolve().parents[1] / "training"
F12 = TRAIN / "final_models/artifacts/build_multisource_datasets.py"
F3 = TRAIN / "model3_phase5/extract_physiological_features.py"
pytestmark = pytest.mark.skipif(not (F12.exists() and F3.exists()), reason="training/ not present")


def _load(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ns = {"np": np, "math": math, "signal": signal, "welch": welch, "butter": butter,
          "filtfilt": filtfilt, "TARGET_FS": 360}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return [ns[n] for n in names]


def _windows():
    for kind in ("NSR", "TACHY", "BRADY", "VF", "VT"):
        for seed in range(3):
            yield kind, P.preprocess(synth_ecg(kind, seed=seed)[:2500])


def test_m1_m2_features_identical_to_training():
    (ref,) = _load(F12, ["extract_features_single_window"])
    for kind, w in _windows():
        a, b = F.features_m1_m2(w), ref(w)
        for k in b:
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12), (kind, k)


def test_m3_features_identical_to_training():
    ac, cx = _load(F3, ["compute_autocorrelation_features", "compute_complexity_features_strategy_b"])
    for kind, w in _windows():
        a = F.features_m3(w)
        r = {**ac(w, 360), **cx(w)}
        for k in a:
            assert a[k] == pytest.approx(r[k], rel=1e-7, abs=1e-9), (kind, k)
