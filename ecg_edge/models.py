"""ONNX model wrappers with explicit, tested label mappings.

Label mappings (verified against training/final_models/common/data_loader.py,
train_model2.py and the Model-3 extraction script):
  M1 class id: 0 = NON_SHOCKABLE, 1 = SHOCKABLE
  M2 class id: 0 = NSR, 1 = TACHY, 2 = BRADY_ASY, 3 = VENTRICULAR
  M3 class id: 0 = VT,  1 = VF
The ONNX files expose `output_label` (int64) and `output_probability`
(a list of {class_id: prob} dicts).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort

from . import config as C
from .features import M1_FEATURES, M2_FEATURES, M3_FEATURES, features_m1_m2, features_m3

M1_CLASSES = ("NON_SHOCKABLE", "SHOCKABLE")
M2_CLASSES = ("NSR", "TACHY", "BRADY_ASY", "VENTRICULAR")
M3_CLASSES = ("VT", "VF")


@dataclass
class ModelOutput:
    label: str
    confidence: float
    probs: Dict[str, float]


def _session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def _run(sess, feats: Dict[str, float], names, classes) -> ModelOutput:
    x = np.array([[feats[n] for n in names]], dtype=np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})
    pd = out[1][0]
    probs = {classes[i]: float(pd[i]) for i in range(len(classes))}
    label = max(probs, key=probs.get)
    # cross-check: the model's own argmax label must agree with ours
    assert classes[int(out[0][0])] == label or abs(max(probs.values()) - sorted(probs.values())[-2]) < 1e-6, \
        "ONNX label / probability mismatch - label map is wrong"
    return ModelOutput(label, probs[label], probs)


class Cascade:
    """M1 + M2 always; M3 only when either model flags a ventricular rhythm."""

    def __init__(self):
        for p in (C.M1_PATH, C.M2_PATH, C.M3_PATH):
            if not p.exists():
                raise FileNotFoundError(f"model file missing: {p}")
        self.s1, self.s2, self.s3 = _session(C.M1_PATH), _session(C.M2_PATH), _session(C.M3_PATH)
        self.calls = {"m1": 0, "m2": 0, "m3": 0}      # test hook

    def classify(self, window360: np.ndarray):
        f12 = features_m1_m2(window360)
        m1 = _run(self.s1, f12, M1_FEATURES, M1_CLASSES); self.calls["m1"] += 1
        m2 = _run(self.s2, f12, M2_FEATURES, M2_CLASSES); self.calls["m2"] += 1
        m3: Optional[ModelOutput] = None
        if m1.label == "SHOCKABLE" or m2.label == "VENTRICULAR":
            m3 = _run(self.s3, features_m3(window360), M3_FEATURES, M3_CLASSES); self.calls["m3"] += 1
        return m1, m2, m3, {"m1_m2": f12}
