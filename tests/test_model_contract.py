"""ONNX I/O contract + label-map checks."""
import numpy as np
import onnxruntime as ort

from ecg_edge import config as C
from ecg_edge.features import M1_FEATURES, M2_FEATURES, M3_FEATURES
from ecg_edge.models import M1_CLASSES, M2_CLASSES, M3_CLASSES


def test_shapes_and_outputs():
    for path, feats, classes in ((C.M1_PATH, M1_FEATURES, M1_CLASSES), (C.M2_PATH, M2_FEATURES, M2_CLASSES),
                                 (C.M3_PATH, M3_FEATURES, M3_CLASSES)):
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        n = s.get_inputs()[0].shape[1]
        assert n == len(feats)
        assert [o.name for o in s.get_outputs()] == ["output_label", "output_probability"]
        out = s.run(None, {"X": np.zeros((1, n), np.float32)})
        assert set(out[1][0].keys()) == set(range(len(classes)))
        assert abs(sum(out[1][0].values()) - 1.0) < 1e-4


def test_label_maps_match_training_constants():
    assert M1_CLASSES == ("NON_SHOCKABLE", "SHOCKABLE")
    assert M2_CLASSES == ("NSR", "TACHY", "BRADY_ASY", "VENTRICULAR")
    assert M3_CLASSES == ("VT", "VF")


def test_onnx_label_equals_argmax_of_probabilities(cascade):
    rng = np.random.default_rng(0)
    for sess, n in ((cascade.s1, 10), (cascade.s2, 12)):
        for _ in range(50):
            lab, prob = sess.run(None, {"X": rng.normal(0, 1, (1, n)).astype(np.float32)})
            p = prob[0]
            assert int(lab[0]) == max(p, key=p.get)
