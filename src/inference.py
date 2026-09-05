"""ONNX Runtime wrappers and the three-stage ECG decision flow."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import FEATURE_NAMES


class ModelContractError(RuntimeError):
    """Raised when a trained model does not match the deployment contract."""


class OnnxClassifier:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        if len(shape) != 2 or shape[-1] != len(FEATURE_NAMES):
            raise ModelContractError(
                f"{self.path.name} expects {shape}; deployment requires [batch, 10]"
            )

    def predict(self, features: np.ndarray) -> dict:
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (1, len(FEATURE_NAMES)):
            raise ValueError(f"Expected feature shape (1, 10), got {features.shape}")
        outputs = self.session.run(None, {self.input.name: features})
        return {"label": outputs[0], "probabilities": outputs[1]}


def classify_window(features: np.ndarray, model_dir: Path) -> dict:
    """Run Model 2 always and Model 3 only when Model 1 is shockable."""
    model_dir = Path(model_dir)
    model_1 = OnnxClassifier(model_dir / "shockable_classifier.onnx")
    model_2 = OnnxClassifier(model_dir / "arrhythmia_multiclass.onnx")
    results = {"model_1": model_1.predict(features), "model_2": model_2.predict(features)}
    shockable = int(np.asarray(results["model_1"]["label"]).reshape(-1)[0]) == 1
    results["model_3"] = (
        OnnxClassifier(model_dir / "vf_vt_subtype_classifier.onnx").predict(features)
        if shockable
        else None
    )
    return results