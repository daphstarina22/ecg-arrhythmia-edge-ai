"""Feature-contract smoke test and ONNX compatibility check."""

from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_NAMES, extract_features


def main() -> None:
    time = np.arange(1800, dtype=np.float64) / 360
    signal = 0.5 * np.sin(2 * np.pi * 1.2 * time)
    features = extract_features(signal)
    print("FEATURE_NAMES:", FEATURE_NAMES)
    print("FEATURE_VECTOR:", features.tolist())
    print("FEATURE_SHAPE:", features.shape)
    assert features.shape == (1, 10)

    for model_path in sorted((ROOT / "models").glob("*.onnx")):
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_metadata = session.get_inputs()[0]
        output_metadata = session.get_outputs()
        outputs = session.run(None, {input_metadata.name: features})
        print("MODEL:", model_path.name)
        print("INPUT_NAME:", input_metadata.name)
        print("INPUT_SHAPE:", input_metadata.shape)
        print("OUTPUT_NAMES:", [output.name for output in output_metadata])
        print("OUTPUT_SHAPES:", [output.shape for output in output_metadata])
        print("PREDICTED_LABEL:", outputs[0].tolist())
        print("PROBABILITIES:", outputs[1].tolist())


if __name__ == "__main__":
    main()