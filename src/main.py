"""Command-line Raspberry Pi inference entry point."""

import argparse
import json
from pathlib import Path

import numpy as np

from .features import extract_features
from .inference import classify_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("signal", type=Path, help=".npy or whitespace-delimited ECG file")
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()
    signal = np.load(args.signal) if args.signal.suffix == ".npy" else np.loadtxt(args.signal)
    features = extract_features(signal)
    print("features:", features.tolist())
    print("feature_shape:", features.shape)
    results = classify_window(features, args.models)
    print(json.dumps(results, default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()