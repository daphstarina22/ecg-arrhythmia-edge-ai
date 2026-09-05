# ECG Arrhythmia Edge AI

Raspberry Pi inference baseline for three ONNX ECG classifiers. The trained models are not modified or retrained by this repository setup.

## Input Contract

Each inference input is one ECG window with:

- 5 seconds of signal
- 360 Hz sampling rate
- exactly 1,800 samples
- the same signal amplitude convention used during training

The deployment feature contract is exactly 10 `float32` features with shape `(1, 10)`. The order is fixed and must never change:

```text
f0  = mean
f1  = std
f2  = ptp
f3  = zero_crossings
f4  = peak_count
f5  = mean_rr
f6  = rr_cv
f7  = qrs_width
f8  = dominant_freq
f9  = vf_band_power_ratio
```

## Preprocessing

QRS peaks are detected without annotations using the training notebook's Pan-Tompkins pipeline:

1. 5-15 Hz, first-order Butterworth bandpass
2. derivative
3. squaring
4. 150 ms moving-window integration
5. 300 ms refractory period
6. threshold at 30% of the maximum integrated signal
7. symmetric 75 ms correction to the local maximum in the original signal

Frequency features use Welch PSD with `nperseg=min(256, len(window))`. QRS width is the average half-amplitude width in a 100 ms search region.

## Models

| Model | Purpose | Intended classes |
|---|---|---|
| Model 1: `shockable_classifier.onnx` | Primary shockable decision | `0 = non-shockable`, `1 = shockable` |
| Model 2: `arrhythmia_multiclass.onnx` | Informational rhythm classification; does not gate Model 1 | `0 = Asystole`, `1 = Normal`, `2 = Tachycardia`, `3 = VF`, `4 = VT` |
| Model 3: `vf_vt_subtype_classifier.onnx` | VF/VT subtype, run only when Model 1 is shockable | `0 = VT`, `1 = VF` |

The intended execution order is Model 1, Model 2, and then Model 3 only when Model 1 predicts shockable.

## Verified Training Contract

The downloaded notebook's latest export code uses these 10 features and does not use `rr_entropy`:

```text
mean, std, ptp, zero_crossings, peak_count, mean_rr, rr_cv,
qrs_width, dominant_freq, vf_band_power_ratio
```

Direct ONNX graph inspection confirms that all three existing models expect `[batch, 10]`:

```text
shockable_classifier.onnx       input [?, 10], probabilities [?, 2]
arrhythmia_multiclass.onnx       input [?, 10], probabilities [?, 5]
vf_vt_subtype_classifier.onnx    input [?, 10], probabilities [?, 2]
```

The deployment code emits and validates this verified 10-feature contract. `rr_entropy` is intentionally absent from the current deployment vector because the existing models were trained without it.

## Layout

```text
models/
  shockable_classifier.onnx
  arrhythmia_multiclass.onnx
  vf_vt_subtype_classifier.onnx
src/
  preprocessing.py
  features.py
  inference.py
  main.py
tests/
  smoke_test.py
requirements.txt
```

## Installation

Use Python 3.12 for deployment. It is supported by the pinned ONNX Runtime version and is available for Raspberry Pi OS deployments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Smoke Test

The smoke test generates one 1,800-sample signal, extracts the 10-feature vector, loads all three models, and runs actual inference:

```bash
python tests/smoke_test.py
```

It prints each model's input name, output names and shapes, predicted label, and probabilities. The validated local run used Python 3.12.14 with `onnxruntime==1.20.1`.

## Command-Line Inference

Run a `.npy` file or a whitespace-delimited text file containing exactly 1,800 samples:

```bash
python -m src.main signal.npy --models models
```
