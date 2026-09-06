# Audit Report: ONNX Deployment & Edge Contract

**Directory Inspected**: `models/` and `src/`  
**Deployment Target**: Raspberry Pi / Embedded Linux Inference  
**Evaluator Status**: Strict Read-Only Audit  

---

## 1. Executive Summary

This audit inspected the three deployed ONNX binary files in `models/` using pure protobuf graph inspection, verified their mathematical contract against `src/preprocessing.py`, `src/features.py`, `src/inference.py`, and `src/main.py`, and validated their traceability back to the training notebook.

All three deployed ONNX models expect a **10-dimensional `float32` feature vector** (`shape: ['?', 10]`) extracted from 5-second ECG windows. While the feature extraction dimensions align, the audit identified **four critical deployment deficiencies**:
1. **Missing Label Maps**: The ONNX models output integer class indices (`0, 1, 2, 3, 4`), but `src/inference.py` has no label dictionaries and outputs raw integer arrays.
2. **Ambiguous Class Index Zero**: In Model 1, `0 = Non-Shockable`. In Model 2, `0 = Asystole`. In Model 3, `0 = VT`. A caller receiving `0` cannot interpret it without referencing external documentation.
3. **Session Re-Creation on Every Window**: `src/inference.py` creates new `InferenceSession` instances on every 5-second window, repeatedly reloading 4.3 MB from disk.
4. **Disconnection from Phase 5–7 Findings**: The deployed Model 3 ONNX file (`vf_vt_subtype_classifier.onnx`) is the obsolete Phase 4 hand-crafted 10-feature XGBoost model (which has known D2 generalization failure), rather than the validated Phase 5–7 physiological complexity model.

---

## 2. Byte-Level ONNX Graph Inspection

| Parameter | Model 1: `shockable_classifier.onnx` | Model 2: `arrhythmia_multiclass.onnx` | Model 3: `vf_vt_subtype_classifier.onnx` |
| :--- | :--- | :--- | :--- |
| **File Size** | **409,920 bytes** (0.39 MB) | **3,750,881 bytes** (3.58 MB) | **165,923 bytes** (0.16 MB) |
| **IR Version** | 8 | 8 | 8 |
| **Producer** | OnnxMLTools 1.16.0 | OnnxMLTools 1.16.0 | OnnxMLTools 1.16.0 |
| **Primary OpType** | `TreeEnsembleClassifier` | `TreeEnsembleClassifier` | `TreeEnsembleClassifier` |
| **Input Tensor Name** | `'input'` | `'input'` | `'input'` |
| **Input Shape** | `['?', 10]` | `['?', 10]` | `['?', 10]` |
| **Input Element Type** | `FLOAT` (1) | `FLOAT` (1) | `FLOAT` (1) |
| **Output 1 (Label)** | name: `'label'`, shape: `['?']`, type: `INT64` (7) | name: `'label'`, shape: `['?']`, type: `INT64` (7) | name: `'label'`, shape: `['?']`, type: `INT64` (7) |
| **Output 2 (Probabilities)** | name: `'probabilities'`, shape: `['?', 2]`, type: `FLOAT` (1) | name: `'probabilities'`, shape: `['?', 5]`, type: `FLOAT` (1) | name: `'probabilities'`, shape: `['?', 2]`, type: `FLOAT` (1) |
| **Class Labels (`int64s`)** | `[0, 1]` | `[0, 1, 2, 3, 4]` | `[0, 1]` |
| **Trained Architecture** | XGBoost (300 trees, max_depth 5) | XGBoost (300 trees, max_depth 6) | XGBoost (200 trees, max_depth 4) |
| **MD5 Hash** | `bf2dce9c53fe11c340a1621d35756fed` | `5258de9135e33143a2f17e5211d9b5db` | `d2d244bf6dc1a7387cc7f9ddeb743cee` |
| **Provenance Match** | 100% match with `ecg_models.zip` | 100% match with `ecg_models.zip` | 100% match with `ecg_models.zip` |

---

## 3. Comparison with Repository Source Code (`src/`)

### A. Preprocessing (`src/preprocessing.py`)
- **Sampling Rate & Windowing**: Target frequency $360\text{ Hz}$, $5.0\text{ s}$ duration $\implies 1,800$ samples. This matches the training configuration in Cell 102.
- **Pan-Tompkins Implementation**: The code in `src/preprocessing.py` matches the notebook's final QRS detector:
  - 1st order Butterworth bandpass (5–15 Hz)
  - First difference and squaring
  - 150 ms moving average integration
  - 300 ms refractory period, 30% peak threshold
  - Symmetric 75 ms local maximum correction
- **Finding**: Preprocessing code is mathematically identical to the training notebook.

### B. Feature Extraction (`src/features.py`)
- **Feature Names and Order**:
  ```python
  FEATURE_NAMES = [
      "mean", "std", "ptp", "zero_crossings", "peak_count",
      "mean_rr", "rr_cv", "qrs_width", "dominant_freq", "vf_band_power_ratio"
  ]
  ```
- **Dimension Check**: Emits shape `(1, 10)` with `dtype=float32`.
- **Finding**: Exact feature order and shape contract is satisfied across all three models.

### C. Inference Logic (`src/inference.py`)
- **Class Contract Enforcement**:
  `OnnxClassifier` checks that the input dimension equals 10:
  ```python
  if len(shape) != 2 or shape[-1] != len(FEATURE_NAMES):
      raise ModelContractError(...)
  ```
- **Deficiency 1: Missing Class Label Interpretation**:
  The function `classify_window()` returns:
  ```python
  results = {
      "model_1": {"label": array([0]), "probabilities": array([[0.99, 0.01]])},
      "model_2": {"label": array([1]), "probabilities": array([[...]])},
      "model_3": None
  }
  ```
  Neither `src/inference.py` nor `src/main.py` provides human-readable translation:
  - `model_1` output `0` $\to$ Non-Shockable
  - `model_2` output `1` $\to$ Normal
  - `model_3` output `0` $\to$ Ventricular Tachycardia (VT)
  A downstream clinical API receives only bare integers with conflicting semantics.

- **Deficiency 2: Severe Runtime Session Waste**:
  In `src/inference.py` (lines 38–47):
  ```python
  def classify_window(features: np.ndarray, model_dir: Path) -> dict:
      model_1 = OnnxClassifier(model_dir / "shockable_classifier.onnx")
      model_2 = OnnxClassifier(model_dir / "arrhythmia_multiclass.onnx")
      ...
  ```
  Calling `OnnxClassifier(...)` inside `classify_window()` re-instantiates `ort.InferenceSession()` on **every window invocation**.
  - Total model size: **4.33 MB**.
  - On a Raspberry Pi 4/5, loading and parsing 4.3 MB from SD card storage takes **200–500 ms**, introducing unnecessary latency and memory thrashing.
  - Fix required: Models must be instantiated once at system startup and passed into `classify_window`.

### D. Model 3 Disconnect with Research Findings
- The deployed `vf_vt_subtype_classifier.onnx` is an XGBoost model trained on the 10 hand-crafted features.
- In Phase 4 and Phase 5, this model was proven to suffer from **Pan-Tompkins QRS breakdown on VF**, causing cross-database D2 ROC-AUC to drop to **0.6076**.
- The repository has already developed and validated the Phase 5/6/7 raw-waveform physiological feature pipeline (Sample Entropy, Lempel-Ziv complexity, autocorrelation decay), but the production deployment directory `models/` was intentionally left untouched per project governance rules.
- When the 3-model system is finalized, Model 3 must be upgraded from the obsolete 10-feature XGBoost model to the validated physiological representation.

---

## 4. Summary Table of Deployment Verification

| Check Item | Status | Finding | Required Action |
| :--- | :---: | :--- | :--- |
| **Input Shape Match** | **PASS** | All models expect `[batch, 10]`, `features.py` produces `(1, 10)` | None |
| **Data Type Match** | **PASS** | `features.py` casts to `float32`; models expect `float32` | None |
| **Sampling Rate Match** | **PASS** | Target 360 Hz used uniformly across all files | None |
| **Label Mapping** | **FAIL** | Zero label dictionaries in `inference.py` | Add dictionary mappings for all 3 models |
| **Session Lifecycle** | **WARNING** | Sessions re-instantiated on every window | Persist session objects at startup |
| **Model 3 Freshness** | **OBSOLETE** | Deployed ONNX is old Phase 4 hand-crafted model | Upgrade to Phase 5–7 physiological pipeline |
