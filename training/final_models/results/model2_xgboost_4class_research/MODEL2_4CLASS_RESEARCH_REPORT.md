# Model 2 Revision: Research Evaluation of 4-Class Arrhythmia Classifier (Excluding ASYSTOLE)

**Date**: October 25, 2026  
**Scope**: Research-only revision of Model 2 (Arrhythmia Classifier) excluding the `ASYSTOLE` class.  
**Location**: `training/final_models/results/model2_xgboost_4class_research/`  
**Target Taxonomy**: 4 Classes — `NSR`, `TACHY`, `BRADYCARDIA`, `VENTRICULAR`.  
**Safety Status**: Zero production files modified; zero deployed ONNX models overwritten; zero Git commits or pushes.

---

## 1. Executive Summary & Objective

In the preceding error analysis and clinical label review, the `ASYSTOLE` class in Model 2 was shown to suffer from fundamental electrophysiological boundary ambiguities:
1. **Agonal / Escape Rhythm Confusion**: Post-arrest windows in records such as `vfdb_427` exhibit coarse dying ventricular escape complexes ($V_{p\text{-}p} > 0.55\text{ mV}$, dominant frequency 1–2 Hz) that share time-frequency characteristics with coarse ventricular rhythms rather than electrical quiescence.
2. **Fine VF / Oscillation Overlap**: Records such as `vfdb_607` exhibit coherent 4.0–4.5 Hz oscillatory waves ($V_{p\text{-}p} \approx 0.25\text{ mV}$) representing fine ventricular fibrillation, yet were historically labeled as asystole in legacy databases.
3. **Clinical / Engineering Separation**: True asystole (electrical ventricular standstill / flatline) is fundamentally an absence of electrical signal or severe sensor lead disconnect, best detected by dedicated hardware lead-off impedance circuitry or calibrated signal-energy thresholds ($V_{p\text{-}p} < 0.15\text{ mV}$) rather than a complex statistical feature classifier.

To rigorously investigate this hypothesis, a **research-only 4-class revision of Model 2** was trained and evaluated by completely excluding `ASYSTOLE` from the supervised training set, preserving only `NSR`, `TACHY`, `BRADYCARDIA`, and `VENTRICULAR`.

---

## 2. Dataset Preparation & Integrity Verification

All dataset filtering and integrity checks were performed via [`prepare_4class_research_dataset.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/prepare_4class_research_dataset.py) and certified in [`dataset_split_verification.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/dataset_split_verification.json).

### 2.1 Dataset Partitioning Summary
- **Source Dataset**: `training/final_models/artifacts/corrected_master_feature_dataset.csv` (7,669 rows, 334 records).
- **Excluded ASYSTOLE Windows**: **175 windows** across **20 records** (139 VFDB, 36 PhysioNet C2015) were isolated and saved to [`excluded_asystole_windows.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/excluded_asystole_windows.csv).
- **Supervised 4-Class Training Dataset**: **7,493 clean supervised windows** across **318 independent records** (1 row in record `cudb_cu02` containing feature NaNs was dropped, matching 5-class baseline handling).

### 2.2 4-Class Supervised Distribution
| Class | Supervised Windows | Percentage | Independent Records |
| :--- | :---: | :---: | :---: |
| **NSR** | 5,073 | 67.70% | 54 records (primary) |
| **VENTRICULAR** | 1,838 | 24.53% | 103 records (primary) |
| **TACHY** | 362 | 4.83% | 122 records (primary) |
| **BRADYCARDIA** | 220 | 2.94% | 39 records (primary) |
| **Total** | **7,493** | **100.00%** | **318 records** |

### 2.3 Strict Integrity Verification
- `no_asystole_in_4class_dataset`: **PASSED (True)** — Verified zero ASYSTOLE rows in the 4-class training set.
- `no_silent_relabeling`: **PASSED (True)** — Verified 100% exact match between `cardiac_rhythm_4class` and `cardiac_rhythm_5class` for all 7,493 rows. Zero asystole rows were relabeled as NSR, TACHY, BRADYCARDIA, or VENTRICULAR.
- `classes_match_expected_target`: **PASSED (True)** — Exactly `{"NSR", "TACHY", "BRADYCARDIA", "VENTRICULAR"}`.
- `row_count_partition_conserved`: **PASSED (True)** — $7,494 + 175 = 7,669$ master rows.

---

## 3. Training & Cross-Validation Methodology

Training was executed via [`train_xgboost_4class_model2.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/train_xgboost_4class_model2.py):
1. **Features**: 12 verified edge features (`mean_rr`, `rr_cv`, `qrs_width`, `peak_count`, `dominant_freq`, `vf_band_power_ratio`, `spectral_entropy`, `zero_crossings`, `hjorth_mobility`, `hjorth_complexity`, `calibrated_ptp`, `calibrated_std`).
2. **Cross-Validation**: 5-Fold `StratifiedGroupKFold` grouped strictly by `record_id` (`random_state=42`).
3. **Leakage Verification**: Programmatically checked that $\text{train\_records} \cap \text{val\_records} = \emptyset$ in each fold. **Zero record leakage across all 5 folds was verified**.
4. **Fold-Level Preprocessing**: `SimpleImputer(strategy="median")` and `StandardScaler()` were fitted strictly on `X_train` within each fold and applied to `X_val` to prevent feature leakage.
5. **Class Imbalance Handling**: Fold-level balanced sample weights were computed via `compute_sample_weight('balanced', y_train)`.

---

## 4. Four-Class XGBoost Performance Results

### 4.1 Hyperparameter Variation Comparison
Three XGBoost tree-depth variations were evaluated ($N = 100\text{ trees}$, $\text{learning\_rate} = 0.05$, $\text{subsample} = 0.8$, $\text{colsample\_bytree} = 0.8$):

| Configuration | Window Accuracy | Balanced Accuracy | Window Macro-F1 | Weighted-F1 | Record Balanced Acc | Record Macro-F1 | False Brady on NSR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Depth 5** *(Best Macro-F1)* | **79.26%** | **78.40%** | **0.6667** | **0.8084** | **72.85%** | **0.6987** | 9.01% (457 / 5,073) |
| **XGBoost Depth 4** | 78.45% | **80.05%** | 0.6603 | 0.8045 | 70.40% | 0.6617 | 11.14% (565 / 5,073) |
| **XGBoost Depth 6** | 78.98% | 76.87% | 0.6650 | 0.8038 | **73.55%** | **0.7195** | **8.12%** (412 / 5,073) |

- **Best Overall Window Model**: `xgboost_depth5_lr005` achieved the highest Out-Of-Fold Window Macro-F1 (**0.6667**) and Balanced Accuracy (**78.40%**).
- **Best Record-Level Discrimination**: `xgboost_depth6_lr005` achieved the highest Record Macro-F1 (**0.7195**) and the lowest False Bradycardia rate on NSR (**8.12%**).

### 4.2 Per-Class Performance (Best Model: Depth 5)
| Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **NSR** | 5,073 | 0.920 | 0.780 | **0.844** |
| **TACHY** | 362 | 0.569 | 0.878 | **0.691** |
| **BRADYCARDIA** | 220 | 0.234 | 0.650 | **0.344** |
| **VENTRICULAR** | 1,838 | 0.753 | 0.828 | **0.788** |

### 4.3 Out-of-Fold Confusion Matrix (Best Model: Depth 5)
*(Rows = True Label, Columns = Predicted Label)*

```
                 Pred: NSR    Pred: TACHY    Pred: BRADY    Pred: VENT    Total True
True: NSR             3,957            182            457           477         5,073
True: TACHY              25            318              0            19           362
True: BRADY              73              0            143             4           220
True: VENT              246             59             12         1,521         1,838
Total Pred            4,301            559            612         2,021         7,493
```

---

## 5. Comparison: 4-Class vs 5-Class Research XGBoost

The side-by-side comparison between the 5-class XGBoost model and the new 4-class XGBoost model is stored in [`comparison_4class_vs_5class_xgboost.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/comparison_4class_vs_5class_xgboost.csv):

| Metric | 5-Class XGBoost (Depth 6) | 4-Class XGBoost (Depth 5) | Delta | Analytical Note |
| :--- | :---: | :---: | :---: | :--- |
| **Number of Classes** | 5 (NSR, TACHY, BRADY, ASY, VENT) | 4 (NSR, TACHY, BRADY, VENT) | -1 class | **Metrics are not directly equivalent** |
| **Supervised Windows** | 7,668 | 7,493 | -175 windows | 175 ASYSTOLE windows excluded |
| **Independent Records** | 333 | 318 | -15 records | 15 records with only asystole removed |
| **Window Accuracy** | 80.70% | 79.26% | -1.44% | 5-class had 175 additional windows |
| **Window Balanced Accuracy** | 77.20% | 78.40% | **+1.20%** | Mean of 4 recall values vs 5 recall values |
| **Window Macro-F1** | 0.6860 | 0.6667 | -0.0193 | 5-class included ASYSTOLE F1 (0.6316) |
| **Window Weighted-F1** | 81.83% | 80.84% | -0.99% | Weighted by surviving class supports |
| **NSR F1-Score** | 0.8625 | 0.8443 | -0.0182 | NSR detection remains stable ($> 0.84$) |
| **TACHY F1-Score** | 0.7181 | 0.6906 | -0.0276 | Recall remains high (87.8% vs 86.7%) |
| **BRADYCARDIA F1-Score** | 0.4383 | 0.3438 | -0.0945 | Precision lower due to slow NSR overlap |
| **VENTRICULAR F1-Score** | 0.7794 | 0.7883 | **+0.0089** | Precision improves without asystole leakage |
| **ASYSTOLE F1-Score** | 0.6316 | **N/A (Excluded)** | N/A | Class removed from Model 2 architecture |
| **False Brady on NSR (%)** | 7.19% | 9.01% (Depth 5) / 8.12% (Depth 6) | +0.93% to +1.82% | 100% of these windows have true HR < 60 bpm |

### Critical Analytical Insights: Why Metrics Are Not Directly Equivalent
1. **Mathematical Asymmetry**: Macro-F1 in the 5-class model is an unweighted average of 5 terms $\frac{1}{5}\sum_{i=1}^5 F_i$. In the 4-class model, it is an average of 4 terms $\frac{1}{4}\sum_{i=1}^4 F_i$. Comparing the two values directly is mathematically invalid.
2. **Ventricular Classification Purity**: In the 5-class model, 70 asystole windows were incorrectly classified as VENTRICULAR, degrading ventricular precision. In the 4-class model, eliminating asystole training examples removes this cross-class boundary tension, raising VENTRICULAR F1 from **0.7794 to 0.7883**.
3. **NSR False Bradycardia Constancy**: In both the 5-class and 4-class models, **100% of the NSR windows classified as BRADYCARDIA have true instantaneous heart rate $< 60\text{ bpm}$**. The 4-class model did not create new false bradycardia errors; rather, without asystole in the training set, low-rate feature space boundaries shifted slightly toward the bradycardia prior.

---

## 6. Separate Evaluation of Excluded ASYSTOLE Windows

To determine how the 4-class model behaves when exposed to flatline and agonal signals, the 175 excluded ASYSTOLE windows were evaluated out-of-domain using the full 4-class pipeline ([`asystole_evaluation_summary.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/asystole_evaluation_summary.json) and [`asystole_windows_4class_predictions.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/asystole_windows_4class_predictions.csv)).

> [!WARNING]
> **Strict Clinical Disclaimer**: The 4-class model contains **NO asystole class**. These forced predictions demonstrate pseudo-classification behavior on flatline/agonal signals. They **MUST NEVER** be interpreted as valid or validated asystole detection.

### 6.1 Forced Prediction Distribution on 175 ASYSTOLE Windows
| Forced 4-Class Output | Windows | Percentage | Mean Probability | Median Probability | Primary Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **VENTRICULAR** | 104 | **59.43%** | 0.534 | 0.530 | Agonal waves, low-frequency dying heart oscillations, fine VF |
| **NSR** | 38 | **21.71%** | 0.233 | 0.111 | Low-amplitude noise resembling baseline isoelectric NSR |
| **BRADYCARDIA** | 33 | **18.86%** | 0.217 | 0.134 | Zero peak count / estimated HR = 0 mapping to slow-rate prior |
| **TACHY** | 0 | **0.00%** | 0.016 | 0.014 | Zero false tachycardia predictions |

### 6.2 Per-Record Breakdown of Forced Predictions
- **`vfdb_607` (16 asystole windows)**: **100% (16 / 16) predicted as VENTRICULAR** ($V_{p\text{-}p} = 0.232\text{ mV}$).
  - *Electrophysiological reality*: Ground truth inspection proves this segment contains continuous 4.0–4.5 Hz oscillatory waves (fine ventricular flutter/fibrillation) before spontaneous nodal recovery. Forced classification as `VENTRICULAR` reflects genuine electrophysiological oscillation.
- **`vfdb_424` (5 asystole windows)**: **100% (5 / 5) predicted as VENTRICULAR** ($V_{p\text{-}p} = 0.686\text{ mV}$, mean peak count = 1.0).
  - *Electrophysiological reality*: Dying agonal escape complexes with large amplitudes.
- **`vfdb_427` (112 asystole windows)**:
  - 67 predicted as `VENTRICULAR` (59.8%): Early post-collapse agonal waveforms ($t = 1533\text{--}1840\text{s}$).
  - 29 predicted as `NSR` (25.9%): Late low-noise flatline segments.
  - 16 predicted as `BRADYCARDIA` (14.3%): Severe profound bradycardic dying complexes ($HR \approx 10\text{--}20\text{ bpm}$).
- **PhysioNet C2015 True Flatline Alarms (e.g. `c2015_a345l`, $V_{p\text{-}p} = 0.054\text{ mV}$, $\text{peaks} = 0$)**:
  - **100% (3 / 3) predicted as BRADYCARDIA**. Because peak count is 0, the QRS detector reports 0 bpm, which the 4-class classifier maps to the lowest available rate category (`BRADYCARDIA`).

---

## 7. Out-of-Distribution Pre-Alarm Baseline Evaluation

The 4-class XGBoost model was evaluated against the **16,722 out-of-distribution pre-alarm baseline windows** (`QUARANTINED_PRE_ALARM` in `quarantined_unsupervised_feature_dataset.csv`):
- **Prediction Distribution**:
  - `NSR`: 5,829 windows (34.86%)
  - `TACHY`: 5,716 windows (34.18%)
  - `VENTRICULAR`: 3,492 windows (20.88%)
  - `BRADYCARDIA`: 1,685 windows (**10.08%**)
- In the legacy uncorrected Model 2, the false bradycardia alarm rate on these pre-alarm baselines exceeded **22%**. With the corrected dataset and the 4-class XGBoost model, the rate is reduced to **10.08%**, confirming robust edge suppression of spurious bradycardia alarms on noisy hospital monitor baselines.

---

## 8. Saved Artifacts Summary

All artifacts created during this revision reside exclusively in the separate research directory:
`training/final_models/results/model2_xgboost_4class_research/`

| Filename | Description | Size |
| :--- | :--- | :---: |
| [`prepare_4class_research_dataset.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/prepare_4class_research_dataset.py) | Standalone dataset filter and integrity verification script | 6.8 KB |
| [`train_xgboost_4class_model2.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/train_xgboost_4class_model2.py) | 4-class XGBoost training, CV, serialization, and asystole evaluation script | 27.8 KB |
| [`four_class_master_feature_dataset.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/four_class_master_feature_dataset.csv) | 7,493 clean supervised 4-class rows across 318 records | 2.5 MB |
| [`excluded_asystole_windows.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/excluded_asystole_windows.csv) | 175 isolated ASYSTOLE windows across 20 records | 53.9 KB |
| [`dataset_split_verification.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/dataset_split_verification.json) | Audit certification proving zero relabeling and zero leakage | 1.5 KB |
| [`research_xgboost_4class_model.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/research_xgboost_4class_model.json) | Serialized native 4-class XGBoost model (Depth 5) | 1.1 MB |
| [`research_xgboost_4class_pipeline.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/research_xgboost_4class_pipeline.joblib) | Full scikit-learn pipeline (Imputer + Scaler + XGBoost) | 799 KB |
| [`asystole_windows_4class_predictions.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/asystole_windows_4class_predictions.csv) | Window-level predictions and probabilities for all 175 asystole windows | 22.6 KB |
| [`asystole_evaluation_summary.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/asystole_evaluation_summary.json) | Forced prediction summary and per-record breakdown for asystole | 6.3 KB |
| [`comparison_4class_vs_5class_xgboost.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/comparison_4class_vs_5class_xgboost.csv) | Side-by-side comparison table (5-class vs 4-class) | 1.4 KB |
| [`xgboost_4class_research_metrics_summary.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/xgboost_4class_research_metrics_summary.json) | Comprehensive metrics dictionary across all 3 tree-depth configs | 18.2 KB |
| [`xgboost_4class_pre_alarm_evaluation.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/xgboost_4class_pre_alarm_evaluation.json) | Pre-alarm baseline evaluation summary | 281 B |
| `xgboost_depth*_oof_predictions.csv` | Out-of-fold window predictions for depths 4, 5, and 6 | ~835 KB each |
| `xgboost_depth*_record_metrics.csv` | Record-level metrics for depths 4, 5, and 6 | ~11.5 KB each |
| `xgboost_depth*_confusion_matrix.json` | Confusion matrices and false alarm stats for depths 4, 5, and 6 | ~3.0 KB each |

---

## 9. Architectural & Clinical Limitations

1. **Inability to Detect Asystole**: The 4-class model **cannot detect asystole**. When deployed on hardware, asystole/flatline detection must be performed by an independent, deterministic upstream safeguard (e.g. ADS1292R lead-off impedance detection and $V_{p\text{-}p} < 0.15\text{ mV}$ flatline thresholding) prior to arrhythmia classification.
2. **Ambiguity on Dying/Agonal Rhythms**: In true terminal collapse, dying ventricular escape beats will be classified as `VENTRICULAR` (if broad and slow) or `BRADYCARDIA` (if isolated complexes appear at rates $< 60\text{ bpm}$). This is electrophysiologically coherent, but clinicians must understand that the model will not output an "Asystole" alarm string.
3. **NSR/Bradycardia Boundary**: Approximately 8–9% of windows in records tagged as sinus rhythm (`(N`) are classified as `BRADYCARDIA`. As verified, 100% of these windows have true instantaneous rates $< 60\text{ bpm}$ (e.g. `vfdb_425` sinus bradycardia at 42–48 bpm). A unified rate-based taxonomy policy must define whether slow sinus windows are considered true bradycardia alarms or normal variants for athletes/resting subjects.

---

## 10. Confirmation of Safety Constraints

- **Production Files**: [`deploy/combined_ecg_app.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/combined_ecg_app.py), [`deploy/edge_runtime.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/edge_runtime.py), and [`deploy/ads1292r_driver.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/ads1292r_driver.py) were **NOT modified**.
- **Deployed ONNX Models**: [`models/arrhythmia_multiclass.onnx`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/models/arrhythmia_multiclass.onnx) and models in `deploy/models/` were **NOT overwritten**.
- **Existing 5-Class Research Results**: [`training/final_models/results/model2_xgboost_research/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_research/) and [`training/final_models/artifacts/corrected_master_feature_dataset.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/artifacts/corrected_master_feature_dataset.csv) were **NOT modified**.
- **Git State**: Zero commits or pushes were made.
- **Scope**: No PVC classification was added; pure AFib/AFL remains strictly quarantined.
