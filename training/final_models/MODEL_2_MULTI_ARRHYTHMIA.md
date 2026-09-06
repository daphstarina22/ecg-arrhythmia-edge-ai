# Model 2: Multi-Arrhythmia Classifier Specification

**Date**: September 6, 2026  
**Status**: Specification & Experimental Design  
**Task**: Multi-Class Rhythm Classification  
**Target Platform**: Edge Devices (Raspberry Pi 4 / Embedded Linux, CPU Execution)  

---

## 1. Objectives & Clinical Rationale

The objective of **Model 2** is to categorize ECG rhythms into actionable clinical diagnoses across both non-shockable and ventricular arrhythmias.

In the original notebook, Model 2 suffered from two severe flaws:
1. **Source Entanglement**: `Tachycardia` ($N=15,210$) came 100% from Challenge 2015, while `Normal` ($N=31,933$) came 0% from Challenge 2015. The model separated them using uncalibrated voltage scaling.
2. **VF Precision Collapse**: Model 2's standalone VF precision dropped to **36.0%** (64% false alarms) due to insufficient independent VF records ($N=16$) and lack of specialized chaos features.

Model 2 addresses these issues by enforcing **physical millivolt calibration** across all sources and evaluating two well-grounded class taxonomy configurations.

---

## 2. Empirical Class Taxonomy Formulation

Based on the multi-database audit documented in `DATA_AUDIT.md`, the candidate taxonomies are structured as follows:

### Primary Taxonomy: 4-Class Consolidated Rhythm System (Recommended)

| ML Class Index | Class Name | Included Rhythms & Annotations | Contributing Databases | Independent Records ($N$) | Window Count ($N$) | Clinical Action |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **0** | **Normal Sinus** (`NSR`) | `(N`, `(NSR` | MIT-BIH, VFDB, CUDB | **61 records** | $>31,000$ | Routine monitoring |
| **1** | **Supraventricular Tachycardia** (`TACHY`) | `Tachycardia` (true), `(SVTA` | Challenge 2015, MIT-BIH, VFDB | **141 records** | $>15,000$ | Vagal maneuvers / Antiarrhythmics |
| **2** | **Bradycardia & Asystole** (`BRADY_ASY`) | `Bradycardia` (true), `Asystole` (true), `(ASYS`, `(SBR` | Challenge 2015, VFDB, MIT-BIH | **75 records** | $>10,000$ | CPR / Atropine / Pacing |
| **3** | **Ventricular Arrhythmia** (`VENTRICULAR`) | `Ventricular_Tachycardia` (true), `Ventricular_Flutter_Fib` (true), `(VT`, `(VF`, `(VFL` | Challenge 2015, VFDB, CUDB, MIT-BIH | **120 records** | $14,704$ | Immediate Triage / Route to Model 3 |

> [!NOTE]
> **Clinical & Architectural Justification**:
> Merging VT, VF, and VFL into a unified `Ventricular Arrhythmia` class ($N=120$ records) provides substantial statistical power and eliminates the 17-record VF sample constraint in Model 2. Model 2 reliably identifies that the patient is in ventricular distress, while **Model 3** (the dedicated shockable subtype classifier) performs the definitive VT vs. VF decision.

### Secondary Taxonomy: 5-Class Granular System (Evaluated in Parallel)

| ML Class Index | Class Name | Records ($N$) | Limitation / Flag |
| :---: | :--- | :---: | :--- |
| **0** | `Normal Sinus Rhythm` | 61 | Robust |
| **1** | `Tachycardia / SVT` | 141 | Robust |
| **2** | `Bradycardia / Asystole` | 75 | Robust |
| **3** | `Ventricular Tachycardia` (VT) | 103 | Robust |
| **4** | `Ventricular Fibrillation / Flutter` (VF/VFL) | **17** | **FLAGGED: High variance expected due to small patient sample size ($N=17$)** |

---

## 3. Feature Pipeline

Model 2 utilizes a balanced 12-dimensional feature set combining rhythm timing, spectral characteristics, and amplitude dynamics:

| Index | Feature Name | Description | Purpose |
| :---: | :--- | :--- | :--- |
| **0** | `mean_rr` | Average RR interval from Pan-Tompkins (s) | Heart rate calculation (Tachycardia vs. Bradycardia) |
| **1** | `rr_cv` | Coefficient of variation of RR intervals | Rhythm regularity (Sinus vs. Arrhythmia) |
| **2** | `qrs_width` | Average duration of detected QRS complexes (s) | Narrow-complex (SVT) vs. Broad-complex (VT) |
| **3** | `peak_count` | Number of detected R-peaks in 5 seconds | Rate confirmation |
| **4** | `dominant_freq` | Peak spectral frequency from Welch PSD (Hz) | Fundamental oscillatory frequency |
| **5** | `vf_band_power_ratio` | Power in 3–9 Hz band divided by total power | Ventricular vs. Supraventricular power distribution |
| **6** | `spectral_entropy` | Shannon entropy of power spectrum | Rhythm complexity / randomness |
| **7** | `zero_crossings` | Zero-baseline crossing count in 5 seconds | Waveform frequency |
| **8** | `hjorth_mobility` | Mean frequency estimator via first derivative | Signal variance progression |
| **9** | `hjorth_complexity` | Spectral bandwidth estimator via second derivative | Waveform shape change |
| **10** | `calibrated_ptp` | Peak-to-peak amplitude in physical $mV$ | Identifies Asystole ($<0.1\text{ mV}$) |
| **11** | `calibrated_std` | Signal standard deviation in physical $mV$ | Total signal energy |

---

## 4. Candidate Model Architectures

1. **Multinomial Logistic Regression (L2 Penalty)**:
   - Softmax multi-class output with `StandardScaler`.
2. **Balanced Random Forest**:
   - 100 trees, `max_depth=7`, `class_weight='balanced'`.
3. **Histogram Gradient Boosting (`HistGradientBoostingClassifier`)**:
   - 100 iterations, `max_depth=5`, `learning_rate=0.05`, `class_weight='balanced'`.

---

## 5. Evaluation & Reporting Protocol

- **Grouping**: Strict 5-fold `GroupKFold` grouped by `record_id`.
- **Metrics**:
  - Global Accuracy and Balanced Accuracy
  - Macro F1 and Weighted F1
  - Per-Class Precision, Recall, and F1 Score (for every class)
  - Multi-class Confusion Matrix (normalized by true support)
  - Record-Level Aggregated Metrics
- **Limitation Auditing**: If class 4 (VF/VFL) in the 5-class model exhibits low precision, the report will explicitly document this limitation and compare it against the consolidated 4-class configuration.
