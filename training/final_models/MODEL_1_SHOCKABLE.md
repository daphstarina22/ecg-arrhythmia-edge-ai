# Model 1: Shockable Rhythm Classifier Specification

**Date**: September 6, 2026  
**Status**: Specification & Experimental Design  
**Task**: Binary Classification — Shockable vs. Non-Shockable ECG Rhythms  
**Target Platform**: Edge Devices (Raspberry Pi 4 / Embedded Linux, CPU Execution)  

---

## 1. Clinical Objective & Target Contract

The primary objective of **Model 1** is to serve as the critical first-tier decision gate for automated external defibrillators (AEDs) and wearable monitoring systems:

$$\text{"Is this ECG rhythm shockable?"}$$

### Clinical Targets:
- **Shockable (Class 1)**: Ventricular Fibrillation (`VF`), Ventricular Flutter (`VFL`), Sustained Ventricular Tachycardia (`VT`).
  - *Clinical Mandate*: High sensitivity is paramount ($\text{Recall} \ge 95\%$). Failure to recognize a shockable rhythm results in untreated cardiac arrest and patient demise.
- **Non-Shockable (Class 0)**: Normal Sinus Rhythm (`NSR`), Supraventricular Tachycardia (`SVT`), Sinus Tachycardia, Sinus Bradycardia, Asystole, Atrial Fibrillation/Flutter, Ectopic Beats.
  - *Clinical Mandate*: High specificity ($\text{Specificity} \ge 90\%$) to prevent inappropriate shocks delivered to conscious patients in supraventricular rhythms.

---

## 2. Eliminating Historic Domain Confounding

The original notebook model achieved an artificial 99% accuracy because **100% of its non-shockable windows originated from MIT-BIH**, while **100% of its shockable windows originated from VFDB and CUDB**. The classifier memorized hardware-specific noise and digitizer gain differences.

Model 1 resolves this issue through three strict design controls:

1. **Balanced Multi-Source Corpus**:
   - Shockable class includes 95 Challenge 2015 records, 22 VFDB records, 3 CUDB records, and 3 MIT-BIH records (120 total independent records).
   - Non-shockable class includes 199 Challenge 2015 records (Tachycardia, Bradycardia, Asystole), 15 VFDB records, 4 CUDB records, and 48 MIT-BIH records ($>250$ total independent records).
2. **Physical Millivolt Calibration**:
   - All raw digital integers are converted to calibrated physical millivolts ($mV$) using official record header metadata before feature extraction.
3. **QRS-Independent Feature Selection**:
   - Because Pan-Tompkins QRS peak detection fails catastrophically during Ventricular Fibrillation (which lacks discrete QRS complexes), Model 1 excludes QRS-dependent interval features (`mean_rr`, `rr_cv`, `qrs_width`) for shockable detection and relies on continuous spectral and chaos dynamics.

---

## 3. Feature Representation (10 Edge-Friendly Features)

Model 1 utilizes a compact 10-dimensional feature vector extracted from 5-second windows (1800 samples at 360 Hz):

| Index | Feature Name | Computation Domain | Physiological Rationale for Shockable Discrimination |
| :---: | :--- | :---: | :--- |
| **0** | `vf_band_power_ratio` | Frequency (FFT / Welch) | Ratio of energy in lethal fibrillatory band (3–9 Hz) vs. total band (0.5–30 Hz). High in VF/VT. |
| **1** | `dominant_freq` | Frequency (Welch PSD) | Frequency of maximum spectral peak. In VT/VF, typically 3–8 Hz; in NSR, corresponds to heart rate (1–2 Hz). |
| **2** | `spectral_entropy` | Information Theory | Measure of spectral flatness/disorganization. Normalized Shannon entropy over power spectrum. |
| **3** | `spectral_peak_purity` | Frequency (Peak Prominence) | Ratio of dominant peak power to surrounding spectral noise. High in monomorphic VT, low in chaotic VF. |
| **4** | `zero_crossings` | Time Domain | Count of zero-baseline crossings in 5 seconds. Captures oscillatory frequency without peak detection. |
| **5** | `hjorth_mobility` | Time / Differential | Square root of the variance of the first derivative divided by variance of amplitude. |
| **6** | `hjorth_complexity` | Time / Differential | Ratio of mobility of first derivative to mobility of signal. Reflects waveform irregularity. |
| **7** | `lz_complexity` | Nonlinear Dynamics | Lempel-Ziv binary sequence complexity (downsampled to 90 Hz, $N=450$). Measures pattern randomness. |
| **8** | `calibrated_ptp` | Amplitude (Physical $mV$) | Peak-to-peak amplitude in millivolts. Distinguishes asystole ($<0.1\text{ mV}$) from coarse VF/VT. |
| **9** | `calibrated_std` | Amplitude (Physical $mV$) | Signal standard deviation in millivolts. Calibrated energy metric. |

---

## 4. Candidate Model Architectures

To ensure robust generalization and low edge inference latency ($<10\text{ ms}$ on Raspberry Pi CPU):

1. **Regularized Logistic Regression (ElasticNet / L2)**:
   - Linear baseline with `StandardScaler` inside a strict `Pipeline`.
   - Highly interpretable coefficients and monotonic probability outputs.
2. **Balanced Random Forest**:
   - 100 trees, `max_depth=6`, `class_weight='balanced'`.
   - Non-linear decision boundaries with built-in resistance to outliers.
3. **Histogram-based Gradient Boosting (`HistGradientBoostingClassifier`)**:
   - Modern, lightweight tree-boosting architecture.
   - `max_iter=100`, `max_depth=4`, `learning_rate=0.05`, `class_weight='balanced'`.

---

## 5. Evaluation Protocol

### A. Primary Evaluation: 5-Fold Stratified Group Cross-Validation
- Stratified by class while grouping strictly by `record_id`.
- **Zero Leakage**: No window from any patient record can exist in both training and test folds.

### B. Secondary Evaluation: Cross-Database Generalization
- **Split D1**: Train on Challenge 2015 $\implies$ Test on VFDB + CUDB + MIT-BIH.
- **Split D2**: Train on VFDB + CUDB + MIT-BIH $\implies$ Test on Challenge 2015.

### C. Required Metrics
- Overall Accuracy and Balanced Accuracy
- Macro F1 and Weighted F1
- Shockable Precision, Recall (Sensitivity), and F1 Score
- Non-Shockable Specificity
- Area Under the ROC Curve (ROC-AUC)
- Confusion Matrix (window-level and record-level aggregated)
- Per-Record Performance Audit (flagging any records with $<80\%$ accuracy)
