# Audit Report: Model 1 (Shockable vs. Non-Shockable Classifier)

**Model Artifact**: `models/shockable_classifier.onnx`  
**Training Source**: `notebook346a8689af (1).ipynb` (Cell 102, lines 624–695)  
**Evaluator Status**: Strict Read-Only Audit  

---

## 1. Executive Summary

Model 1 is the primary emergency decision gate in the edge pipeline, intended to classify an incoming 5-second ECG window as `0 = non_shockable` or `1 = shockable` (Ventricular Fibrillation or Ventricular Tachycardia).

While the training notebook reported an impressive **99.0% accuracy** and **0.9997 ROC-AUC**, this audit reveals that this performance is **synthetically inflated by extreme dataset confounding**. Specifically, the negative class was drawn 100% from one database (MIT-BIH), while the positive class was drawn 100% from two other databases (VFDB and CUDB). Non-shockable rhythms from VFDB and CUDB were deliberately omitted, and Challenge 2015 was excluded because testing on it caused precision to collapse to 38%.

---

## 2. Dataset Composition Audit

### A. Window and Class Distribution
- **Total Windows**: 35,971
- **Class 0 (Non-Shockable)**: 32,773 windows (**91.11%**)
- **Class 1 (Shockable)**: 3,198 windows (**8.89%**)
- **Imbalance Ratio**: $10.25 : 1$ (Non-shockable to Shockable windows)

### B. Independent Record and Patient Representation
A critical distinction must be made between overlapping 5-second window slices ($N=35,971$) and true independent physical patients ($N=73$):

| Source Database | Assigned Class | Window Count | % of Class | Independent Records | Mean Windows / Record | Recording Context |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **MIT-BIH Arrhythmia (mitdb)** | Non-Shockable (`0`) | 32,773 | **100.0%** | **48 records** | 682.8 | 1970s Boston ambulatory Holter tapes |
| **MIT-BIH Malignant VF (vfdb)** | Shockable (`1`) | 3,060 | **95.68%** | **22 records** | 139.1 | 1980s Boston Holter tapes |
| **Creighton University (cudb)** | Shockable (`1`) | 138 | **4.32%** | **3 records** | 46.0 | Omaha, NE acute ICU monitor tapes |
| **Challenge 2015 (c2015)** | *Excluded* | 0 | 0.0% | 0 records | — | PhysioNet ICU multi-center alarms |
| **Total** | | **35,971** | **100%** | **73 records** | | |

### C. Major Contributing Records
- **Non-Shockable**: 48 records contribute an average of 682.8 windows each (e.g. `mitdb_100`, `mitdb_101`, `mitdb_105`).
- **Shockable**:
  - `vfdb_430`: 564 windows (17.6% of shockable cohort)
  - `vfdb_426`: 272 windows (8.5% of shockable cohort)
  - `vfdb_422`: 279 windows (8.7% of shockable cohort)
  - `cudb_cu01`, `cudb_cu02`, `cudb_cu15`: contribute only 138 total windows across 3 patients.

---

## 3. Leakage & Confounding Audit

### Leakage Verdict: **FAIL (Critical Confounder / Synthetic Shortcut)**

#### 1. 100% Co-Extensiveness Between Database Identity and Class Label
The training script implements the following selection logic (Cell 102, lines 636–660):
- Only genuine VF and VT segments from VFDB and CUDB are extracted into the positive class.
- All non-shockable segments (Normal, Asystole, peri-arrhythmic transitions) from VFDB and CUDB are **intentionally excluded**.
- The entire non-shockable training set is harvested exclusively from MIT-BIH (`build_mitdb_binary_nonshockable`).

**Consequence**:
$$\text{Class Label} \equiv \text{Database Source}$$
$$\text{Label} = 0 \iff \text{Source} = \text{MIT-BIH}$$
$$\text{Label} = 1 \iff \text{Source} \in \{\text{VFDB, CUDB}\}$$

The classifier was not trained to recognize the electrophysiological difference between sinus rhythm and ventricular fibrillation; it was trained to recognize the **digitization, lead placement, and noise characteristics of MIT-BIH tapes versus VFDB/CUDB tapes**.

#### 2. Amplitude Distribution Bias
The first three features in the 10-feature vector are raw signal amplitude statistics: `mean`, `std`, and `ptp`.
Cell 102 line 614 documents the following amplitude statistics:
- **MIT-BIH** (`mitdb`): `std` mean = **0.1750 mV**, `ptp` mean = **1.6075 mV**
- **VFDB** (`vfdb`): `std` mean = **0.4682 mV**, `ptp` mean = **2.1200 mV**
- **CUDB** (`cudb`): `std` mean = **0.4533 mV**, `ptp` mean = **2.2773 mV**

Because MIT-BIH ECG signals have an average standard deviation **2.7 times smaller** than VFDB/CUDB signals, a shallow decision tree using only `std > 0.30` or `ptp > 1.85` can separate MIT-BIH from VFDB with near-perfect accuracy without ever inspecting heart rate, spectral power, or QRS morphology.

#### 3. Reason for Challenge 2015 Exclusion
The notebook author explicitly noted in Cell 102 (lines 625–633):
> *"NOTE: deliberately VFDB + CUDB + mitdb ONLY, not Challenge2015. Challenge2015's ADC gain (CHALLENGE2015_GAIN=200) is UNVERIFIED... Including it here dropped precision from 99% (VFDB+mitdb only, proven) to 38%..."*

When exposed to an independent dataset (Challenge 2015), Model 1's precision collapsed from **99% down to 38%**, demonstrating that the model's apparent precision was an artifact of the clean, source-segregated sandbox.

---

## 4. Generalization Audit

- **Cross-Record Evaluation**: The notebook evaluated Model 1 with a single `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on `record_id`. Within this split, train and test records did not overlap.
- **Cross-Source Generalization**: **Completely Absent**. 
  - Train and test sets both contained 80% MIT-BIH (as negatives) and 80% VFDB/CUDB (as positives).
  - The model was never evaluated in a cross-source transfer setting (e.g. train on MITDB + VFDB, test on CUDB or Challenge 2015).
  - The moment an independent source (Challenge 2015) was introduced, the decision boundary collapsed.

---

## 5. Metric Audit

The following window-level metrics were reported in Cell 102 output:

| Class | Precision | Recall | F1-Score | Test Support (Windows) |
| :--- | :---: | :---: | :---: | :---: |
| **Non-Shockable (`0`)** | 1.00 | 1.00 | 1.00 | 7,210 |
| **Shockable (`1`)** | 0.98 | 0.98 | 0.98 | 1,050 |
| **Overall Accuracy** | — | — | **0.99** | **8,260** |
| **Macro Average** | 0.99 | 0.99 | 0.99 | 8,260 |
| **Weighted Average** | 0.99 | 0.99 | 0.99 | 8,260 |
| **ROC-AUC** | — | — | — | **0.9997** |

### Audit Commentary:
- **Metrics are 100% Window-Level**: 8,260 overlapping 5-second slices from a handful of records (~15 test records).
- **Zero Record-Level Metrics**: No patient-level accuracy or patient-level false alarm rate was computed.
- **Metric Validity**: Completely ungrounded due to the 100% source-label confound.

---

## 6. Model 1 Verdict

$$\text{\Huge \textbf{RETRAIN WITH CORRECTED DATASET & SPLIT}}$$

### Justification:
`shockable_classifier.onnx` cannot be deployed in any reliable edge system. It operates as a database discriminator rather than a clinical rhythm classifier.

### Mandatory Remediation Requirements:
1. **Source Balancing**: Non-shockable rhythms must be sourced from VFDB (Normal segments), CUDB (Normal segments), and Challenge 2015, not solely from MIT-BIH.
2. **Gain Normalization**: Features `mean`, `std`, and `ptp` must either be normalized per window (e.g. z-score or IQR standardization) or replaced by scale-invariant waveform features.
3. **Cross-Source Validation**: The retrained model must be evaluated by holding out entire databases (e.g., Leave-One-Database-Out or cross-source transfer D1/D2) before acceptance.
