# Audit Report: Model 2 (Multiclass Arrhythmia Classifier)

**Model Artifact**: `models/arrhythmia_multiclass.onnx`  
**Training Source**: `notebook346a8689af (1).ipynb` (Cell 102, lines 698–772)  
**Evaluator Status**: Strict Read-Only Audit  

---

## 1. Executive Summary

Model 2 is deployed as a 5-class XGBoost classifier intended to provide informational rhythm categorization on Raspberry Pi. It operates on the same 10 hand-crafted features extracted from 5-second ECG windows.

While the training notebook reported an overall window accuracy of **85.0%**, this audit identifies three critical systemic failures:
1. **Severe Class-Source Confounding**: `Tachycardia` ($N=15,210$) comes **100% from Challenge 2015**, while `Normal` ($N=31,933$) comes **0% from Challenge 2015**. Because Challenge 2015 was extracted at 8x–20x higher voltage amplitude due to an uncalibrated gain factor, the classifier separates Normal from Tachycardia primarily via signal voltage rather than heart rate or rhythm features.
2. **Extreme Class Imbalance & Critical Failure on VF**: The shockable class `VF` contains only 2,026 windows across only 16 independent records. Under test evaluation, Model 2's VF precision dropped to **36.0%** (almost two-thirds of VF alarms were false positives).
3. **Clinical Role Conflict**: Model 2 independently predicts VF and VT, competing with Model 1 and Model 3 and generating contradictory alarm states.

---

## 2. Exact Class Taxonomy & Source Mapping

The taxonomy implemented in Cell 102 comprises exactly 5 classes (AFIB, AFL, SVT, and Bradycardia were removed in earlier iterations):

| Class Index | Class Label | Clinical / Electrophysiological Definition | Contributing Datasets | Window Count | % of Dataset | Independent Records |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **0** | `Asystole` | Cardiac arrest; absence of electrical activity ($<0.1\text{ mV}$) | Challenge 2015 ($2,716$), VFDB ($300$) | 3,016 | 4.65% | ~24 records (22 C2015, 2 VFDB) |
| **1** | `Normal` | Normal Sinus Rhythm (NSR, 60–100 bpm) | MITDB ($24,767$), VFDB ($6,627$), CUDB ($539$) | 31,933 | **49.19%** | ~105 records (48 MITDB, 22 VFDB, 35 CUDB) |
| **2** | `Tachycardia` | Supraventricular or Sinus Tachycardia ($>100\text{ bpm}$) | **Challenge 2015 ONLY** ($15,210$) | 15,210 | 23.43% | ~122 records (122 C2015, 0 others) |
| **3** | `VF` | Ventricular Fibrillation / Flutter (chaotic shockable) | VFDB ($1,080$), Challenge 2015 ($756$), CUDB ($132$) | 2,026 | **3.12%** | **16 records** (8 VFDB, 6 C2015, 2 CUDB) |
| **4** | `VT` | Sustained Ventricular Tachycardia (wide shockable) | Challenge 2015 ($10,664$), VFDB ($1,980$), CUDB ($6$) | 12,732 | 19.61% | 106 records (86 C2015, 19 VFDB, 1 CUDB) |
| **Total** | | | | **64,917** | **100%** | **~250–300 records** |

---

## 3. Class Imbalance & Representation Audit

- **Dominant Class**: `Normal` (31,933 windows, 49.19% of total corpus).
- **Rarest Class**: `VF` (2,026 windows, 3.12% of total corpus).
- **Imbalance Ratio**: $31,933 / 2,026 = \mathbf{15.76 : 1}$.
- **Patient Cohort Deficit**:
  - While `Normal`, `Tachycardia`, and `VT` have $>100$ independent records each, `VF` has only **16 independent patient records** across all combined sources.
  - `Asystole` has only ~24 records (mostly ICU flatline alarms from Challenge 2015).
- **Majority-Class Dominance**:
  - `Normal` + `Tachycardia` represent **72.6%** of all test evaluation windows.
  - High overall accuracy (85%) is heavily driven by correctly predicting abundant Normal and Tachycardia windows, while performance on life-threatening VF is severely impaired.

---

## 4. Source Bias & Confounding Analysis

### A. The Tachycardia vs. Normal Source Trap
- **100% of Tachycardia** windows ($15,210$) come from Challenge 2015 ICU records. Zero Tachycardia was mapped from MITDB, VFDB, or CUDB.
- **0% of Normal** windows come from Challenge 2015. All 31,933 Normal windows come from MITDB ($24,767$), VFDB ($6,627$), and CUDB ($539$).
- **The Amplitude Mismatch**:
  Cell 102 diagnostic output records:
  - Challenge 2015 (`c2015`): `ptp` mean = **20.88 mV**, `std` mean = **3.75 mV**
  - MITDB (`mitdb`): `ptp` mean = **1.61 mV**, `std` mean = **0.18 mV**
  - VFDB (`vfdb`): `ptp` mean = **2.12 mV**, `std` mean = **0.47 mV**
- **Mechanism of Failure**:
  Because Challenge 2015 was scaled with an unverified gain of 200, its signal amplitudes are roughly **10x to 20x larger** than the standard mV Holter recordings. The decision tree splits on `std > 2.0` or `ptp > 10.0` to trivially separate Tachycardia from Normal. 
  If an ambulatory Holter record from MIT-BIH or VFDB experiences sinus tachycardia, its `ptp` is ~1.6 mV; Model 2 will fail to classify it as Tachycardia because it has never seen a low-amplitude Tachycardia recording.

---

## 5. Performance Audit & Error Analysis

The test evaluation metrics reported in Cell 102 output (under `GroupShuffleSplit`, test support = 12,954 windows) are:

| Class Index | Class Name | Precision | Recall | F1-Score | Test Support (Windows) | Primary Error Mode |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `0` | **Asystole** | 0.80 | 0.65 | 0.72 | 602 | 35% false negative (confused with low-amplitude Normal/VF) |
| `1` | **Normal** | **0.99** | **0.95** | **0.97** | 6,361 | Artificially elevated by MITDB amplitude isolation |
| `2` | **Tachycardia** | 0.83 | 0.91 | 0.87 | 2,822 | Artificially elevated by C2015 amplitude isolation |
| `3` | **VF** | **0.36** | 0.64 | **0.46** | 547 | **Severe false positive rate (64% of VF alarms are false)** |
| `4` | **VT** | 0.71 | 0.61 | 0.66 | 2,622 | Confused with Tachycardia and VF |
| | **Overall Accuracy** | — | — | **0.85** | **12,954** | |
| | **Macro Average** | 0.74 | 0.75 | **0.74** | 12,954 | |
| | **Weighted Average** | 0.86 | 0.85 | **0.85** | 12,954 | |

### Key Performance Failures:
1. **VF False Alarm Hazard**: A precision of 0.36 means that out of every 100 windows Model 2 labels as Ventricular Fibrillation, **64 are false alarms**. In an emergency system, this is an unacceptable error profile.
2. **VT Under-Sensitivity**: Recall for Ventricular Tachycardia is only **61%**, missing nearly 4 out of every 10 true VT episodes.
3. **QRS Feature Inadequacy**: The 10 hand-crafted features rely on Pan-Tompkins QRS detection. During true VF, discrete peaks do not exist; Pan-Tompkins detects irregular baseline noise, corrupting `mean_rr` and `qrs_width`, which explains the heavy confusion between VF and VT/Tachycardia.

---

## 6. Record/Patient Leakage Evaluation

- **Grouping Mechanism**: The split used `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on `record_id`. Within that single split, train and test record IDs were disjoint.
- **Cross-Source Evaluation**: Never performed. The model was not tested across databases.
- **Classification**: **WARNING**. While direct record leakage within the split was prevented, the intense class-to-source entanglement renders the evaluation unrepresentative of real-world generalization.

---

## 7. Model 2 Verdict

$$\text{\Huge \textbf{RESTRUCTURE TAXONOMY & RETRAIN}}$$

### Recommendations:
1. **De-duplicate Shockable Detection from Model 2**:
   - Model 2 should **not** attempt to distinguish VF from VT. That task belongs exclusively to the specialized Model 3 (which uses validated raw-waveform physiological complexity features).
   - In Model 2, collapse shockable ventricular rhythms into a single category (`Ventricular Arrhythmia` or delegate entirely to Model 1).
2. **Harmonize Tachycardia and Normal Across Sources**:
   - Normal and Tachycardia must be sampled from multiple datasets with standardized z-score or IQR amplitude scaling to prevent amplitude shortcut learning.
3. **Evaluate via Leave-One-Database-Out**:
   - No multiclass model should be approved without proving cross-database transfer.
