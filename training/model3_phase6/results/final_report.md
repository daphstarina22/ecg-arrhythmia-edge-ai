# Phase 6B Final Report: MIT-BIH Dataset Expansion & Patient-Level Limitation Audit

**Date**: September 6, 2026  
**Status**: Completed  
**Artifact Directory**: `training/model3_phase6/artifacts/`  
**Results Directory**: `training/model3_phase6/results/`  
**Primary Output Table**: `training/model3_phase6/artifacts/expanded_physiological_feature_table.csv`  

---

## 1. Executive Summary & Context

Phase 5 demonstrated that raw-waveform physiological features (spectral organization, autocorrelation periodicity, Hjorth dynamics, downsampled Lempel-Ziv, and sample entropy via Strategy B) significantly improved cross-database performance on Direction 2 ($\text{c2015} + \text{cudb} \to \text{vfdb}$) from ROC-AUC $\approx 0.60$ to $0.6825$. However, no model configuration passed the predefined generalization gate ($\text{ROC-AUC} \ge 0.75$, record recall $\ge 70\%$ for both classes).

The root-cause analysis in Phase 6A hypothesized that this cross-domain failure is primarily driven by an **insufficient cohort of independent VF patients** ($N=16$ physical records total) rather than a deficiency in feature extraction or model architecture.

In Phase 6B, we executed the targeted integration of the three verified sustained arrhythmia records from the MIT-BIH Arrhythmia Database (`MITBIH.zip`):
- **Record 205**: Sustained monomorphic VT ($1$ episode, $8.66\text{ s}$, $2$ windows $\ge 5.0\text{ s}$)
- **Record 207**: Sustained ventricular flutter ($5$ episodes $\ge 5.0\text{ s}$, total $139.49\text{ s}$, $49$ windows)
- **Record 223**: Sustained wide-complex VT ($2$ episodes $\ge 5.0\text{ s}$, total $94.05\text{ s}$, $35$ windows)

A total of **86 new windows** were extracted, calibrated to physical millivolts, and appended to the validated Phase 5 feature table to create the new expanded dataset:
$$\text{\textbf{expanded\_physiological\_feature\_table.csv}} \quad (14,704 \text{ windows, } 120 \text{ physical records, } 28 \text{ columns})$$

---

## 2. Exact Verified Rhythm Segments & Annotation Evidence

All segment boundaries were established using WFDB reference annotations (`atr`), verified down to individual sample timestamps, and audited against clinical ECG descriptions. All transient runs of non-sustained VT ($<5.0\text{ s}$) were strictly excluded to prevent window pollution with baseline sinus rhythm.

| Record ID | Database | Source Rhythm | Project Label | Target | Start Sample | End Sample | Duration (s) | Windows ($\ge 5\text{s}$) | Clinical Annotation Evidence |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **mitdb_205** | MIT-BIH | `(VT` | `VT` | 0 | 527,698 | 530,816 | 8.66 s | 2 | Sustained monomorphic VT episode terminating at sample 530,816 with `(N` |
| **mitdb_207** | MIT-BIH | `(VFL` | `VF` | 1 | 14,689 | 18,396 | 10.30 s | 3 | Sustained rapid sinusoidal ventricular flutter ($>250\text{ bpm}$) terminating at `(N` |
| **mitdb_207** | MIT-BIH | `(VFL` | `VF` | 1 | 19,760 | 21,827 | 5.74 s | 1 | Sustained ventricular flutter terminating at `(N` |
| **mitdb_207** | MIT-BIH | `(VFL` | `VF` | 1 | 89,320 | 94,220 | 13.61 s | 4 | Sustained ventricular flutter terminating at `(N` |
| **mitdb_207** | MIT-BIH | `(VFL` | `VF` | 1 | 97,051 | 101,185 | 11.48 s | 3 | Sustained ventricular flutter terminating at `(N` |
| **mitdb_207** | MIT-BIH | `(VFL` | `VF` | 1 | 554,740 | 590,149 | 98.36 s | 38 | Long uninterrupted ventricular flutter run terminating at `(IVR` |
| **mitdb_223** | MIT-BIH | `(VT` | `VT` | 0 | 208,178 | 228,004 | 55.07 s | 21 | Sustained wide-complex VT terminating at `(N` |
| **mitdb_223** | MIT-BIH | `(VT` | `VT` | 0 | 375,556 | 389,589 | 38.98 s | 14 | Sustained wide-complex VT terminating at `(N` |

*Complete machine-readable table saved at:* [`training/model3_phase6/results/segment_audit_table.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/model3_phase6/results/segment_audit_table.csv).

---

## 3. Strict Preprocessing & Feature Extraction Parity

1. **Lead Selection**: Channel `'MLII'` (modified limb lead II, Lead II equivalent).
2. **Calibration**: Standard MIT-BIH header parameters applied:
   $$V(\text{mV}) = \frac{\text{digital} - 1024.0}{200.0}$$
3. **Sampling Rate**: Native $f_s = 360\text{ Hz}$ (matches target $f_s$ exactly; zero rational resampling distortion).
4. **Windowing & Stride**: Fixed 5.0-second sliding windows ($1,800$ samples) with 2.5-second stride ($900$ samples).
5. **Strategy B Feature Extraction**: Extracted the exact 21 physiological features validated in Phase 5:
   - Spectral Organization: `dominant_freq`, `dominant_peak_power`, `dominant_peak_prominence`, `spectral_peak_power_ratio`, `spectral_entropy`, `spectral_flatness`, `spectral_centroid`, `spectral_bandwidth`, `spectral_concentration`, `spectral_peak_purity`.
   - Autocorrelation / Periodicity: `ac_max_peak_ratio`, `ac_first_secondary_peak`, `ac_decay_time`, `ac_periodicity_strength`, `ac_zero_crossing_lag`.
   - Complexity / Nonlinear Dynamics: `hjorth_activity`, `hjorth_mobility`, `hjorth_complexity`, `lz_complexity` (90 Hz binary LZC, $N=450$), `permutation_entropy` ($m=3, \tau=2$), `sample_entropy` (45 Hz downsampled, $N=225, m=2, r=0.2\sigma$).

---

## 4. 12-Point Data Integrity Audit Results

The programmatic 12-point integrity audit executed on the expanded table yielded an uncompromised **12 / 12 PASS**:

| # | Integrity Audit Point | Pre-Expansion (Phase 5) | Post-Expansion (Phase 6B) | Delta / Check | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Total Windows | 14,618 | 14,704 | +86 windows | **PASS** |
| **2** | Total Physical Records | 117 | 120 | +3 records (`mitdb_205`, `mitdb_207`, `mitdb_223`) | **PASS** |
| **3** | Unique VT Records | 106 | 108 | +2 records (`mitdb_205`, `mitdb_223`) | **PASS** |
| **4** | Unique VF/VFL Records | 16 | 17 | +1 record (`mitdb_207`) | **PASS** |
| **5** | Windows per Added Record | — | — | `mitdb_205`: 2, `mitdb_207`: 49, `mitdb_223`: 35 | **PASS** |
| **6** | Label Distribution | VT: 12,650 (86.5%)<br>VF: 1,968 (13.5%) | VT: 12,687 (86.3%)<br>VF: 2,017 (13.7%) | VT: +37 windows<br>VF: +49 windows | **PASS** |
| **7** | Source Distribution | `c2015`: 11,420<br>`vfdb`: 3,060<br>`cudb`: 138 | `c2015`: 11,420<br>`vfdb`: 3,060<br>`cudb`: 138<br>`mitdb`: 86 | `mitdb`: 86 windows | **PASS** |
| **8** | Missing / Non-Finite Features | 0 NaN, 0 Inf | 0 NaN, 0 Inf | 100% clean finite numerical floats | **PASS** |
| **9** | Duplicate Windows | 0 duplicates | 0 duplicates | Zero `(record_id, window_index)` collisions | **PASS** |
| **10** | Record ID Collisions | — | 0 collisions | Distinct `mitdb_` namespace | **PASS** |
| **11** | Feature Schema Consistency | 28 columns | 28 columns | Identical column names, order, dtypes | **PASS** |
| **12** | Original Data Immutability | 14,618 rows | 14,618 rows | 0 mismatches against Phase 5 table | **PASS** |

*Audit JSON persisted at:* [`training/model3_phase6/results/expansion_audit.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/model3_phase6/results/expansion_audit.json).

---

## 5. Patient-Level Diversity & Limitation Analysis

The primary scientific objective of Phase 6 is to evaluate whether dataset expansion meaningfully resolves the patient diversity bottleneck.

### Quantitative Change in Patient Cohort
* **Baseline Cohort (Phase 5)**:
  - Total physical records: **117 records**
  - Independent VT patients: **106 patients**
  - Independent VF/VFL patients: **16 patients**
  - Patient Imbalance Ratio: **6.63 : 1** (VT to VF patients)
* **Expanded Cohort (Phase 6B)**:
  - Total physical records: **120 records** (+2.56%)
  - Independent VT patients: **108 patients** (+2 patients, +1.89%)
  - Independent VF/VFL patients: **17 patients** (+1 patient, +6.25%)
  - Patient Imbalance Ratio: **6.35 : 1** (VT to VF patients)

### Scientific Evaluation of the Limitation
1. **The VF Patient Bottleneck Remains Acute**:
   - Integrating MIT-BIH added only **one** additional shockable rhythm patient (`mitdb_207`, Ventricular Flutter).
   - A total cohort of **17 independent VF patients** is fundamentally inadequate for deep statistical learning or domain-invariant representation learning.
   - Under grouped cross-validation ($k=5$), each validation fold contains only **3 to 4 VF patients**.
   - In cross-database evaluation, the target test sets remain microscopic at the patient level:
     - Direction 1 ($\to \text{c2015}$): Only **6 VF patients** in the test set.
     - Direction 2 ($\to \text{vfdb}$): Only **8 VF patients** in the test set.
2. **Window Volume vs. Patient Independence**:
   - The expanded dataset contains **2,017 VF windows**, but these windows are clustered across just 17 individuals (average of ~118 windows per patient).
   - High window-level sample size gives an illusion of statistical power that collapses under patient-grouped or cross-domain evaluation.
3. **VT Diversity is Already Saturated**:
   - The VT cohort (108 patients, 12,687 windows) is reasonably diverse, but further additions of VT patients without matched VF patients only exacerbates the class imbalance.

---

## 6. Source Overlap & Data Leakage Check

A rigorous provenance audit was conducted across all datasets:

1. **MIT-BIH Arrhythmia Database (`mitdb`)**:
   - Recorded between 1975 and 1979 at Beth Israel Hospital (Boston, MA) from 4,000 24-hour ambulatory Holter tapes (Mark & Moody, 1982).
   - Records 205, 207, and 223 are distinct ambulatory outpatient Holter recordings.
2. **Cross-Database Overlap Audit**:
   - **Creighton University Database (`cudb`)**: Recorded in Omaha, Nebraska in the late 1980s from Holter recordings of 35 ICU/resuscitation patients. Zero institutional or patient overlap.
   - **Computing in Cardiology Challenge 2015 (`c2015`)**: Recorded decades later (2010–2014) from bedside intensive care monitors. Zero patient identity overlap with 1970s Holter tapes.
   - **MIT-BIH Malignant Ventricular Arrhythmia Database (`vfdb`)**: Records 418–615 were derived from the Greenwald (1986) tapes. Records 205, 207, and 223 were not part of this series.
   - **SDDB (Disqualified)**: As established in Phase 6A, SDDB has 100% duplicate patient identity overlap with VFDB and remains strictly excluded.

**LEAKAGE AUDIT VERDICT**:
$$\text{\textbf{LEAKAGE CHECK: PASS}}$$
*(Zero patient identity overlap, zero duplicate recordings, zero feature leakage across dataset boundaries).*

---

## 7. Remaining Limitations

1. **CUDB Bracketed Episodes Remain Quarantined**:
   - 25 unused CUDB records contain ~2,800 seconds of sustained tachyarrhythmias (39 episodes), but lack rhythm subtype annotations (`[` and `]` without differentiating VT from VF). They cannot be added without expert clinical re-adjudication.
2. **Non-Sustained VT (NSVT) Bursts Excluded**:
   - 10 MIT-BIH records with brief NSVT bursts ($<5.0\text{ s}$) were excluded. While clinically significant, they cannot produce pure 5-second windows without sinus rhythm pollution.
3. **Malignant Rhythm Scarcity in Public Repositories**:
   - Open-access, high-resolution ECG databases universally suffer from extreme scarcity of true Ventricular Fibrillation episodes due to the rapid, lethal nature of VF and the requirement for immediate defibrillation.

---

## 8. Final Decision Gate

In accordance with the Phase 6B research specification:

$$\text{\Large \textbf{DATASET EXPANSION INSUFFICIENT — ACQUIRE ADDITIONAL INDEPENDENT VF DATA}}$$

### Justification:
The integration of verified MIT-BIH records added only **+1 independent VF/VFL patient** (increasing the VF cohort from 16 to 17 patients, a marginal gain of +6.25%). The dataset remains fundamentally underpowered at the patient level for Ventricular Fibrillation. 

Proceeding to Phase 6C evaluation with only 17 VF patients would merely re-confirm the cross-database operating-point shift and small-sample variance observed in Phase 5 without providing sufficient statistical power to validate genuine domain invariance. Meaningful generalization requires expanding the independent VF patient cohort through:
1. Clinical re-adjudication of the 25 quarantined CUDB records (`cu04`–`cu35`, containing 39 sustained episodes).
2. Sourcing additional independent clinical databases containing verified VF episodes (e.g., AHA ECG Database, Sudden Cardiac Death Holter Database).
