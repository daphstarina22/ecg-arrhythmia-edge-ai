# Model 3: Shockable Subtype Classifier (VT vs. VF/VFL)

**Date**: September 6, 2026  
**Status**: Specification & Empirical Feature Protocol  
**Task**: Binary Classification — Ventricular Tachycardia (VT) vs. Ventricular Fibrillation/Flutter (VF/VFL)  
**Dataset**: `training/model3_phase6/artifacts/expanded_physiological_feature_table.csv`  
**Target Platform**: Edge Devices (Raspberry Pi 4 / Embedded Linux, CPU Execution)  

---

## 1. Scientific Context & Evolutionary Findings

Model 3 addresses the difficult clinical decision within shockable rhythms:

$$\text{"Given a confirmed shockable rhythm, is it organized VT or chaotic VF/VFL?"}$$

### Key Findings from Prior Phases:
1. **Phase 4 Breakdown**: The original 10 hand-crafted features relied on Pan-Tompkins QRS peak detection. Because VF lacks discrete QRS complexes, the detector triggered on random high-frequency oscillations, causing cross-database D2 ROC-AUC to collapse to **0.6076**.
2. **Phase 5 Breakthrough**: Replacing QRS features with **21 raw-waveform physiological complexity and spectral organization features** (Sample Entropy, Lempel-Ziv complexity, autocorrelation decay, spectral purity) improved D2 cross-database ROC-AUC to **0.6825** and stabilized grouped cross-validation.
3. **Phase 5 Feature Audit Findings**:
   - Exactly **12 out of 21 features showed physical direction inversion** between Challenge 2015 and VFDB (e.g. `dominant_freq`, `spectral_entropy`, `spectral_peak_purity`, `ac_periodicity_strength`).
   - Physiological reason: Challenge 2015 VF alarms are dominated by narrow-band sinusoidal ventricular flutter (high concentration, low entropy), whereas VFDB VF represents chaotic, disorganized fibrillation (low concentration, high entropy).
   - Frequency-domain spectral features inverted sign across sources and are heavily confounded by recording hardware bandwidth.
   - Autocorrelation decay and nonlinear complexity features (`lz_complexity`, `ac_decay_time`, `ac_zero_crossing_lag`, `ac_max_peak_ratio`, `sample_entropy`, `hjorth_mobility`) preserved uniform physical direction across all databases.
4. **Phase 6B Dataset Expansion**: Incorporating verified sustained ventricular rhythms from MIT-BIH (records 205, 207, 223) expanded the corpus to **120 independent records** and **14,704 windows**.
5. **Fundamental Sample Size Constraint**: The entire available ECG corpus contains only **17 independent physical records with sustained VF/VFL** across all four databases. High window counts ($N=2,017$) must not be mistaken for patient diversity. Model 3 remains a research prototype whose performance limits must be transparently reported.

---

## 2. Dataset & Target Labels

- **Input File**: `training/model3_phase6/artifacts/expanded_physiological_feature_table.csv`
- **Total Windows**: 14,704
- **Independent Records**: 120
- **Classes**:
  - **Class 0 (VT)**: 12,687 windows across **103 records** (86 C2015, 14 VFDB, 1 CUDB, 2 MIT-BIH)
  - **Class 1 (VF/VFL)**: 2,017 windows across **17 records** (6 C2015, 8 VFDB, 2 CUDB, 1 MIT-BIH)
- **Data Quality**: 0 missing values, 0 infinite values, 0 duplicate window indices.

---

## 3. Corrected Empirical Feature Protocol

To strictly satisfy the Phase 5 empirical audit, candidate features are partitioned into three rigorous subsets:

### Complete Phase 5 Feature Audit & Selection Table

| Feature Name | Category | Status in Set B | Phase 5 Discriminative Cohen's $d$ (C2015 / VFDB) | Direction Consistent? | Source Confounding Mean $\eta^2$ | Signal-to-Confounder Ratio | Reason for Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `lz_complexity` | Complexity | **INCLUDED** | $-0.3905$ / $-0.1471$ | **True** | **0.0257** | **0.8253** | Rank 1: Low source confounding ($\eta^2 \le 0.03$), consistent direction. |
| `ac_decay_time` | Autocorr | **INCLUDED** | $+0.5438$ / $+0.1957$ | **True** | **0.0653** | **0.6183** | Rank 2: Consistent decay direction, low source confounding. |
| `ac_first_secondary_peak` | Autocorr | **EXCLUDED** | $+1.1106$ / $+0.3872$ | True | **0.2388** | 0.5759 | Excluded: High source confounding ($\eta^2 = 0.24 > 0.20$). |
| `ac_zero_crossing_lag` | Autocorr | **INCLUDED** | $+0.4867$ / $+0.2378$ | **True (All 3)** | **0.0762** | **0.5675** | Rank 4: Consistent across C2015, VFDB, and CUDB; low confounding. |
| `ac_max_peak_ratio` | Autocorr | **INCLUDED** | $-0.6091$ / $-0.3131$ | **True (All 3)** | **0.1398** | **0.5539** | Rank 5: Consistent across all 3 sources; moderate confounding ($\le 0.20$). |
| `sample_entropy` | Complexity | **INCLUDED** | $+0.7270$ / $+0.4754$ | **True (All 3)** | **0.1936** | **0.5219** | Rank 6: Consistent chaos metric across all 3 databases; $\eta^2 \le 0.20$. |
| `hjorth_mobility` | Complexity | **INCLUDED** | $-0.6148$ / $-0.0550$ | **True** | **0.1741** | **0.3132** | Rank 7: Consistent direction; source $\eta^2 \le 0.20$. |
| `spectral_flatness` | Spectral | **EXCLUDED** | $-0.6776$ / $-0.1610$ | True | **0.3242** | 0.2643 | Excluded: Severe source confounding ($\eta^2 = 0.32 \gg 0.20$). |
| `hjorth_complexity` | Complexity | **EXCLUDED** | $+0.5387$ / $+0.1995$ | True (All 3) | **0.2782** | 0.2483 | Excluded: Severe source confounding ($\eta^2 = 0.28 > 0.20$). |
| `hjorth_activity` | Complexity | **EXCLUDED** | $+0.2893$ / $-0.8518$ | **False (Inverted)** | 0.0723 | 0.9052 | Excluded: Direction inverted between C2015 and VFDB. |
| `permutation_entropy` | Complexity | **EXCLUDED** | $-1.3144$ / $+0.4663$ | **False (Inverted)** | 0.1810 | 0.9017 | Excluded: Direction inverted between C2015 and VFDB. |
| `dominant_peak_prominence`| Spectral | **EXCLUDED** | $+0.5562$ / $-0.4790$ | **False (Inverted)** | 0.0747 | 0.8706 | Excluded: Direction inverted between C2015 and VFDB. |
| `dominant_freq` | Spectral | **EXCLUDED** | $-0.3894$ / $+0.1414$ | **False (Inverted)** | 0.0952 | 0.7670 | Excluded: Direction inverted between C2015 and VFDB. |
| `dominant_peak_power` | Spectral | **EXCLUDED** | $+0.3795$ / $-0.7955$ | **False (Inverted)** | 0.1224 | 0.6905 | Excluded: Direction inverted between C2015 and VFDB. |
| `ac_periodicity_strength` | Autocorr | **EXCLUDED** | $+0.9460$ / $-0.0119$ | **False (Inverted)** | 0.1823 | 0.5293 | Excluded: Direction inverted between C2015 and VFDB. |
| `spectral_peak_power_ratio`| Spectral | **EXCLUDED** | $+1.0229$ / $-0.2356$ | **False (Inverted)** | 0.2323 | 0.4818 | Excluded: Direction inverted and high source confounding ($\eta^2 > 0.20$). |
| `spectral_concentration` | Spectral | **EXCLUDED** | $+1.0229$ / $-0.2356$ | **False (Inverted)** | 0.2323 | 0.4818 | Excluded: Direction inverted and high source confounding ($\eta^2 > 0.20$). |
| `spectral_peak_purity` | Spectral | **EXCLUDED** | $+0.6598$ / $-0.2153$ | **False (Inverted)** | 0.2550 | 0.3156 | Excluded: Direction inverted and high source confounding ($\eta^2 = 0.26$). |
| `spectral_entropy` | Spectral | **EXCLUDED** | $-0.7874$ / $+0.1471$ | **False (Inverted)** | 0.3504 | 0.2686 | Excluded: Direction inverted and massive source confounding ($\eta^2 = 0.35$). |
| `spectral_centroid` | Spectral | **EXCLUDED** | $-0.5408$ / $+0.1233$ | **False (Inverted)** | 0.2347 | 0.2558 | Excluded: Direction inverted and high source confounding ($\eta^2 = 0.23$). |
| `spectral_bandwidth` | Spectral | **EXCLUDED** | $-0.6550$ / $+0.0901$ | **False (Inverted)** | 0.3516 | 0.2202 | Excluded: Direction inverted and massive source confounding ($\eta^2 = 0.35$). |

---

### Corrected Model 3 Feature Subsets

#### Set A: Full Physiological Feature Set (21 Features)
- All 21 candidate features (spectral, autocorrelation, and nonlinear complexity). Evaluated to measure baseline capacity when all features are available.

#### Set B: Validated Domain-Stable Subset (6 Features)
Strictly filtered to features that demonstrated:
1. **Physical Direction Consistency**: Zero sign inversion between Challenge 2015 and VFDB.
2. **Controlled Source Confounding**: Mean source $\eta^2 \le 0.20$.
- Features:
  1. `lz_complexity` (Nonlinear sequence complexity)
  2. `ac_decay_time` (Autocorrelation decay duration)
  3. `ac_zero_crossing_lag` (Autocorrelation zero-crossing lag)
  4. `ac_max_peak_ratio` (Secondary-to-primary autocorrelation peak ratio)
  5. `sample_entropy` (Downsampled Sample Entropy)
  6. `hjorth_mobility` (Waveform mobility)

#### Set C: Minimal Edge Subset (4 Features)
Top 4 consistent features by signal-to-confounder ratio:
- `lz_complexity`
- `ac_decay_time`
- `ac_zero_crossing_lag`
- `sample_entropy`

---

## 4. Candidate Model Architectures

1. **Regularized Logistic Regression (L2 / Ridge)**:
   - Linear decision boundary with `StandardScaler` inside a leak-free pipeline.
   - Robust against overfitting on the small 17-record VF minority class.
2. **Balanced Random Forest**:
   - 100 trees, `max_depth=5`, `min_samples_leaf=10`, `class_weight='balanced'`.
3. **Histogram-based Gradient Boosting (`HistGradientBoostingClassifier`)**:
   - `max_iter=100`, `max_depth=4`, `learning_rate=0.05`, `class_weight='balanced'`.

---

## 5. Rigorous Evaluation Protocol

### A. 5-Fold Stratified Group Cross-Validation
- Folds grouped strictly by `record_id`.
- Stratification maintains consistent VT vs. VF record ratios across folds.
- Window-level and record-level aggregated metrics computed for every fold.

### B. Cross-Database Validation
- **Direction 1 (D1)**: Train on Challenge 2015 ($N=92$ records) $\implies$ Test on VFDB + CUDB + MIT-BIH ($N=28$ records).
- **Direction 2 (D2)**: Train on VFDB + CUDB + MIT-BIH ($N=28$ records) $\implies$ Test on Challenge 2015 ($N=92$ records).

### C. Persistent Prediction Artifacts
Every test window will be persisted to CSV containing:
- `record_id`
- `dataset_source`
- `true_label`
- `prediction`
- `predicted_probability`

### D. Per-Record Performance & Failure Analysis
- Generate a record-level audit table ranking all 120 records by accuracy.
- Specifically isolate the 17 VF records and document any misclassified episodes.
- Maintain conservative clinical stance: Do NOT claim deployment readiness unless cross-database ROC-AUC meets acceptable clinical margins ($\ge 0.85$).
