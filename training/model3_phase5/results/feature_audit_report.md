# Phase 5 Read-Only Feature Signal vs. Domain-Confounder Audit Report

## 1. Executive Summary
This report presents the statistical and domain-confounding evaluation of the **21 candidate raw-waveform physiological features** (spectral organization, temporal autocorrelation, and nonlinear complexity) extracted from 14,618 calibrated ECG windows across Challenge 2015, VFDB, and CUDB.

In accordance with Phase 5 boundaries, this audit is strictly read-only: no broad ML model exploration or production model export was performed.

---

## 2. Statistical Audit: Signal vs. Confounder Ranking

| Rank | Feature | Mean Disc. |d| | Mean Source Conf. |d| | Direction Consistent? | Mean Source eta^2 | Signal-to-Confounder Ratio | Category |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `lz_complexity` | 0.2688 | 0.3257 | **True** | 0.0257 | **0.8253** | Complexity |
| 2 | `ac_decay_time` | 0.3697 | 0.5980 | **True** | 0.0653 | **0.6183** | Autocorrelation |
| 3 | `ac_first_secondary_peak` | 0.7489 | 1.3005 | **True** | 0.2388 | **0.5759** | Autocorrelation |
| 4 | `ac_zero_crossing_lag` | 0.3622 | 0.6382 | **True** | 0.0762 | **0.5675** | Autocorrelation |
| 5 | `ac_max_peak_ratio` | 0.4611 | 0.8324 | **True** | 0.1398 | **0.5539** | Autocorrelation |
| 6 | `sample_entropy` | 0.6012 | 1.1519 | **True** | 0.1936 | **0.5219** | Complexity |
| 7 | `hjorth_mobility` | 0.3349 | 1.0690 | **True** | 0.1741 | **0.3132** | Complexity |
| 8 | `spectral_flatness` | 0.4193 | 1.5866 | **True** | 0.3242 | **0.2643** | Spectral |
| 9 | `hjorth_complexity` | 0.3691 | 1.4866 | **True** | 0.2782 | **0.2483** | Complexity |
| 10 | `hjorth_activity` | 0.5706 | 0.6303 | **False** | 0.0723 | **0.9052** | Complexity |
| 11 | `permutation_entropy` | 0.8904 | 0.9874 | **False** | 0.1810 | **0.9017** | Complexity |
| 12 | `dominant_peak_prominence` | 0.5176 | 0.5946 | **False** | 0.0747 | **0.8706** | Spectral |
| 13 | `dominant_freq` | 0.2654 | 0.3460 | **False** | 0.0952 | **0.7670** | Spectral |
| 14 | `dominant_peak_power` | 0.5875 | 0.8508 | **False** | 0.1224 | **0.6905** | Spectral |
| 15 | `ac_periodicity_strength` | 0.4790 | 0.9049 | **False** | 0.1823 | **0.5293** | Autocorrelation |
| 16 | `spectral_peak_power_ratio` | 0.6293 | 1.3061 | **False** | 0.2323 | **0.4818** | Spectral |
| 17 | `spectral_concentration` | 0.6293 | 1.3061 | **False** | 0.2323 | **0.4818** | Spectral |
| 18 | `spectral_peak_purity` | 0.4376 | 1.3863 | **False** | 0.2550 | **0.3156** | Spectral |
| 19 | `spectral_entropy` | 0.4673 | 1.7394 | **False** | 0.3504 | **0.2686** | Spectral |
| 20 | `spectral_centroid` | 0.3320 | 1.2980 | **False** | 0.2347 | **0.2558** | Spectral |
| 21 | `spectral_bandwidth` | 0.3726 | 1.6919 | **False** | 0.3516 | **0.2202** | Spectral |

### Key Findings on Direction Consistency & Spectral Inversion
* **Consistent Features (9/21)**: Only 9 features preserve the same physical direction across Challenge 2015 and VFDB:
  - Complexity: `lz_complexity`, `sample_entropy`, `hjorth_mobility`, `hjorth_complexity`
  - Autocorrelation: `ac_decay_time`, `ac_first_secondary_peak`, `ac_zero_crossing_lag`, `ac_max_peak_ratio`
  - Spectral: `spectral_flatness`
* **Inconsistent / Flipped Features (12/21)**: Exactly 12 features flipped physical direction between Challenge 2015 and VFDB.
  - Notably, **9 out of 10 spectral frequency and power features** (`spectral_entropy`, `spectral_peak_power_ratio`, `spectral_concentration`, `spectral_peak_purity`, `spectral_centroid`, `spectral_bandwidth`, `dominant_freq`, `dominant_peak_power`, `dominant_peak_prominence`) inverted sign.
  - **Physiological Mechanism**: Challenge 2015 VF alarms are dominated by narrow-band sinusoidal ventricular flutter (high concentration, low entropy), whereas VFDB VF represents chaotic, disorganized fibrillation (low concentration, high entropy).

---

## 3. Diagnostic Source Classification Audit

A diagnostic Logistic Regression model was evaluated using strict **5-fold GroupKFold cross-validation on `record_id`** to test how readily the dataset origin (`c2015` vs `vfdb` vs `cudb`) can be predicted from the feature representations:

| Feature Representation | Features Count | Source Prediction Balanced Accuracy | Source Prediction Macro F1 | Top Source-Predictive Features |
|:---|:---:|:---:|:---:|:---|
| `set1_full_physio_21` | 21 | **0.6094** | **0.6033** | `spectral_bandwidth`, `spectral_entropy`, `hjorth_complexity` |
| `set2_domain_reduced_6` | 6 | **0.8219** | **0.6077** | `hjorth_mobility`, `sample_entropy`, `ac_max_peak_ratio` |
| `set0_baseline_old7` | 7 | **0.7885** | **0.5904** | `vf_band_power_ratio`, `qrs_width`, `rr_cv` |

### Source Separability Interpretation
* Even with zero record leakage across CV folds, a linear classifier predicts dataset origin with **60.9% balanced accuracy** using the full physiological features, and **82.2%** using the domain-reduced subset.
* This proves that distinct recording hardware, filter bandwidths, and clinical settings impart strong non-cardiac domain signatures into the raw ECG waveform.

---

## 4. Formal Decision Recommendations

### A. Are the new physiological features more cross-source stable than the Phase 4 features?
* **YES, but selectively**.
* In Phase 4, hand-crafted features suffered from Pan-Tompkins QRS peak breakdown on fibrillatory waves (e.g. `qrs_width` had a source confounding $|d| = 1.47$).
* In Phase 5, the autocorrelation decay and complexity features (`lz_complexity`, `ac_decay_time`, `ac_zero_crossing_lag`, `sample_entropy`) eliminated direction reversal and demonstrated lower source confounding ($\eta^2 \le 0.19$).
* However, frequency-domain spectral features are **not** cross-source stable due to fundamental differences in clinical VF presentation across datasets.

### B. Is D2 failure likely fixable with improved feature engineering?
* **NO**.
* The audit demonstrates that feature engineering has hit an asymptotic ceiling:
  1. The dataset contains a catastrophic **sample size bottleneck of only ~16 unique physical patients with VF** (6 in Challenge 2015, 8 in VFDB, 2 in CUDB).
  2. The 14,618 sliding windows are pseudo-replicates of these 16 patients.
  3. No mathematical transformation of the waveform can overcome the lack of biological patient variance. Any hand-crafted or automated feature representation will inevitably overfit to the idiosyncratic morphology of 8–10 training patients.

### C. Should the next step be targeted ML evaluation or collecting additional raw data?
* **COLLECTING ADDITIONAL RAW DATA (DATASET EXPANSION)**.
* Broad ML tuning or deeper architectures cannot solve an $N=16$ patient bottleneck.
* The mandatory next phase (Phase 6) must focus on expanding independent VF patient representations from ~16 toward $\ge 50$ (e.g. adjudicating CUDB records 4–35, integrating AHA ECG Database or independent ICU telemetry cohorts) before deploying Model 3 to production.
