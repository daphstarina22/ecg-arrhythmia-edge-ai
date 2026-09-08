# Phase 5 Research Report: Raw Waveform Physiological Representation

## 1. Executive Summary & Governance Audit

* **Phase 5 Objective**: Test whether candidate physiological organization and complexity features (spectral, temporal autocorrelation, and nonlinear dynamics) extracted directly from raw calibrated ECG waveforms resolve the severe cross-database generalization collapse identified in Phase 4.2.
* **Integrity & Boundaries**:
  * Executed strictly under `training/model3_phase5/`.
  * Zero modifications to production source code (`src/`), deployed ONNX models (`models/`), unit tests (`tests/`), or historical artifacts (`model3_phase2*`, `model3_phase3`, `model3_phase4`).
  * Strict grouped-record separation: training and test splits contain zero overlapping patient records.
* **Dataset Inventory Reconciled**:
  * Exactly **117 unique physical patient records** (92 Challenge 2015, 22 VFDB, 3 CUDB).
  * 5 records in VFDB contain both VT and VF segments (`vfdb_422`, `vfdb_426`, `vfdb_429`, `vfdb_430`, `vfdb_609`), explaining the earlier notation of 122 record-label slices ($92 + 27 + 3 = 122$).
  * Total corpus: exactly **14,618 windows** (12,650 VT, 1,968 VF; 360 Hz, 5.0 s duration, 2.5 s step). Zero records were excluded.

---

## 2. Phase 5C: Feature Signal vs. Domain-Confounder Statistical Audit

All 21 candidate raw-waveform features were audited across within-database effect size (Cohen's $d$, ROC-AUC, Mutual Information) and within-class cross-database source shift (source Cohen's $d$, ANOVA $\eta^2$, Signal-to-Confounder ratio):

| Rank | Feature | Mean Disc. $|d|$ | Mean Source Conf. $|d|$ | Direction Consistent? | Mean Source $\eta^2$ | Signal-to-Confounder Ratio | Category |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | `lz_complexity` | 0.2688 | 0.3257 | **True** | **0.0257** | **0.8253** | Complexity |
| **2** | `ac_decay_time` | 0.3697 | 0.5980 | **True** | **0.0653** | **0.6183** | Autocorrelation |
| **3** | `ac_first_secondary_peak` | 0.7489 | 1.3005 | **True** | 0.2388 | **0.5759** | Autocorrelation |
| **4** | `ac_zero_crossing_lag` | 0.3622 | 0.6382 | **True** | **0.0762** | **0.5675** | Autocorrelation |
| **5** | `ac_max_peak_ratio` | 0.4611 | 0.8324 | **True** | **0.1398** | **0.5539** | Autocorrelation |
| **6** | `sample_entropy` | 0.6012 | 1.1519 | **True** | **0.1936** | **0.5219** | Complexity |
| **7** | `hjorth_mobility` | 0.3349 | 1.0690 | **True** | **0.1741** | **0.3132** | Complexity |
| **8** | `spectral_flatness` | 0.4193 | 1.5866 | **True** | 0.3242 | **0.2643** | Spectral |
| **9** | `hjorth_complexity` | 0.3691 | 1.4866 | **True** | 0.2782 | **0.2483** | Complexity |
| **10** | `hjorth_activity` | 0.5706 | 0.6303 | **FALSE** | 0.0723 | 0.9052 | Complexity (Failed) |
| **11** | `permutation_entropy` | 0.8904 | 0.9874 | **FALSE** | 0.1810 | 0.9017 | Complexity (Failed) |
| **12** | `dominant_peak_prominence` | 0.5176 | 0.5946 | **FALSE** | 0.0747 | 0.8706 | Spectral (Failed) |
| **13** | `dominant_freq` | 0.2654 | 0.3460 | **FALSE** | 0.0952 | 0.7670 | Spectral (Failed) |
| **14** | `dominant_peak_power` | 0.5875 | 0.8508 | **FALSE** | 0.1224 | 0.6905 | Spectral (Failed) |
| **15** | `ac_periodicity_strength` | 0.4790 | 0.9049 | **FALSE** | 0.1823 | 0.5293 | Autocorrelation (Failed) |
| **16** | `spectral_peak_power_ratio` | 0.6293 | 1.3061 | **FALSE** | 0.2323 | 0.4818 | Spectral (Failed) |
| **17** | `spectral_concentration` | 0.6293 | 1.3061 | **FALSE** | 0.2323 | 0.4818 | Spectral (Failed) |
| **18** | `spectral_peak_purity` | 0.4376 | 1.3863 | **FALSE** | 0.2550 | 0.3156 | Spectral (Failed) |
| **19** | `spectral_entropy` | 0.4673 | 1.7394 | **FALSE** | 0.3504 | 0.2686 | Spectral (Failed) |
| **20** | `spectral_centroid` | 0.3320 | 1.2980 | **FALSE** | 0.2347 | 0.2558 | Spectral (Failed) |
| **21** | `spectral_bandwidth` | 0.3726 | 1.6919 | **FALSE** | 0.3516 | 0.2202 | Spectral (Failed) |

### Key Findings from the Audit
1. **Pervasive Spectral Signal Inversion**:
   * 9 out of 10 spectral features completely inverted their physical relationship between Challenge 2015 and VFDB.
   * In Challenge 2015, VF segments exhibit high spectral concentration ($d = +1.02$) and low spectral entropy ($d = -0.79$) because the alarms are driven by organized, narrow-band ventricular flutter.
   * In VFDB, VF segments represent disorganized, broadband fibrillation, reversing the sign ($d = -0.24$ for concentration, $d = +0.15$ for entropy).
   * **Conclusion**: Conventional frequency-domain spectral features cannot generalize across datasets due to disparate clinical VF presentations.
2. **Domain-Robust Physiological Candidates**:
   * Autocorrelation decay and zero-crossing lag consistently distinguish rhythmic VT from disorganized VF across both databases without sign reversal.
   * Lempel-Ziv Complexity and Sample Entropy maintain consistent effect direction with low source confounding ($\eta^2 < 0.20$).

---

## 3. Phase 5D & 5E: ML Modeling & Cross-Database Performance

Four feature subsets were evaluated across Logistic Regression, Random Forest, and SVM-RBF models in directions D1 and D2:
* **`set1_full_physio`**: All 21 physiological features.
* **`set2_domain_reduced`**: 6 features passing consistency and low confounding (`lz_complexity`, `ac_decay_time`, `ac_zero_crossing_lag`, `ac_max_peak_ratio`, `sample_entropy`, `hjorth_mobility`).
* **`set3_best_low_conf`**: Top 4 features by signal-to-confounder ratio (`lz_complexity`, `ac_decay_time`, `ac_first_secondary_peak`, `ac_zero_crossing_lag`).
* **`set0_baseline_old7`**: The 7 reduced hand-crafted features from Phase 4.2.

### Comprehensive Model Robustness Ranking

$$\text{Robustness} = 0.35 \times \overline{\text{Macro F1}} + 0.35 \times \overline{\text{VF Recall}} + 0.30 \times \min(\text{Macro F1})$$

| Rank | Feature Set | Model | D1 ROC-AUC | D2 ROC-AUC | D1 Rec. Macro F1 | D2 Rec. Macro F1 | D1 VF Recall | D2 VT Recall | Robustness Score | Gate Passed? |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `set1_full_physio` | Logistic Regression | **0.8411** | 0.6458 | 0.7501 | 0.3173 | 43.0% | 16.2% | **0.5322** | **False** |
| **2** | `set1_full_physio` | Random Forest | 0.5861 | 0.5898 | 0.6802 | 0.3592 | 39.3% | 29.3% | **0.5285** | **False** |
| **3** | `set2_domain_reduced` | Logistic Regression | 0.6960 | **0.6825** | **0.9129** | 0.2986 | 31.5% | 25.3% | **0.5234** | **False** |
| **4** | `set2_domain_reduced` | Random Forest | 0.6606 | **0.6792** | 0.8137 | 0.3476 | 32.0% | 29.3% | **0.5178** | **False** |
| **5** | `set0_baseline_old7` | Logistic Regression | 0.7233 | 0.6076 | 0.8428 | 0.3120 | 21.5% | 19.0% | **0.5062** | **False** |
| **6** | `set3_best_low_conf` | Random Forest | 0.5636 | 0.5694 | 0.3806 | 0.3943 | **58.3%** | 37.6% | **0.4971** | **False** |
| **7** | `set1_full_physio` | SVM-RBF | 0.6560 | 0.6753 | 0.9054 | 0.4865 | 24.8% | 80.8% | **0.4865** | **False** |
| **8** | `set3_best_low_conf` | Logistic Regression | 0.6758 | 0.5378 | 0.7631 | 0.2652 | 22.7% | 10.5% | **0.4732** | **False** |
| **9** | `set0_baseline_old7` | SVM-RBF | 0.7471 | 0.6388 | 0.8494 | 0.3945 | 18.2% | 59.8% | **0.4712** | **False** |
| **10** | `set2_domain_reduced` | SVM-RBF | 0.6585 | 0.6295 | 0.8988 | 0.4441 | 18.3% | 73.7% | **0.4633** | **False** |
| **11** | `set0_baseline_old7` | Random Forest | 0.7114 | 0.6035 | 0.7452 | 0.2451 | 27.6% | 11.7% | **0.4628** | **False** |
| **12** | `set3_best_low_conf` | SVM-RBF | 0.6240 | 0.5304 | 0.8135 | 0.4854 | 8.7% | 96.2% | **0.3940** | **False** |

---

## 4. Detailed Answers to Phase 5F Research Questions

### Question 1: Does any feature set or model achieve D2 ROC-AUC > 0.70?
* **Answer**: **NO**.
* The baseline 7 features produced D2 ROC-AUC of **0.6076** (Logistic Regression) and **0.6035** (Random Forest).
* The Domain-Reduced physiological feature set (`set2_domain_reduced`) raised D2 ROC-AUC to **0.6825** (+0.075 for Logistic Regression) and **0.6792** (+0.076 for Random Forest). Full physiological SVM reached **0.6753**.
* While physiological features demonstrate a measurable gain over the baseline, D2 ROC-AUC remains strictly bounded below **0.70** (and well below the target decision gate of 0.75).

### Question 2: Does record-level balanced accuracy exceed 65% in both directions?
* **Answer**: **NO**.
* Cross-database testing displays extreme polarity distortion:
  * In **D1** (Train VFDB+CUDB $\to$ Test Challenge 2015): Models heavily overpredict VT. While VT record recall is 91%–99%, VF record recall drops to 31.5%–43.0% (missing 4 to 5 of the 6 VF records).
  * In **D2** (Train Challenge 2015+CUDB $\to$ Test VFDB): Models heavily overpredict VF. While VF record recall is 95%–100%, VT record recall collapses to 16.2%–25.3% (misclassifying 11 to 13 of the 19 VT records as VF).
  * Consequently, D2 Record Macro F1 is capped at **0.2986 – 0.3592**, failing the balanced generalization requirement.

### Question 3: Why do physiological features also fail cross-database generalization?
Three fundamental root causes explain this ceiling:
1. **Critical Clinical Patient Bottleneck (~16 VF Patients)**:
   * Across the entire combined corpus, there are only **16 unique physical patient records** with Ventricular Fibrillation (6 in Challenge 2015, 8 in VFDB, 2 in CUDB).
   * While sliding windows produce 1,968 VF samples, repeated overlapping windows from 16 individuals cannot substitute for morphological diversity. When a model trains on 8-10 patients in one domain, it overfits to individual waveform idiosyncrasies rather than universal VF physiology.
2. **Clinical Phenotype Mismatch Across Datasets**:
   * **Challenge 2015**: Recorded in ICU/bedside monitoring environments; VF alarms are predominantly triggered by organized, high-amplitude, monomorphic ventricular flutter or regular tachyarrhythmia precursors.
   * **VFDB**: Recorded from out-of-hospital Holter monitors during sudden cardiac arrest; VF segments are chaotic, coarse/fine, fragmented fibrillatory waves.
   * Because the underlying cardiac states labeled as "VF" differ physiologically between these two datasets, features trained on one will inevitably misinterpret the other.
3. **Instrumentation & Amplitude Discrepancies**:
   * Even after unit calibration to physical millivolts, baseline wander, electrode filtering, and dynamic range vary significantly between telemetry monitors and ambulatory tape recorders.

---

## 5. Decision Gate Verdict & Recommendations

### Decision: **REJECTED: CRITERIA NOT MET**
* **D2 ROC-AUC Gate (>0.75)**: FAILED (Max achieved: 0.6825).
* **Record Balanced Recall Gate (>70% in both classes)**: FAILED (D1 VF record recall = 31.5%; D2 VT record recall = 25.3%).
* **Domain Disentanglement**: PARTIAL. Complexity and autocorrelation features eliminate sign inversion, but the ~16 patient VF corpus remains too small to support robust domain generalization.

### Next Research Step (Phase 6)
* **Corpus Expansion Prerequisite**: Before investing further in feature engineering or edge deep neural networks, the project must acquire additional independent clinical VT/VF records (e.g., Creighton University Tachyarrhythmia Database expansion, MIT-BIH Malignant Ventricular Arrhythmia Database, or AHA ECG Database) to expand the independent VF patient count from 16 to $\ge 50$.
* **Do Not Deploy to Production**: Model 3 must remain a research prototype. It should not replace the existing production baseline or be exported to production ONNX runtime until cross-database generalization is validated on an expanded cohort.
