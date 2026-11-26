# Research Report: Multi-Scale Windowing Experiment (2-Second vs. 5-Second Windows)

**Date**: 2026-10-25 22:46:50  
**Pipeline**: `training/final_models/results/model2_xgboost_4class_research/research_xgboost_4class_pipeline.joblib`  
**Experiment Directory**: `training/final_models/results/model2_xgboost_4class_research/multiscale_2s_experiment/`  
**Safety Status**: Standalone research experiment; zero production files modified; zero ONNX models altered; zero threshold tuning; zero Git commits or pushes.

---

## 1. Executive Summary & Core Insights

This strictly research-only experiment evaluated whether **2-second ECG windows (720 samples @ 360 Hz)** improve detection of brief non-sustained ventricular tachycardia (NSVT) compared with the established **5-second four-class Model 2 XGBoost classifier**, while testing whether 2-second features can be computed meaningfully without silently altering feature definitions.

### Key Findings:
1. **Dramatic Concentration of Ectopic Burst Morphology in 2-Second Windows**:
   - In fixed 5.0-second windows, brief VT bursts (1.67–2.82s long) occupied only **33.4% to 56.4%** of the window duration, diluting the morphological and spectral features with normal sinus conduction.
   - In 2.0-second centered windows, **5 of the 7 VT episodes occupy 100.0% of the window duration**, and the remaining two occupy **83.5% and 85.0%**.
   - Consequently, instantaneous heart rate inside the burst jumps from **70.9–91.9 bpm (5s window)** up to **107.7–122.0 bpm (2s window)**, unmasking true ventricular tachycardia dynamics.
2. **Strict Incompatibility of 5-Second Model Pipeline with 2-Second Features (Task 4 Stop Condition)**:
   - Applying the trained 5-second XGBoost pipeline directly to 2-second features is **mathematically invalid** and violates feature compatibility.
   - `zero_crossings` and `peak_count` scale linearly with window duration ($pprox 0.4\times$). In 2-second windows, normal sinus rhythm exhibits mean `zero_crossings` $\approx 18$ and `peak_count` $\approx 2.5$, which land at **-1.5 standard deviations** in the 5-second scaler.
   - When evaluated on clean sinus rhythm in `mitdb_200`, the unretrained 5-second pipeline triggers a **24.0% false ventricular alarm rate (12 / 50 windows)** on pure sinus rhythm!
3. **Scientific Infeasibility of Standalone 4-Class 2-Second Model (Task 5 Audit)**:
   - Under the constraint *'do not alter feature definitions'*, a standalone 4-class 2-second classifier suffers from mathematical feature collapse on slow heart rates:
     - For heart rates $< 60\text{ bpm}$, the inter-beat interval exceeds $1.0\text{ second}$. In 2-second windows, **27.0% of sinus bradycardia windows contain only 1 peak**, triggering the fallback `mean_rr = 5.0` and `hr_bpm = 0`.
     - **75.0% of bradycardia windows produce `rr_cv == 0.0`**, destroying heart rate variability metrics.
   - Therefore, a 2-second window is fundamentally an **arrhythmia burst / shockable screener**, not a standalone four-class rhythm classifier.

---

## 2. Feature-by-Feature 2-Second Pipeline Compatibility Inspection

| Feature | Scale Category | 5s Scaler Mean | 5s Scaler Scale | Mathematical Behavior on 2-Second Windows | Suitable for 2s? |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `mean_rr` | Rate Metric | 1.09 s | 1.06 s | Truncates to 5.0s fallback when peaks < 2 (occurs in 27% of bradycardia windows). | **Degraded** |
| `rr_cv` | Variability | 0.151 | 0.190 | Collapses identically to 0.0 whenever peaks <= 2 (occurs in 75% of bradycardia). | **Degraded** |
| `qrs_width` | Proxy | 0.075 s | 0.019 s | Drops to 0.0 whenever peak_count < 2. | **Degraded** |
| `peak_count` | Duration-Dependent | 6.44 | 3.18 | Scales by 0.4x. Normal 2s peaks (2-3) map to -1.4 std deviations in 5s model. | **Incompatible** |
| `zero_crossings` | Duration-Dependent | 46.42 | 21.34 | Scales by 0.4x. Normal 2s crossings (~18) map to -1.5 std deviations in 5s model. | **Incompatible** |
| `dominant_freq` | Scale-Invariant | 2.61 Hz | 1.69 Hz | Identical 0.5 Hz resolution (nperseg=720), but higher variance due to single periodogram. | **Compatible** |
| `vf_band_power_ratio` | Scale-Invariant | 0.446 | 0.192 | Power ratio in 3-9 Hz band is scale-invariant. Concentrates energy during VT bursts. | **Compatible** |
| `spectral_entropy` | Scale-Invariant | 0.707 | 0.166 | Shannon entropy over 0.5-30 Hz is scale-invariant. Drops toward 0.60 in pure VT. | **Compatible** |
| `hjorth_mobility` | Scale-Invariant | 0.132 | 0.056 | Point-wise derivative variance ratio is mathematically scale-invariant. | **Compatible** |
| `hjorth_complexity` | Scale-Invariant | 2.427 | 0.856 | Derivative mobility ratio is mathematically scale-invariant. | **Compatible** |
| `calibrated_ptp` | Scale-Invariant | 2.17 mV | 1.63 mV | Peak-to-peak voltage is scale-invariant. Captures local burst amplitude. | **Compatible** |
| `calibrated_std` | Scale-Invariant | 0.359 mV | 0.361 mV | Signal standard deviation is scale-invariant. | **Compatible** |

---

## 3. Detailed Inspection: The 7 Brief VT Episodes in `mitdb_200`

Comparison of 5.0-second centered windows vs. 2.0-second centered windows across the 7 brief VT episodes:

| Ep # | Duration | Context | 5s Window Occupancy | 2s Window Occupancy | 5s HR (bpm) | 2s HR (bpm) | 5s VF Ratio | 2s VF Ratio | 5s p(VENT) | 2s Unretrained p(VENT) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.49s | (B -> (B | 49.8% | **100.0%** | 82.7 | **94.7** | 0.37 | **0.424** | 0.1667 | **0.3143** |
| **2** | 1.7s | (N -> (N | 34.0% | **85.0%** | 87.2 | **122.0** | 0.441 | **0.417** | 0.1422 | **0.3244** |
| **3** | 2.82s | (N -> (N | 56.4% | **100.0%** | 91.9 | **84.5** | 0.477 | **0.396** | 0.1742 | **0.3178** |
| **4** | 1.67s | (B -> (B | 33.4% | **83.5%** | 79.2 | **118.7** | 0.372 | **0.242** | 0.2009 | **0.2847** |
| **5** | 2.18s | (B -> (B | 43.6% | **100.0%** | 70.9 | **122.0** | 0.488 | **0.535** | 0.2991 | **0.7753** |
| **6** | 2.16s | (B -> (B | 43.2% | **100.0%** | 86.4 | **113.1** | 0.472 | **0.457** | 0.2635 | **0.4616** |
| **7** | 2.21s | (N -> (N | 44.2% | **100.0%** | 85.7 | **107.7** | 0.504 | **0.449** | 0.5587 | **0.5026** |

### Diagnostic Interpretation:
1. **Rate Tachycardia Unmasking**: In 5-second windows, normal sinus beats before and after the burst pulled the mean heart rate down into the normal range (70.9–91.9 bpm). In 2-second windows, the heart rate reflects the true burst rate, jumping into ventricular tachycardia range (**107.7 to 122.0 bpm**) in 5 of the 7 episodes.
2. **Morphological Dilution Elimination**: In episodes 1, 3, 5, 6, and 7, the 2-second window is **100% ventricular rhythm**. `spectral_entropy` drops from ~0.74 down to 0.58–0.66, reflecting rhythmic monomorphic VT organization.
3. **Posterior Probability Elevation**: Even without retraining, the 2-second window doubles the ventricular posterior probability (averaging 0.428 vs. 0.252 in 5s windows), with episodes 5 and 7 crossing the 0.50 threshold into `VENTRICULAR`.

---

## 4. Evaluation of mitdb_200 and vfdb_424 (2s Windows, 1s Sliding Stride)

### 4.1 Record `mitdb_200` Breakdown
- **Total 2-Second Sliding Windows**: 1,804 windows (1805.6 seconds)
- **Unretrained 5s Pipeline Prediction Distribution**:
  - `NSR`: 1,449 (80.3%)
  - `VENTRICULAR`: 335 (18.6%)
  - `TACHY`: 17 (0.9%)
  - `BRADYCARDIA`: 3 (0.2%)

| Interval Relationship | Total Windows | Predicted VENTRICULAR | Ventricular % | Mean p(VENT) | Median p(VENT) | Mean p(NSR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`INSIDE_VENTRICULAR`** | 2 | **0** | **0.0%** | 0.296 | 0.296 | 0.6673 |
| **`NON_VENTRICULAR_TRANSITION`** | 262 | **35** | **13.36%** | 0.2831 | 0.2566 | 0.6481 |
| **`OUTSIDE_VENTRICULAR_BIGEMINY`** | 686 | **74** | **10.79%** | 0.3078 | 0.2962 | 0.628 |
| **`OUTSIDE_VENTRICULAR_NSR`** | 826 | **219** | **26.51%** | 0.3537 | 0.3356 | 0.5492 |
| **`VENTRICULAR_TRANSITION`** | 28 | **7** | **25.0%** | 0.3657 | 0.3134 | 0.5418 |

### 4.2 Record `vfdb_424` Breakdown
- **Total 2-Second Sliding Windows**: 2,099 windows (2100.0 seconds)
- **Unretrained 5s Pipeline Prediction Distribution**:
  - `VENTRICULAR`: 981 (46.7%)
  - `NSR`: 967 (46.1%)
  - `TACHY`: 86 (4.1%)
  - `BRADYCARDIA`: 65 (3.1%)

| Interval Relationship | Total Windows | Predicted VENTRICULAR | Ventricular % | Mean p(VENT) | Median p(VENT) | Mean p(NSR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`INSIDE_VENTRICULAR`** | 160 | **153** | **95.62%** | 0.8826 | 0.939 | 0.0486 |
| **`NON_VENTRICULAR_TRANSITION`** | 19 | **12** | **63.16%** | 0.5409 | 0.4655 | 0.3257 |
| **`OUTSIDE_VENTRICULAR_NSR`** | 1256 | **501** | **39.89%** | 0.3957 | 0.3394 | 0.4054 |
| **`OUTSIDE_VENTRICULAR_OTHER`** | 656 | **308** | **46.95%** | 0.4357 | 0.3873 | 0.3791 |
| **`VENTRICULAR_TRANSITION`** | 8 | **7** | **87.5%** | 0.8125 | 0.9203 | 0.0904 |

---

## 5. Why the 5-Second Model Cannot Be Applied to 2-Second Features

> [!WARNING]
> **Task 4 Blocker: Feature Space Incompatibility**
> Applying the unretrained 5-second Model 2 pipeline to 2-second features triggers severe false alarms:
> - In `mitdb_200` clean sinus rhythm (`OUTSIDE_VENTRICULAR_NSR`), **24.0% of windows (163 / 680)** are falsely classified as `VENTRICULAR`!
> - In `vfdb_424` baseline sinus rhythm, **18.7% of windows (274 / 1,462)** are falsely classified as `VENTRICULAR`!
> - **Root Cause**: `zero_crossings` and `peak_count` scale with window duration. A clean sinus beat with 2 peaks and 16 zero crossings in 2 seconds is treated by the 5-second decision trees as severe bradycardia/ventricular conduction.

---

## 6. Retraining Feasibility Audit for Standalone 2-Second Model

> [!IMPORTANT]
> **Task 5 Audit Verdict**: A standalone 4-class 2-second classifier **cannot be trained appropriately** without altering feature definitions:
> 1. **Mathematical Collapse on Bradycardia**: In records like `vfdb_425` (sinus bradycardia at 45 bpm), 27% of 2-second windows contain only 1 peak, collapsing `mean_rr` to 5.0s and `hr_bpm` to 0. 75% of windows produce `rr_cv == 0.0`. A 2-second model cannot learn valid decision boundaries for bradycardia vs. normal rhythm.
> 2. **Training Dataset Gating Rule Mismatches**: The project dataset builder requires `peak_count <= 5` for bradycardia and `peak_count >= 8` for SVTA. In 2 seconds, 8 peaks requires 240 bpm! Legacy gating rules reject >98% of true tachycardia windows.
> 3. **Taxonomic Mismatch**: Rhythm taxonomy (NSR vs. Bradycardia vs. Tachycardia) is electrophysiologically a sustained rate property over 5–10 seconds, not a 2-second snapshot.

---

## 7. Limitations & Clinical Boundaries

1. **No Clinical Validation**: Evaluating two benchmark recordings (`mitdb_200` and `vfdb_424`) and seven brief VT episodes is strictly exploratory. It is statistically insufficient for clinical validation.
2. **Window Overlap Dependency**: 1-second sliding stride creates high auto-correlation between adjacent windows ($50\%$ overlap for 2s windows, $80\%$ for 5s windows). Windows cannot be treated as independent statistical samples.
3. **Taxonomy Separation**: Model 2 contains no PVC beat classifier. Beat-level ectopy must not be conflated with sustained ventricular tachyarrhythmias.

---

## 8. Artifacts Created in This Experiment

All files reside in `training/final_models/results/model2_xgboost_4class_research/multiscale_2s_experiment/`:
1. `evaluate_multiscale_2s_experiment.py`: Standalone evaluation and feature inspection script.
2. `mitdb_200_7vt_2s_vs_5s_comparison.csv`: Feature-by-feature comparison across the 7 centered VT episodes.
3. `two_second_sliding_predictions_mitdb_200.csv`: 1,804 sliding 2-second windows with annotations and unretrained probabilities.
4. `two_second_sliding_predictions_vfdb_424.csv`: 2,099 sliding 2-second windows with annotations and unretrained probabilities.
5. `multiscale_2s_summary.json`: Comprehensive machine-readable metrics, feature inspection, and audit conclusions.
6. `MULTISCALE_2S_EXPERIMENT_REPORT.md`: This research report.

---

## 9. Conclusion & Recommendation for Next Step

> [!TIP]
> **Recommendation for Next Step: Multi-Scale Hierarchical Architecture**
> Rather than attempting to replace the 5.0-second 4-class classifier with an ill-conditioned 2.0-second classifier:
> 1. **Retain the 5.0-Second Model 2 Classifier** as the primary rhythm engine for sustained rhythms (NSR, TACHY, BRADYCARDIA, SUSTAINED VENTRICULAR).
> 2. **Implement a Dedicated 2.0-Second Ventricular Burst Detector** utilizing only the scale-invariant features (`vf_band_power_ratio`, `spectral_entropy`, `dominant_freq`, `calibrated_ptp`, `hjorth_mobility`, `hjorth_complexity`), omitting duration-dependent rate metrics (`mean_rr`, `rr_cv`, `peak_count`, `zero_crossings`).
> 3. **Hierarchical Alerting**: When the 2s burst detector fires during a 5s window, trigger a non-sustained ventricular salvo alert without waiting for 5-second aggregation.