# Preregistered Evaluation Protocol: Held-Out Test Evaluation of Causal Patient-Adaptive 5-Second AFib Classifier

**Status**: PREREGISTERED & FROZEN (UNEXECUTED)  
**Parent Study**: [`ADAPTIVE_BASELINE_FEASIBILITY_REPORT.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/ADAPTIVE_BASELINE_FEASIBILITY_REPORT.md)  
**Target Evaluation Cohort**: Quarantined AFDB Held-Out Test Cohort (4 patients: `04908`, `07162`, `08215`, `08219`)  
**Data File**: [`training/research/afib_5s_classifier/data/afdb_5s_features_test.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/data/afdb_5s_features_test.csv) ($N = 29,260$ non-overlapping windows)  
**Protocol Rule**: ZERO TEST SAMPLES MAY BE PROCESSED, EVALUATED, OR ACCESSED PRIOR TO EXPLICIT USER AUTHORIZATION.

---

## 1. Study Objective & Research Hypothesis

### 1.1 Objective
To empirically determine on the quarantined, untouched held-out test cohort whether the **causal patient-adaptive baseline pipeline (Config C, 26 features)** resolves the cross-patient logit inversion and score uncalibration that degraded the original model's test performance ($\text{ROC-AUC} = 0.6615$).

### 1.2 Hypotheses Under Evaluation
1. **Primary Hypothesis ($H_1$)**: Incorporating causal running baseline features will normalize baseline resting rate differences (e.g. 66 bpm in `07162` vs. 97 bpm in `08219`) and increase the pooled held-out test ROC-AUC from **0.6615** to **$\ge 0.8500$**.
2. **Secondary Hypothesis ($H_2$)**: Patient `08219`'s sinus arrhythmia false alarms ($2,235$ alarms in the original model) will be reduced by at least $40\%$ due to baseline-normalized RR CV tracking.
3. **Secondary Hypothesis ($H_3$)**: Sensitivity on sustained slow AFib (`07162`, originally $25.07\%$) will increase to $\ge 50.0\%$ because low resting rates will no longer drag predictions below the detection threshold.

---

## 2. Frozen Feature Specification (26 Features)

The feature vector for every window $k$ shall consist of exactly 26 features, constructed by concatenating the 21 existing physical features with the 5 causal adaptive features:

### 2.1 The 21 Original Physical Features
1. `n_beats`
2. `hr_mean_bpm`
3. `hr_std_bpm`
4. `rr_cv`
5. `rr_rmssd`
6. `rr_mad`
7. `rr_pnn50`
8. `rr_sample_dispersion`
9. `rr_turning_point_ratio`
10. `alternating_sign_index`
11. `f_band_power_ratio`
12. `f_band_flatness`
13. `f_band_peak_purity`
14. `p_band_power_ratio`
15. `qrs_power_ratio`
16. `hjorth_mobility`
17. `hjorth_complexity`
18. `ac_std_mv`
19. `ac_ptp_mv`
20. `zero_crossings`
21. `lz_complexity`

### 2.2 The 5 Causal Patient-Adaptive Baseline Features
22. `delta_hr_bpm`: $\text{HR}_t - \text{HR}_{\text{base}, t}$
23. `rr_cv_relative`: $\text{RR\_CV}_t / (\text{RR\_CV}_{\text{base}, t} + 0.01)$
24. `hr_z_robust`: $(\text{HR}_t - \text{HR}_{\text{base}, t}) / (1.4826 \times \text{MAD}_{\text{hr}, t} + 2.0)$
25. `afib_index_relative`: $\text{RR\_CV}_{\text{relative}, t} \times (f\text{\_band}_t / (f\text{\_band}_{\text{base}, t} + 0.05))$
26. `baseline_warmup_fraction`: $\min(1.0, N_{\text{buffered}} / 12)$

---

## 3. Frozen Tracker & Preprocessing Hyperparameters

The following parameters are permanently locked based on training cohort estimation:

```json
{
  "tracker_parameters": {
    "history_window_sec": 300.0,
    "min_warmup_windows": 12,
    "max_gap_sec": 300.0,
    "anti_contamination": true,
    "rate_percentile": 25.0,
    "cv_percentile": 20.0,
    "gating_cv_threshold": 0.18,
    "epsilons": {
      "eps_cv": 0.01,
      "eps_mad": 2.0,
      "eps_f": 0.05
    },
    "clipping_bounds": {
      "hr_baseline_min_max": [40.0, 160.0],
      "rr_cv_baseline_min_max": [0.015, 0.25],
      "f_band_baseline_min_max": [0.05, 0.90]
    }
  },
  "population_priors_from_train": {
    "hr_mean_bpm": 83.5591,
    "rr_cv": 0.1016,
    "f_band_power_ratio": 0.3952
  }
}
```

* **Scaler**: `StandardScaler` fitted strictly on the 15 training patients ($N = 217,851$ windows). No test samples shall enter scaler mean/variance computation.
* **Classifier Model**: Frozen L2-regularized balanced Logistic Regression trained on Config C.
  - Intercept: $-0.4126$
  - Learned weights as documented in [`adaptive_study_validation_metrics.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/adaptive_study_validation_metrics.json).

---

## 4. Frozen Operating Thresholds

In accordance with strict preregistration rules, test evaluation shall be conducted at two explicitly predefined operating thresholds without post-hoc tuning:

1. **Fixed Legacy Threshold**:
   $$\theta^* = 0.560$$
   *Direct backwards compatibility with the original frozen 5-second model report.*
2. **High-Specificity Triage Threshold**:
   $$\theta_{\text{spec95}} = 0.760$$
   *Selected exclusively from the validation sweep for $\text{Specificity} \ge 95.0\%$.*

---

## 5. Primary and Secondary Success Criteria

The test evaluation will be considered a **Confirmed Success** if:
1. **Global Pooled ROC-AUC**: Improves from **$0.6615$** to **$\ge 0.8500$** (Statistically significant at $\alpha = 0.01$).
2. **False Alarm Rate**: Decreases from **$157.45\text{ FP/hr}$** to **$\le 75.0\text{ FP/hr}$** at $\theta^* = 0.560$.
3. **Record `08219` Specificity**: Improves from **$60.77\%$** to **$\ge 80.0\%$**.
4. **Record `07162` Sensitivity**: Improves from **$25.07\%$** to **$\ge 50.0\%$**.
5. **Patient-Level ROC-AUC**: Retains $> 0.85$ on all patients with rhythm transitions (`04908`, `08215`, `08219`).

---

## 6. Execution Safeguards & Anti-Contamination Rules

1. **Single-Shot Execution**: The test evaluation script must execute exactly once. Iterative re-running or parameter adjustment against test metrics is strictly forbidden.
2. **Zero In-Memory Leakage**: When iterating over the 4 test records, `tracker.reset(record_id)` must be called before the first window of each record.
3. **Window Stride**: Held-out test evaluation shall use the established non-overlapping $5.0\text{s}$ stride ($N = 29,260$ windows) as recorded in `afdb_5s_features_test.csv`.
4. **No Model Re-Selection**: The model architecture, feature weights, and threshold are frozen. No alternative models shall be substituted after observing test metrics.
