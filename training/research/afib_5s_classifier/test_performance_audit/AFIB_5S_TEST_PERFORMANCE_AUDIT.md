# Comprehensive Audit: Root-Cause Investigation of the 5-Second AFib Held-Out Test Performance Discrepancy

**Audit Scope**: Root-cause diagnostic audit of the 5-second Atrial Fibrillation held-out test evaluation  
**Audit Directory**: [`training/research/afib_5s_classifier/test_performance_audit/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/test_performance_audit/)  
**Primary Execution Audit Script**: [`audit_held_out_discrepancy.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/test_performance_audit/audit_held_out_discrepancy.py)  
**Evaluated Artifacts**:
- Frozen Model Package: [`afib_5s_model_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afib_5s_model_package.joblib)
- Training Configuration: [`afib_5s_training_config.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afib_5s_training_config.json)
- Held-Out Test Feature Table: [`afdb_5s_features_test.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/data/afdb_5s_features_test.csv) ($N = 29,260$ windows)
- Validation Feature Table: [`afdb_5s_features_val.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/data/afdb_5s_features_val.csv) ($N = 58,645$ windows)
- Split Manifest: [`afdb_5s_split_manifest.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afdb_5s_split_manifest.json)
- Research Report: [`AFIB_5S_CLASSIFIER_REPORT.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/AFIB_5S_CLASSIFIER_REPORT.md)

**Strict Engineering & Safety Policy Bounds**:
- Zero models trained or retrained.
- Zero decision thresholds tuned or altered.
- Zero labels or window definitions modified.
- Zero production code, deployed ONNX models, or Model 2 touched.
- The held-out test split was NOT altered or used to select any new model.

---

## 1. Executive Summary & The Single Highest-Priority Root Cause

### 1.1 The Single Highest-Priority Cause: Cross-Patient Baseline Shift and Uncalibrated Inter-Patient Logit Inversion (Simpson's Paradox)

The investigation confirms that the collapse of validation performance ($\text{ROC-AUC} = \mathbf{0.9832}$) to held-out test performance ($\text{ROC-AUC} = \mathbf{0.6615}$) is **NOT** caused by an inability of the model to distinguish AFib from sinus rhythm within an individual patient. 

Instead, the single highest-priority cause is an **inter-patient score uncalibration induced by heavy model reliance on absolute heart rate (`hr_mean_bpm`) and absolute RR variance (`rr_cv`) without patient-specific baseline centering**:

```
+------------------------------------------------------------------------------------------------------------------------+
|                                    CROSS-PATIENT SCORE INVERSION (SIMPSON'S PARADOX)                                   |
+------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                        |
|   PATIENT 08219 (Tachycardic Sinus Baseline):                                                                         |
|   True Normal Sinus Rhythm (HR ~ 97 bpm, RR CV ~ 0.117)  -------> Mean P(AFib) = 0.4108  (Median = 0.3932)            |
|                                                                         |                                              |
|                                                              HIGHER LOGIT SCORE                                        |
|                                                                    INVERSION                                           |
|                                                                         v                                              |
|   PATIENT 07162 (Normofrequent / Bradycardic AFib):                                                                    |
|   True Sustained Atrial Fibrillation (HR ~ 66 bpm)        -------> Mean P(AFib) = 0.3491  (Median = 0.2845)            |
|                                                                                                                        |
|   Empirical Result: In 52.47% of cross-patient pairs, Patient 07162's TRUE AFIB has a LOWER predicted score             |
|   than Patient 08219's TRUE NORMAL SINUS RHYTHM!                                                                       |
|                                                                                                                        |
+------------------------------------------------------------------------------------------------------------------------+
```

When evaluated within individual test patients with rhythm transitions, the model demonstrates high discriminability:
- **Patient `04908`**: **Patient ROC-AUC = 0.9686**, **Sensitivity = 98.00%**, **Specificity = 88.73%**
- **Patient `08215`**: **Patient ROC-AUC = 0.9363**, Specificity = 97.48%
- **Patient `08219`**: **Patient ROC-AUC = 0.8944**, **Sensitivity = 94.37%**

However, because the logistic regression assigns large positive weights to heart rate ($\beta_{\text{hr}} = +1.7423$) and RR interval CV ($\beta_{\text{rr\_cv}} = +5.2012$), Patient `08219`'s sinus tachycardia with sinus arrhythmia is shifted upward into the AFib decision space, while Patient `07162`'s slow, low-variability AFib is shifted downward into the normal sinus decision space.

When all 29,260 windows are pooled into a single global ROC curve, **thousands of normal sinus windows from Patient `08219` are ranked higher than thousands of true AFib windows from Patient `07162` and `08215`**. This cross-patient ranking inversion artificially destroys the global pooled ROC-AUC ($0.6615$), despite high intra-patient discriminatory power.

---

## 2. Detailed Investigation of the Five Assigned Questions

### Question 1: Why Validation Performance (ROC-AUC 0.9832) Fell to Held-Out Test ROC-AUC 0.6615

#### Empirical Validation Breakdown
In the validation set, all 4 patients (`04126`, `05121`, `07879`, `07910`) presented with relatively uniform baseline physiology:
- Mean validation heart rate across records was **78.27 bpm** (standard deviation across patient means was $< 6.5\text{ bpm}$).
- When AFib occurred in validation patients, ventricular conduction rates were consistently elevated ($> 90\text{ bpm}$), aligning with the global scaler mean.
- Per-record validation ROC-AUCs were uniformly high and concordant:
  - Record `04126`: $\text{AUC} = \mathbf{0.9954}$ ($N = 14,670$, prevalence = $3.6\%$)
  - Record `05121`: $\text{AUC} = \mathbf{0.9550}$ ($N = 14,569$, prevalence = $63.2\%$)
  - Record `07879`: $\text{AUC} = \mathbf{0.9973}$ ($N = 14,714$, prevalence = $60.3\%$)
  - Record `07910`: $\text{AUC} = \mathbf{0.9869}$ ($N = 14,692$, prevalence = $15.9\%$)
- Because validation patients had overlapping, well-aligned normal sinus baselines and separated AFib clusters, the pooled validation ROC-AUC reached **0.9832**.

#### Empirical Test Set Divergence
In the quarantined held-out test cohort (`04908`, `07162`, `08215`, `08219`), patient baselines diverged sharply:
- Patient `07162` had a mean heart rate of **66.44 bpm** (bradycardic/normofrequent AFib).
- Patient `04908` had a mean heart rate of **103.28 bpm** (tachycardic baseline).
- Patient `08219` had a mean heart rate of **99.45 bpm** with chronic sinus arrhythmia (`rr_cv` = 0.1168 during normal sinus rhythm, exceeding the training dataset average of 0.10).

Because the linear classifier treats an elevated heart rate as evidence for AFib ($\beta_{\text{hr\_mean\_bpm}} = +1.7423$) and a reduced heart rate as evidence for sinus rhythm, the model's posterior probabilities shifted systematically by patient rather than by rhythm:
- Patient `08219` Non-AFib windows received an average probability of **$0.4108$** (with 10% of sinus windows reaching $> 0.885$).
- Patient `07162` AFib windows received an average probability of only **$0.3491$** (with 50% of AFib windows falling below $0.285$).

In ROC analysis, every pair where a negative sample has a higher score than a positive sample decreases the area under the curve. Because Patient `07162` contributed **7,363 AFib windows** and Patient `08219` contributed **5,697 Non-AFib windows**, the cross-record comparison between these two patients alone generated $7,363 \times 5,697 = 41,946,011$ pairwise comparisons, of which **52.47% were inverted**. This cross-patient score shift directly explains the global AUC degradation.

---

### Question 2: Variation by AFib Patterns, Label Alignment, Distribution Shifts, or Class Imbalance

We examined the four potential hypotheses systematically:

```
+------------------------------------+-----------------------+-------------------------------------------------------------+
| Hypothesized Cause                 | Evaluation Status     | Empirical Evidence & Diagnostic Audit                       |
+------------------------------------+-----------------------+-------------------------------------------------------------+
| 1. Patient-Specific AFib Patterns  | PRIMARY ROOT CAUSE    | Normofrequent AFib in 07162 evaded global rate features.   |
|                                    |                       | Sinus arrhythmia in 08219 mimicked AFib irregularity.       |
| 2. Feature Distribution Shifts     | CRITICAL CONTRIBUTOR  | Baseline HR varied by 37 bpm across patients (+/-2.0 sigma) |
|                                    |                       | Signal amplitude in 07162 was 0.62 mV vs 2.86 mV train mean |
| 3. Class Imbalance Across Patients | SECONDARY DRIVER      | 07162 was 100% AFib; 04908 was 8.2% AFib.                   |
|                                    |                       | 07162 and 08215 contributed 86.3% of all test AFib windows  |
| 4. Label / Window Misalignment     | RULED OUT             | 2.5s transition guard bands eliminated boundary leakage     |
|                                    |                       | 1,847 mixed windows were audited and rejected               |
+------------------------------------+-----------------------+-------------------------------------------------------------+
```

#### Detailed Breakdown of Hypotheses:

1. **Patient-Specific AFib Patterns (Verified as Primary)**:
   - Atrial Fibrillation exhibits marked electrophysiological heterogeneity:
     - In **Rapid AFib** (Patient `04908`), ventricular conduction is rapid ($113.7\text{ bpm}$) and chaotic (`rr_cv` = $0.1841$). The model achieved **98.00% sensitivity**.
     - In **Controlled AFib** (Patient `07162`), AV nodal filtering keeps ventricular rate low ($66.4\text{ bpm}$) and intervals relatively regular. The model achieved only **25.07% sensitivity** under the global threshold ($\theta^* = 0.560$).
     - In **Sinus Arrhythmia** (Patient `08219`), sinus rhythm intervals varied with respiration (`rr_cv` = $0.1168$), causing **39.2% false alarms** on non-AFib segments.

2. **Feature Distribution Shifts (Verified as Critical Contributor)**:
   - Inspecting the raw feature distributions across splits:
     - `hr_mean_bpm`: Train mean = **83.56 bpm**, Val mean = **78.27 bpm**. In Test, Patient `07162` was **66.44 bpm** ($-1.03\sigma$) while Patient `04908` was **103.28 bpm** ($+1.12\sigma$). This represents a $2.15\sigma$ spread across patients.
     - `ac_ptp_mv`: Train mean = **2.86 mV**. Patient `07162` had a mean peak-to-peak amplitude of only **0.62 mV** ($-0.78\sigma$), significantly reducing f-wave signal-to-noise ratio and decreasing `f_band_power_ratio` to $0.2286$ (vs $0.40$ in training data).

3. **Label and Window Alignment (Ruled Out as Cause)**:
   - Transition guard bands of 2.5 seconds were strictly enforced around all WFDB rhythm markers.
   - 1,847 windows falling within transition zones were automatically quarantined into [`afdb_5s_rejected_windows.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/data/afdb_5s_rejected_windows.csv).
   - Every retained window in `afdb_5s_features_test.csv` was verified to reside entirely within a single pure rhythm segment.

4. **Class Imbalance (Secondary Amplifying Factor)**:
   - The test set has a macro-prevalence of $52.65\%$, but individual patient burdens vary drastically:
     - `04908`: 8.18% AFib ($N = 600$ AFib windows)
     - `08219`: 20.95% AFib ($N = 1,510$ AFib windows)
     - `08215`: 80.62% AFib ($N = 5,931$ AFib windows)
     - `07162`: 100.00% AFib ($N = 7,363$ AFib windows)
   - Because `07162` contains no non-AFib windows, its 5,517 false negatives entered the global sensitivity calculation without contributing any true negatives to offset them.

---

### Question 3: Whether Reported Window-Level Metrics Are Dominated by a Small Number of Long Records

**Audit Finding: YES. The window-level metrics are heavily dominated by two records for positives and two records for negatives.**

Although each patient has an identical total recording duration ($\sim 10\text{ hours}$, yielding $\sim 7,200\text{--}7,360$ non-overlapping 5-second windows per patient), their rhythm compositions create extreme sample concentration:

#### Positive AFib Window Dominance
- Total AFib windows in held-out test cohort: **15,404 windows**.
- Record `07162`: **7,363 AFib windows (47.80% of all test AFib windows)**.
- Record `08215`: **5,931 AFib windows (38.50% of all test AFib windows)**.
- Record `08219`: 1,510 AFib windows (9.80%).
- Record `04908`: 600 AFib windows (3.90%).

> **Critical Observation**: Records `07162` and `08215` account for **86.30% of all true AFib windows in the held-out test set** ($13,294 / 15,404$). Because both patients had low-to-moderate sensitivity ($25.07\%$ and $36.84\%$), their sheer sample volume completely dominates the global window-level sensitivity ($39.24\%$), masking the high sensitivity achieved on `04908` ($98.00\%$) and `08219` ($94.37\%$).

#### Negative Non-AFib Window Dominance
- Total Non-AFib windows in held-out test cohort: **13,856 windows**.
- Record `04908`: **6,733 non-AFib windows (48.59%)**.
- Record `08219`: **5,697 non-AFib windows (41.12%)**.
- Record `08215`: 1,426 non-AFib windows (10.29%).
- Record `07162`: 0 non-AFib windows (0.00%).

> **Critical Observation**: Records `04908` and `08219` account for **89.71% of all non-AFib windows** ($12,430 / 13,856$). Furthermore, of the 3,030 total false alarms in the test set, Record `08219` alone produced **2,235 false alarms (73.76% of all false alarms)**. The global specificity ($78.13\%$) and false alarm rate ($157.45\text{ FP/hr}$) are effectively measurements of Patient `08219`'s sinus arrhythmia.

---

### Question 4: Whether the Threshold and Reported False Alarms per Hour Are Calculated Consistently

**Audit Finding: The mathematical rate definition ($\text{FP} / \text{hours}$) was calculated consistently, but there is an important operational difference between overlapping validation windows and non-overlapping test windows.**

#### Mathematical Verification of the Calculation
1. **Held-Out Test Set**:
   - Stride: $5.0\text{ seconds}$ (non-overlapping).
   - Total Non-AFib windows: $13,856$.
   - Total Non-AFib time:
     $$\text{Duration} = \frac{13,856 \times 5.0\text{ s}}{3,600\text{ s/hr}} = 19.2444\text{ hours}$$
   - False Alarms at $\theta^* = 0.560$: $3,030\text{ windows}$.
   - Reported False Alarms per Hour:
     $$\text{FP / hr} = \frac{3,030}{19.2444\text{ hours}} = \mathbf{157.45\text{ FP/hr}}$$
   - **Verification**: Exactly matches the reported metric in [`afib_5s_test_metrics.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afib_5s_test_metrics.json).

2. **Validation Set**:
   - Stride: $2.5\text{ seconds}$ (overlapping).
   - Total Non-AFib windows: $37,708$.
   - Total Non-AFib time:
     $$\text{Duration} = \frac{37,708 \times 2.5\text{ s}}{3,600\text{ s/hr}} = 26.1861\text{ hours}$$
   - False Alarms at $\theta^* = 0.560$: $1,781\text{ windows}$.
   - Reported False Alarms per Hour:
     $$\text{FP / hr} = \frac{1,781}{26.1861\text{ hours}} = \mathbf{68.01\text{ FP/hr}}$$
   - **Verification**: Exactly matches the reported metric in [`afib_5s_training_config.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afib_5s_training_config.json).

#### Operational Nuance in Stride Comparison
While the dimensional units ($\text{alarms} / \text{hour}$) are identical, the opportunity frequency differed:
- With a $2.5\text{s}$ stride (validation), there are $1,440\text{ window evaluations per hour}$. A rate of $68.01\text{ FP/hr}$ corresponds to a per-window false positive rate of **$4.72\%$** ($68.01 / 1440$), which matches the reported validation specificity of $95.28\%$ ($1 - 0.0472$).
- With a $5.0\text{s}$ stride (test), there are $720\text{ window evaluations per hour}$. A rate of $157.45\text{ FP/hr}$ corresponds to a per-window false positive rate of **$21.87\%$** ($157.45 / 720$), which matches the reported test specificity of $78.13\%$ ($1 - 0.2187$).

Therefore, the metric is mathematically consistent, and the increase in false alarms per hour accurately reflects the true drop in specificity from $95.28\%$ on validation to $78.13\%$ on held-out test data.

---

### Question 5: Which Findings Can Be Verified Without Retraining or Changing Thresholds

Every finding reported in this audit was mathematically verified directly from the existing frozen model, serialized predictions, and dataset tables without retraining, tuning, or changing labels:

1. **Intra-Patient ROC-AUCs ($0.9686$, $0.9363$, $0.8944$)**:
   - Calculated directly from `prob_afib` column in [`afdb_5s_features_test.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/data/afdb_5s_features_test.csv).
2. **Cross-Patient Score Inversion Rate ($52.47\%$)**:
   - Evaluated by directly comparing Patient `07162` true AFib probabilities ($N = 7,363$) against Patient `08219` true normal sinus probabilities ($N = 5,697$).
3. **AFib Window Concentration ($86.30\%$)**:
   - Audited directly from class counts in `afdb_5s_features_test.csv` (13,294 / 15,404 windows reside in `07162` and `08215`).
4. **False Alarm Concentration ($73.76\%$)**:
   - Audited directly: 2,235 of the 3,030 total false alarms originate in Record `08219`.
5. **Impact of Removing Record `07162`**:
   - Without retraining, recomputing sensitivity on the remaining 3 patients (`04908`, `08215`, `08219`) increases sensitivity from **39.24% to 52.21%**, while global ROC-AUC rises from **0.6615 to 0.7350**.
6. **Feature Distribution Shifts**:
   - Heart rate divergence (66.4 bpm in `07162` vs 103.3 bpm in `04908`) and voltage attenuation (0.62 mV in `07162` vs 2.86 mV train mean) are verified from existing feature columns.

---

## 3. Comprehensive Synthesis & Audit Verdict

```
+------------------------------------------------------------------------------------------------------------------------+
|                                    AUDIT SUMMARY: WHY 5-SECOND TEST AUC WAS 0.6615                                     |
+------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                        |
|   1. The model DOES NOT fail at detecting AFib within a patient.                                                      |
|      -> Patient AUCs: 0.9686 (04908), 0.9363 (08215), 0.8944 (08219).                                                 |
|                                                                                                                        |
|   2. The model FAILS at inter-patient score calibration.                                                               |
|      -> Heavy beta weights on absolute heart rate (+1.74) and absolute RR CV (+5.20) cause patient baselines           |
|         to shift logits by +/- 3.5 units.                                                                              |
|                                                                                                                        |
|   3. Simpson's Paradox destroys the global pooled ROC-AUC.                                                             |
|      -> 52.47% of Patient 07162's true AFib windows score LOWER than Patient 08219's normal sinus windows.             |
|                                                                                                                        |
|   4. Two records dominate the positive class; two records dominate the negative class.                                 |
|      -> Records 07162 and 08215 contain 86.3% of all test AFib windows.                                                |
|      -> Record 08219 generates 73.8% of all false alarms.                                                              |
|                                                                                                                        |
+------------------------------------------------------------------------------------------------------------------------+
```

---

## 4. Single Recommended Next Step

> [!IMPORTANT]
> **Single Recommended Next Step: Implement Patient-Adaptive Baseline Tracking (Running Median Centering) for Rate and Dispersion Features**
>
> To resolve cross-patient score uncalibration and Simpson's Paradox without violating causal edge constraints:
> 1. Compute heart rate and interval dispersion relative to a **causal, patient-specific running baseline** (e.g., $\Delta \text{HR} = \text{HR}_{\text{5s}} - \text{HR}_{\text{running\_median\_5min}}$ and $\text{RR\_CV}_{\text{relative}} = \text{RR\_CV}_{\text{5s}} / \text{RR\_CV}_{\text{running\_baseline}}$).
> 2. Normalizing features by each patient's own resting baseline will prevent tachycardic sinus patients (like `08219`) from producing chronic false alarms and prevent bradycardic AFib patients (like `07162`) from being penalized into false negatives.
> 3. This retains 5-second responsive detection while eliminating the inter-patient logit shifts that degraded pooled test performance.

---

## 5. Catalog of Untouched Files

In strict compliance with instructions, the following files were left completely untouched:

### Production Code & Deployed Models (Untouched)
- `deploy/edge_runtime.py`
- `deploy/ads1292r_driver.py`
- `deploy/ads1292r_stream.py`
- `deploy/combined_ecg_app.py`
- `models/shockable_classifier.onnx`
- `models/arrhythmia_multiclass.onnx`
- `models/vf_vt_subtype_classifier.onnx`
- `training/final_models/export_candidates/shockable_classifier_candidate.onnx`
- `training/final_models/export_candidates/arrhythmia_multiclass_4class_candidate.onnx`
- `training/final_models/export_candidates/vf_vt_subtype_classifier_candidate.onnx`
- `training/final_models/results/model2_xgboost_4class_research/research_xgboost_4class_pipeline.joblib`

### Existing Datasets & Prior Research Phases (Untouched)
- `training/research/afib_experiment/afdb_model_package.joblib`
- `training/research/afib_experiment/compatibility_experiment/AFIB_COMPATIBILITY_REPORT.md`
- `training/research/afib_experiment/compatibility_experiment/afib_compatibility_predictions.csv`
- `training/research/afib_experiment/data/afdb/raw/*` (all 23 AFDB raw record files)
- `cu-ventricular-tachyarrhythmia-database-1.0.0.zip`
- `MITBIH.zip`
- `VFDB.zip`

### Completed 5-Second Experiment Artifacts (Audited in Read-Only Mode, Untouched)
- `training/research/afib_5s_classifier/afdb_5s_split_manifest.json`
- `training/research/afib_5s_classifier/afib_5s_model_package.joblib`
- `training/research/afib_5s_classifier/afib_5s_training_config.json`
- `training/research/afib_5s_classifier/afib_5s_test_metrics.json`
- `training/research/afib_5s_classifier/afib_5s_confounder_metrics.json`
- `training/research/afib_5s_classifier/afib_5s_confounder_predictions.csv`
- `training/research/afib_5s_classifier/safety_gate_evaluation_metrics.json`
- `training/research/afib_5s_classifier/safety_gate_confounder_decisions.csv`
- `training/research/afib_5s_classifier/safety_gate_true_afib_audit.json`
- `training/research/afib_5s_classifier/data/afdb_5s_features_train.csv`
- `training/research/afib_5s_classifier/data/afdb_5s_features_val.csv`
- `training/research/afib_5s_classifier/data/afdb_5s_features_test.csv`
- `training/research/afib_5s_classifier/data/afdb_5s_rejected_windows.csv`
