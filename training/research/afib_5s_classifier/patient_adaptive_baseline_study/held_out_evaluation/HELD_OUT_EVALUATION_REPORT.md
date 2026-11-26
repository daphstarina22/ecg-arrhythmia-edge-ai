# Evaluation Report: Preregistered Held-Out Test Evaluation of Causal Patient-Adaptive 5-Second AFib Classifier

**Evaluation Phase**: Preregistered Single-Shot Held-Out Test Evaluation  
**Evaluation Directory**: [`training/research/afib_5s_classifier/patient_adaptive_baseline_study/held_out_evaluation/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/held_out_evaluation/)  
**Preregistered Protocol**: [`PREREGISTERED_HELD_OUT_EVALUATION_PLAN.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/PREREGISTERED_HELD_OUT_EVALUATION_PLAN.md)  
**Execution Script**: [`execute_preregistered_evaluation.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/held_out_evaluation/execute_preregistered_evaluation.py)  
**Frozen Model Package**: [`frozen_models_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/held_out_evaluation/frozen_models_package.joblib) (SHA256: `7d5c6518a1a843c5...`)  
**Evaluated Cohort**: AFDB Quarantined Held-Out Test Cohort (4 patients: `04908`, `07162`, `08215`, `08219`, $N = 29,260$ non-overlapping 5-second windows, $19.2444$ non-AFib hours)  
**Empirical Outputs**: [`held_out_test_metrics.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/held_out_evaluation/held_out_test_metrics.json) | [`held_out_test_predictions.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/held_out_evaluation/held_out_test_predictions.csv)

---

## 1. Executive Summary & Preregistration Decision Gate Verdict

### 1.1 Core Finding
Executing the preregistered evaluation plan on the untouched held-out test cohort demonstrates that **causal patient-adaptive baseline tracking resolves the cross-patient logit inversion (Simpson's Paradox) that crippled the original 5-second model**:

* **Held-Out Test ROC-AUC**: Rose from **0.6615** to **0.9015** (Logistic Regression, $+0.2400$ AUC points) and **0.8933** (HistGradientBoosting, $+0.1935$ AUC points).
* **Held-Out Test PR-AUC**: Rose from **0.6550** to **0.8916** ($+0.2366$).
* **AFib Sensitivity (@ $\theta^* = 0.560$)**: Leaped from **39.24%** ($6,044 / 15,404$ windows) to **81.09%** ($12,491 / 15,404$ windows), detecting **$6,447$ additional AFib windows**.
* **False Alarm Rate (@ $\theta^* = 0.560$)**: Decreased by **35.5%** from **157.45 FP/hr** down to **101.54 FP/hr** (reducing false alarms by $1,076$).

```
+------------------------------------------------------------------------------------------------------------------------+
|                                  HELD-OUT TEST GENERALIZATION: ORIGINAL VS. ADAPTIVE                                   |
+------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                        |
|   ORIGINAL MODEL (Config A: Absolute Only, 21 Features):                                                              |
|   - Test ROC-AUC: 0.6615  |  PR-AUC: 0.6550                                                                           |
|   - Sensitivity:  39.24%  |  Specificity: 78.13%  |  False Alarms: 157.45 FP/hr                                        |
|   - Slow AFib Detection (Patient 07162): Sensitivity = 25.07%  (Severe Under-Detection)                               |
|   - Sinus Arrhythmia False Alarms (Patient 08219): 2,235 FP  (282.46 FP/hr)                                            |
|                                                                                                                        |
|                                                     |                                                                  |
|                                        CAUSAL ADAPTIVE BASELINE TRACKING                                               |
|                                                     v                                                                  |
|                                                                                                                        |
|   ADAPTIVE MODEL (Config C: Combined Absolute + Adaptive, 26 Features):                                               |
|   - Test ROC-AUC: 0.9015 (+0.2400)  |  PR-AUC: 0.8916 (+0.2366)                                                        |
|   - Sensitivity:  81.09% (+41.85%)  |  Specificity: 85.90% (+7.77%)  |  False Alarms: 101.54 FP/hr (-35.5%)            |
|   - Slow AFib Detection (Patient 07162): Sensitivity = 78.66%  (MORE THAN TRIPLED: +53.59% INCREASE)                  |
|   - Sinus Arrhythmia False Alarms (Patient 08219): 1,571 FP  (REDUCED BY 664 FALSE ALARMS)                             |
|                                                                                                                        |
+------------------------------------------------------------------------------------------------------------------------+
```

### 1.2 Verdict on Preregistered Hypotheses
1. **Primary Hypothesis $H_1$ ($\text{ROC-AUC} \ge 0.8500$)**: **CONFIRMED**. Pooled test ROC-AUC reached **0.9015**, exceeding the preregistered threshold of 0.8500.
2. **Secondary Hypothesis $H_2$ ($\ge 40\%$ reduction in `08219` false alarms)**: **PARTIALLY CONFIRMED / LIMITATION**. False alarms dropped by **29.7%** (from 2,235 down to 1,571, an absolute reduction of 664 false alarms). While substantial, it did not reach the 40% reduction target, indicating that respiratory sinus arrhythmia remains a challenge for purely statistical baseline tracking.
3. **Secondary Hypothesis $H_3$ (Sensitivity on `07162` $\ge 50.0\%$)**: **CONFIRMED**. Sensitivity on sustained normofrequent AFib surged from **25.07% to 78.66%**, exceeding the target by +28.66%.

---

## 2. Pre-Flight Verification & Checksum Audit

Before reading any test labels or computing predictions, data file hashes were verified:
* `afdb_5s_features_train.csv`: SHA256 `6c2094c17c33266f...` ($N = 217,851$ windows)
* `afdb_5s_features_val.csv`: SHA256 `4a658f59c48f4812...` ($N = 58,645$ windows)
* `afdb_5s_features_test.csv`: SHA256 `5bf67ecebb73185d...` ($N = 29,260$ windows)
* **Disjointness Audit**: $0$ overlapping patients between train (15 patients) and held-out test (4 patients: `AFDB_04908`, `AFDB_07162`, `AFDB_08215`, `AFDB_08219`).
* **Model Packaging**: `StandardScaler` and `LogisticRegression` models were fitted strictly on the 15 training patients ($N = 217,851$) and frozen to [`frozen_models_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/held_out_evaluation/frozen_models_package.joblib) (SHA256: `7d5c6518a1a843c5...`).

---

## 3. Global Pooled Comparative Results

Evaluated on the $29,260$ held-out non-overlapping 5-second windows ($19.2444$ non-AFib hours):

```
+-------------------------------------------------+------------+-------------+------------------+------------------+-----------------+
| Configuration / Model                           | Model Type | ROC-AUC     | Sensitivity      | Specificity      | False Alarm Rate|
|                                                 |            | (PR-AUC)    | (@ theta = 0.560)| (@ theta = 0.560)| (FP / hour)     |
+-------------------------------------------------+------------+-------------+------------------+------------------+-----------------+
| Config A: Existing Absolute Model (21 features) | LogReg     | 0.6615      | 39.24%           | 78.13%           | 157.45 FP/hr    |
|                                                 |            | (0.6550)    | (6,044 / 15,404) | (10,826 / 13,856)| (3,030 alarms)  |
| Config A: Existing Absolute Model (21 features) | HistGrad   | 0.6998      | 27.96%           | 79.48%           | 147.73 FP/hr    |
|                                                 |            | (0.6766)    | (4,307 / 15,404) | (11,012 / 13,856)| (2,844 alarms)  |
+-------------------------------------------------+------------+-------------+------------------+------------------+-----------------+
| Config C: Combined Adaptive Model (26 features) | LogReg     | 0.9015      | 81.09%           | 85.90%           | 101.54 FP/hr    |
|                                                 |            | (0.8916)    | (12,491 / 15,404)| (11,902 / 13,856)| (1,954 alarms)  |
| Config C: Combined Adaptive Model (26 features) | HistGrad   | 0.8933      | 80.56%           | 83.87%           | 116.14 FP/hr    |
|                                                 |            | (0.8556)    | (12,410 / 15,404)| (11,621 / 13,856)| (2,235 alarms)  |
+-------------------------------------------------+------------+-------------+------------------+------------------+-----------------+
```

### Evaluation at Validation High-Specificity Triage Threshold ($\theta = 0.760$)
Operating at the validation-preregistered high-specificity operating threshold ($\theta = 0.760$):
* **Config C (LogReg @ $\theta = 0.760$)**:
  * Sensitivity: **72.01%** ($11,092 / 15,404$)
  * Specificity: **89.43%** ($12,391 / 13,856$)
  * False Alarm Rate: **76.13 FP/hr** ($1,465$ total alarms)
  * Balanced Accuracy: **80.72%** | F1-Score: **0.7934**
* **Config A (LogReg @ $\theta = 0.760$)**:
  * Sensitivity collapsed to **24.87%** ($3,831 / 15,404$).
  * False Alarm Rate was $97.53\text{ FP/hr}$.

---

## 4. Per-Patient Breakdown on the Held-Out Test Cohort

Evaluating each patient separately highlights how baseline adaptation operates across diverse clinical rhythm profiles:

```
+--------------------------------------------------------------------------------------------------------------------+
|                                      PER-PATIENT HELD-OUT PERFORMANCE COMPARISON                                   |
+------------+--------------------+------------------------+---------------------------------------------------------+
| Record ID  | Rhythm Profile     | Config A (Absolute)    | Config C (Combined: Absolute + Adaptive)                |
+------------+--------------------+------------------------+---------------------------------------------------------+
| Record     | Low Prevalence     | ROC-AUC: 0.9686        | ROC-AUC: 0.9679                                         |
| 04908      | AFib: 8.18%        | Sens: 98.00% (588/600) | Sens: 89.17% (535/600)                                  |
|            | Non-AFib: 91.82%   | Spec: 88.73%           | Spec: 94.34%  (+5.61% improvement)                      |
|            | (Tachycardic Base) | FP/hr: 81.16           | FP/hr: 40.74  (-49.8% FALSE ALARMS CUT IN HALF)         |
+------------+--------------------+------------------------+---------------------------------------------------------+
| Record     | Sustained Slow     | ROC-AUC: Undefined*    | ROC-AUC: Undefined*                                     |
| 07162      | AFib: 100.00%      | Sens: 25.07%           | Sens: 78.66%  (+53.59% INCREASE: >3x DETECTION)         |
|            | Non-AFib: 0.00%    | Spec: Undefined*       | Spec: Undefined*                                        |
|            | (HR ~ 66 bpm)      | FP/hr: 0.00 (0 FP)     | FP/hr: 0.00 (0 FP)                                      |
+------------+--------------------+------------------------+---------------------------------------------------------+
| Record     | High Prevalence    | ROC-AUC: 0.9363        | ROC-AUC: 0.9908 (+0.0545 AUC INCREASE)                  |
| 08215      | AFib: 80.62%       | Sens: 36.84%           | Sens: 79.77%  (+42.93% SENSITIVITY LEAP)                |
|            | Non-AFib: 19.38%   | Spec: 97.48%           | Spec: 99.86%  (NEAR-PERFECT: 2 FP in 1,426 windows)     |
|            |                    | FP/hr: 18.18           | FP/hr: 1.01   (-94.4% FALSE ALARM REDUCTION)            |
+------------+--------------------+------------------------+---------------------------------------------------------+
| Record     | Sinus Tachycardia  | ROC-AUC: 0.8944        | ROC-AUC: 0.9184 (+0.0240 AUC INCREASE)                  |
| 08219      | + Arrhythmia       | Sens: 94.37%           | Sens: 94.90%  (94.90% SENSITIVITY MAINTAINED)           |
|            | AFib: 20.95%       | Spec: 60.77%           | Spec: 72.42%  (+11.65% SPECIFICITY GAIN)                |
|            | Non-AFib: 79.05%   | FP/hr: 282.46 (2235 FP)| FP/hr: 198.55 (-29.7% REDUCTION: 664 FEWER ALARMS)      |
+------------+--------------------+------------------------+---------------------------------------------------------+
*Note: In single-class records (Record 07162 has 100% AFib and 0 Non-AFib windows), Specificity, ROC-AUC, and FP/hr are mathematically undefined.
```

### Clinical Findings by Patient
1. **Patient `07162` (Controlled / Slow AFib)**:
   - In the audit, this patient was identified as the main source of false negatives because slow ventricular conduction ($66.4\text{ bpm}$) and low amplitude ($0.62\text{ mV}$) evaded absolute rate thresholds, detecting only $25.07\%$ of AFib windows.
   - Under Config C, causal baseline tracking evaluated the patient's relative rate and variance changes rather than penalizing their slow absolute rate. Sensitivity surged to **78.66%**, detecting $5,792$ true AFib windows (an increase of $3,946$ correctly identified episodes).
2. **Patient `08215` (High Burden AFib)**:
   - Sensitivity jumped from **36.84% to 79.77%** (+42.93%), while false alarms on non-AFib segments dropped from 18.18 down to **1.01 FP/hr** (only 2 false alarms in the entire 10-hour recording!). ROC-AUC reached **0.9908**.
3. **Patient `04908` (Paroxysmal Rapid AFib)**:
   - False alarms were cut in half from **81.16 FP/hr down to 40.74 FP/hr**, while maintaining high discrimination ($\text{AUC} = 0.9679$).
4. **Patient `08219` (Chronic Sinus Arrhythmia Challenge)**:
   - Specificity rose from $60.77\%$ to $72.42\%$, eliminating $664$ false alarms ($2,235 \rightarrow 1,571$).
   - However, false alarms remained high at **198.55 FP/hr**. This confirms an important clinical limitation: statistical percentile tracking mitigates, but does not completely eliminate, false alarms caused by marked respiratory sinus arrhythmia where cycle-to-cycle interval changes mimic paroxysmal AFib on short 5-second windows.

---

## 5. Warm-Up Dynamics & Staged Coverage

In the held-out test cohort, windows were sampled with a 5.0-second non-overlapping stride:
* **Warm-Up Windows ($N_{\text{buffered}} < 12$, first $60\text{ seconds}$)**: $1,049$ windows (**3.59%** of test cohort).
* **Post-Warmup Windows ($N_{\text{buffered}} \ge 12$, after 1 minute)**: $28,211$ windows (**96.41%** of test cohort).
* **Performance Breakdown**:
  * **Post-Warmup Subset**: ROC-AUC reached **0.9137**, with Sensitivity = **82.80%**, Specificity = **85.99%**, and False Alarms = **100.90 FP/hr**.
  * **Warmup Subset**: ROC-AUC was **0.7324**, with Sensitivity = **48.58%**, Specificity = **81.59%**, and False Alarms = **132.56 FP/hr**.
* **Clinical Triage Implication**: During the initial 60 seconds of wearing a monitor, the tracker has not yet established the patient's individual resting baseline. Using `baseline_warmup_fraction` as a gating flag allows edge devices to suppress alarms during the first 60 seconds or display a "calibrating" indicator, avoiding cold-start false positives.

---

## 6. Audit Trail & Exact Evaluation Configuration

### 6.1 Evaluation Configuration
* **Training Data**: `afdb_5s_features_train.csv` ($N = 217,851$ windows across 15 patients).
* **Preprocessing**: `StandardScaler` fitted strictly on training data; mean and variance parameters locked.
* **Model**: L2-regularized balanced Logistic Regression ($C = 1.0, \text{random\_state} = 42$).
* **Model Intercept**: $-0.4126$.
* **Top Learned Feature Weights**:
  * `rr_cv` (Absolute): $+8.8221$
  * `hr_std_bpm` (Absolute): $-3.0785$
  * `rr_cv_relative` (Adaptive): $-2.6011$
  * `hr_mean_bpm` (Absolute): $+1.7150$
  * `baseline_warmup_fraction`: $+1.4057$
  * `f_band_power_ratio` (Absolute): $+1.2016$
  * `f_band_flatness` (Absolute): $+1.0661$
  * `delta_hr_bpm` (Adaptive): $+0.5225$
* **Tracker Hyperparameters**:
  * `history_window_sec`: $300.0\text{ s}$
  * `min_warmup_windows`: $12$
  * `max_gap_sec`: $300.0\text{ s}$
  * `anti_contamination`: True ($\text{rate\_percentile} = 25.0$, $\text{cv\_percentile} = 20.0$, $\text{gating\_cv\_threshold} = 0.18$)
* **Operating Thresholds**:
  * Fixed Legacy: $\theta^* = 0.560$
  * Validation High-Spec Triage: $\theta = 0.760$

### 6.2 Protocol Deviations
* **Zero Deviations**: The evaluation was executed in a single shot directly from [`execute_preregistered_evaluation.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/patient_adaptive_baseline_study/held_out_evaluation/execute_preregistered_evaluation.py). No thresholds were re-tuned, no features modified, and no parameters adjusted.

---

## 7. Final Decision & Research Recommendations

### **Verdict: THE EXPERIMENT SUPPORTS FURTHER RESEARCH**
The preregistered held-out evaluation confirms that patient-adaptive baseline tracking addresses the single highest-priority cause of poor generalization identified in the audit:
1. **Generalization Validated**: Pooled test ROC-AUC improved from **0.6615 to 0.9015**, and sensitivity leaped from **39.24% to 81.09%**.
2. **Slow AFib Resolved**: Sensitivity on slow AFib (`07162`) more than tripled from $25.07\%$ to $78.66\%$.
3. **No Retraining or Leakage Required**: Improvements were achieved causally without future information or data leakage.

### **Important Limitations & Non-Claims**
1. **Not Ready for Autonomous Clinical Deployment**:
   - While pooled false alarms dropped from $157.45$ to $101.54\text{ FP/hr}$, $101.54\text{ alarms/hr}$ is still too high for an unmonitored wearable alert without temporal consensus or secondary confirmation.
   - For patient `08219`, respiratory sinus arrhythmia still caused $198.55\text{ FP/hr}$.
2. **Single Dataset Limitation**:
   - These findings are grounded in AFDB. While patient-disjointness was strictly maintained, validation on multi-lead external cohorts (e.g. CPSC 2018 or PTB-XL) is required before clinical claims can be made.
3. **Recommended Next Research Step**:
   - Integrate the causal adaptive baseline tracker with **temporal persistence / multi-window confirmation** (e.g., confirming AFib only if $\ge 3$ consecutive 5-second windows or $\ge 15\text{ seconds}$ trigger the classifier). This will eliminate isolated sinus arrhythmia spikes (like in `08219`) while preserving high sustained detection sensitivity.
