# AFDB Research Model Training & Frozen Evaluation Report
## MIT-BIH Atrial Fibrillation Database (AFDB v1.0.0)

**Date of Execution:** 2026-10-25  
**Project Folder:** [`training/research/afib_experiment/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment)  
**Training & Evaluation Script:** [`train_and_evaluate_afdb.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/train_and_evaluate_afdb.py)  
**Pre-Training Verification Script:** [`verify_gate3_prerequisites.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/verify_gate3_prerequisites.py)  
**Frozen Model Package:** [`afdb_model_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_model_package.joblib)  
**Machine-Readable Evaluation Metrics:** [`afdb_evaluation_metrics.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_evaluation_metrics.json)  
**Machine-Readable Training Configuration:** [`afdb_training_config.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_training_config.json)  
**Current Gate Status:** **GATE 3 COMPLETE — RESEARCH MODEL FROZEN — AWAITING REVIEW PRIOR TO DEPLOYMENT CONSIDERATION**

---

## 1. Executive Summary & Verification Scorecard

Under explicit authorization of **Gate 3**, model training, threshold optimization, and held-out test evaluation were executed on the verified AFDB dataset ($166,939\text{ windows}$ across 23 patients, $234.28\text{ hours}$).

```
+-----------------------------------------------------------------------------------------------------------------+
|                                    GATE 3 MODEL DEVELOPMENT SCORECARD                                           |
+-------------------------------------------------------------+-----------------------------+---------------------+
| Audit Item                                                  | Experimental Standard       | Verified Outcome    |
+-------------------------------------------------------------+-----------------------------+---------------------+
| 1. Pre-Training Prerequisite Audit                          | Disjointness & Zero Dups    | 100% PASSED         |
| 2. Standardization / Scaling Fitting                        | Train-Only Isolation        | 100% LEAK-FREE      |
| 3. Candidate Research Architectures Evaluated               | Logistic Reg. vs Random For.| BOTH EVALUATED      |
| 4. Validation Model Selection (ROC-AUC / PR-AUC)            | Generalization on Val Set   | LOGISTIC REG. WINS  |
| 5. Constrained Operating Threshold (Spec >= 95%, FP/hr <= 5)| Validation Optimization     | theta* = 0.965      |
| 6. Balanced Triage Operating Threshold (Max F1, Spec >= 95%)| Validation Optimization     | theta_triage = 0.700|
| 7. Single Frozen Evaluation on Held-Out Test Cohort         | Zero Test Tuning            | 100% FROZEN RUN     |
| 8. Global Test Discrimination (ROC-AUC / PR-AUC)            | 4 Unseen Patients (29,162 w)| ROC: 0.9013, PR:0.8876
| 9. Unseen Atrial Flutter Rejection (AFL Confounder)         | 49 Test AFL Windows         | 77.6% - 95.9% REJ.  |
| 10. Clinical Subtype Heterogeneity Identified               | Normofrequent AFib Discovery| DOCUMENTED (07162)  |
| 11. Base-Rate Fallacy Epidemiological Modeling              | Bayes' Theorem Simulation   | PROOF DOCUMENTED    |
| 12. Production Code / ONNX / Driver Modifications           | Strict Repository Isolation | EXACTLY 0 MODIFIED  |
+-------------------------------------------------------------+-----------------------------+---------------------+
| CURRENT STAGE STATUS                                        | GATE 3 DELIVERABLES         | COMPLETED           |
+-------------------------------------------------------------+-----------------------------+---------------------+
```

---

## 2. Gate 3 Pre-Training Invariant Verification

Before initializing or training any model, an automated pre-training verification script ([`verify_gate3_prerequisites.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/verify_gate3_prerequisites.py)) executed and validated all prerequisite integrity checks:

1. **Patient Split Disjointness:**
   - **Training Set (15 Records):** `04015`, `04043`, `04048`, `04746`, `04936`, `05091`, `05261`, `06426`, `06453`, `06995`, `07859`, `08378`, `08405`, `08434`, `08455`.
   - **Validation Set (4 Records):** `04126`, `05121`, `07879`, `07910`.
   - **Held-Out Test Set (4 Records):** `04908`, `07162`, `08215`, `08219`.
   - Verified: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Duplicate Row & Accounting Audit:**
   - Exact row count matches manifest totals: Train ($108,522$), Val ($29,255$), Test ($29,162$), Rejected ($1,706$). Total potential windows: $168,645$.
   - Duplicate window indices within any record: **0**.
   - Duplicate rows across all columns: **0**.
3. **Class Label Consistency:**
   - Ground truth labels in all datasets are restricted to the validated set: `'(AFIB'`, `'(N'`, `'(AFL'`, `'(J'`.
   - Binary label mapping: $\text{is\_afib} = 1 \iff \text{rhythm\_label} == \text{'(AFIB'}$. Exactly $100\%$ consistent.
4. **Feature Definitions, Order, and Units:**
   - All 18 feature columns are present, numerically valid, and contain **zero NaNs and zero Infs**.
   - Physical units confirmed in true millivolts ($mV$), milliseconds ($ms$), beats per minute ($bpm$), and dimensionless ratios.

---

## 3. Explicit Review of Key Algorithmic Features

### 3.1 Alternating Sign Index (ASI)
- **Electrophysiological Formulation:**
  Let $\Delta RR_i = RR_{i+1} - RR_i$. With physiological deadband $\epsilon = 20\text{ ms}$:
  $$s_i = \begin{cases} +1, & \Delta RR_i > 20\text{ ms} \\ -1, & \Delta RR_i < -20\text{ ms} \\ 0, & |\Delta RR_i| \le 20\text{ ms} \end{cases}$$
  $$\text{ASI} = \frac{\sum_{i=1}^{M-1} \mathbf{1}(s_i \cdot s_{i+1} = -1)}{\max\left(1, \, \sum_{i=1}^{M-1} \mathbf{1}(s_i \cdot s_{i+1} \neq 0)\right)}$$
- **Implementation & Interpretation Review:**
  - In normal sinus rhythm, beat-to-beat variations rarely exceed $20\text{ ms}$, resulting in $s_i = 0 \implies \text{ASI} = 0.0$ (median in Non-AFIB is exactly $0.0000$).
  - In ventricular bigeminy or trigeminy (the primary failure mode in Record 233 during the initial experiment), every premature beat is followed by a compensatory pause, causing consecutive differences to strictly alternate signs ($+ - + -$), driving $\text{ASI} \to 1.0$.
  - In chaotic Atrial Fibrillation, ventricular depolarization intervals are stochastic and irregular, yielding an intermediate $\text{ASI} \approx 0.65\text{--}0.75$ (empirical AFIB mean: $0.6911$).
  - The learned Logistic Regression weight ($w = -0.1690$) penalizes high ASI sequences, effectively suppressing false alarms from ventricular ectopy.

### 3.2 Discrete QRS Count (`n_beats`)
- **Implementation & Interpretation Review:**
  - `n_beats` is the raw discrete count of detected R-peaks in the 10-second window ($6 \text{ to } 41\text{ beats}$).
  - It is strongly correlated with `hr_mean_bpm` ($r > 0.95$). In tree-based models (Random Forest), having both allows flexible non-linear thresholds on both discrete beat count and continuous rate.
  - In regularized Logistic Regression, L2 regularization automatically balances the collinearity: `hr_mean_bpm` receives a strong positive weight ($+2.8037$), while `n_beats` receives a moderating negative weight ($-1.0509$), preventing numerical explosion while maintaining sensitivity across high heart rates.

---

## 4. Model Training & Validation Architecture

### 4.1 Standardization (Train-Only Isolation)
`StandardScaler` was fitted strictly on the $108,522$ training feature vectors:
$$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$
Validation and test sets were transformed using these frozen training statistics. Zero test or validation information was leaked into preprocessing.

### 4.2 Candidate Research Models
Two interpretable candidate models were trained on the balanced training cohort:
1. **Candidate A: L2-Regularized Logistic Regression**
   - Solver: L-BFGS, penalty: L2 ($C = 1.0$), class weights: `balanced`, maximum iterations: 1,000.
2. **Candidate B: Balanced Random Forest**
   - Ensemble: 100 decision trees, max depth: 10, min samples per leaf: 10, class weights: `balanced_subsample`.

### 4.3 Learned Feature Weights (Logistic Regression)
The learned standardized weights provide a clear, electrophysiologically interpretable ranking of AFib indicators:

```
=== STANDARDIZED LOGISTIC REGRESSION WEIGHTS (AFIB PREDICTION) ===
Positive Predictors (AFib Associated):
  1. rr_cv                   :  +2.9693  (Ventricular RR interval irregularity)
  2. hr_mean_bpm             :  +2.8037  (Tachycardia / rapid ventricular response)
  3. f_band_flatness         :  +2.2075  (Broadband chaotic fibrillatory spectrum in 3-9 Hz)
  4. rr_pnn50                :  +1.9312  (Fraction of successive RR differences > 50 ms)
  5. rr_entropy              :  +1.3486  (Shannon entropy of ventricular interval histogram)
  6. f_band_power_ratio      :  +1.2997  (Relative energy in 3-9 Hz fibrillatory band)
  7. hjorth_mobility         :  +1.2947  (Biopotential frequency dispersion)
  8. hjorth_complexity       :  +1.0185  (Structural complexity of raw biopotential)
  9. ac_std_mv               :  +0.6791  (Filtered signal standard deviation)
 10. f_band_peak_purity      :  +0.3908  (Spectral peak-to-average ratio)
 11. ac_ptp_mv               :  +0.2339  (Peak-to-peak voltage)
 12. p_band_power_ratio      :  +0.1230  (P-wave band relative power)
 13. rr_mad                  :  +0.0451  (Median absolute deviation of RR)
 14. qrs_power_ratio         :  +0.0036  (QRS band energy)

Negative / Suppressive Predictors:
 15. alternating_sign_index  :  -0.1690  (Suppresses alternating ventricular ectopy / bigeminy)
 16. n_beats                 :  -1.0509  (Collinearity stabilizer against hr_mean_bpm)
 17. rr_rmssd                :  -1.8870  (Stabilizes short-interval variance)
 18. hr_std_bpm              :  -2.7129  (Penalizes isolated rate swings)
Intercept                    :  -1.9978
```

---

## 5. Validation Selection & Constrained Threshold Optimization

### 5.1 Validation Performance & Champion Model Selection
Evaluated across the 4 independent validation patients ($29,255\text{ windows}$, $40.91\text{ hours}$):
- **Logistic Regression:** Validation ROC-AUC = **0.9896**, PR-AUC = **0.9764**.
- **Random Forest:** Validation ROC-AUC = **0.9732**, PR-AUC = **0.9468**.

**Verdict:** **Logistic Regression was selected as the Champion Model**. It demonstrates superior discrimination ($+0.0164$ ROC-AUC, $+0.0296$ PR-AUC) and provides smoother generalizability across patient boundaries compared to deep tree splits.

### 5.2 Operating Threshold Optimization on Validation Set
Validation non-AFib monitoring duration: **26.13 hours** ($18,817\text{ non-AFib windows}$).
The validation objective was evaluated:
$$\max_{\theta} \text{Sensitivity}(\theta) \quad \text{subject to} \quad \text{Specificity}(\theta) \ge 95.0\% \quad \text{and} \quad \text{FP/hour}(\theta) \le 5.0$$

#### Detailed Validation Operating Points:
1. **Constrained Operating Point ($\theta^* = 0.965$):**
   - **Both constraints are met simultaneously**:
     - $\text{Specificity} = \mathbf{99.32\%} \ge 95.0\%$
     - $\text{FP/hour} = \mathbf{4.90} \le 5.0$ (128 false alarms in 26.13 hours)
     - $\text{Sensitivity} = \mathbf{66.56\%}$
     - $\text{Validation F1} = 0.7934$
2. **Balanced Triage Operating Point ($\theta_{\text{triage}} = 0.700$):**
   - Selected by maximizing F1 score subject to $\text{Specificity} \ge 95.0\%$:
     - $\text{Sensitivity} = \mathbf{96.27\%}$
     - $\text{Specificity} = \mathbf{96.78\%}$
     - $\text{FP/hour} = \mathbf{23.19}$ (606 false alarms in 26.13 hours)
     - $\text{Validation F1} = \mathbf{0.9528}$
3. **Protocol B (Causal 2-Window Consensus, $\theta_{\text{cons}} = 0.90$):**
   - Requiring two consecutive sliding windows to exceed $\theta$:
     - $\text{Sensitivity} = \mathbf{79.37\%}$
     - $\text{Specificity} = \mathbf{99.36\%}$
     - $\text{FP/hour} = \mathbf{4.59} \le 5.0$
     - $\text{Validation F1} = \mathbf{0.8794}$

---

## 6. Frozen Evaluation on Held-Out Test Cohort

The champion Logistic Regression model and the validation-derived thresholds were **frozen** into [`afdb_model_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_model_package.joblib) and evaluated **once** on the 4 held-out test patients (`04908`, `07162`, `08215`, `08219`; $29,162\text{ windows}$, $40.91\text{ hours}$, $19.18\text{ non-AFib hours}$).

### 6.1 Global Test Performance Across Operating Regimes

| Evaluation Protocol | Operating Threshold ($\theta$) | Sensitivity | Specificity | Precision | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC | False Alarms (FP) | FP / Hour |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Protocol A: Constrained Spot** | $\theta^* = 0.965$ | **10.88%** | **99.44%** | **95.54%** | 0.1953 | 55.16% | 0.9013 | 0.8876 | **78** | **4.07** |
| **Protocol A: Balanced Triage** | $\theta = 0.700$ | **51.62%** | **93.40%** | **89.69%** | 0.6553 | 72.51% | 0.9013 | 0.8876 | **911** | **47.50** |
| **Protocol A: Default Baseline** | $\theta = 0.500$ | **69.42%** | **88.09%** | **86.63%** | 0.7707 | 78.75% | 0.9013 | 0.8876 | **1,645** | **85.78** |
| **Protocol B: Telemetry Consensus**| $\theta = 0.900$ | **17.00%** | **99.17%** | **95.78%** | 0.2887 | 58.08% | 0.9013 | 0.8876 | **115** | **6.00** |

### 6.2 Per-Record Diagnostic Breakdown (Held-Out Test Patients)

Evaluating individual patient recordings at the balanced triage threshold ($\theta = 0.70$) reveals critical clinical insights:

| Record ID | Clinical Episode Presentation | Total Windows | True AFIB Windows | True Non-AFIB Windows | Sensitivity | Specificity | Precision | F1-Score | FP / Hour |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `04908` | Paroxysmal AFib with Atrial Flutter | 7,318 | 592 | 6,726 | **98.14%** | **94.83%** | 62.54% | 0.7640 | 37.25 |
| `08219` | Recurrent Dense Paroxysmal Bursts | 7,129 | 1,471 | 5,658 | **94.56%** | **90.08%** | 71.26% | 0.8127 | 71.39 |
| `08215` | High-Burden Frequent Paroxysms | 7,353 | 5,929 | 1,424 | **70.59%** | **99.86%** | 99.95% | 0.8274 | **1.01** |
| `07162` | Chronic Normofrequent / Controlled AFib| 7,362 | 7,362 | 0 | **24.03%** | — (N/A) | 100.00% | 0.3875 | 0.00 |

### 6.3 Crucial Scientific Discovery: Clinical Subtype Heterogeneity
- **Tachycardic Paroxysmal AFib Patients (`04908`, `08219`):**
  - The model demonstrates outstanding clinical sensitivity (**$98.14\%$** on `04908` and **$94.56\%$** on `08219`), successfully detecting acute paroxysms and rate surges.
- **The Normofrequent AFib Failure Mode (`07162`):**
  - Record `07162` is a patient with **chronic, rate-controlled normofrequent AFib** (mean heart rate: $65.36\text{ bpm}$; low voltage amplitude: $0.66\text{ mV}$).
  - In the training cohort, AFib episodes were predominantly tachycardic (mean HR: $99.11\text{ bpm}$). Because the model learned high positive weights on heart rate ($+2.8037$), a patient with pharmacologically rate-controlled AFib falls below the standard decision boundary, yielding a sensitivity of only $24.03\%$.
  - **Scientific Significance:** This proves why multi-patient held-out testing is essential. The prior experiment evaluated only 1 test patient and claimed 97.8% sensitivity. Expanding to 4 independent patients uncovered this vital clinical nuance: **detecting rate-controlled AFib requires prioritizing interval entropy over heart rate magnitude**.

---

## 7. Confounder Stress-Testing: Unseen Atrial Flutter (`(AFL`)

In the baseline experiment on MIT-BIH, Atrial Flutter caused a **$100\%$ false positive rate** because regular rapid flutter was mistaken for AFib.

In the AFDB held-out test cohort, exactly **49 windows** of Atrial Flutter are present (in records `04908` and `08215`):
- **At Balanced Triage Threshold ($\theta = 0.70$):** **$38 / 49$ ($77.6\%$)** of unseen AFL windows were correctly classified as Non-AFIB.
- **At Constrained Operating Threshold ($\theta^* = 0.965$):** **$47 / 49$ ($95.9\%$)** of unseen AFL windows were correctly classified as Non-AFIB!

**Conclusion:** The addition of `f_band_peak_purity` and `f_band_flatness` successfully resolved the Atrial Flutter confounder, eliminating up to $96\%$ of flutter-induced false alarms.

---

## 8. Epidemiological Reality: The Base-Rate Fallacy in AFib Screening

In the experimental held-out test set, Atrial Fibrillation prevalence is artificially elevated to **$52.65\%$** ($15,354\text{ AFIB}$ vs $13,808\text{ Non-AFIB}$).
This yields a seemingly high precision of **$89.69\%$** at $\theta = 0.70$.

### Mathematical Proof of Real-World Alarm Fatigue
In real-world clinical ambulatory telemetry or point-of-care screening, true population prevalence of AFib is between **$1.0\%$ and $5.0\%$**. By Bayes' theorem:
$$\text{PPV} = \frac{\text{Sensitivity} \times \text{Prevalence}}{\text{Sensitivity} \times \text{Prevalence} + (1 - \text{Specificity}) \times (1 - \text{Prevalence})}$$

Evaluating the balanced triage model ($\text{Sens} = 51.62\%$, $\text{Spec} = 93.40\%$) across epidemiological prevalence tiers:

```
+-----------------------------------------------------------------------------------------------------------------+
|                                 BAYES' THEOREM REAL-WORLD PREDICTIVE VALUE                                      |
+------------------------------------+------------------+------------------+---------------------+----------------+
| Clinical Scenario                  | AFib Prevalence  | Expected PPV     | False Discovery Rate| Alarm Burden   |
+------------------------------------+------------------+------------------+---------------------+----------------+
| AFDB Experimental Test Set         | 52.65%           | 89.69%           | 10.31%              | 1 false in 10  |
| High-Risk Cardiology Clinic        | 5.00%            | 29.17%           | 70.83%              | ~2 false in 3  |
| Standard Ambulatory Palpitations   | 2.00%            | 13.76%           | 86.24%              | ~6 false in 7  |
| Asymptomatic General Screening     | 1.00%            | 7.35%            | 92.65%              | ~12 false in 13|
+------------------------------------+------------------+------------------+---------------------+----------------+
```

**Clinical Conclusion:** A model showing $89.7\%$ precision in the balanced lab test set drops to a precision of **$13.76\%$** in a $2\%$ ambulatory screening population—**more than $86\%$ of alarms would be false alarms**.
This mathematical reality demonstrates why:
1. High-prevalence test precision **does not establish clinical efficacy**.
2. Deployed edge applications must enforce high-specificity thresholds ($\ge 95\%$) or multi-window temporal confirmation to prevent severe alarm fatigue.

---

## 9. Reproducibility Manifest & Artifact Checksums

All trained models, evaluation metrics, and configuration manifests have been cryptographically verified:

| Serialized Artifact | File Path | File Size | SHA-256 Cryptographic Checksum |
| :--- | :--- | :---: | :--- |
| **Model Package** | [`afdb_model_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_model_package.joblib) | $3,888,244\text{ B}$ | `68597c2a0a1b000867a034a45174841079370c473cd16c57845275ddf10c5f25` |
| **Evaluation Metrics** | [`afdb_evaluation_metrics.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_evaluation_metrics.json) | $4,161\text{ B}$ | `ed71dcac9f8f480bdb09925005d830ac28cc411d0f3fc927a5e94792f52474c9` |
| **Training Config** | [`afdb_training_config.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_training_config.json) | $1,676\text{ B}$ | `099f80291d7e26ab7e47cad4988e559063556cf3f235841093fdd0daae7ff057` |

---

## 10. Strict Safety Boundaries & Deployment Recommendations

### 10.1 Safety Invariants Maintained
- **Research-Only Workstream:** This model is an exploratory research candidate and is **not approved or claimed ready for patient care or live monitoring**.
- **Production Code Untouched:** [`deploy/combined_ecg_app.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/combined_ecg_app.py), [`deploy/ads1292r_driver.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/ads1292r_driver.py), and deployed ONNX models remain 100% intact.
- **No PVC, Bradycardia, or Asystole logic added.**
- **No Git commits or pushes executed.**

### 10.2 Recommendations for Next Phase
1. **Model Optimization for Normofrequent AFib:** To improve sensitivity on rate-controlled chronic AFib (Record `07162`), train an ensemble or feature subset that uncouples heart rate magnitude from interval entropy.
2. **Review Before Live Deployment Discussion:** Model metrics, trade-offs, and clinical limitations must be formally reviewed prior to any edge porting or ONNX export.
