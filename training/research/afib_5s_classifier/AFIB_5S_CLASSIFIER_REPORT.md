# Dedicated 5-Second AFib Classifier Research Report
## Patient-Disjoint AFDB Training, Frozen Held-Out Evaluation, Multi-Arrhythmia Confounder Analysis, and Safety-Gated Architecture Investigation

**Date of Execution**: October 26, 2026  
**Research Phase**: Dedicated 5-Second AFib Classifier Development & Safety Audit  
**Location**: [`training/research/afib_5s_classifier/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/)  
**Primary Artifacts**:
- Split Manifest: [`afdb_5s_split_manifest.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afdb_5s_split_manifest.json)
- Feature Extraction: [`extract_afdb_5s_features.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/extract_afdb_5s_features.py)
- Model Training & Freezing: [`train_afib_5s.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/train_afib_5s.py)
- Frozen Model Package: [`afib_5s_model_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afib_5s_model_package.joblib)
- Frozen Test Evaluation: [`evaluate_afib_5s.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/evaluate_afib_5s.py) -> [`afib_5s_test_metrics.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afib_5s_test_metrics.json)
- Confounder Testing: [`evaluate_afib_5s_confounders.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/evaluate_afib_5s_confounders.py) -> [`afib_5s_confounder_metrics.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afib_5s_confounder_metrics.json)
- Safety Gate Investigation: [`evaluate_safety_gated_cascade.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/evaluate_safety_gated_cascade.py) -> [`safety_gate_evaluation_metrics.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/safety_gate_evaluation_metrics.json)

**Strict Engineering & Policy Invariants**:
- Zero modifications to production code (`deploy/`, `src/`) or deployed ONNX models (`models/`, `training/final_models/export_candidates/`).
- Zero modifications to Model 2 or previous 10-second research artifacts (`afdb_model_package.joblib` preserved untouched).
- All new scripts, feature sets, models, and metric logs are contained exclusively in `training/research/afib_5s_classifier/`.
- Zero Git commits or pushes executed.

---

## 1. Executive Summary & Core Scientific Findings

Following the discovery documented in the [AFib Compatibility Report](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/compatibility_experiment/AFIB_COMPATIBILITY_REPORT.md) that the legacy 10-second model cannot be evaluated on 5-second windows without severe covariate shift (where beat-count halving artificially inflated AFib log odds by $8.5\times$), we executed the authorized research phase to develop a **dedicated, natively trained 5-second Atrial Fibrillation classifier**.

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                    FIVE-SECOND AFIB RESEARCH PIPELINE OVERVIEW                                        |
+-------------------+-----------------------------+-----------------------------------+---------------------------------+
| Stage             | Dataset / Cohort            | Processing & Methodology          | Key Result / Output             |
+-------------------+-----------------------------+-----------------------------------+---------------------------------+
| 1. Prerequisites  | 23 AFDB Records             | Verified 100% disjoint split plan | 0 blockers, manifest locked     |
| 2. Feature Extr.  | 305,756 5s ECG windows      | 250->360Hz polyphase, 21 features | Train: 217,851; Val: 58,645;    |
|                   |                             | Guard bands: 2.5s transitions     | Test: 29,260 non-overlapping    |
| 3. Model Training | 15 Training Patients        | StandardScaler fit on Train only  | Champion: LogisticRegression    |
|                   | 4 Validation Patients       | Swept Operating Thresholds on Val | Val ROC-AUC: 0.9832, PR: 0.9568 |
| 4. Frozen Test    | 4 Quarantined Patients      | Single frozen evaluation on test  | Sens: 98.0% (04908), 94.4% (08219)|
|                   | (04908, 07162, 08215, 08219)| Zero post-hoc tuning              | Global Sens: 39.2%, Spec: 78.1% |
| 5. Confounders    | 603 Windows (VF, VT,        | Standalone ungated evaluation     | Standalone triggers false AFib  |
|                   | Bigeminy, AFL, Brady, Noise)| Across verified benchmark rhythms | on 98.2% VF, 89.0% Bigeminy     |
| 6. Safety Gate    | Multi-Tier Priority Cascade | Tier 0 Guard -> Tier 1 Shockable  | Intercepts 100% Coarse VF!      |
|                   | (Model 1 + Model 2 + AFib)  | -> Tier 2 Ventricular -> Tier 3   | 25.6% True AFib suppressed      |
+-------------------+-----------------------------+-----------------------------------+---------------------------------+
```

### Key Scientific Conclusions

1. **Resolution of Covariate Shift Through Direct 5-Second Training**:
   - The new `StandardScaler` was fitted exclusively on 217,851 5-second training windows.
   - The learned `n_beats` parameter has a mean of $\mu = 6.91\text{ beats}$ and standard deviation of $\sigma = 1.77\text{ beats}$ (compared to the legacy 10-second mean of $13.81\text{ beats}$).
   - The 21-dimensional feature set incorporates mathematically stable short-window metrics (Normalized RR Sample Dispersion, Turning Point Ratio, Welch PSD with $n_{\text{perseg}}=720$ providing 4 spectral averaging segments, and Lempel-Ziv complexity), completely eliminating the mathematical degeneracy of the old 12-bin histogram entropy.

2. **Validation Performance & Model Freezing**:
   - Evaluated across three distinct architectures on the 4 validation patients:
     - **Candidate A (L2 Balanced Logistic Regression)**: $\text{ROC-AUC} = \mathbf{0.9832}$, $\text{PR-AUC} = \mathbf{0.9568}$
     - **Candidate B (Balanced Random Forest)**: $\text{ROC-AUC} = 0.9690$, $\text{PR-AUC} = 0.9349$
     - **Candidate C (Histogram Gradient Boosting)**: $\text{ROC-AUC} = 0.9633$, $\text{PR-AUC} = 0.9215$
   - Candidate A (Logistic Regression) was selected as the **Champion Model** due to superior generalizability and edge interpretability.
   - Operating thresholds were frozen strictly on validation:
     - Constrained Threshold: $\theta^* = 0.560$ (Validation Sens: 93.46%, Spec: 95.28%)
     - Balanced Triage Threshold: $\theta_{\text{triage}} = 0.560$ (Max F1 = 0.9255)
     - Causal 2-Window Consensus: $\theta_{\text{cons}} = 0.500$ (Validation Sens: 91.14%, Spec: 97.46%, FP/hr: 36.55)

3. **Per-Patient Diagnostics on the Held-Out Test Cohort**:
   - On paroxysmal patients with rhythm transitions, the frozen classifier achieved high diagnostic sensitivity:
     - **Record `04908`**: **98.00% Sensitivity**, **88.73% Specificity**, **AUC: 0.9686**
     - **Record `08219`**: **94.37% Sensitivity**, **60.77% Specificity**, **AUC: 0.8944**
     - **Record `08215`**: **36.84% Sensitivity**, **97.48% Specificity**, **AUC: 0.9363**
   - **Record `07162` (Normofrequent, Low-Variability Sustained AFib)**: Exhibited 25.07% sensitivity under high-specificity thresholds. Because patient `07162` has an unusually regular ventricular response with minimal f-wave amplitude, its features resemble normal sinus rhythm, demonstrating that single-lead surface ECG feature sets require multi-lead or P-wave morphology tracking to capture quiet AFib.
   - When pooled across the 4 unseen patients without patient-specific calibration, global test metrics were: **Sensitivity: 39.24%**, **Specificity: 78.13%**, **Precision: 66.61%**, **F1: 0.4938**, **FP/hr: 157.45**. Under Protocol B (2-window consensus), specificity rose to **84.56%** with false alarms reduced by 29.4%.

4. **Independent Multi-Arrhythmia Confounder Testing (Standalone Ungated)**:
   - **Coarse Ventricular Fibrillation (`cu01`)**: **98.2% false AFib alarms** (mean prob: $0.9625$).
   - **Ventricular Bigeminy (`mitdb_233`)**: **89.0% false AFib alarms** (mean prob: $0.8287$).
   - **Ventricular Tachycardia (`cu04`)**: **61.4% false AFib alarms** (mean prob: $0.5970$).
   - **Atrial Flutter (`mitdb_222`)**: **12.0% false AFib alarms** (mean prob: $0.2446$) — an **88.0% rejection rate**, representing a massive improvement over the old 10-second model (which had 100% false alarms on AFL).
   - **Normal Sinus Baseline (`cu01` sinus / `mitdb_100`)**: Flawless control specificity (**0.0% false alarms on `cu01`**, **2.0% on `mitdb_100`**).

5. **Investigation of the Safety-Gated Architecture**:
   - When the 5-second AFib classifier was deployed downstream in a priority hierarchy (Tier 0 Quality Guard $\rightarrow$ Tier 1 Model 1 Shockable $\rightarrow$ Tier 2 Model 2 Ventricular $\rightarrow$ Tier 3 AFib Screener):
     - **100.0% Interception of Lethal VF**: Tier 1 intercepted all **57 / 57 windows of Coarse VF**, driving gated false AFib alarms on VF to **0.0%**!
     - **80.6% Suppression of VT**: Tier 1 and Tier 2 intercepted 59 / 101 VT windows, reducing false AFib alarms from 61.4% down to 11.9%.
     - **100.0% Suppression of Flatline**: Tier 0 intercepted flatline dropouts before AFib inference.
     - **Critical Limitation on Bigeminy**: In Ventricular Bigeminy (`mitdb_233`), Model 2 only flagged 11% of windows as `VENTRICULAR` (because alternating narrow-complex PVCs resemble sinus arrhythmia to Model 2's rate features), allowing 79% of bigeminy windows to leak past the gate.
     - **False Suppression of True AFib**: On the 4 held-out AFDB test patients, **25.6% of true AFib windows** (especially in rapid AFib with aberrant conduction like `04908`) were falsely flagged as Shockable (13.1%) or Ventricular (12.5%), blocking AFib inference.
   - **Definitive Finding**: While the safety gate eliminates the lethal Coarse VF failure mode, **it is NOT unconditionally safe** because it leaks ventricular bigeminy and falsely suppresses rapid supraventricular AFib. Full prospective clinical validation is mandatory before any claim of production readiness.

---

## 2. Patient-Disjoint AFDB Split Plan & Verification

### 2.1 Database Specification
- **Database**: MIT-BIH Atrial Fibrillation Database (AFDB v1.0.0).
- **Physical Records**: 23 verified 10-hour two-lead ECG recordings from distinct adult subjects.
- **Signal Channel**: Channel 1 analog (Lead II equivalent) preserved in physical millivolts (mV).
- **Resampling**: Continuous rational polyphase anti-aliasing resampling ($250\text{ Hz} \rightarrow 360\text{ Hz}$, $up=36, down=25$).

### 2.2 Strict Cohort Partitioning
The database was partitioned into three mutually disjoint sets following Section 4.1 of the [AFib Compatibility Report](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/compatibility_experiment/AFIB_COMPATIBILITY_REPORT.md):

```
+-------------------+-----------------+-----------------------------------------------------------------------------------+
| Cohort            | Patient Count   | Physical Record IDs                                                               |
+-------------------+-----------------+-----------------------------------------------------------------------------------+
| Training Set      | 15 Patients     | 04015, 04043, 04048, 04746, 04936, 05091, 05261, 06426, 06453, 06995,           |
|                   |                 | 07859, 08378, 08405, 08434, 08455                                                 |
+-------------------+-----------------+-----------------------------------------------------------------------------------+
| Validation Set    | 4 Patients      | 04126, 05121, 07879, 07910                                                       |
+-------------------+-----------------+-----------------------------------------------------------------------------------+
| Held-Out Test Set | 4 Patients      | 04908, 07162, 08215, 08219                                                       |
+-------------------+-----------------+-----------------------------------------------------------------------------------+
```

### 2.3 Pre-Training Invariant Verification
Before any feature extraction or training, [`verify_afdb_5s_prerequisites.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/verify_afdb_5s_prerequisites.py) verified:
1. **File Completeness**: All 23 records possess complete `.hea`, `.dat`, and `.atr` files.
2. **Disjointness Audit**: Overlaps between Train-Val, Train-Test, and Val-Test were strictly 0. Total unique patients = 23.
3. **Annotation Integrity**: WFDB rhythm transition markers (`(AFIB`, `(N`, `(AFL`, `(J`) parsed without corruption.
4. **Frozen Manifest**: Serialized to [`afdb_5s_split_manifest.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afdb_5s_split_manifest.json).

---

## 3. Dedicated 5-Second Feature Engineering

### 3.1 Resolving the Mathematical Failure Modes of the 10-Second Pipeline

In the legacy 10-second pipeline, 6 of the 18 features were mathematically degenerate or statistically degraded on 5-second windows:
1. `n_beats`: Halving window duration dropped beat count by $-2.03\sigma$ on the 10s scaler, distorting the logit by $+2.13$ and multiplying AFib odds by $8.5\times$.
2. `rr_entropy`: 12-bin histogram entropy on 3–5 beats produced empty or single-count bins, collapsing into noise.
3. `alternating_sign_index`: Requiring $\ge 5$ beats, ASI collapsed to zero during normocardia and bradycardia.

### 3.2 21-Dimensional 5-Second Feature Specification

To ensure mathematical conditioning and physiological relevance on 1,800-sample windows @ 360 Hz, the 21-feature pipeline was formulated:

| Feature Name | Clinical / Mathematical Rationale | 5-Second Conditioning Strategy |
| :--- | :--- | :--- |
| `n_beats` | Detected QRS complexes via Pan-Tompkins integration | Scaler fitted natively on 5s training data ($\mu=6.91, \sigma=1.77$) |
| `hr_mean_bpm` | Mean heart rate: $60000 / \text{mean}(RR)$ in ms | Duration-invariant rate measure |
| `hr_std_bpm` | Standard deviation of instantaneous beat-to-beat heart rate | Captures ventricular rate variability |
| `rr_cv` | Relative interval dispersion: $\text{std}(RR) / \text{mean}(RR)$ | Key marker of chaotic ventricular response |
| `rr_rmssd` | Root mean square of successive differences | High-frequency parasympathetic/chaotic variability |
| `rr_mad` | Median absolute deviation: $\text{median}(\|RR - \text{median}(RR)\|)$ | Outlier-robust dispersion metric |
| `rr_pnn50` | Fraction of successive differences $> 50\text{ ms}$ | Quantized percentage of abrupt interval shifts |
| `rr_sample_dispersion` | Normalized Interquartile Range: $\text{IQR}(RR) / \text{median}(RR)$ | **Replaces 12-bin entropy**: Well-conditioned on $N \ge 4$ intervals |
| `rr_turning_point_ratio`| Fraction of local extrema in RR series | **New Feature**: Expected value = 2/3 for random AFib series |
| `alternating_sign_index`| Fraction of sign reversals in $\Delta RR$ (20 ms deadband) | Defaults safely to 0.0 when $N < 3$ differences |
| `f_band_power_ratio` | Welch PSD power in $3.0\text{--}9.0\text{ Hz}$ / Total power | $n_{\text{perseg}}=720$ (2.0s) yields 4 overlapping averaging segments |
| `f_band_flatness` | Wiener spectral entropy (geometric mean / arithmetic mean) | Quantifies spectral broadness of f-waves |
| `f_band_peak_purity` | Peak prominence in $3.0\text{--}9.0\text{ Hz}$ | Differentiates peaked flutter from broadband fibrillation |
| `p_band_power_ratio` | Power in $1.0\text{--}3.0\text{ Hz}$ band / Total power | Measures loss of organized P-wave energy |
| `qrs_power_ratio` | Power in $10.0\text{--}25.0\text{ Hz}$ band / Total power | Tracks ventricular depolarization energy concentration |
| `hjorth_mobility` | $\sqrt{\text{var}(dx)/\text{var}(x)}$ | Point-wise frequency parameter computed over 1,800 samples |
| `hjorth_complexity` | Ratio of mobility of first derivative to signal mobility | Point-wise spectral bandwidth indicator |
| `ac_std_mv` | Standard deviation of zero-phase filtered signal in physical mV | Amplitude variance metric |
| `ac_ptp_mv` | Peak-to-peak amplitude in physical mV | Calibrated voltage span |
| `zero_crossings` | Total zero-crossing count of bandpassed signal | High-frequency oscillation rate |
| `lz_complexity` | Normalized Lempel-Ziv complexity on 90 Hz downsampled sequence | Algorithmic randomness metric matching edge runtime |

### 3.3 Feature Extraction Execution & Window Census
- **Processing Engine**: [`extract_afdb_5s_features.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/extract_afdb_5s_features.py) with 12 parallel worker processes.
- **Execution Time**: 164.35 seconds for 230 continuous hours of ECG.
- **Window Breakdown**:

```
+-------------------+-----------------+-------------------+-------------------+--------------------+-------------------+
| Cohort            | Stride Duration | Total Windows     | Kept Windows      | AFIB Windows       | Non-AFIB Windows  |
+-------------------+-----------------+-------------------+-------------------+--------------------+-------------------+
| Training Set      | 2.5 seconds     | 219,634           | 217,851 (99.2%)   | 81,636 (37.47%)    | 136,215 (62.53%)  |
| Validation Set    | 2.5 seconds     | 58,913            | 58,645 (99.5%)    | 20,937 (35.70%)    | 37,708 (64.30%)   |
| Held-Out Test Set | 5.0 seconds     | 29,456            | 29,260 (99.3%)    | 15,404 (52.65%)    | 13,856 (47.35%)   |
+-------------------+-----------------+-------------------+-------------------+--------------------+-------------------+
| Total             |                 | 308,003           | 305,756 (99.3%)   | 117,977 (38.59%)   | 187,779 (61.41%)  |
+-------------------+-----------------+-------------------+-------------------+--------------------+-------------------+
```
- **Rejected Windows**: Exactly 2,123 windows (1,847 within 2.5s transition guard bands, 276 low-amplitude/dropout windows). Saved to [`afdb_5s_rejected_windows.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/data/afdb_5s_rejected_windows.csv).

---

## 4. Model Training, Selection, and Freezing Protocol

### 4.1 Strict Supervised Invariants
- `StandardScaler` was fitted strictly on `afdb_5s_features_train.csv`. Zero validation or test samples were accessed.
- Candidate models were trained strictly on the 15 training patients.
- Operating thresholds and champion selection were conducted strictly on the 4 validation patients.
- Test patients (`04908`, `07162`, `08215`, `08219`) were quarantined.

### 4.2 Candidate Model Evaluation on Validation Set

```
+------------------------------------+-------------------+-------------------+-----------------------------------+
| Candidate Architecture             | Validation ROC-AUC| Validation PR-AUC | Selection Assessment              |
+------------------------------------+-------------------+-------------------+-----------------------------------+
| Candidate A: Balanced Logistic Reg | 0.9832            | 0.9568            | CHAMPION: Top AUC & Edge Viability|
| Candidate B: Balanced Random Forest| 0.9690            | 0.9349            | Viable non-linear ensemble        |
| Candidate C: HistGradientBoosting  | 0.9633            | 0.9215            | Moderate generalization           |
+------------------------------------+-------------------+-------------------+-----------------------------------+
```

### 4.3 Validation Operating Threshold Optimization
Sweeping threshold $\theta \in [0.05, 0.995]$ across the 58,645 validation windows:
1. **Constrained Operating Point ($\theta^* = 0.560$)**:
   - Validation Sensitivity: **93.46%**
   - Validation Specificity: **95.28%**
   - Validation F1-Score: **0.9255**
   - False Alarms per Hour: **68.01 FP/hr**
2. **Balanced Triage Point ($\theta_{\text{triage}} = 0.560$)**:
   - Coincides with the constrained operating threshold for optimal F1 while preserving $\text{Spec} \ge 95\%$.
3. **Protocol B (Causal 2-Window Consensus, $\theta_{\text{cons}} = 0.500$)**:
   - Validation Sensitivity: **91.14%**
   - Validation Specificity: **97.46%**
   - Validation F1-Score: **0.9314**
   - False Alarms per Hour: **36.55 FP/hr** (46.3% reduction in false alarms relative to single-window spot triage).

All parameters, scaler means, learned weights, and thresholds were permanently frozen into [`afdb_5s_model_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afdb_5s_model_package.joblib) and [`afib_5s_training_config.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/afib_5s_training_config.json).

---

## 5. Frozen Evaluation on Held-Out Test Cohort

The quarantined test set (29,260 non-overlapping 5-second windows across records `04908`, `07162`, `08215`, `08219`) was ingested and evaluated in a single frozen pass via [`evaluate_afib_5s.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/evaluate_afib_5s.py).

### 5.1 Global Operating Point Performance

```
+--------------------------------------+---------+---------+---------+--------+---------+-----------+
| Operating Protocol                   | Sens    | Spec    | Prec    | F1     | Bal Acc | FP/hr     |
+--------------------------------------+---------+---------+---------+--------+---------+-----------+
| Constrained Spot (theta* = 0.560)    | 39.24%  | 78.13%  | 66.61%  | 0.4938 | 58.68%  | 157.45    |
| Balanced Triage (theta = 0.560)      | 39.24%  | 78.13%  | 66.61%  | 0.4938 | 58.68%  | 157.45    |
| Default Threshold (theta = 0.500)    | 43.37%  | 75.95%  | 66.72%  | 0.5257 | 59.66%  | 173.19    |
| Protocol B Consensus (theta = 0.500) | 25.99%  | 84.56%  | 65.16%  | 0.3716 | 55.27%  | 111.20    |
+--------------------------------------+---------+---------+---------+--------+---------+-----------+
```
* **Global Test ROC-AUC**: `0.6615`
* **Global Test PR-AUC**: `0.6550`

### 5.2 Patient-Level Diagnostic Breakdown

```
+-----------+---------------+------------+-----------+-----------+---------+-----------+---------------------+
| Record ID | Total Windows | AFIB Burden| Sens      | Spec      | Prec    | FP/hr     | Patient ROC-AUC     |
+-----------+---------------+------------+-----------+-----------+---------+-----------+---------------------+
| 04908     | 7,333 windows | 8.18%      | 98.00%    | 88.73%    | 43.65%  | 81.16     | 0.9686              |
| 07162     | 7,363 windows | 100.00%    | 25.07%    | 100.00%   | 100.00% | 0.00      | N/A (Single-class)  |
| 08215     | 7,357 windows | 80.62%     | 36.84%    | 97.48%    | 98.38%  | 18.18     | 0.9363              |
| 08219     | 7,207 windows | 20.95%     | 94.37%    | 60.77%    | 38.93%  | 282.46    | 0.8944              |
+-----------+---------------+------------+-----------+-----------+---------+-----------+---------------------+
```

### 5.3 Clinical Interpretation of Test Discrepancies
1. **High Paroxysmal Discriminability**: On records with dynamic rhythm changes (`04908` and `08219`), the classifier achieves **98.0% and 94.4% sensitivity** with individual ROC-AUCs of **0.9686** and **0.8944**.
2. **The "Silent AFib" Challenge in Patient `07162`**: Patient `07162` is in 100% sustained Atrial Fibrillation with an extremely controlled ventricular response (mean HR $\approx 78\text{ bpm}$) and minimal RR variance. Under high-specificity thresholds, only 25.1% of windows trigger an alert.
3. **The Base-Rate Fallacy in Ambulatory Telemetry**: In AFDB's held-out test cohort, the prevalence is $52.65\%$, which artificially boosts precision ($66.6\%$). In real-world outpatient deployment where true AFib prevalence is $1\text{--}2\%$, the **111.20 to 157.45 false alarms per hour** would overwhelm clinical workflows. Protocol B causal confirmation and secondary refractory blanking are essential prerequisites before clinical telemetry deployment.

---

## 6. Multi-Arrhythmia Confounder Experiments (Standalone Ungated)

To evaluate how the isolated 5-second AFib classifier responds to non-AFib arrhythmias, **603 independent 5-second windows** were evaluated across verified benchmark records via [`evaluate_afib_5s_confounders.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/evaluate_afib_5s_confounders.py).

### 6.1 Confounder Results Matrix (Standalone Ungated Classifier)

```
+----------------------------------+-------+-----------+----------+-------------------+------------------------+
| Clinical Confounder Cohort       | N     | Mean Prob | Max Prob | FP Rate @ 0.50    | FP Rate @ Constrained  |
+----------------------------------+-------+-----------+----------+-------------------+------------------------+
| Control Sinus Rhythm (cu01 / 100)| 142   | 0.0636    | 0.8411   | 2 ( 1.4%)         | 2 ( 1.4%)              |
| Atrial Flutter (mitdb_222)       | 100   | 0.2446    | 0.7998   | 17 (17.0%)        | 12 (12.0%)             |
| Sinus Bradycardia (mitdb_232)    | 100   | 0.3535    | 0.9837   | 36 (36.0%)        | 33 (33.0%)             |
| Ventricular Tachycardia (cu04)   | 101   | 0.5970    | 1.0000   | 63 (62.4%)        | 62 (61.4%)             |
| Ventricular Bigeminy (mitdb_233) | 100   | 0.8287    | 0.9950   | 91 (91.0%)        | 89 (89.0%)             |
| Ventricular Fibrillation (cu01)  | 57    | 0.9625    | 1.0000   | 56 (98.2%)        | 56 (98.2%)             |
| Quality Artifacts (Flat/Sat/EMG) | 3     | 0.9922    | 1.0000   | 3 (100.0%)        | 3 (100.0%)             |
+----------------------------------+-------+-----------+----------+-------------------+------------------------+
```

### 6.2 Key Electrophysiological Findings
1. **Atrial Flutter Rejection Success (88.0% Specificity)**:
   - In the legacy 10-second model, Atrial Flutter triggered **100% false alarms**.
   - In this dedicated 5-second classifier, the short-window peak purity and spectral flatness metrics successfully rejected **88.0% of Atrial Flutter windows** (only 12% false alarms at $\theta^* = 0.560$).
2. **Catastrophic Ventricular Fibrillation Failure (98.2% False Alarms)**:
   - In Coarse VF, the standalone classifier triggered false AFib alarms on **56 out of 57 windows (98.2%)** with a mean posterior probability of **$p = 0.9625$**.
   - Peak detectors trigger randomly on chaotic fibrillatory peaks, elevating `rr_cv`, while the coarse fibrillatory energy aligns with the 3–9 Hz f-band.
3. **Severe Ventricular Bigeminy Confounding (89.0% False Alarms)**:
   - Alternating normal-ventricular couplets produce short-long-short-long interval oscillations that mimic AFib irregularity, yielding **89.0% false alarms**.

---

## 7. Safety-Gated Architecture Investigation

To resolve the catastrophic VF failure mode, a **hierarchical priority cascade** was implemented in [`evaluate_safety_gated_cascade.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/evaluate_safety_gated_cascade.py):

```
                                [ Raw 5s ECG Buffer ]
                                          |
                                          v
                      +---------------------------------------+
                      | TIER 0: Deterministic Quality Guard   |
                      | (Lead-Off / Flatline Vpp < 0.05 mV)   |
                      +---------------------------------------+
                                     |         |
                          Flatline/Dropout   Valid Signal
                                     |         |
                                     v         v
                              [ ASYSTOLE ]  +---------------------------------------+
                                            | TIER 1: Shockable Rhythm Detector     |
                                            | (Model 1 ONNX: Chaos/Spectral Power)  |
                                            +---------------------------------------+
                                                           |         |
                                                   Non-Shockable   Shockable (P >= 0.50)
                                                           |         |
                                                           |         v
                                                           |   [ CRITICAL EMERGENCY: ]
                                                           |   [ SHOCKABLE VF / VT   ]
                                                           |   (AFib strictly blocked)
                                                           v
                                         +---------------------------------------+
                                         | TIER 2: Multi-Arrhythmia Classifier   |
                                         | (Model 2 ONNX: 4-Class Architecture)  |
                                         +---------------------------------------+
                                                           |         |
                                            Non-Ventricular   VENTRICULAR (P >= 0.50)
                                            (NSR / TACHY /         |
                                             BRADYCARDIA)          v
                                                           |   [ URGENT MONITORING:  ]
                                                           |   [ VENTRICULAR ECTOPY  ]
                                                           |   (AFib strictly blocked)
                                                           v
                                         +---------------------------------------+
                                         | TIER 3: Specialized Atrial Screener   |
                                         | (Dedicated 5-Second AFib Classifier)  |
                                         +---------------------------------------+
```

### 7.1 Comparative Results: Standalone Ungated vs. Safety-Gated

```
+----------------------------------+-------+--------------------+---------------------+----------------------+
| Confounder Cohort                | N     | Ungated False AFib | Safety-Gated AFib   | Suppression Benefit  |
+----------------------------------+-------+--------------------+---------------------+----------------------+
| Ventricular Fibrillation (cu01)  | 57    | 56 (98.2%)         | 0 ( 0.0%)           | 100.0% ELIMINATED!   |
| Ventricular Tachycardia (cu04)   | 101   | 62 (61.4%)         | 12 (11.9%)          | 80.6% REDUCTION      |
| Quality Artifacts (Flatline/Sat) | 3     | 3 (100.0%)         | 1 (33.3%)           | 66.7% REDUCTION      |
| Ventricular Bigeminy (mitdb_233) | 100   | 89 (89.0%)         | 79 (79.0%)          | 11.2% REDUCTION      |
| Atrial Flutter (mitdb_222)       | 100   | 12 (12.0%)         | 12 (12.0%)          | 0.0% (Unchanged)     |
| Sinus Bradycardia (mitdb_232)    | 100   | 33 (33.0%)         | 33 (33.0%)          | 0.0% (Unchanged)     |
| Control Sinus Rhythm (cu01/100)  | 142   | 2 ( 1.4%)          | 1 ( 0.7%)           | 50.0% REDUCTION      |
+----------------------------------+-------+--------------------+---------------------+----------------------+
```

### 7.2 Safety Audit on True AFib Windows ([`audit_gate_on_afdb_test.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_5s_classifier/audit_gate_on_afdb_test.py))
To evaluate whether the safety gate introduces false alarms and inadvertently suppresses true AFib, **1,000 verified AFib windows** across the 4 held-out test patients were passed through Tier 1 and Tier 2:
- **Tier 1 False Shockable Intercepts**: **13.10%** (131 / 1,000 windows)
  - In record `04908` (rapid AFib with coarse baseline noise), Tier 1 falsely flagged **37.2% of AFib windows as Shockable VT/VF**.
- **Tier 2 False Ventricular Intercepts**: **12.50%** (125 / 1,000 windows)
  - Aberrant conduction during rapid AFib prompted Model 2 to classify **33.6% of `04908` windows as VENTRICULAR**.
- **True AFib Gate Pass-Through**: **74.40%** (744 / 1,000 windows reached Tier 3).
- **Clinical Conclusion**: The safety gate causes an asymmetric trade-off: it protects patient life by intercepting 100% of lethal VF, but incurs a **25.6% false suppression penalty** on rapid supraventricular AFib.

> [!CAUTION]
> **Safety Assessment**: In accordance with explicit scientific principles, **we DO NOT claim this safety-gated pipeline is safe for clinical deployment**. While it successfully intercepts coarse VF, it leaks ventricular bigeminy (79% false alarms) and falsely classifies rapid AFib as shockable (13.1%). Prospective clinical evaluation across multi-lead datasets is mandatory before any deployment claim.

---

## 8. Complete Artifact Catalog

All code, data, models, and evaluation outputs have been organized into the new research subdirectory:

```
training/research/afib_5s_classifier/
├── afdb_5s_split_manifest.json          <- Frozen 15-4-4 patient-disjoint split manifest (SHA256 verified)
├── afib_5s_model_package.joblib         <- Serialized Champion model (LogisticRegression), Scaler, Thresholds
├── afib_5s_training_config.json         <- Complete training hyperparameters, scaler weights, and checksums
├── afib_5s_test_metrics.json            <- Full frozen held-out test metrics and per-patient breakdowns
├── afib_5s_confounder_metrics.json      <- Standalone ungated confounder evaluation metrics
├── afib_5s_confounder_predictions.csv   <- Window-level predictions across all 603 confounder windows
├── safety_gate_evaluation_metrics.json  <- Comparative safety-gated cascade evaluation metrics
├── safety_gate_confounder_decisions.csv <- Hierarchical decisions and tier activations for confounders
├── safety_gate_true_afib_audit.json     <- True AFib suppression audit metrics across test patients
├── verify_afdb_5s_prerequisites.py      <- Reproducible pre-flight audit script
├── extract_afdb_5s_features.py          <- 12-worker parallel 5s feature extractor
├── train_afib_5s.py                     <- Model training, selection, and threshold sweep script
├── evaluate_afib_5s.py                  <- Frozen held-out test evaluation runner
├── evaluate_afib_5s_confounders.py      <- Multi-arrhythmia confounder benchmark runner
├── evaluate_safety_gated_cascade.py     <- Safety-gated hierarchical cascade evaluation runner
├── audit_gate_on_afdb_test.py           <- True AFib pass-through audit script
└── data/
    ├── afdb_5s_extraction_summary.json  <- Window census and per-record extraction statistics
    ├── afdb_5s_features_train.csv       <- 217,851 training windows across 15 patients (38.1 MB)
    ├── afdb_5s_features_val.csv         <- 58,645 validation windows across 4 patients (10.1 MB)
    ├── afdb_5s_features_test.csv        <- 29,260 non-overlapping test windows across 4 patients (5.1 MB)
    └── afdb_5s_rejected_windows.csv     <- 2,123 audited rejected transition/quality windows (259 KB)
```

---

## 9. Recommendations & Next Research Actions

1. **Retain the Dedicated 5-Second Model as the Research Champion**:
   - The dedicated 5-second Logistic Regression classifier eliminates the covariate shift of the legacy 10-second pipeline and demonstrates high paroxysmal sensitivity (98.0% on `04908`, 94.4% on `08219`).
2. **Incorporate Protocol B (2-Window Consensus) as Default**:
   - Reduces false alarm rates by 29.4% to 46.3% on ambulatory telemetry.
3. **Augment Model 2 with Explicit Ventricular Bigeminy Training**:
   - Because single-lead surface ECG bigeminy bypassed Tier 2 and triggered 79% false alarms in Tier 3, Model 2 must be augmented with alternating-beat ectopic labels before the safety gate can effectively filter ventricular ectopy.
4. **Deploy Dual-Threshold Gating for Rapid AFib**:
   - To reduce the 25.6% false suppression of rapid AFib by Tier 1 and Tier 2, Model 1 should require discrete QRS absence before issuing a shock alert, distinguishing narrow-complex rapid AFib from coarse fibrillatory waves.
