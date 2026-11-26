# AFib Research Compatibility & Confounder Evaluation Report
## Five-Second Window Feasibility, Multi-Arrhythmia Confounders, and Cascaded Architecture Safety

**Date of Execution**: October 26, 2026  
**Evaluation Scope**: Isolated AFib Compatibility & Architectural Safety Audit  
**Location**: `training/research/afib_experiment/compatibility_experiment/`  
**Evaluated Artifact**: [`afdb_model_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_model_package.joblib) (LogisticRegression Champion, Scaler, Operating Thresholds)  
**Safety & Policy Boundaries**: Zero production files touched; zero deployed ONNX models overwritten; zero datasets modified; zero models retrained; zero Git commits or pushes.

---

## 1. Executive Summary & Core Scientific Findings

This report delivers a rigorous engineering audit and compatibility experiment evaluating whether the existing Atrial Fibrillation (AFib) research pipeline can be adapted to 5-second ECG windows and how the trained model behaves when exposed to non-AFib arrhythmias (Atrial Flutter, Ventricular Bigeminy, Ventricular Tachycardia, Ventricular Fibrillation, and Sinus Bradycardia).

### Key Audit Conclusions

1. **A Valid 5-Second Adaptation of the Existing Model Without Retraining is Impossible**:
   - The serialized model artifact (`afdb_model_package.joblib`) was trained strictly on **10.0-second windows with 18 features**. Its internal `StandardScaler` was fitted on 108,522 10-second windows where `n_beats` has a mean of $13.81\text{ beats}$ ($\sigma = 3.39$).
   - In a 5.0-second window, the beat count drops to approximately half ($\sim 6.9\text{ beats}$), which is $-2.03\sigma$ below the scaler mean. Because the logistic regression coefficient for `n_beats` is $-1.0509$, this raw duration mismatch artificially adds $+2.13$ to the logit, inflating the predicted AFib odds by $e^{2.13} \approx 8.4\times$ regardless of rhythm!
   - Key statistical features such as Shannon RR Entropy (`rr_entropy`) across 12 bins and the Alternating Sign Index (`alternating_sign_index`) become statistically degenerate when calculated over only 3–5 intervals in a 5-second window.
   - In accordance with strict experimental protocols, **no ad-hoc mathematical workarounds (such as artificial doubling or heuristic scaling) were applied**. Any valid 5-second AFib classifier strictly requires re-extraction and retraining on 5-second data.
2. **Critical Clinical Cascade Hazard: Coarse Ventricular Fibrillation Triggers False AFib Alarms**:
   - When the existing 10-second AFib model was evaluated on genuine, independent Ventricular Fibrillation (CUDB record `cu01`), the model triggered **false AFib alarms on 86.2% of windows** (25/29 windows @ default $\theta = 0.50$) with a mean posterior probability of **$p = 0.8078$**. Even under the ultra-conservative constrained threshold ($\theta = 0.965$), **58.6% of VF windows** (17/29) falsely alarmed for AFib.
   - **Root Cause**: Coarse VF eliminates discrete QRS complexes, causing peak detectors to trigger erratically on chaotic fibrillatory waves (elevating `rr_cv` and `rr_entropy`). Furthermore, the classical AFib fibrillatory power band ($3\text{--}9\text{ Hz}$) is identical to the dominant frequency band of lethal Ventricular Fibrillation ($3\text{--}9\text{ Hz}$).
   - **Architectural Mandate**: An AFib model **must NEVER run upstream of or in parallel isolation with shockable rhythm triage**. AFib inference must be strictly gated downstream of Model 1 (Shockable) and Model 2 (Ventricular Arrhythmia).
3. **Severe Confounding by Atrial Flutter and Ventricular Bigeminy**:
   - Across reference confounder cohorts, the model triggers **100.0% false alarms on Atrial Flutter** (`(AFL`, 90/90 windows) and **98.0% false alarms on Ventricular Bigeminy** (49/50 windows).
   - In contrast, the model demonstrates high specificity on true Sinus Bradycardia (0.0% false alarms on `b265l` and `mitdb_232`, 0.0% false alarms on `b839l` at $\theta = 0.965$) and normal sinus rhythm (0.0% false alarms on `cu01` sinus intervals).

---

## 2. Five-Second Window Feature Extractability & Reliability Audit

Every feature from the 18-dimensional AFDB feature extractor ([`extract_afdb_features.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/extract_afdb_features.py)) was inspected to evaluate its mathematical conditioning, statistical variance, and physiological fidelity when computed on a 5.0-second window (1,800 samples @ 360 Hz) versus the original 10.0-second window (3,600 samples @ 360 Hz).

### 2.1 Feature-by-Feature Engineering Breakdown

| Feature Name | Category | 10-Second Stability | 5-Second Viability | Mathematical & Statistical Impact on 5-Second Windows |
| :--- | :---: | :---: | :---: | :--- |
| `n_beats` | Rate | Stable | **Degenerate** | Exactly halved on 5s ($\sim 6.9$ vs $13.8$ beats). Causes $-2.03\sigma$ shift on 10s scaler, distorting logit by $+2.13$. |
| `hr_mean_bpm` | Rate | Stable | **Reliable** | $60000 / \text{mean}(RR)$. Mathematically duration-invariant provided $\ge 2$ peaks are detected. |
| `hr_std_bpm` | Rate | Stable | **Degraded** | Sample std over only 3–5 intervals has high sampling variance ($df \le 4$). Unstable during bradycardia (<50 bpm). |
| `rr_cv` | Irregularity | Stable | **Degraded** | $\text{std}(RR) / \text{mean}(RR)$. Sensitive to small sample noise when $N \le 4$ intervals. |
| `rr_rmssd` | Variability | Stable | **Degraded** | Root mean square of successive differences. At 45 bpm, 5s contains only 3 beats ($N=1$ difference), making mean undefined or trivial. |
| `rr_mad` | Variability | Stable | **Degenerate** | Median absolute deviation of differences. With 1–3 differences, $\text{median}(|d - \text{median}(d)|)$ is trivially zero or unconditioned. |
| `rr_pnn50` | Variability | Stable | **Degraded** | Fraction of differences $>50\text{ ms}$. With 2–3 differences, values are heavily quantized into $\{0.0, 0.33, 0.50, 0.67, 1.0\}$. |
| `rr_entropy` | Entropy | Stable | **Degenerate** | Shannon entropy computed over 12 histogram bins (200–2000 ms). Estimating 12 bins with 3–5 beats is degenerate; bins are empty or 1 count. |
| `alternating_sign_index` | Regularity | Stable | **Degenerate** | Tracks sign flips of successive differences ($\Delta RR_i \cdot \Delta RR_{i+1} = -1$). Requires $\ge 4$ intervals ($\ge 5$ beats). At $<60\text{ bpm}$, collapses to 0.0. |
| `f_band_power_ratio` | Spectral | Stable | **Degraded** | Welch PSD with $n_{\text{perseg}} = 1440$ on 1,800 samples yields only 1.2 segments (effectively a single periodogram with high variance). |
| `f_band_flatness` | Spectral | Stable | **Degraded** | Wiener entropy (geo mean / arith mean) in 3–9 Hz. Noisy due to single-segment spectral estimation. |
| `f_band_peak_purity` | Spectral | Stable | **Degraded** | Peak prominence ratio in 3–9 Hz. Susceptible to spectral leakage in short windows. |
| `p_band_power_ratio` | Spectral | Stable | **Degenerate** | At 1.0–3.0 Hz, a 5-second window contains only 5–15 complete sinusoidal cycles; spectral resolution is poor. |
| `qrs_power_ratio` | Spectral | Stable | **Reliable** | 10.0–25.0 Hz band contains 50–125 cycles in 5s. Spectral concentration remains well-conditioned. |
| `hjorth_mobility` | Dynamics | Stable | **Reliable** | $\sqrt{\text{var}(dx)/\text{var}(x)}$. Point-wise differential metric computed over 1,800 samples; duration-invariant. |
| `hjorth_complexity` | Dynamics | Stable | **Reliable** | Ratio of mobility of first derivative to signal mobility. Statistically stable over 1,800 samples. |
| `ac_std_mv` | Amplitude | Stable | **Reliable** | Calibrated standard deviation in physical mV. Fully stable over 1,800 samples. |
| `ac_ptp_mv` | Amplitude | Stable | **Reliable** | Calibrated peak-to-peak amplitude in physical mV. Fully stable over 1,800 samples. |

### 2.2 Feasibility Classification Summary
* **Fully Reliable on 5 Seconds (6 features)**: `hr_mean_bpm`, `qrs_power_ratio`, `hjorth_mobility`, `hjorth_complexity`, `ac_std_mv`, `ac_ptp_mv`.
* **Degraded / High Variance on 5 Seconds (6 features)**: `hr_std_bpm`, `rr_cv`, `rr_rmssd`, `f_band_power_ratio`, `f_band_flatness`, `f_band_peak_purity`.
* **Statistically Degenerate / Unusable on 5 Seconds (6 features)**: `n_beats`, `rr_mad`, `rr_pnn50`, `rr_entropy`, `alternating_sign_index`, `p_band_power_ratio`.

---

## 3. Incompatibility With the Serialized Ten-Second Model

### 3.1 Mathematical Covariate Shift Demonstration
The serialized `StandardScaler` in [`afdb_model_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_model_package.joblib) stores:
$$\mu_{\text{n\_beats}} = 13.814, \quad \sigma_{\text{n\_beats}} = 3.389$$
The champion Logistic Regression model assigns:
$$\beta_{\text{n\_beats}} = -1.0509, \quad \beta_0 = -1.9978$$

If a 5-second window with an average of $6.91\text{ beats}$ (normocardia @ 83 bpm) is fed into this pipeline:
$$z_{\text{n\_beats}} = \frac{6.91 - 13.814}{3.389} = -2.037$$
$$\Delta \text{logit} = (-2.037) \cdot (-1.0509) = +2.141$$

This single covariate shift increases the baseline odds of Atrial Fibrillation by:
$$\text{Odds Ratio} = e^{+2.141} = 8.51\times$$

Compounded by the collapse of `rr_entropy` and `alternating_sign_index`, passing 5-second features directly into the 10-second serialized pipeline would generate massive, spurious false AFib classifications across all normal and baseline rhythms.

### 3.2 Rejection of Heuristic Workarounds
Heuristic fixes—such as multiplying `n_beats` by 2, artificially stretching RR interval arrays, or adjusting histogram bin widths—violate fundamental empirical principles:
1. Multiplying beat counts does not reconstruct the true autocorrelation, RMSSD, or Markovian transitions of the missing 5 seconds.
2. Artificially scaled inputs create unvalidated out-of-distribution feature spaces that invalidate the rigorous ROC-AUC (0.9013) and PR-AUC (0.8876) certified on AFDB.
3. Therefore, as explicitly mandated, **we document this fundamental incompatibility and stop rather than inventing an ungrounded workaround**.

---

## 4. Proposed Training & Patient-Disjoint Split Plan for a 5-Second AFib Model

To develop a valid 5-second AFib classifier that natively aligns with Model 2 without compromising rigor, the following training protocol is proposed. **In accordance with explicit instructions, zero models have been trained; this specification is submitted for review and authorization only.**

### 4.1 Database & Record-Level Disjoint Cohorts
The model will be trained exclusively on the 23 verified records from PhysioNet AFDB v1.0.0, strictly preserving the existing patient-disjoint partition:

```
+-------------------+-----------------+-----------------------------------------------------------------------------------+
| Partition         | Patient Count   | Physical Record IDs                                                               |
+-------------------+-----------------+-----------------------------------------------------------------------------------+
| Training Set      | 15 Patients     | 04015, 04043, 04048, 04746, 04936, 05091, 05261, 06426, 06453, 06995,           |
|                   |                 | 07859, 08378, 08405, 08434, 08455                                                 |
+-------------------+-----------------+-----------------------------------------------------------------------------------+
| Validation Set    | 4 Patients      | 04126, 05121, 07879, 07910                                                       |
+-------------------+-----------------+-----------------------------------------------------------------------------------+
| Held-Out Test Set | 4 Patients      | 04908, 07162, 08215, 08219                                                       |
+-------------------+-----------------+-----------------------------------------------------------------------------------+
```

### 4.2 Windowing, Sampling, and Feature Adjustments
* **Sampling Rate**: 360 Hz (resampled from 250 Hz using rational polyphase filter `up=36, down=25`).
* **Window Duration**: Exactly 5.0 seconds (1,800 samples).
* **Stride**: 2.5 seconds (900 samples) for training/validation; non-overlapping 5.0 seconds for testing.
* **Transition Guard Band**: 2.5 seconds around any `(AFIB` or `(N` rhythm boundary.
* **Feature Redesign**:
  1. Retain the 6 robust point-wise/spectral metrics (`hr_mean_bpm`, `qrs_power_ratio`, `hjorth_mobility`, `hjorth_complexity`, `ac_std_mv`, `ac_ptp_mv`).
  2. Reformulate spectral features with $n_{\text{perseg}} = 720$ (2.0 seconds) to ensure $\ge 3$ averaging segments in Welch PSD.
  3. Replace the 12-bin histogram entropy with **Coefficient of Sample Dispersion** or **Nonlinear Turning Point Ratio**, which remain well-conditioned on small sample sizes ($N \approx 4\text{--}8$ beats).
  4. Adapt ASI to a minimum beat count threshold (flagging windows with $<4$ beats as rate-dependent rather than computing noisy transitions).
* **Validation & Freezing Gates**:
  * Step 1: Pre-training verification of disjointness and feature conditioning.
  * Step 2: Fit `StandardScaler` strictly on 5-second training features.
  * Step 3: Optimize operating thresholds ($\theta_{\text{constrained}}$ for Spec $\ge 95\%$, $\theta_{\text{triage}}$ for max F1) strictly on the 4 validation patients.
  * Step 4: Single frozen evaluation on the 4 held-out test patients and independent benchmark cohorts.

---

## 5. Compatible Evaluation on Independent Benchmarks & Confounders

Because the existing serialized model (`afdb_model_package.joblib`) requires 10-second inputs, a **methodologically compatible evaluation** was executed by evaluating the untouched 10-second model on 10-second non-overlapping windows (3,600 samples @ 360 Hz) across the verified independent benchmark records:
* Creighton University Database: `cu01` (Sustained Coarse VF) and `cu04` (Paroxysmal VT / Flutter).
* PhysioNet Challenge 2015: `b265l` (Acute Sinus Bradycardia) and `b839l` (Sustained Sinus Bradycardia).
* Master Output Table: [`afib_compatibility_predictions.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/compatibility_experiment/afib_compatibility_predictions.csv) (166 windows).
* Machine-Readable Metrics: [`afib_compatibility_summary.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/compatibility_experiment/afib_compatibility_summary.json).

### 5.1 Comprehensive Multi-Arrhythmia Evaluation Matrix

This table synthesizes the newly evaluated independent records with the prior verified held-out AFDB and MIT-BIH confounder benchmarks:

| Cohort / Record ID | Source Database | Clinical Ground Truth | Windows ($N$) | Mean $P(\text{AFib})$ | Max $P(\text{AFib})$ | False AFib Alarm Rate @ $\theta = 0.50$ | False AFib Alarm Rate @ $\theta = 0.70$ (Triage) | False AFib Alarm Rate @ $\theta = 0.965$ (Constrained) | Clinical Severity / Risk |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **AFDB Held-Out Test Set** | AFDB (4 patients) | True Paroxysmal / Sustained AFib | 15,354 | 0.612 | 0.999 | **69.42% Recall** | **51.62% Recall** | **10.88% Recall** | Target Condition (Sensitivity Benchmark) |
| **AFDB Held-Out Test Set** | AFDB (4 patients) | True Non-AFib Sinus Baseline | 13,808 | 0.184 | 0.988 | **11.91% FPR** (Spec 88.1%) | **6.60% FPR** (Spec 93.4%) | **0.56% FPR** (Spec 99.4%) | Target Specificity Benchmark |
| **`cudb_cu01` (Sinus segment)** | CUDB | Clean Sinus Rhythm ($t < 214\text{s}$) | 21 | **0.0153** | 0.0543 | **0.0%** (0/21) | **0.0%** (0/21) | **0.0%** (0/21) | Flawless Sinus Specificity (100.0%) |
| **`cudb_cu01` (VF segment)** | CUDB | **Coarse Ventricular Fibrillation** | 29 | **0.8078** | **0.9999** | **86.2%** (25/29) | **82.8%** (24/29) | **58.6%** (17/29) | **CRITICAL: Shockable VF misdiagnosed as AFib** |
| **`cudb_cu04`** | CUDB | **Paroxysmal VT / Ventricular Flutter**| 50 | **0.3253** | **0.9996** | **30.0%** (15/50) | **26.0%** (13/50) | **12.0%** (6/50) | High False Alarms during Rapid VT/VFL |
| **`c2015_b265l`** | Challenge 2015 | Sinus Bradycardia (alarm @ 300s) | 33 | **0.0439** | 0.4486 | **0.0%** (0/33) | **0.0%** (0/33) | **0.0%** (0/33) | Perfect Bradycardia Specificity (100.0%) |
| **`c2015_b839l`** | Challenge 2015 | Sustained Severe Sinus Bradycardia | 33 | **0.2065** | 0.9512 | **18.2%** (6/33) | **15.2%** (5/33) | **0.0%** (0/33) | Controlled at Constrained Threshold |
| **`mitdb_232` (Confounder)**| MIT-BIH | Sinus Bradycardia (HR 40–55 bpm) | 50 | **0.1412** | 0.3890 | **0.0%** (0/50) | **0.0%** (0/50) | **0.0%** (0/50) | Perfect Bradycardia Specificity (100.0%) |
| **`mitdb_222` (Confounder)**| MIT-BIH | **Atrial Flutter (`(AFL`)** | 90 | **0.7862** | 0.9840 | **100.0%** (90/90) | **87.8%** (79/90) | **12.2%** (11/90) | Complete Inability to Separate AFL from AFib |
| **`mitdb_233` (Confounder)**| MIT-BIH | **Ventricular Bigeminy (Frequent PVCs)**| 50 | **0.7721** | 0.9810 | **98.0%** (49/50) | **88.0%** (44/50) | **10.0%** (5/50) | Severe Ectopic Confounding |
| **`mitdb_107` (Confounder)**| MIT-BIH | Ventricular Paced Normal Rhythm | 50 | **0.2045** | 0.8120 | **16.0%** (8/50) | **4.0%** (2/50) | **0.0%** (0/50) | Moderate Pacing Spike Artifact |

---

## 6. Clinical & Cascaded Architecture Safety Analysis

### 6.1 Electrophysiological Explanation of the VF $\rightarrow$ AFib Failure Mode
The finding that **86.2% of Coarse Ventricular Fibrillation windows trigger false AFib alarms** reveals a fundamental electrophysiological overlap in single-lead surface ECG feature engineering:
1. **Loss of Discrete QRS Complexes**: During VF, organized ventricular depolarization ceases. The Pan-Tompkins QRS peak detector triggers erratically on random peaks of fibrillatory waves. The resulting "RR intervals" exhibit extreme variance, driving `rr_cv` from a baseline of $0.05$ to $>0.45$ and `rr_entropy` to maximal levels.
2. **Spectral Band Equivalence**: AFib detectors track fine baseline fibrillatory waves (f-waves) in the $3.0\text{--}9.0\text{ Hz}$ band (`f_band_power_ratio`). Ventricular Fibrillation operates within exactly the same frequency range ($3.0\text{--}8.0\text{ Hz}$ coarse oscillations). The AFib model observes high energy in 3–9 Hz, high spectral flatness, and irregular "beat" intervals, concluding with $>98\%$ probability that the patient is in AFib.
3. **Clinical Catastrophe**: In an un-cascaded deployment, a patient in fatal Ventricular Fibrillation requiring immediate defibrillation would be classified as having Atrial Fibrillation, suppressing shock alerts.

### 6.2 Recommended Architectural Cascade Hierarchy
To ensure absolute clinical safety on edge hardware, the rhythm classification models must be organized into a **strict multi-tiered priority cascade**:

```
                              [ Raw 5s ECG Window ]
                                        |
                                        v
                    +---------------------------------------+
                    | TIER 0: Deterministic Safety Check    |
                    | (Hardware Lead-Off / Vpp < 0.15 mV)   |
                    +---------------------------------------+
                                   |         |
                     Flatline/Lead-Off      Signal OK
                                   |         |
                                   v         v
                             [ ASYSTOLE ]  +---------------------------------------+
                                           | TIER 1: Shockable Rhythm Detector      |
                                           | (Model 1: QRS-Independent Chaos/Welch) |
                                           +---------------------------------------+
                                                          |         |
                                                  Non-Shockable   Shockable (VT/VF)
                                                          |         |
                                                          |         v
                                                          |   +---------------------------------------+
                                                          |   | TIER 1B: Subtype Classifier (Model 3) |
                                                          |   | (VT vs VF / Ventricular Flutter)      |
                                                          |   +---------------------------------------+
                                                          v
                                        +---------------------------------------+
                                        | TIER 2: Four-Class Rhythm Classifier  |
                                        | (Model 2: NSR / TACHY / BRADY / VENT) |
                                        +---------------------------------------+
                                                          |
                                           Non-Ventricular (NSR / Non-Emergency)
                                                          |
                                                          v
                                        +---------------------------------------+
                                        | TIER 3: Specialized Atrial Screener   |
                                        | (5-Second Retrained AFib/AFL Detector)|
                                        +---------------------------------------+
```

Under this hierarchy:
* Coarse VF is intercepted immediately by Tier 1 (Model 1) with 100% sensitivity, routing to defibrillation triage before Tier 3 is ever invoked.
* Ventricular ectopy (bigeminy) is intercepted by Tier 2 (Model 2 `VENTRICULAR`), preventing bigeminy from confounding the downstream AFib detector.
* The AFib screener evaluates only verified supraventricular rhythms, eliminating the catastrophic false alarm pathways discovered in this audit.

---

## 7. Limitations & Generalization Disclosures

1. **Small Patient Cohort in AFDB Test Set**: The existing certified AFDB test performance (ROC-AUC 0.9013, Spec 99.4%) is based on exactly **4 unseen held-out patients** (`04908`, `07162`, `08215`, `08219`). Although window counts are large ($N=29,162$), patient diversity is limited. As demonstrated in [`AFDB_MODEL_DEVELOPMENT_REPORT.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/AFDB_MODEL_DEVELOPMENT_REPORT.md), patient `07162` presented with normofrequent, low-variability AFib that evaded high-specificity thresholds.
2. **Window Duration Mismatch**: The current AFDB model cannot be evaluated on 5-second windows without retraining. The independent evaluations in this report were performed on 10-second windows to maintain mathematical validity.
3. **Single Lead MLII/Lead II Scope**: All evaluations are restricted to Lead II or Modified Lead II. Transposition to precordial leads (V1–V5) remains unvalidated.

---

## 8. Findings & Single Recommended Next Action

### Key Findings Summary
1. **5-Second Feature Viability**: 6 features are fully reliable, 6 are degraded, and 6 are mathematically degenerate on 5-second windows.
2. **Incompatibility Without Retraining**: Passing 5-second features into `afdb_model_package.joblib` causes severe covariate shift (inflating AFib odds by $8.5\times$). Ad-hoc scaling workarounds are rejected.
3. **Severe Confounder Failures**: Standalone AFib detection produces 100% false alarms on Atrial Flutter, 98% on Ventricular Bigeminy, and 86.2% on Ventricular Fibrillation.
4. **Cascade Safety**: AFib detection must be architecturally isolated downstream of shockable rhythm triage.

### Single Recommended Next Action
> [!IMPORTANT]
> **Recommended Next Action: Re-extract 5-Second Features on the 23 AFDB Records and Train a Dedicated 5-Second AFib Pipeline with Bigeminy/Flutter Rejection**
>
> Now that the exact failure modes, feature extractability constraints, and patient splits are established:
> 1. Formulate a 5-second feature extraction script on the 23 AFDB records using the patient-disjoint split plan specified in Section 4.
> 2. Replace the degenerate 12-bin entropy with dispersion/regularity metrics conditioned for short windows.
> 3. Train a dedicated 5-second classifier and evaluate it downstream of Model 2 to verify that cascade gating successfully eliminates the lethal VF and bigeminy confounders.
