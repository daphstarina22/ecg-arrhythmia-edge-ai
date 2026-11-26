# Dedicated Atrial Fibrillation (AFib) Data Preparation & Research Plan

**Author:** Senior ECG Machine Learning Researcher & Edge-AI Deployment Engineer  
**Date:** October 25, 2026  
**Repository:** `ecg-arrhythmia-edge-ai`  
**Deliverable Document:** `training/research/AFIB_DATA_PREPARATION_PLAN.md`  
**Status:** Pre-Training Research Plan & Data Specification  
**Hardware Target:** Raspberry Pi 4B+ (ARM64 Debian Bookworm) + Texas Instruments ADS1292R 24-bit Biopotential AFE (Dual Channel @ 360 Hz)  

---

## 1. Executive Summary & Policy Compliance

This document establishes the technical, statistical, and clinical specifications for developing a dedicated **Atrial Fibrillation (AFib) rhythm classifier** for the ECG Arrhythmia Edge AI system.

### Scope & Invariant Protection Rules:
- **AFib Priority:** AFib is the sole focus of this plan. Bradycardia and Asystole remain separate future workstreams.
- **Strict PVC Exclusion:** In accordance with explicit user instructions, Premature Ventricular Contraction (PVC) detection is **strictly excluded**. No PVC models, beat segmenters, or PVC pipelines are developed, trained, or integrated in this task.
- **Production Code Freeze:** Production acquisition code ([`deploy/ads1292r_driver.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/ads1292r_driver.py)), deployment runtime ([`deploy/combined_ecg_app.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/combined_ecg_app.py)), signal-quality thresholds, and deployed ONNX models remain strictly identical to HEAD.
- **No Premature Model Training:** This document specifies dataset acquisition, windowing, and evaluation protocols without training or downloading data.

---

## 2. Local Dataset Audit: Verified AFib Annotations

A detailed audit of the local archives ([`MITBIH.zip`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/MITBIH.zip), [`VFDB.zip`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/VFDB.zip), and [`mitbih-vtvtf.zip`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/mitbih-vtvtf.zip)) was executed via [`training/research/inspect_afib_episodes.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/inspect_afib_episodes.py).

### 2.1 Verified Local MIT-BIH Arrhythmia Database AFib Inventory
In `MITBIH.zip` (48 records, 30 minutes each @ 360 Hz), rhythm-level Atrial Fibrillation (`(AFIB`) and Atrial Flutter (`(AFL`) annotations are confined to exactly **8 records**:

```
+--------+------------------+------------------+--------------------+-------------------------------------------+
| Record | AFIB Episodes    | AFIB Total Time  | AFL Total Time     | Episode Duration Range                    |
+--------+------------------+------------------+--------------------+-------------------------------------------+
| 201    | 3 episodes       | 605.8s (10.1 m)  | 0.0s               | 36.8s to 378.2s                           |
| 202*   | 4 episodes       | 586.3s ( 9.8 m)  | 47.9s ( 0.8 m)     | 15.2s to 230.4s                           |
| 203    | 21 episodes      | 1455.0s (24.2 m) | 314.3s ( 5.2 m)    | 0.6s to 292.9s (Frequent short bursts)    |
| 210    | 9 episodes       | 1769.9s (29.5 m) | 0.0s               | 4.3s to 545.5s (Sustained AFib)           |
| 217    | 24 episodes      | 252.4s ( 4.2 m)  | 0.0s               | 2.1s to 49.7s (Brief paroxysms)           |
| 219    | 10 episodes      | 1426.7s (23.8 m) | 0.0s               | 2.4s to 508.9s                            |
| 221    | 12 episodes      | 1756.6s (29.3 m) | 0.0s               | 10.7s to 787.4s (Sustained AFib)          |
| 222    | 24 episodes      | 103.7s ( 1.7 m)  | 423.0s ( 7.0 m)    | 0.9s to 25.6s (Very short bursts / AFL)   |
+--------+------------------+------------------+--------------------+-------------------------------------------+
| Total  | 107 episodes     | 7,956.4s (2.21 h)| 785.2s (13.1 m)    | Overall range: 0.6s to 787.4s             |
+--------+------------------+------------------+--------------------+-------------------------------------------+
* Note: Records 201 and 202 originate from the exact same patient (68-year-old male).
```

### 2.2 Verified Local VFDB & Challenge 2015 Inventory
- **VFDB (`VFDB.zip`):** Contains `(AFIB` annotations in only **3 records** (`423`, `609`, `615`), totaling 9 brief episodes.
- **Challenge 2015 (`mitbih-vtvtf.zip`):** Contains zero rhythm-level AFib annotations. Bedside alarms are categorized only as *Tachycardia*, *Bradycardia*, *Asystole*, *Ventricular_Tachycardia*, and *Ventricular_Flutter_Fib*.
- **Creighton University Database (`CUDB.zip`):** Contains generic `(AF` tags in 2 records (`cu04`, `cu06`), but lacks detailed rhythm boundaries.

### 2.3 Verdict on Local Dataset Capacity:
- Total unique patients with verified AFib across all local archives: **10 patients** (7 in MIT-BIH, 3 in VFDB).
- Total available AFib duration: **2.21 hours**.
- **Conclusion:** While local data provides a useful smoke-test benchmark, training a production edge classifier on 10 patients violates medical-AI generalization requirements. Accessing a dedicated external database is mandatory.

---

## 3. External Dataset Acquisition Plan: MIT-BIH Atrial Fibrillation Database (AFDB)

To achieve robust clinical generalization without patient memorization, the project will acquire the gold-standard PhysioNet MIT-BIH Atrial Fibrillation Database.

### 3.1 Authoritative Source & Access Specifications
- **PhysioNet Resource:** MIT-BIH Atrial Fibrillation Database (AFDB v1.0.0).
- **Official URL:** `https://physionet.org/content/afdb/1.0.0/`
- **Licensing & Access:** Open Access under the Open Data Commons Public Domain Dedication and License (PDDL v1.0). Requires no credentialed access or DUA.
- **Physical Volume:** Approximately **75 MB (compressed)** across 25 records.

### 3.2 Recording Selection & Provenance
- **Total Subjects:** 25 human subjects with paroxysmal or persistent atrial fibrillation.
- **Recording Duration:** Each recording contains **10 continuous hours** of two-channel ECG (totaling ~250 hours of ambulatory data).
- **Curated Selection Protocol:**
  - **23 Validated Records:** `04015`, `04043`, `04048`, `04126`, `04746`, `04908`, `04936`, `05091`, `05121`, `05261`, `06426`, `06453`, `06995`, `07162`, `07859`, `07879`, `07910`, `08215`, `08219`, `08378`, `08405`, `08434`, `08455`.
  - **2 Excluded Records:** Records `00735` and `03665` contain uncurated or incomplete rhythm annotations in PhysioNet documentation and must be discarded from the training/evaluation corpus.

### 3.3 Signal Leads & Sampling Characteristics
- **Native Sampling Frequency ($f_s$):** **250 Hz**.
- **Resolution:** 12-bit analog-to-digital conversion over $\pm 10\text{ mV}$ dynamic range ($200\text{ ADC units/mV}$).
- **Lead Configuration:** Two bipolar ambulatory leads (Lead I and modified Lead II / Lead V1 equivalent).
- **Target System Hardware:** Raspberry Pi with Texas Instruments ADS1292R operating at **360 Hz**.

### 3.4 Polyphase Resampling Requirements (250 Hz to 360 Hz)
The edge AI inference pipeline and feature extraction filters are locked to 360 Hz. Converting 250 Hz AFDB recordings requires rational polyphase resampling:
$$\frac{f_{\text{target}}}{f_{\text{source}}} = \frac{360}{250} = \frac{36}{25}$$

```python
from scipy.signal import resample_poly
# Up-sample by 36, apply anti-aliasing lowpass filter, down-sample by 25
calibrated_360 = resample_poly(raw_signal_250, up=36, down=25)
```

#### What Resampling Fixes vs. What It CANNOT Fix:
1. **What It Fixes:** Exactly aligns sample rates, sample step sizes ($2.77\text{ ms}$), and digital filter cutoff normalized frequencies ($\frac{f}{f_s}$) with the live ADS1292R buffer.
2. **What It CANNOT Fix:**
   - *Frequency Ceiling:* Resampling up to 360 Hz cannot synthesize physiological high-frequency biopotentials above the original 250 Hz Nyquist boundary ($125\text{ Hz}$).
   - *Phase and Gain Differences:* AFDB was digitized with 1980s analog tape filters ($0.1\text{--}40\text{ Hz}$ bandwidth); the ADS1292R has a modern sigma-delta 24-bit architecture with $0.05\text{--}100\text{ Hz}$ passband.
   - *Lead Vector Geometry:* Ambulatory tape electrodes placed across the sternum do not match the exact dipole orientation of ADS1292R Lead II chest pads.

---

## 4. Rhythm Interval to Window Conversion Protocol

A primary cause of label noise in ECG machine learning is placing fixed-length windows across transitions between different rhythms.

```
ECG Timeline:  [------ NORMAL SINUS RHYTHM ------] | TRANSITION | [------ ATRIAL FIBRILLATION ------]
Window A:      |======== 100% NSR ========| (Label: Non-AFib)
Window B (BAD):                     |=== 40% NSR / 60% AFib ===| -> LABEL CONTAMINATION!
Window C:                                                 |======== 100% AFIB ========| (Label: AFib)
```

### 4.1 Window Length Specification
- **Selected Window Length:** **10.0 seconds** (3,600 samples at 360 Hz).
  - *Clinical Rationale:* Unlike ventricular fibrillation (which exhibits high-frequency chaotic oscillations detectable in 2–5 seconds), Atrial Fibrillation requires measuring **ventricular rate irregularity** (RR interval variability, entropy). In a 5-second window at a normal or slow rate, only 4–6 beats occur, which is insufficient to compute stable statistical variance. A 10-second window captures 10–25 cardiac cycles, providing robust coefficient of variation ($\text{CV}_{\text{RR}}$) metrics.
- **Overlap Strategy:** 50% overlap (step size = 5.0 seconds = 1,800 samples) to match the edge application's temporal smoothing cadence.

### 4.2 Transition Buffer & Guard Band Policy
1. **Annotation Event Parsing:**
   - Load rhythm start and end sample timestamps from `.atr` files.
   - For every continuous rhythm segment $[S_{\text{start}}, S_{\text{end}}]$:
     - Discard any segment shorter than $10.0\text{ seconds}$ ($3,600\text{ samples}$). In MIT-BIH, 18 of the 107 AFib episodes are $<10\text{s}$ in duration (e.g. Record 203 has 0.6s bursts, Record 222 has 0.9s bursts). These brief bursts cannot fill a 10s window and must be excluded from training.
2. **Strict Guard Bands:**
   - Apply a **2.5-second guard band** at the beginning and end of every rhythm transition to ensure electrical settling.
   - Windows must be extracted strictly **within** a single homogenous rhythm interval:
     $$S_{\text{window\_start}} \ge S_{\text{start}} + 2.5\text{s} \quad \text{and} \quad S_{\text{window\_end}} \le S_{\text{end}} - 2.5\text{s}$$
3. **Purity Thresholding:**
   - Any window that intersects a rhythm change marker is **rejected from the training and evaluation corpus** (`is_transition_contaminated = True`).
   - Every admitted window has **100% label purity**.

---

## 5. Patient-Separated Partitioning Strategy

To eliminate optimistic memorization bias and ensure the model learns true physiological dynamics:

### 5.1 Pre-Windowing Record Partitioning
The 23 validated AFDB patient recordings will be split into three non-overlapping cohorts **before generating any feature windows**:

| Cohort | Fraction | Record Count | Designated AFDB Patient Records | Role in Machine Learning Pipeline |
| :--- | :---: | :---: | :--- | :--- |
| **Training Set** | ~65% | **15 records** | `04015`, `04043`, `04048`, `04126`, `04746`, `04908`, `04936`, `05091`, `05121`, `05261`, `06426`, `06453`, `06995`, `07162`, `07859` | Model parameter optimization and feature selection. |
| **Validation Set** | ~17% | **4 records** | `07879`, `07910`, `08215`, `08219` | Hyperparameter tuning, probability threshold calibration. |
| **Held-Out Test Set** | ~18% | **4 records** | `08378`, `08405`, `08434`, `08455` | Final performance reporting; strictly isolated during development. |

### 5.2 Stratification Balance Across Splits
The partition ensures that each split contains a balanced ratio of:
1. High AFib burden patients (sustained AFib $>50\%$ of recording time).
2. Paroxysmal AFib patients (frequent short episodes).
3. Low AFib burden patients (predominantly sinus rhythm with isolated paroxysms).

### 5.3 Leakage Prevention Invariants
- **Assertion 1:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Assertion 2:** `StandardScaler` must calculate mean $\mu$ and standard deviation $\sigma$ strictly on the Training set. Validation and Test sets are transformed using training parameters.
- **Assertion 3:** Overlapping windows ($50\%$ step) exist only within an individual record and never cross partition boundaries.

---

## 6. AFib vs. Non-AFib Labeling & Confounding Rhythms

### 6.1 Class Definition Contract
The target task is defined as a binary rhythm classification problem:
- **Class 1 (AFib):** Verified Atrial Fibrillation (`(AFIB`).
- **Class 0 (Non-AFib):** Normal Sinus Rhythm (`(N`), Atrial Flutter (`(AFL`), Junctional Rhythm (`(J`), and baseline supraventricular rhythms.

### 6.2 Critical Physiological Distinctions (Why Irregular RR $\neq$ AFib)
Relying solely on heart rate irregularity creates high false-positive rates due to the following confounding rhythms:

```
+------------------------------------+--------------------------+---------------------------+-----------------------------------------------+
| Cardiac Rhythm                     | Ventricular RR Pattern   | Atrial Electrical Activity| Danger of False Misclassification             |
+------------------------------------+--------------------------+---------------------------+-----------------------------------------------+
| **Atrial Fibrillation (AFib)**     | Irregularly irregular    | Absent P-waves; chaotic   | **Target Class 1**                            |
|                                    | (high variance & entropy)| fibrillatory waves (3-9Hz)|                                               |
+------------------------------------+--------------------------+---------------------------+-----------------------------------------------+
| **Atrial Flutter (AFL)**           | Regularly irregular or   | Rapid organized sawtooth  | **Confounder:** Can have irregular conduction |
|                                    | regular (2:1, 3:1, 4:1)  | F-waves at 250-350 bpm    | (variable block) mimicking AFib RR variance.  |
+------------------------------------+--------------------------+---------------------------+-----------------------------------------------+
| **Respiratory Sinus Arrhythmia**  | Smooth cyclical variance | Normal distinct P-waves   | **Confounder:** High RR variance in young/fit |
|                                    | (aligned with breathing) | preceding every QRS       | individuals; must not trigger AFib alarms.    |
+------------------------------------+--------------------------+---------------------------+-----------------------------------------------+
| **Frequent Premature Atrial Beats**| Irregular with pauses    | Abnormal premature P'     | **Confounder:** Bigeminy/trigeminy causes high|
|                                    | (compensatory pause)     | waves followed by narrow  | RR variance despite underlying sinus rhythm.  |
+------------------------------------+--------------------------+---------------------------+-----------------------------------------------+
| **Rapid Ventricular Response (RVR)**| Extremely rapid          | Fibrillatory baseline     | **Life Safety Risk:** Wide aberrant QRS beats |
|                                    | (>150 bpm) with aberrancy| obscured by high rate     | can mimic polymorphic VT.                      |
+------------------------------------+--------------------------+---------------------------+-----------------------------------------------+
```

### 6.3 Required Feature Vectors to Disentangle Confounders
To prevent false alarms from non-AFib irregular rhythms, the feature vector must couple **timing dynamics** with **spectral atrial organization**:
1. **Ventricular Rhythm Entropy (Time Domain):**
   - $\text{CV}_{\text{RR}}$: Standard deviation of RR divided by mean RR.
   - $\text{MAD}_{\Delta\text{RR}}$: Median absolute deviation of consecutive RR differences (robust to isolated ectopic beats).
   - Shannon Entropy of RR Differences: Very high in chaotic AFib, low in organized flutter or sinus arrhythmia.
2. **Atrial Sub-Band Spectral Density (Frequency Domain):**
   - Power ratio in the $3\text{--}9\text{ Hz}$ band after Pan-Tompkins QRS blanking (avoids QRS energy dominance).
   - Peak purity in the atrial band: Distinguishes narrow-band Atrial Flutter ($4\text{--}5\text{ Hz}$ sharp peak) from wide-band chaotic Atrial Fibrillation.

---

## 7. Rigorous Evaluation Protocol & Performance Metrics

### 7.1 Quantitative Benchmark Metrics
Because Atrial Fibrillation episodes occur intermittently, evaluating models solely on window-level accuracy is misleading (a patient with 95% sinus rhythm yields 95% accuracy by predicting 100% normal). The evaluation must report:

1. **Class-Specific Sensitivity & Specificity:**
   $$\text{AFib Sensitivity (Recall)} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad (\text{Target} \ge 90.0\%)$$
   $$\text{AFib Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} \quad (\text{Target} \ge 95.0\%)$$
2. **Precision (Positive Predictive Value) & F1-Score:**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad \text{and} \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
3. **Temporal False Alarm Rate ($\text{FP/hour}$):**
   $$\text{False Positives per Hour} = \frac{\text{Total False Positive Windows}}{\text{Total Non-AFib Monitoring Hours}}$$
   - *Clinical Requirement:* In wearable and emergency monitoring, $\text{FP/hour} < 1.0$ is mandatory to prevent clinical alarm fatigue.
4. **Episode Onset Detection Latency:**
   - Time elapsed from physical onset of an AFib episode until the classifier emits its first confirmed positive window (Target $< 15.0\text{ seconds}$).
5. **Per-Record Performance Audit:**
   - Disaggregated table reporting Sensitivity, Specificity, and Precision for every individual test patient record to identify anatomical outliers.

---

## 8. Dataset vs. Live ADS1292R Signal Differences

When transitioning models from historical PhysioNet databases to live hardware acquisition on the Raspberry Pi + ADS1292R, four physical domain shifts must be managed:

| Dimension | PhysioNet AFDB Characteristics | Live Hardware ADS1292R Characteristics | Engineering Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Lead Placement & Vector** | Modified bipolar chest leads (Lead I / modified V1). | Dual-channel front-end; Channel 1 configured as **Lead II equivalent** (Right Arm to Left Leg vector). | Normalize features by standard deviation; avoid raw voltage amplitude thresholds; rely on dimensionless ratios. |
| **ADC Resolution & Dynamic Range** | 12-bit ADC ($200\text{ counts/mV}$, linear range $\pm 10\text{ mV}$). | **24-bit Delta-Sigma ADC** ($48.08\text{ nV/count}$, linear range $\pm 403.33\text{ mV}$ at Gain 6). | Scale raw integers strictly to calibrated physical millivolts ($mV$) prior to feature computation. |
| **Analog Filtering & Bandwidth** | Analog tape recorder filtering ($0.1\text{--}40\text{ Hz}$). | Low-noise differential inputs with software **research-grade filter** ($0.05\text{--}100\text{ Hz}$ bandpass + $50\text{ Hz}$ India notch). | Standardize pre-processing filters across both training archives and live acquisition buffers. |
| **Electromagnetic & Motion Artifacts** | Controlled ambulatory Holter monitoring; mostly baseline drift and tape flutter. | Drone airframe vibration ($10\text{--}120\text{ Hz}$) and brushless motor PWM electromagnetic interference ($20\text{--}40\text{ kHz}$). | Integrate strict **Signal Quality Inspection (SQI)**; halt AFib classification during high AC saturation or mechanical motion bursts. |

---

## 9. Prioritized Next-Step Research Roadmap

```mermaid
flowchart TD
    subgraph Step 1: Data Ingestion & Calibration (Sprint 1)
        A1["Download Official MIT-BIH AFDB (23 Validated Records)"]
        A2["Apply Polyphase Resampling (250 Hz to 360 Hz)"]
        A3["Calibrate Signals to Physical Millivolts"]
    end

    subgraph Step 2: Leak-Free Pre-Windowing Partitioning (Sprint 1)
        B1["Execute Patient Split: Train (15 recs), Val (4 recs), Test (4 recs)"]
        B2["Apply 2.5s Guard Bands Around Rhythm Transitions"]
        B3["Extract 10s Non-Contaminated Windows (100% Label Purity)"]
    end

    subgraph Step 3: Feature Engineering & Baseline Model (Sprint 2)
        C1["Extract RR Timing Dynamics (CV_RR, MAD_RR, Shannon Entropy)"]
        C2["Extract Residual Atrial Spectral Density (3-9 Hz Bandpower)"]
        C3["Train Balanced Baseline Logistic Regression & Random Forest"]
    end

    subgraph Step 4: Independent Validation & ONNX Export (Sprint 2-3)
        D1["Evaluate on Held-Out Test Cohort (Records 08378, 08405, 08434, 08455)"]
        D2["Calculate False Positives per Hour & Per-Record Sensitivity"]
        D3["Export Research Candidate: afib_classifier_candidate.onnx"]
    end

    Step 1 --> Step 2
    Step 2 --> Step 3
    Step 3 --> Step 4
```

---

## 10. Repository Integrity & Final File Verification

### Git Status (`git status --short`)
```text
?? DATASET_ARRHYTHMIA_FEASIBILITY_REPORT.md
?? NEXT_STEPS_RESEARCH_REPORT.md
?? training/research/
```

### Files Created Under `training/research/`:
1. [`training/research/inspect_afib_episodes.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/inspect_afib_episodes.py) — Read-only script auditing exact episode intervals, durations, and window capacities.
2. [`training/research/AFIB_DATA_PREPARATION_PLAN.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/AFIB_DATA_PREPARATION_PLAN.md) — Comprehensive dataset preparation and research plan.

### Absolute Safety Rule Invariants Confirmed:
- **No PVC Classifier Was Developed or Evaluated:** Completely excluded from all scripts and plans.
- **Production Acquisition Driver ([`deploy/ads1292r_driver.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/ads1292r_driver.py)):** **STRICTLY UNCHANGED (identical to HEAD).**
- **Production Application ([`deploy/combined_ecg_app.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/combined_ecg_app.py)):** **STRICTLY UNCHANGED.**
- **Existing ONNX Models ([`deploy/models/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/models/)):** **STRICTLY UNCHANGED.**
- **Signal-Quality Thresholds & Live Runtime:** **100% PRESERVED.**
- **No Git commits or pushes were executed.**
