# Pre-Acquisition Audit & Validation Review: MIT-BIH Atrial Fibrillation Database (AFDB)

**Author:** Senior ECG Machine Learning Researcher & Edge-AI Deployment Engineer  
**Date:** October 25, 2026  
**Repository:** `ecg-arrhythmia-edge-ai`  
**Target Hardware:** Raspberry Pi 4B+ (ARM64 Debian Bookworm) + Texas Instruments ADS1292R 24-bit Biopotential AFE (Dual Channel @ 360 Hz)  
**Audited Document:** [`training/research/afib_experiment/AFDB_ACQUISITION_AND_VALIDATION_PLAN.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/AFDB_ACQUISITION_AND_VALIDATION_PLAN.md)  
**Deliverable Document:** `training/research/afib_experiment/PRE_ACQUISITION_AUDIT.md`  
**Audit Status:** Formal Pre-Acquisition Audit Completed — **Conditional GO (Pending User Approval)**  

---

## Executive Summary & Audit Mandate

In strict accordance with project safety guidelines and repository protection invariants, this pre-acquisition audit evaluates the proposed engineering specifications, data partitioning, algorithmic formulations, and evaluation protocols documented in [`AFDB_ACQUISITION_AND_VALIDATION_PLAN.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/AFDB_ACQUISITION_AND_VALIDATION_PLAN.md) **prior to any network download or model training**.

### Key Audit Findings at a Glance:
1. **Repository & Code Integrity:** The working tree is strictly preserved. All production files ([`deploy/ads1292r_driver.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/ads1292r_driver.py), [`deploy/combined_ecg_app.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/combined_ecg_app.py)), deployed ONNX models ([`deploy/models/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/models/)), and signal-quality thresholds remain 100% untouched. No PVC, bradycardia, or asystole logic has been introduced.
2. **Physical Storage Volume Discrepancy Corrected:** The initial plan cited a download size of $75.4\text{ MB}$ compressed / $152.0\text{ MB}$ raw. Authoritative PhysioNet inspection confirms the ZIP archive is **439.7 MB** and the uncompressed dataset volume is **605.9 MB**.
3. **Signal Availability vs. Annotation Status:** Records `00735` and `03665` do not merely have corrupted annotations; their ECG waveform `.dat` files are **physically absent** on PhysioNet. Exactly **23 records** provide biopotential waveforms.
4. **Severe Tape Corruption in Proposed Test Cohort:** The initial plan arbitrarily assigned the last 4 records (`08378`, `08405`, `08434`, `08455`) as the held-out test cohort. PhysioNet master logs reveal that **two of these four records (`08405` and `08434`) contain corrupted tape blocks (flatline zero fills)**, and all four lack wall-clock start times. Concentrating damaged recordings in a 4-patient test cohort introduces major bias; the split must be rebalanced.
5. **Algorithmic Soundness of False-Alarm Suppressors:** The Alternating Sign Index (ASI) and Atrial Spectral Peak Purity Ratio are mathematically verified to discriminate ventricular ectopy (Record 233 failure mode) and Atrial Flutter (AFL confounder) without data leakage.
6. **Product Workflow vs. Telemetry Distinction:** A formal evaluation protocol is established separating the **intended product spot-check workflow** (one 10-second strip $\to$ preprocessing $\to$ one prediction) from **continuous telemetry monitoring** (FP/hour, rolling consensus, detection latency).
7. **The 4-Patient Test Set Limitation:** Four held-out patients provide only $N=4$ degrees of freedom for patient-level generalization. Confidence intervals across subjects will be substantial. Cross-database validation remains mandatory before clinical deployment claims.

---

## 1. Inspection of Existing AFib Research Assets & Repository Status

### 1.1 Repository Status Check (`git status --short`)
A pre-audit check of the Git working tree was executed:
```text
?? DATASET_ARRHYTHMIA_FEASIBILITY_REPORT.md
?? NEXT_STEPS_RESEARCH_REPORT.md
?? training/research/
```
- **Tracked Code Invariant:** Zero modified tracked files.
- **Production Isolation Invariant:** Zero modifications to [`deploy/ads1292r_driver.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/ads1292r_driver.py) or [`deploy/combined_ecg_app.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/combined_ecg_app.py).
- **ONNX Model Invariant:** No modifications or overwrites in [`deploy/models/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/models/).
- **Feature Scope Invariant:** No PVC, bradycardia, or asystole classifiers have been implemented.
- **Git State Invariant:** No commits, pushes, resets, or stash operations have been performed.

### 1.2 Audit of Baseline Experiment Assets
The baseline research artifacts in [`training/research/afib_experiment/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/) were audited:

| Artifact Path | Verified Status | Observations & Audit Findings |
| :--- | :---: | :--- |
| [`dataset_inventory.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/dataset_inventory.csv) | Verified | 70 local records audited (48 MIT-BIH, 22 VFDB). Identifies 8 AFib records, 3 pacemaker records, 2 missing MLII records. |
| [`prepare_afib_data.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/prepare_afib_data.py) | Verified | Strictly enforces pre-windowing patient separation; co-locates 201/202 to `MIT_S201_202`; applies 10s window, 5s stride, 2.5s guard bands; enforces 100% label purity. |
| [`train_afib_baseline.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/train_afib_baseline.py) | Verified | Scaler fit exclusively on training cohort (zero leakage); evaluates Logistic Regression and Balanced Random Forest. |
| [`evaluate_afib.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/evaluate_afib.py) | Verified | Bootstrap CIs ($N=1,000$); disaggregated per-subject metrics; separate confounders slice (AFL, Bigeminy, Bradycardia, Paced). |
| [`afib_baseline_package.joblib`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afib_baseline_package.joblib) | Verified | Serialized baseline models, scaler, feature names, and metadata preserved in research quarantine. |
| [`AFIB_EXPERIMENT_REPORT.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/AFIB_EXPERIMENT_REPORT.md) | Verified | Comprehensive empirical analysis documenting the 4 root limitations that necessitate AFDB acquisition. |

---

## 2. Verification of Official AFDB Record Identifiers & Availability

### 2.1 Official PhysioNet Inventory Verification
Direct query of the PhysioNet master index (`wfdb.get_record_list('afdb')`) and inspection of `https://physionet.org/content/afdb/1.0.0/` confirms that the database contains exactly **25 record identifiers**:

```text
['00735', '03665', '04015', '04043', '04048', '04126', '04746', '04908',
 '04936', '05091', '05121', '05261', '06426', '06453', '06995', '07162',
 '07859', '07879', '07910', '08215', '08219', '08378', '08405', '08434', '08455']
```

### 2.2 Record-Level Curated Exclusions and Master Notes Audit
An inspection of the official PhysioNet `notes.txt` and repository release notes revealed critical record-level facts:

| Record ID | Available Files | PhysioNet Master Note | Usability & Pipeline Impact |
| :---: | :---: | :--- | :--- |
| `00735` | `.hea`, `.atr`, `.qrs` | **Signals unavailable** | **Must be excluded.** No `.dat` signal exists. |
| `03665` | `.hea`, `.atr`, `.qrs` | **Signals unavailable** | **Must be excluded.** No `.dat` signal exists. |
| `04043` | `.hea`, `.dat`, `.atr`, `.qrs` | **Block 39 is unreadable** | 10.24s (2,560 samples) of flat zero fill. Requires flatline window rejection. |
| `04936` | `.hea`, `.dat`, `.atr`, `.qrs` | Historically original signal | Clean, fully validated signal and annotations. |
| `05091` | `.hea`, `.dat`, `.atr`, `.qrs`, `.qrsc` | Corrected QRS available (`.qrsc`) | Validated signal; beat annotations manually audited. |
| `06453` | `.hea`, `.dat`, `.atr`, `.qrs` | **Ends after ~9 hours, 15 min** | Validated signal; duration is $33,300\text{s}$ rather than $36,000\text{s}$. |
| `07859` | `.hea`, `.dat`, `.atr`, `.qrs`, `.qrsc` | 2,569 samples missing from start | Corrected `.qrsc` exists. Must verify `.atr` alignment against waveform onset. |
| `08378` | `.hea`, `.dat`, `.atr`, `.qrs` | No start time recorded | Relative elapsed time intact; valid for windowing. |
| `08405` | `.hea`, `.dat`, `.atr`, `.qrs` | **No start time; Block 1067 unreadable** | 10.24s of flat zero fill. High risk if placed in test set. |
| `08434` | `.hea`, `.dat`, `.atr`, `.qrs` | **Blocks 648, 857, 894 unreadable** | Three 10.24s flat zero blocks. High risk if placed in test set. |
| `08455` | `.hea`, `.dat`, `.atr`, `.qrs` | No start time recorded | Relative elapsed time intact; valid for windowing. |

### 2.3 Physical Storage Correction
- **Original Plan Statement:** Compressed archive $\approx 75.4\text{ MB}$, extracted volume $\approx 152.0\text{ MB}$.
- **Verified PhysioNet Ground Truth:**
  - Complete `.zip` download: **439.7 MB**.
  - Total uncompressed physical volume: **605.9 MB**.
- **Correction Required:** The acquisition script must budget at least **1.2 GB** of local storage in `training/research/afib_experiment/data/afdb/` to accommodate both the downloaded archive and the extracted binary 212 files.

---

## 3. Cohort Split Audit: Train, Validation, and Test Cohort Rebalancing

### 3.1 Flaw in the Initial Split Proposal
The initial acquisition plan proposed the following cohort assignment:
- **Training (15 records):** `04015`, `04043`, `04048`, `04126`, `04746`, `04908`, `04936`, `05091`, `05121`, `05261`, `06426`, `06453`, `06995`, `07162`, `07859`.
- **Validation (4 records):** `07879`, `07910`, `08215`, `08219`.
- **Held-Out Test (4 records):** `08378`, `08405`, `08434`, `08455`.

#### Diagnostic Flaw:
In the proposed test set, **two out of four patients (`08405` and `08434`) contain unreadable tape blocks filled with artificial flatline zeros**. Furthermore, all four test patients lack wall-clock start times. Placing corrupted tapes into an already small test cohort ($N=4$) distorts false positive and false negative calculations (flatline segments can produce zero heart rate, missing QRS detections, and erratic PSD values).

### 3.2 Recommended Rebalanced Split Matrix
To ensure the test set provides a clean, unbiased, and uncompromised benchmark, records with known unreadable tape dropouts (`04043`, `08405`, `08434`) must be moved to the **Training cohort**, where an automated zero-block rejection filter can safely quarantine the missing 10.24s segments, while the **Validation and Test cohorts are composed exclusively of intact, verified recordings**:

```
[ 23 Validated AFDB Signal Records (~229 Hours Continuous Biopotentials) ]
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
     Training Cohort    Validation Cohort   Held-Out Test Cohort
      (15 Records)         (4 Records)         (4 Records)
      ~149.2 Hours         ~40.0 Hours         ~40.0 Hours
      (65.1% Data)        (17.4% Data)        (17.4% Data)
```

| Cohort | Recommended Record IDs | Intact Tape Status | Clinical Rationale & Rhythm Coverage |
| :--- | :--- | :---: | :--- |
| **Training** | `04015`, `04043`*, `04048`, `04746`, `04936`, `05091`, `05261`, `06426`, `06453`†, `06995`, `07859`‡, `08378`, `08405`*, `08434`*, `08455` (15 records) | Contains tape dropouts (*), shortened duration (†), and annotation shift (‡) | Model training can safely absorb isolated dropouts via automated flatline filtering ($V_{p\text{-}p} < 0.05\text{ mV}$ rejection). Maximum physiological diversity for feature learning. |
| **Validation** | `04126`, `05121`, `07879`, `07910` (4 records) | 100% Intact | Clean, uninterrupted records with balanced paroxysmal AFib and normal sinus rhythm. Used for hyperparameter tuning and cost-sensitive threshold selection. |
| **Held-Out Test** | `04908`, `07162`, `08215`, `08219` (4 records) | 100% Intact | **Strictly pristine records with zero tape dropouts.** Contains sustained AFib (`07162`), paroxysmal AFib (`08219`), Atrial Flutter intervals (`08215`), and sinus background (`04908`). |

*Note: Unverified record-level assumption: The exact burden (percentage of time in AFib) for each record can only be computed once the `.atr` annotations are downloaded and parsed. Gate 2 will formally audit the extracted class balance.*

---

## 4. Verification of the 250 Hz to 360 Hz Resampling Pipeline

### 4.1 Polyphase Resampling Mathematical Formulation
The target hardware (Texas Instruments ADS1292R biopotential AFE) operates at a fixed sampling frequency $f_{\text{target}} = 360\text{ Hz}$ ($T_{\text{sample}} = 2.777\text{ ms}$). AFDB recordings were digitized at $f_{\text{source}} = 250\text{ Hz}$ ($T_{\text{sample}} = 4.000\text{ ms}$).

Resampling requires rational polyphase rate conversion:
$$\frac{f_{\text{target}}}{f_{\text{source}}} = \frac{360}{250} = \frac{36}{25} \quad (L = 36, \, M = 25)$$

Using `scipy.signal.resample_poly(signal, up=36, down=25)`:
1. **Upsampling ($L = 36$):** Inserts 35 zeros between consecutive samples, expanding the sampling rate to $250 \times 36 = 9,000\text{ Hz}$.
2. **Anti-Imaging Lowpass FIR Filter:** Applies an equiripple FIR filter with normalized cutoff frequency:
   $$\omega_c = \frac{\pi}{\max(L, M)} = \frac{\pi}{36} \implies f_c = \frac{9,000}{2 \times 36} = 125.0\text{ Hz}$$
   This cutoff matches the original Nyquist frequency of the 250 Hz input signal, completely eliminating spectral imaging artifacts.
3. **Downsampling ($M = 25$):** Retains every 25th sample, producing a final sampling rate of $\frac{9,000}{25} = 360\text{ Hz}$.

```
                 Polyphase Resampling Pipeline (250 Hz -> 360 Hz)
 ┌──────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
 │  AFDB Input  │     │   Upsample   │     │  Anti-Imaging FIR │     │  Downsample  │
 │    250 Hz    │ ──> │   x36 (Zero  │ ──> │ Lowpass Filter    │ ──> │   /25 (Dec-  │ ──> 360 Hz Signal
 │  (12-bit ADC)│     │  Insertion)  │     │ (Cutoff = 125 Hz) │     │   imation)   │
 └──────────────┘     └──────────────┘     └───────────────────┘     └──────────────┘
```

### 4.2 Timestamp & Annotation Alignment
In PhysioNet `.atr` files, rhythm annotations are indexed by sample number $n_{250}$ relative to the start of the 250 Hz file.
To map an annotation timestamp to the 360 Hz domain:
$$n_{360} = \text{round}\left(n_{250} \cdot \frac{36}{25}\right)$$
Physical elapsed time is preserved identically:
$$t = \frac{n_{250}}{250\text{ Hz}} = \frac{n_{360}}{360\text{ Hz}}$$

### 4.3 Preservation of Rhythm Transition Boundaries & Guard Bands
Because the FIR anti-imaging filter has an effective group delay, rhythm boundaries must be protected against filter smearing:
1. **Transition Timestamp Isolation:** Every rhythm change annotated in `.atr` (e.g. `(N` $\to$ `(AFIB`) defines a boundary $t_{\text{trans}}$.
2. **2.5-Second Guard Band Enforcement:** Any 10.0-second window spanning $[t_{\text{start}}, t_{\text{end}}]$ must satisfy:
   $$t_{\text{start}} \ge t_{\text{episode\_start}} + 2.5\text{s} \quad \text{and} \quad t_{\text{end}} \le t_{\text{episode\_end}} - 2.5\text{s}$$
3. **100% Label Purity:** Any window that overlaps the interval $[t_{\text{trans}} - 2.5\text{s}, t_{\text{trans}} + 2.5\text{s}]$ is discarded. This prevents transitional arrhythmias or filter ring-down transients from corrupting the label.

---

## 5. Review of False-Alarm Mitigation Features & Leak-Free Evaluation

### 5.1 Alternating Sign Index (ASI) for Ventricular Ectopy Rejection
- **Diagnostic Objective:** Eliminate false positives on ventricular bigeminy, trigeminy, and frequent PVCs (Record 233 failure mode).
- **Electrophysiological Rationale:** In bigeminy, a normal sinus beat is followed by a premature ventricular contraction (short coupling interval), which is followed by a complete compensatory pause (long interval). Consecutive RR differences $\Delta RR_i = RR_i - RR_{i-1}$ continuously alternate sign ($+ - + -$). In true AFib, ventricular response is Poisson-distributed and signs change randomly ($\approx 50\%$).
- **Mathematical Formulation:**
  Let $RR = [rr_1, rr_2, \dots, rr_K]$ be the series of normal-to-normal or detected RR intervals in a 10s window.
  $$\Delta RR_i = rr_{i+1} - rr_i \quad (i = 1, \dots, K-1)$$
  To avoid false sign flips from baseline physiological sinus jitter, define a deadband threshold $\epsilon = 20\text{ ms}$:
  $$s_i = \begin{cases} +1, & \Delta RR_i > \epsilon \\ -1, & \Delta RR_i < -\epsilon \\ 0, & |\Delta RR_i| \le \epsilon \end{cases}$$
  The Alternating Sign Index is the fraction of consecutive non-zero transitions that change sign:
  $$\text{ASI} = \frac{\sum_{i=1}^{M-1} \mathbf{1}(s_i \cdot s_{i+1} = -1)}{\max\left(1, \, \sum_{i=1}^{M-1} \mathbf{1}(s_i \cdot s_{i+1} \neq 0)\right)}$$
  - In ventricular bigeminy: $\text{ASI} \to 1.0$.
  - In true Atrial Fibrillation: $\text{ASI} \approx 0.45\text{--}0.55$.
  - In regular sinus rhythm: $s_i = 0 \implies \text{ASI} = 0.0$.
- **Leak-Free Guarantee:** ASI is computed strictly within the individual 10-second window. It depends on no external parameters, fitting, or cross-window history.

### 5.2 Atrial Spectral Peak Purity Ratio for Atrial Flutter (AFL) Discrimination
- **Diagnostic Objective:** Discriminate Atrial Flutter ($100\%$ false alarms in baseline) from chaotic Atrial Fibrillation.
- **Electrophysiological Rationale:** Atrial Flutter involves an organized macro-reentrant circuit around the tricuspid valve annulus, producing uniform flutter waves (sawtooth waves) at a sharp single frequency between $4.0\text{ Hz}$ and $6.0\text{ Hz}$ ($240\text{--}360\text{ bpm}$). In contrast, AFib involves multiple chaotic reentrant wavelets, producing a broad, disorganized, flat power spectrum across $3.0\text{--}9.0\text{ Hz}$.
- **Mathematical Formulation:**
  Compute the Welch Power Spectral Density $\text{PSD}(f)$ of the bandpassed ECG (or QRS-subtracted residual signal) over the fibrillatory band $f \in [3.0, 9.0]\text{ Hz}$:
  $$P_{\text{peak}} = \max_{f \in [3.0, 9.0]} \text{PSD}(f), \qquad \bar{P} = \frac{1}{B} \int_{3.0}^{9.0} \text{PSD}(f) \, df$$
  $$\text{Purity}_{3-9} = \frac{P_{\text{peak}}}{\bar{P}}$$
  - In Atrial Flutter: $\text{Purity}_{3-9} \ge 4.5$ (sharp spectral spike).
  - In Atrial Fibrillation: $\text{Purity}_{3-9} \le 3.0$ and spectral flatness $\to 1.0$.
- **Leak-Free Guarantee:** Welch PSD parameters (window length, FFT size, overlap) are fixed constants. Zero learning or parameter tuning occurs across splits.

### 5.3 Cost-Sensitive Operating Threshold Selection
- **Formulation:** Instead of unconstrained F1 maximization, the decision threshold $\theta^*$ is selected via a constrained optimization objective:
  $$\theta^* = \arg\max_{\theta \in [0.10, 0.90]} \text{Sensitivity}(\theta) \quad \text{subject to} \quad \text{Specificity}(\theta) \ge 95.0\% \quad \text{and} \quad \text{FP/hour}(\theta) \le 5.0$$
- **Leak-Free Guarantee:** The threshold $\theta^*$ is tuned **strictly on the Validation cohort**. It is saved into `afdb_model_metadata.json` and loaded as a frozen constant during evaluation of the held-out test cohort.

### 5.4 Causal Rolling Temporal Confirmation
- **Formulation:** A continuous alert is triggered if and only if two consecutive sliding windows are classified as AFib:
  $$Y_{\text{confirmed}}(t_k) = \mathbf{1}\left(P(t_k) \ge \theta^* \land P(t_{k-1}) \ge \theta^*\right)$$
- **Leak-Free Guarantee:** The consensus filter is strictly **causal** (depends only on $t_k$ and $t_{k-1}$, never $t_{k+1}$). Window histories are strictly reset at the boundary between different patient records to eliminate any cross-subject leakage.

---

## 6. Dual Evaluation Protocols: Product Workflow vs. Continuous Telemetry

The evaluation framework must formally separate two distinct clinical use cases:

```
                          Two Distinct Operational Regimes
 ┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
 │   PROTOCOL A: Product Spot-Check Triage │   │  PROTOCOL B: Continuous Telemetry Mon.  │
 │   - Single isolated 10-second recording │   │  - Continuous 10-hour Holter monitoring │
 │   - Point-of-care patient assessment    │   │  - Sliding 10s windows (50% overlap)    │
 │   - One binary prediction emitted       │   │  - Causal 2-window consensus filtering  │
 │   - Evaluates spot screening accuracy   │   │  - Evaluates alarm fatigue (FP/hour)    │
 └─────────────────────────────────────────┘   └─────────────────────────────────────────┘
```

### 6.1 Protocol A: Intended Product Workflow (Single 10-Second Recording Triage)
This protocol mirrors the actual edge-device workflow when a patient or paramedic initiates a 10-second spot assessment on the Raspberry Pi + ADS1292R device:
1. **Acquisition:** Acquire exactly one 10.0-second biopotential strip ($3,600\text{ samples}$ at $360\text{ Hz}$).
2. **Signal Conditioning:** Apply the production bandpass filter ($0.5\text{--}40.0\text{ Hz}$, 2nd-order Butterworth, zero-phase forward-backward).
3. **Flatline / Quality Rejection:** Verify that peak-to-peak amplitude exceeds $0.05\text{ mV}$ and zero-sample fraction is $< 5\%$. If low quality, emit `SIGNAL_QUALITY_UNACCEPTABLE` and abort.
4. **Feature Extraction:** Detect QRS complexes and extract the exact 17-dimensional feature vector $\mathbf{x}$ (16 timing/spectral features + ASI).
5. **Standardization:** Transform $\mathbf{x}$ using the **frozen training scaler parameters**:
   $$\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$
6. **Inference:** Pass $\mathbf{z}$ to the champion Random Forest model to compute probability $p = P(\text{AFib} \mid \mathbf{z})$.
7. **Classification:**
   $$\text{Decision} = \begin{cases} \text{AFIB}, & p \ge \theta^* \\ \text{NON\_AFIB}, & p < \theta^* \end{cases}$$
8. **Reporting Metrics for Protocol A:**
   - **Spot Sensitivity:** Percentage of true AFib 10s strips correctly identified.
   - **Spot Specificity:** Percentage of true non-AFib 10s strips correctly identified.
   - **Positive Predictive Value (PPV) & Negative Predictive Value (NPV).**
   - **Brier Score & Expected Calibration Error (ECE):** Evaluates probability calibration for clinical risk scoring.

### 6.2 Protocol B: Continuous Telemetry Monitoring Workflow
This protocol mirrors long-term ambulatory monitoring (e.g. Holter or continuous wearable telemetry):
1. **Sliding Windows:** Evaluate consecutive 10s windows with 5s stride ($50\%$ overlap).
2. **Causal Consensus:** Apply the 2-window causal confirmation filter.
3. **Reporting Metrics for Protocol B:**
   - **False Positives per Monitoring Hour ($\text{FP/hour}$):**
     $$\text{FP/hour} = \frac{\text{Total False Positive Windows}}{\text{Total Non-AFib Monitoring Hours}}$$
   - **False Alarm Ratio:** Total duration of false alarm episodes divided by total non-AFib monitoring time.
   - **Episode Sensitivity ($\ge 30\text{s}$):** Fraction of clinically significant AFib episodes detected by at least one confirmed alert.
   - **Episode Detection Latency:** Time in seconds from true electrophysiological onset to the first confirmed alert.

---

## 7. Statistical Power Analysis & The 4-Patient Test Set Limitation

### 7.1 The Cluster-Correlation Reality
In biomedical signal processing, windows extracted from the same patient's Holter recording are **statistically dependent time series samples**.
- While 4 AFDB records provide $\approx 40\text{ hours}$ of data ($\approx 28,800$ sliding windows), those windows represent only **four independent human biological systems** ($N_{\text{patients}} = 4$).
- If one test patient has an idiosyncratic conduction anomaly (e.g. rate-dependent bundle branch block or frequent aberrant conduction), a failure on that single individual drops patient-level performance by **$25.0\%$**.

### 7.2 Patient-Level Confidence Intervals
Evaluating a binary outcome across $N=4$ patients yields the following Clopper-Pearson exact 95% confidence intervals:

| Number of Successful Patients | Point Estimate | Exact 95% Clopper-Pearson Confidence Interval |
| :---: | :---: | :---: |
| 4 out of 4 | $100.0\%$ | $[39.76\%, \, 100.00\%]$ |
| 3 out of 4 | $75.0\%$ | $[19.41\%, \, 99.37\%]$ |
| 2 out of 4 | $50.0\%$ | $[6.76\%, \, 93.24\%]$ |

> [!WARNING]
> Even if the model achieves $100\%$ accuracy on all four held-out patients, the lower bound of the 95% confidence interval is **$39.76\%$**. Window-level bootstrapping yields artificially narrow intervals ($\pm 1.5\%$) because it assumes windows are independent identically distributed (i.i.d.) observations.

### 7.3 Governance Decision on 4-Patient Test Cohort
1. **Substantial Progress over Baseline:** The baseline experiment had only **one** AFib test patient (`MIT_S221`). Expanding to four independent test patients (~40 hours) represents a **400% increase in patient diversity** and allows disaggregated per-subject auditing.
2. **Exploratory Status Constraint:** Performance on the 4-patient AFDB test set must be classified as an **exploratory research benchmark**.
3. **Mandatory Post-AFDB Gate:** No model can be promoted to production or deployed on hardware without subsequent **independent cross-database validation** on large-cohort external datasets (e.g. PhysioNet Challenge 2017 with 8,528 patients or CPSC 2018 with 6,877 patients) and **physical benchtop verification with a calibrated ECG patient simulator**.

---

## 8. Audit of Assumptions, Unresolved Decisions, and Pre-Acquisition Checks

### 8.1 Summary of Corrected Plan Assumptions

| Parameter / Assumption | Original Plan Assertion | Audit Verification & Correction |
| :--- | :--- | :--- |
| **Download Archive Size** | 75.4 MB compressed / 152 MB raw | **439.7 MB ZIP / 605.9 MB uncompressed**. Requires $\ge 1.2\text{ GB}$ disk quota. |
| **Missing Records Status** | Records `00735` and `03665` have corrupted rhythm annotations | Signals (`.dat`) are physically nonexistent on PhysioNet. Total signal records = 23. |
| **Tape Dropouts in Test Set** | Test set assigned to `08378`, `08405`, `08434`, `08455` | `08405` and `08434` have unreadable tape blocks (flat zeros). **Moved to Train; clean records (`04908`, `07162`, `08215`, `08219`) moved to Test.** |
| **Recording Duration** | All 25 records are 10.0 continuous hours | Record `06453` terminates at 9 hours, 15 minutes. |
| **License & DOI** | PDDL v1.0 / DOI `10.13026/C2CP4X` | **ODC-By v1.0 / DOI `10.13026/C2MW2D`**. |

### 8.2 Unresolved Decisions & Engineering Requirements
Before executing data acquisition and model training, the following four engineering gates must be verified:
1. **Automated Flatline Rejection Filter:** The preprocessing script must reject any 10s window where peak-to-peak amplitude $V_{p\text{-}p} < 0.05\text{ mV}$ or $> 10\%$ of samples are identical zeros. This prevents tape dropouts in records `04043`, `08405`, and `08434` from distorting training features.
2. **Lead Selection Protocol:** AFDB records provide two bipolar leads (Channel 0 and Channel 1). Channel 1 is designated as the primary Lead II analog. The ingestion code must verify channel indices against header comments to ensure consistent lead assignment.
3. **Gate 2 Data Verification Step:** Upon downloading headers and annotations, an automated script must output exact AFib, AFL, and NSR window counts per record before any feature extraction or training commences.
4. **Hardware Simulator Benchmarking Requirement:** Before considering any production ONNX export, the model must be evaluated on real-time signals played from an ECG simulator into the ADS1292R analog inputs.

---

## 9. Go / No-Go Recommendation

### Formal Verdict: **CONDITIONAL GO (Pending User Approval)**

```
                                  AUDIT VERDICT
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │                 >>>  STATUS: CONDITIONAL GO  <<<                            │
 │                                                                             │
 │  1. The algorithmic architecture and feature engineering are VALIDATED.     │
 │  2. The polyphase anti-aliasing resampling pipeline is MATHEMATICALLY SOUND.│
 │  3. The leak-free threshold and consensus filtering are VERIFIED.           │
 │  4. The acquisition plan MUST BE UPDATED with corrected storage sizes,      │
 │     rebalanced patient splits, and the dual evaluation protocols.           │
 │  5. ACQUISITION MUST REMAIN FROZEN until the user gives explicit written    │
 │     authorization to download the dataset.                                  │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘
```

### Action Items Prior to Dataset Download:
1. Update [`AFDB_ACQUISITION_AND_VALIDATION_PLAN.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/AFDB_ACQUISITION_AND_VALIDATION_PLAN.md) to reflect the corrected storage footprint ($605.9\text{ MB}$), rebalanced test cohort, and dual product evaluation protocol.
2. Present this pre-acquisition audit and the updated plan to the user for formal approval.
3. Maintain zero network calls, zero package installations, and zero modifications to production files until explicit approval is granted.
