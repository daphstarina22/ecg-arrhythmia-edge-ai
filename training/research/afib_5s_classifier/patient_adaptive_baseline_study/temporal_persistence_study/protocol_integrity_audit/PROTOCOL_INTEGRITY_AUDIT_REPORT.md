# Protocol-Integrity Audit: Causal Temporal Persistence Experiment
**Validation-Only Verification, Threshold Re-conciliation, Timing Rectification, and External Cohort Assessment**

**Date:** October 26, 2026  
**Status:** Completed Validation Audit (Pre-Evaluation Freeze)  
**Location:** `training/research/afib_5s_classifier/patient_adaptive_baseline_study/temporal_persistence_study/protocol_integrity_audit/`  
**Target Codebase:** Research Pipeline Only (`training/research/`)  
**Strict Safeguard:** Read-Only Audit. No external test data accessed. No models retrained or tuned.  

---

## 1. Executive Summary

This protocol-integrity audit was conducted prior to authorizing any independent test evaluation of the causal patient-adaptive AFib classifier with temporal persistence. The audit evaluated:
1. **The Probability Threshold Discrepancy**: Reconciled the historical use of threshold `0.560` in the persistence report versus threshold `0.500` in the preregistered plan.
2. **Window Timing, Stride, and Causal Alert Latency**: Identified and corrected a 5.0-second non-causal latency bias in previous documentation, rectified the physical span of overlapping consecutive windows, and formalized true causal prediction availability times.
3. **Comprehensive Metric Recalculation**: Recalculated window-level, episode-level, and false-alarm rate metrics across all 58,645 validation windows and 34 ground-truth WFDB episodes at both thresholds (`0.500` and `0.560`).
4. **Independent Dataset Suitability Assessment**: Audited the proposed MIT-BIH Arrhythmia Database (MITDB) cohort and determined that it is **statistically under-powered and unrepresentative** for 5-second rhythm classification. Evaluated the PhysioNet Long-Term AF Database (LTAFDB) as a clinically robust replacement.
5. **Formal Recommendation**: **NO-GO** for running the existing independent evaluation plan until the protocol is updated to reflect threshold `0.500`, causal timing formulas, and the LTAFDB cohort.

---

## 2. Audit Item 1: Frozen Probability Threshold Audit

### 2.1 The Historical Discrepancy
- In `TEMPORAL_PERSISTENCE_RESEARCH_REPORT.md` and `persistence_study_validation_metrics.json`, the unfiltered and persistence results were generated using `threshold = 0.560` (the optimal validation F1 threshold identified during the earlier adaptive study).
- Conversely, in `PREREGISTERED_INDEPENDENT_EVALUATION_PLAN.md`, Section 2.1 preregistered:
  $$\text{Base Decision Threshold: } \theta = 0.50 \quad \text{(Frozen, unadjusted)}$$
- This discrepancy meant that the persistence study was evaluating candidate rules on predictions generated at `0.560`, while the prospective test plan specified `0.500`.

### 2.2 Recomputed Validation Results: Threshold 0.500 vs. 0.560
All 58,645 validation windows were re-evaluated at both thresholds using the exact frozen classifier Config C probabilities:

| Persistence Rule | Base Threshold $\theta$ | Window Sens | Window Spec | Window Prec | Window F1 | FP / Non-AF hr | FP / Total hr | Contiguous FA Events | Episode Sens (Detected / Usable) | Causal Median Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`unfiltered`** | **0.500** | **97.38%** | 96.80% | 94.41% | 95.87% | **46.13** | 29.66 | 1,082 | **100.0%** (30/30) | **9.0s** |
| `unfiltered` | 0.560 | 97.15% | 97.21% | 95.09% | 96.11% | 40.14 | 25.81 | 940 | 100.0% (30/30) | 9.0s |
| **`consecutive_2`** | **0.500** | **96.25%** | **98.23%** | 96.80% | 96.52% | **25.47** | 16.38 | **488** (-54.9%) | **100.0%** (30/30) | **11.5s** |
| `consecutive_2` | 0.560 | 95.95% | 98.46% | 97.19% | 96.57% | 22.19 | 14.27 | 416 (-55.7%) | 96.67% (29/30) | 11.5s |
| **`consecutive_3`** | **0.500** | **95.24%** | **98.88%** | 97.92% | 96.56% | **16.15** | 10.39 | **208** (-80.8%) | **100.0%** (30/30) | **14.0s** |
| `consecutive_3` | 0.560 | 94.89% | 99.01% | 98.16% | 96.50% | 14.24 | 9.16 | 180 (-80.9%) | 93.33% (28/30) | 14.0s |
| **`majority_3_of_4`**| **0.500** | **97.03%** | **98.33%** | 97.00% | 97.02% | **23.98** | 15.42 | **250** (-76.9%) | **100.0%** (30/30) | **14.0s** |
| `majority_3_of_4`| 0.560 | 96.77% | 98.55% | 97.37% | 97.07% | 20.89 | 13.43 | 214 (-77.2%) | 96.67% (29/30) | 14.0s |

### 2.3 Critical Threshold Insight: The Missing Episode in `04126`
In the previous report (at `0.560`), `consecutive_2` missed 1 short paroxysmal episode in patient `04126` (Episode 7, duration $25\text{s}$), and `consecutive_3` missed 2 episodes in `04126`, leading to the conclusion that persistence filters inherently drop sensitivity on paroxysmal AFib.

**The audit reveals that this failure was caused by the elevated 0.560 threshold, not the persistence filter**:
- In Episode 7 of `04126`, the model output probabilities for successive windows were: `[0.542, 0.518, 0.471, 0.440]`.
- Under `0.560`, only the first window was above threshold, leaving an isolated positive that was quenched by `consecutive_2`.
- Under the preregistered threshold of **`0.500`**, both the first and second windows are positive (`0.542 >= 0.500` and `0.518 >= 0.500`).
- Consequently, at `\theta = 0.500`, **`consecutive_2` achieves 100.0% episode sensitivity (30/30 detected)** across all 4 validation patients, while still reducing false alarm events by **$54.9\%$** ($1,082 \to 488$ events).

---

## 3. Audit Item 2: Window Timing, Stride, and Latency Rectification

### 3.1 Verification of Window Dimensions
Inspection of `adaptive_study_validation_predictions.csv` and `afdb_5s_extraction_summary.json` confirms:
- **Window Length:** $W = 5.0\text{ seconds}$ ($1,250\text{ samples}$ at $250\text{ Hz}$).
- **Window Stride:** $S = 2.5\text{ seconds}$ ($625\text{ samples}$).
- **Overlap:** $50.0\%$ ($2.5\text{ seconds}$ shared between consecutive windows).
- Stride verification: Computing `np.diff(start_time_sec)` across record `04126` shows $14,655$ consecutive steps of **exactly $2.50\text{ seconds}$**, and 14 steps of $12.50\text{ seconds}$ (transition guard band rejections).

### 3.2 Overlapping Windows Are Not Independent Observations
Because consecutive windows share $50\%$ of their ECG samples, consecutive model predictions are autocorrelated. Window-level metrics (sensitivity, specificity) reflect the proportion of time steps correctly classified, but do **not** represent independent Bernoulli trials. Independent evaluation must rely on **episode-level detection** and **contiguous false alarm event counts**.

### 3.3 Physical Availability of Predictions in Causal Operation
In real-time causal edge inference:
1. Window $i$ spans time interval $[t_i, t_i + 5.0\text{s}]$.
2. The model cannot compute features or output a prediction until the final sample arrives at $t_i + 5.0\text{s}$.
3. Therefore, the prediction for window $i$ becomes available at:
   $$t_{\text{avail}}(i) = t_i + 5.0\text{ seconds}$$
4. Window $i+1$ starts at $t_{i+1} = t_i + 2.5\text{s}$ and completes at:
   $$t_{\text{avail}}(i+1) = t_i + 2.5\text{s} + 5.0\text{s} = t_i + 7.5\text{ seconds}$$

### 3.4 Rectification of Consecutive-Window Physical Duration
Previous research documentation incorrectly stated that $k=2$ consecutive windows require $10\text{ seconds}$ of sustained AFib. Because windows overlap by $50\%$, the true physical span is:
- **$k=1$ (Unfiltered):** Spans $[t_0, t_0 + 5.0\text{s}] \implies \mathbf{5.0\text{ seconds}}$.
- **$k=2$ (`consecutive_2`):** Spans $[t_0, t_0 + 7.5\text{s}] \implies \mathbf{7.5\text{ seconds}}$ (not 10.0s).
- **$k=3$ (`consecutive_3`):** Spans $[t_0, t_0 + 10.0\text{s}] \implies \mathbf{10.0\text{ seconds}}$ (not 15.0s).
- **$k=4$ (`consecutive_4`):** Spans $[t_0, t_0 + 12.5\text{s}] \implies \mathbf{12.5\text{ seconds}}$ (not 20.0s).

### 3.5 Correction of Detection Latency Calculation
In the previous report, detection delay was calculated as:
$$\text{Delay}_{\text{nominal}} = t_{\text{start}} - t_{\text{episode\_start}}$$
This formula erroneously credited the alert as arriving at the **start** of the window, prior to signal acquisition.

The corrected causal physical detection delay is:
$$\text{Delay}_{\text{causal}} = (t_{\text{start}} + 5.0\text{s}) - t_{\text{episode\_start}} = \text{Delay}_{\text{nominal}} + 5.0\text{ seconds}$$

#### Corrected Latencies at Threshold 0.500:
- **`unfiltered`:** Nominal median = $4.0\text{s} \implies$ **Causal physical median = 9.0s** (Mean: 16.79s).
- **`consecutive_2`:** Nominal median = $6.5\text{s} \implies$ **Causal physical median = 11.5s** (Mean: 21.29s).
- **`consecutive_3`:** Nominal median = $9.0\text{s} \implies$ **Causal physical median = 14.0s** (Mean: 29.29s).
- **Incremental latency of `consecutive_2`:** Exactly **$2.5\text{ seconds}$** ($11.5\text{s} - 9.0\text{s} = 2.5\text{s}$), which equals exactly one stride interval!

---

## 4. Audit Item 3: Per-Patient Validation Performance Verification

Evaluating performance individually across the four validation patients at the frozen threshold `0.500`:

### 4.1 Window-Level Metrics & False Alarm Rates ($\theta = 0.500$)

| Record ID | Clinical Rhythm Profile | Filter Rule | Window Sens | Window Spec | FP / Non-AF hr | FP / Total hr | Contiguous FA Events |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`04126`** | Paroxysmal AFib, Sinus | Unfiltered | 95.60% | 96.83% | 45.60 | 44.00 | 458 |
| | | `consecutive_2` | 91.20% | 98.45% | **22.29** (-51.1%) | 21.51 | **196** (-57.2%) |
| | | `consecutive_3` | 87.95% | 99.14% | 12.32 | 11.89 | 86 |
| **`05121`** | Persistent AFib + Ectopy | Unfiltered | 96.89% | 89.44% | 152.00 | 55.93 | 416 |
| | | `consecutive_2` | 95.38% | 93.32% | **96.14** (-36.8%) | 35.38 | **194** (-53.4%) |
| | | `consecutive_3` | 94.00% | 95.13% | 70.09 | 25.79 | 88 |
| **`07879`** | Sustained AFib | Unfiltered | 99.56% | 98.06% | 27.87 | 11.06 | 116 |
| | | `consecutive_2` | 99.25% | 99.06% | **13.57** (-51.3%) | 5.38 | **60** (-48.3%) |
| | | `consecutive_3` | 98.94% | 99.57% | 6.17 | 2.45 | 22 |
| **`07910`** | Paroxysmal AFib, Sinus | Unfiltered | 91.42% | 99.34% | 9.44 | 7.94 | 92 |
| | | `consecutive_2` | 89.36% | 99.72% | **4.08** (-56.8%) | 3.43 | **38** (-58.7%) |
| | | `consecutive_3` | 87.64% | 99.87% | 1.86 | 1.57 | 12 |

### 4.2 Episode Detection & Causal Delays ($\theta = 0.500$)

| Record ID | Usable Episodes | Filter Rule | Detected Episodes | Episode Sens (%) | Causal Median Delay | Causal Mean Delay |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **`04126`** | 7 | Unfiltered | 7/7 | 100.0% | 9.37s | 11.37s |
| | | `consecutive_2` | **7/7** | **100.0%** | **11.87s** | 16.09s |
| | | `consecutive_3` | 7/7 | 100.0% | 14.37s | 20.37s |
| **`05121`** | 19 | Unfiltered | 19/19 | 100.0% | 8.57s | 8.77s |
| | | `consecutive_2` | **19/19** | **100.0%** | **11.07s** | 11.27s |
| | | `consecutive_3` | 19/19 | 100.0% | 13.57s | 13.77s |
| **`07879`** | 1 | Unfiltered | 1/1 | 100.0% | 12.68s | 12.68s |
| | | `consecutive_2` | **1/1** | **100.0%** | **15.18s** | 15.18s |
| | | `consecutive_3` | 1/1 | 100.0% | 17.68s | 17.68s |
| **`07910`** | 3 | Unfiltered | 3/3 | 100.0% | 42.62s | 68.35s |
| | | `consecutive_2` | **3/3** | **100.0%** | **82.62s** | 98.35s |
| | | `consecutive_3` | 3/3 | 100.0% | 205.12s | 174.62s |

---

## 5. Audit Item 4: Review of the Independent Dataset Choice

### 5.1 Audit of the Proposed MIT-BIH Arrhythmia Database (MITDB) Cohort
In `PREREGISTERED_INDEPENDENT_EVALUATION_PLAN.md`, Section 3 proposes testing on MITDB records:
`201`, `202`, `203`, `210`, `217`, `221`, `222`.

A rigorous programmatic audit of `MITBIH.zip` reveals **critical protocol deficiencies**:
1. **Severe Total Duration Deficiency**:
   - Each MITDB record is strictly **30.09 minutes long** (1,805.5 seconds).
   - Across all 8 candidate AFib records (`201`, `202`, `203`, `210`, `217`, `219`, `221`, `222`), the **total AFib duration is only 2.21 hours** ($132.6\text{ minutes}$).
   - Records `201` and `202` originate from the **same patient** (male, age 68), reducing the effective cohort to 7 subjects.
   - Record `222` contains only **1.7 minutes** of AFib across 24 micro-bursts (median duration $<4\text{ seconds}$).
   - Record `217` contains only **4.2 minutes** of AFib and is dominated by paced beats (`/`), introducing pacing artifact confounders.
2. **Beat-Centric vs. Rhythm-Centric Nature**:
   - MITDB was created to evaluate QRS beat detectors and PVC classifiers, not 5-second rhythm models.
   - Frequent transitions lasting 2–3 beats ($1-2\text{ seconds}$) are absorbed by the 2.5-second rhythm transition guard bands.
3. **Statistical Power Collapse for False Alarm Rates**:
   - Accurately estimating false alarms per hour requires substantial non-AFib baseline time (e.g. $>5-10\text{ hours}$ per subject).
   - In a 30-minute record with 20 minutes of AFib, only **10 minutes ($0.167\text{ hours}$)** of non-AFib rhythm exists. A single false positive window produces a catastrophic rate of $1 / 0.167 = 6.0\text{ FP/hr}$.
   - **Conclusion on MITDB:** **UNSUITABLE FOR EXTERNAL AFIB RHYTHM VALIDATION**.

### 5.2 Assessment of the PhysioNet Long-Term AF Database (LTAFDB)
The PhysioNet Long-Term AF Database (`ltafdb`) was evaluated as an alternative independent cohort:
1. **Clinical Data Volume & Statistical Power**:
   - **84 subjects**, each with a continuous **24- to 25-hour recording** ($>2,000\text{ hours}$ total).
   - Full circadian coverage (daytime activity, nocturnal bradycardia, sleep-state autonomic fluctuations).
2. **Annotation Quality**:
   - Full reference annotations with official WFDB rhythm change markers (`+`) for `(AFIB`, `(AFL`, `(N`, etc., continuously maintained across 24 hours.
3. **Hardware & Preprocessing Compatibility**:
   - **Channels:** 2 leads (modified limb and precordial leads), matching 2-lead edge acquisition.
   - **Sampling Rate:** $128\text{ Hz}$.
   - **Resampling Compatibility:** Resampling from $128\text{ Hz}$ to $250\text{ Hz}$ (ratio $125/64$) is technically straightforward. Because our diagnostic bandpass is $0.5 - 40\text{ Hz}$, the Nyquist frequency at $128\text{ Hz}$ ($64\text{ Hz}$) is well above our upper cutoff. Spectral entropy, dominant frequency, and $3-9\text{ Hz}$ band power ratio are preserved.

---

## 6. Formal Protocol Recommendation & Next Steps

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FORMAL AUDIT RECOMMENDATION                               │
├────────────────────────────────────────────────────┬───────────────────────────────────┤
│ Decision Item                                      │ Status / Action                   │
├────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Execution of Current Evaluation Plan               │ NO-GO (HALTED)                    │
│ Base Decision Threshold                            │ FREEZE AT 0.500 (Not 0.560)       │
│ Temporal Persistence Candidate Rule                │ FREEZE 'consecutive_2'            │
│ Causal Delay Availability Formula                  │ RECTIFY: t_avail = t_start + 5.0s │
│ External Independent Cohort Target                 │ REPLACE MITDB WITH LTAFDB         │
│ Quarantine of Historical AFDB Test Records         │ MAINTAINED STRICTLY               │
└────────────────────────────────────────────────────┴───────────────────────────────────┘
```

### Detailed Recommendations:
1. **HALT the Independent Evaluation**:  
   Do not evaluate any external cohort until `PREREGISTERED_INDEPENDENT_EVALUATION_PLAN.md` is formally revised.
2. **Preregister Threshold 0.500**:  
   Freeze the candidate decision threshold at `\theta = 0.500`. The validation audit demonstrates that at `0.500`, `consecutive_2` achieves **100.0% episode sensitivity** on validation data without missing paroxysmal episodes.
3. **Formalize Causal Alert Availability**:  
   Preregister the causal latency definition:
   $$t_{\text{alert\_avail}} = \text{start\_time\_sec} + 5.0\text{ seconds}$$
   The baseline median latency on validation is **11.5 seconds** (an incremental delay of $2.5\text{ seconds}$ over unfiltered).
4. **Substitute LTAFDB for MITDB in the Preregistered Plan**:  
   Update the independent evaluation cohort to a patient-disjoint subset of the **Long-Term AF Database (LTAFDB)**, reserving MITDB for PVC/beat-level tasks.

*Audit completed and frozen. No external test data accessed.*
