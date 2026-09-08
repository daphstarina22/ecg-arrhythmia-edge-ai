# Phase 6C Final Report: CUDB Quarantined Record Re-Adjudication & Rhythm Audit

**Date**: September 6, 2026  
**Status**: Completed  
**Artifact Directory**: `training/model3_phase6/phase6c/`  
**Results Directory**: `training/model3_phase6/phase6c/results/`  
**Audit Table**: [`training/model3_phase6/phase6c/results/cudb_episode_audit.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/model3_phase6/phase6c/results/cudb_episode_audit.csv)  
**Visualizations**: [`training/model3_phase6/phase6c/results/plots/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/model3_phase6/phase6c/results/plots/) (45 4-panel diagnostic figures)  

---

## 1. Executive Summary & Audit Mandate

In Phase 5 and Phase 6B, the Model 3 research pipeline demonstrated that cross-database generalization collapse between Ventricular Tachycardia (VT) and Ventricular Fibrillation (VF) is fundamentally driven by an **acute lack of independent patient diversity in the shockable arrhythmia class** (only 17 independent VF/VFL physical patient records across Challenge 2015, VFDB, CUDB, and MIT-BIH).

The Creighton University Ventricular Tachyarrhythmia Database (`CUDB`) contains 35 records. In earlier phases (Phase 2E.2 / Phase 5), only 3 records (`cu01`, `cu02`, `cu03`) were incorporated because their WFDB reference annotations contained explicit subtype strings (`(VF` or `(VT`). The remaining 32 records (`cu04` through `cu35`) were quarantined.

**Phase 6C Objective**:
Execute an evidence-based, read-only audit of all 32 quarantined CUDB records (`cu04`–`cu35`) to determine whether any sustained ventricular episodes can be incorporated into Model 3 without label ambiguity.

### Strict Governance Rules Enforced:
1. **Zero Model Retraining or Dataset Modification**: No sliding windows were extracted into the ML training table; no models were trained or exported.
2. **Anti-Assumption Mandate**: Do not assume CUDB records are VF simply because they originate from CUDB.
3. **Preservation of Ambiguity**: Where reference annotations lack explicit subtype certification, label ambiguity is preserved. No clinical labels were invented.
4. **Patient-Level Accounting**: Diversity is measured strictly by independent physical records ($N$), never by window volume.

---

## 2. Records Inspected & Candidate Episodes Identified

All 32 quarantined records (`cu04` through `cu35`) were unpacked and inspected. Signals were calibrated to physical millivolts ($\text{mV}$) at native $f_s = 250\text{ Hz}$ on channel `'ECG'` ($\text{adc\_gain} = 400.0\text{ ADC/mV}, \text{baseline} = 0$).

* **Records with Zero Episodes**: Exactly **1 record** (`cu14`) contains 0 bracketed episodes (normal sinus rhythm and isolated ectopy only).
* **Records with Candidate Episodes**: Exactly **31 records** contain at least one candidate episode delineated by onset (`[`) and offset (`]`) markers or ending at record termination.
* **Total Candidate Episodes**: **45 distinct episodes** totaling over **2,800+ seconds** of sustained tachyarrhythmia.
* **Duration Filter**: Every one of the 45 episodes has a duration $\ge 12.7\text{ seconds}$ (all exceed the $5.0\text{ s}$ minimum window requirement; $0$ episodes were excluded for being too short).

---

## 3. Waveform Diagnostics & Classification Governance

For each candidate episode, 10 descriptive physical and mathematical diagnostics were computed on a representative 5.0-second central window:
- Dominant frequency ($f_{\text{dom}}$) via Welch PSD ($0.5 - 30.0\text{ Hz}$).
- Normalized spectral entropy and spectral concentration ($\pm 1.0\text{ Hz}$ band).
- Temporal autocorrelation periodicity strength and $1/e$ decay time.
- Sample Entropy ($45\text{ Hz}, m=2, r=0.2\sigma$) and binary Lempel-Ziv Complexity ($90\text{ Hz}, N=450$).
- Hjorth mobility and complexity.
- Signal amplitude statistics (min, max, std, peak-to-peak, RMS in mV).

A 4-panel diagnostic plot was generated and saved for every episode under `results/plots/`, displaying the full waveform, 5.0s window, PSD, and ACF.

### Summary of Episode Classifications

| Proposed Category | Number of Episodes | Unique Records | Distinct Clinical / Waveform Characteristics | Confidence | Recommendation |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **VENTRICULAR_FLUTTER** | **19** | 16 | Rapid sinusoidal oscillations ($f_{\text{dom}} \in [3.5, 6.0]\text{ Hz}$), high periodicity ($>0.5$), high spectral concentration, absence of isoelectric baseline | MEDIUM | `REQUIRES_MANUAL_REVIEW` |
| **MONOMORPHIC_VT** | **10** | 9 | Regular wide-complex tachycardia ($f_{\text{dom}} \in [1.9, 3.5]\text{ Hz}$), discrete QRS return to baseline, strong autocorrelation periodicity | MEDIUM | `REQUIRES_MANUAL_REVIEW` |
| **COARSE_VF** | **1** | 1 (`cu12`) | Chaotic, polymorphic, disorganized fibrillatory waves ($f_{\text{dom}} = 0.98\text{ Hz}$, high entropy $= 0.77$, low periodicity $= 0.21$) | MEDIUM | `REQUIRES_MANUAL_REVIEW` |
| **AMBIGUOUS** | **15** | 13 | Mixed, transitional morphology (VT degenerating into flutter/VF), pacing artifacts, or unresolvable baseline shifts | LOW | `REQUIRES_MANUAL_REVIEW` |
| **Total** | **45** | **31** | | | **100% Manual Review** |

---

## 4. Complete Episode-by-Episode Audit Table

The complete audit results from [`cudb_episode_audit.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/model3_phase6/phase6c/results/cudb_episode_audit.csv) are summarized below:

| Record ID | Episode | Duration | $f_{\text{dom}}$ | Spec. Ent. | Periodicity | Proposed Category | Conf. | Inclusion Recommendation | Waveform / Clinical Evidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| `cudb_cu04` | ep01 | 55.6s | 2.93 Hz | 0.49 | 1.10 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Regular wide QRS tachycardia, clear periodicity |
| `cudb_cu04` | ep02 | 19.8s | 3.91 Hz | 0.51 | 0.69 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Rapid sinusoidal flutter transition |
| `cudb_cu04` | ep03 | 91.4s | 0.98 Hz | 0.47 | 0.23 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Baseline wander and transitional morphology |
| `cudb_cu04` | ep04 | 105.5s | 2.93 Hz | 0.52 | 0.31 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Wide QRS tachycardia with moderate regularity |
| `cudb_cu05` | ep01 | 87.6s | 4.88 Hz | 0.44 | 1.55 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Classic sinusoidal flutter waveform (293 bpm) |
| `cudb_cu06` | ep01 | 123.8s | 4.88 Hz | 0.52 | 0.75 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Rapid ventricular flutter |
| `cudb_cu06` | ep02 | 13.2s | 0.98 Hz | 0.42 | 0.12 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Low frequency baseline shift post-episode |
| `cudb_cu07` | ep01 | 326.9s | 3.91 Hz | 0.46 | 1.15 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Massive sustained sinusoidal flutter (235 bpm) |
| `cudb_cu08` | ep01 | 82.5s | 2.93 Hz | 0.57 | 0.84 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Regular monomorphic VT |
| `cudb_cu09` | ep01 | 57.4s | 0.98 Hz | 0.50 | 0.08 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Known Atrial Fibrillation patient; low periodicity |
| `cudb_cu10` | ep01 | 192.4s | 4.88 Hz | 0.48 | 1.40 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Sustained rapid ventricular flutter |
| `cudb_cu11` | ep01 | 137.7s | 0.98 Hz | 0.61 | 0.16 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Disorganized baseline with low periodicity |
| `cudb_cu12` | ep01 | 194.3s | 0.98 Hz | 0.77 | 0.21 | COARSE_VF | MED | REQUIRES_MANUAL_REVIEW | Paced patient; degenerated into chaotic coarse VF |
| `cudb_cu13` | ep01 | 54.4s | 3.91 Hz | 0.68 | 0.43 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Mixed flutter/tachycardia morphology |
| `cudb_cu15` | ep01 | 102.9s | 1.95 Hz | 0.33 | 1.55 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Paced patient; regular wide-complex VT |
| `cudb_cu16` | ep01 | 95.6s | 4.88 Hz | 0.39 | 1.76 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Pure sinusoidal ventricular flutter |
| `cudb_cu16` | ep02 | 16.1s | 3.91 Hz | 0.43 | 1.61 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Recurrent ventricular flutter run |
| `cudb_cu17` | ep01 | 38.5s | 2.93 Hz | 0.46 | 1.29 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Highly regular monomorphic VT |
| `cudb_cu18` | ep01 | 26.8s | 3.91 Hz | 0.47 | 1.44 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Rapid flutter in patient with underlying AFib |
| `cudb_cu19` | ep01 | 68.5s | 5.86 Hz | 0.51 | 0.88 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Very rapid sinusoidal flutter (350 bpm) |
| `cudb_cu19` | ep02 | 17.5s | 2.93 Hz | 0.46 | 1.39 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Slower monomorphic VT run |
| `cudb_cu20` | ep01 | 264.7s | 0.98 Hz | 0.25 | 0.00 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Asystolic baseline drift post-arrest |
| `cudb_cu21` | ep01 | 13.2s | 3.91 Hz | 0.48 | 1.44 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Transient ventricular flutter burst |
| `cudb_cu21` | ep02 | 35.2s | 4.88 Hz | 0.66 | 0.90 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Mixed polymorphic activity |
| `cudb_cu21` | ep03 | 15.3s | 5.86 Hz | 0.39 | 1.35 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Sinusoidal flutter run |
| `cudb_cu21` | ep04 | 36.3s | 5.86 Hz | 0.56 | 1.11 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Recurrent rapid flutter |
| `cudb_cu21` | ep05 | 35.3s | 0.98 Hz | 0.53 | 0.13 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Disorganized terminal segment |
| `cudb_cu22` | ep01 | 110.0s | 3.91 Hz | 0.76 | 0.42 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Polymorphic/transitional wide complex |
| `cudb_cu23` | ep01 | 102.5s | 3.91 Hz | 0.41 | 1.57 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Regular sinusoidal flutter (235 bpm) |
| `cudb_cu24` | ep01 | 67.6s | 4.88 Hz | 0.37 | 1.74 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Paced patient; rapid sinusoidal flutter |
| `cudb_cu25` | ep01 | 39.0s | 0.98 Hz | 0.33 | 0.00 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Paced patient; artifact-dominated baseline |
| `cudb_cu26` | ep01 | 57.7s | 3.91 Hz | 0.53 | 1.17 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | High amplitude flutter run |
| `cudb_cu26` | ep02 | 16.0s | 0.98 Hz | 0.60 | 0.12 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Low frequency irregular baseline |
| `cudb_cu27` | ep01 | 23.2s | 2.93 Hz | 0.44 | 0.70 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Monomorphic wide QRS tachycardia |
| `cudb_cu28` | ep01 | 12.7s | 2.93 Hz | 0.47 | 0.56 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Monomorphic VT terminating at record end |
| `cudb_cu29` | ep01 | 130.8s | 0.98 Hz | 0.50 | 0.01 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Very low periodicity disorganized rhythm |
| `cudb_cu30` | ep01 | 105.2s | 0.98 Hz | 0.45 | 0.00 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Low amplitude irregular activity |
| `cudb_cu30` | ep02 | 109.2s | 0.98 Hz | 0.38 | 0.06 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Irregular baseline wander |
| `cudb_cu30` | ep03 | 159.6s | 0.98 Hz | 0.31 | 0.00 | AMBIGUOUS | LOW | REQUIRES_MANUAL_REVIEW | Asystolic baseline drift |
| `cudb_cu31` | ep01 | 14.6s | 3.91 Hz | 0.47 | 1.41 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Sinusoidal flutter oscillation |
| `cudb_cu32` | ep01 | 47.3s | 3.91 Hz | 0.45 | 1.51 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Paced patient; flutter with high periodicity |
| `cudb_cu33` | ep01 | 88.6s | 5.86 Hz | 0.61 | 0.98 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Very fast flutter (350 bpm) |
| `cudb_cu34` | ep01 | 12.8s | 2.93 Hz | 0.47 | 1.40 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Discrete monomorphic VT burst |
| `cudb_cu34` | ep02 | 42.3s | 3.91 Hz | 0.42 | 1.53 | VENTRICULAR_FLUTTER | MED | REQUIRES_MANUAL_REVIEW | Sustained ventricular flutter |
| `cudb_cu35` | ep01 | 25.4s | 2.93 Hz | 0.54 | 1.35 | MONOMORPHIC_VT | MED | REQUIRES_MANUAL_REVIEW | Monomorphic VT ending at record limit |

---

## 5. Patient-Level Impact Analysis

The core scientific premise of this audit is that **thousands of sliding windows from a single recording cannot be treated as independent patients**. We evaluate the potential expansion strictly at the patient/record level:

```
                            32 Quarantined CUDB Records (cu04 - cu35)
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
             1 Record with 0 Episodes                          31 Records with Sustained
                    (cu14)                                        Candidate Episodes
                                                                       │
                         ┌─────────────────────────────────────────────┼─────────────────────────────────────────────┐
                         ▼                                             ▼                                             ▼
                 6 Pure Candidate VT                          11 Pure Candidate VF/VFL                      14 Mixed or Ambiguous
                       Records                                       Records                                       Records
            (cu08, cu15, cu17, cu27,                      (cu05, cu07, cu10, cu12,                      (cu04, cu06, cu09, cu11,
                   cu28, cu35)                                   cu16, cu18, cu23,                             cu13, cu19, cu20,
                                                                 cu24, cu31, cu32,                             cu21, cu22, cu25,
                                                                       cu33)                                   cu26, cu29, cu30,
                                                                                                                     cu34)
```

### Potential Impact on Model 3 Patient Cohorts

| Arrhythmia Class | Current Patients (Post-Phase 6B) | Pure Candidate Records in CUDB | Mixed Records in CUDB | Potential Post-Adjudication Patients | Maximum Potential Relative Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ventricular Tachycardia (VT)** | 108 records | +6 records | +3 records (`cu04`, `cu19`, `cu34`) | **114 to 117 records** | $+5.6\% \text{ to } +8.3\%$ |
| **Ventricular Fibrillation / Flutter (VF/VFL)** | 17 records | +11 records | +5 records (`cu04`, `cu06`, `cu19`, `cu21`, `cu26`, `cu34`) | **28 to 33 records** | $\mathbf{+64.7\% \text{ to } +94.1\%}$ |
| **Ambiguous Only** | — | — | 8 records (`cu09`, `cu11`, `cu13`, `cu20`, `cu22`, `cu25`, `cu29`, `cu30`) | Quarantined / Excluded | — |
| **Total Physical Records** | **120 records** | **+17 records** | **+14 records** | **Up to 142 records** | $+18.3\%$ |

### Scientific Significance of the Findings:
1. **The Shockable Class Could Nearly Double**:
   If clinical manual review by a cardiologist confirms the 11 pure candidate VF/VFL records (or a subset thereof), the independent VF/VFL patient cohort would expand from **17 to 28+ patients**. This represents an enormous $+64.7\%$ increase in true shockable patient diversity.
2. **Ventricular Flutter Dominates CUDB**:
   Of the 30 candidate shockable episodes, 19 exhibit classic sinusoidal ventricular flutter morphology ($f_{\text{dom}} \approx 3.5 - 6.0\text{ Hz}$). This is highly consistent with hospital cardiac care unit (CCU) telemetry where ventricular tachycardia rapidly accelerates into ventricular flutter prior to defibrillation.
3. **Pacing and Artifact Confounders**:
   Five records (`cu12`, `cu15`, `cu24`, `cu25`, `cu32`) involve paced patients. `cu12` degenerates into coarse VF; `cu24` and `cu32` exhibit rapid flutter; `cu15` exhibits VT. These records require special care during clinical review to verify that pacing spikes do not distort feature extraction.
4. **Why Automated Labeling is Unacceptable**:
   14 records contain mixed or ambiguous morphology. For example, `cu04` begins with monomorphic VT (`ep01`), accelerates into ventricular flutter (`ep02`), exhibits transitional wander (`ep03`), and returns to VT (`ep04`). Blindly labeling the entire record as "VF" would inject catastrophic label noise into the VT vs. VF decision boundary.

---

## 6. Critical Limitations & Governance Boundary

1. **Source Annotation Ambiguity**:
   The WFDB `.atr` files in `cu04`–`cu35` mark episode onsets with `[` without physician-signed diagnostic notes. Algorithmic waveform diagnostics (even when backed by high periodicity or spectral concentration) cannot substitute for definitive clinical adjudication.
2. **Strict Non-Interference**:
   In strict accordance with Task 10, **no samples from `cu04`–`cu35` have been added to the ML training dataset**. The expanded dataset from Phase 6B (`expanded_physiological_feature_table.csv`, 14,704 windows, 120 records) remains completely preserved and unchanged.

---

## 7. Strict Decision Gate

In accordance with Task 9 of the Phase 6C specification:

$$\text{\Large \textbf{B. PARTIAL EXPANSION — MANUAL REVIEW REQUIRED}}$$

### Justification:
The audit revealed **11 pure candidate VF/VFL records** and **6 pure candidate VT records** (comprising 45 candidate episodes and over 2,800 seconds of sustained tachyarrhythmia) that have the potential to nearly double the project's shockable patient cohort (from 17 to 28+ patients). 

However, because the source annotations delineate episodes with unspecific bracket symbols (`[`, `]`) and lack certified subtype labels, and because 14 records exhibit transitional/ambiguous morphology or pacing, **these episodes cannot be automatically merged into the ML training dataset without expert manual clinical review**.

Proceeding directly to dataset rebuild (Option A) would violate medical AI safety and inject unverified labels. Conversely, concluding that CUDB is entirely insufficient (Option C) would ignore 11 to 16 highly viable shockable patients that are already locally accessible in the repository.

Therefore, the only medically and scientifically sound determination is:
$$\text{\textbf{PARTIAL EXPANSION — MANUAL REVIEW REQUIRED}}$$

---

### Recommendation for Phase 6D (Pending User Approval):
Before any rebuild of `expanded_physiological_feature_table.csv`:
1. Establish an expert review panel protocol (or multi-expert consensus criteria) using the 45 generated diagnostic plots in [`results/plots/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/model3_phase6/phase6c/results/plots/) to adjudicate each candidate episode.
2. Formally exclude ambiguous episodes (e.g. `cu09`, `cu20`, `cu25`, `cu30`) and quarantine paced records where capture cannot be confirmed.
3. Only incorporate episodes whose clinical consensus category is definitively Monomorphic VT (Target 0) or Ventricular Flutter / Fibrillation (Target 1).
