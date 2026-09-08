# Phase 6A: Public ECG Dataset Inventory & Expansion Plan

## 1. Executive Summary & Problem Formulation

### Scientific Context from Phase 5
Phase 5 demonstrated that extracting raw-waveform physiological features (autocorrelation decay, zero-crossing lag, and downsampled Lempel-Ziv complexity / Sample Entropy) eliminated the direction inversion observed in spectral and QRS-derived features, improving cross-database D2 ROC-AUC from ~0.60 to a maximum of **0.6825**. 

However, **no configuration passed the generalization decision gate** (D2 ROC-AUC > 0.75, balanced record recall > 70%). Cross-database evaluation revealed severe polarity distortion:
* Models trained on VFDB heavily underpredict VF on Challenge 2015 (VF record recall: 31.5%–43.0%).
* Models trained on Challenge 2015 heavily overpredict VF on VFDB (VT record recall: 16.2%–25.3%).

### Root Cause: Patient Sample Size Bottleneck
The combined Phase 2E.2 / Phase 3 / Phase 5 corpus contains:
* **117 unique physical patient records** (92 Challenge 2015, 22 VFDB, 3 CUDB)
* **Only 16 independent physical patients with Ventricular Fibrillation** (6 in Challenge 2015, 8 in VFDB, 2 in CUDB)
* **14,618 total 5-second windows** (12,650 VT, 1,968 VF)

**Crucial Scientific Axiom**: 1,968 overlapping sliding windows generated from 16 individuals do **NOT** represent 1,968 independent observations. When a classifier trains on 8–10 VF patients from one database, it memorizes individual patient electrophysiological morphologies, pacing artifacts, and lead axes rather than universal VF dynamics.

### Phase 6 Objective
Systematically identify, audit, and rank publicly available and accessible ECG databases to expand the independent VF patient representation from **~16 toward at least 50 or more**, establishing whether the cross-database generalization barrier is primarily driven by sample diversity.

---

## 2. Comprehensive Candidate Dataset Audit (12 Evaluation Criteria)

Every candidate dataset was audited against the 12 required criteria:

```
Criteria Legend:
1. Independent Patients/Records  5. Number of Leads          9. Patient ID Availability
2. Number of VT Records         6. Signal Duration          10. Overlap with Existing Corpus
3. Number of VF Records         7. Annotation Format        11. Licensing / Access Policy
4. Sampling Frequency (Hz)      8. VT/VF Extractability     12. Pipeline Compatibility
```

---

### Candidate 1: MIT-BIH Malignant Ventricular Arrhythmia Database (`vfdb`)
* **Status in Project**: Currently Integrated (Baseline Transfer Source/Target).
* **1. Independent Patients**: **22 unique physical patients** (records 418 to 615).
* **2. VT Records**: 19 records (1,980 windows).
* **3. VF Records**: 8 records (1,080 windows). Exactly 5 records (`vfdb_422`, `vfdb_426`, `vfdb_429`, `vfdb_430`, `vfdb_609`) contain both VT and VF episodes.
* **4. Sampling Frequency**: 250 Hz.
* **5. Number of Leads**: 2 leads (modified limb and precordial).
* **6. Signal Duration**: ~35 minutes per record.
* **7. Annotation Format**: WFDB `.atr` with explicit rhythm change labels: `(VT`, `(VF`, `(VFL`, `(N`.
* **8. VT/VF Extractability**: High. Exact episode onset and offset boundaries verified.
* **9. Patient ID Availability**: Yes (record IDs 418–615 map 1:1 with patients).
* **10. Overlap with Existing Corpus**: Baseline component.
* **11. Licensing / Access**: Open Access (Open Data Commons Attribution License v1.0).
* **12. Pipeline Compatibility**: Direct (native WFDB; resampled via `resample_poly` from 250 to 360 Hz).

---

### Candidate 2: PhysioNet / Computing in Cardiology Challenge 2015 (`c2015`)
* **Status in Project**: Currently Integrated (Baseline Transfer Source/Target).
* **1. Independent Patients**: **92 unique physical patients** (records v131l to v844s).
* **2. VT Records**: 86 records (10,664 windows).
* **3. VF Records**: 6 records (756 windows).
* **4. Sampling Frequency**: 250 Hz (calibrated from raw ADCs and `.hea` gains).
* **5. Number of Leads**: 2 leads (Lead II and Lead V/pleth).
* **6. Signal Duration**: 5 minutes per record (alarms triggered at minute 4.5).
* **7. Annotation Format**: Expert-verified alarm adjudications in `ALARMS` table (`1` = True alarm, `0` = False alarm).
* **8. VT/VF Extractability**: High (entire 5-minute pre-alarm telemetry strip belongs to the verified alarm class).
* **9. Patient ID Availability**: Yes (anonymized ICU patient record IDs).
* **10. Overlap with Existing Corpus**: Baseline component.
* **11. Licensing / Access**: Open Access (ODC-By v1.0).
* **12. Pipeline Compatibility**: Direct (custom ZIP memory loader and polynomial resampler).

---

### Candidate 3: Creighton University Ventricular Tachyarrhythmia Database (`cudb`)
* **Status in Project**: Partially Integrated (3 of 35 records used: `cu01`, `cu02`, `cu03`).
* **1. Independent Patients**: **35 unique physical patients** (`cu01` to `cu35`).
* **2. VT Records**:
  * *Currently Audited*: 1 record (`cu02`, 6 windows).
  * *Full Archive*: Records 4–35 lack rhythm subtype notes in `.atr`.
* **3. VF Records**:
  * *Currently Audited*: 2 records (`cu01`, `cu03`, 132 windows).
  * *Full Archive*: Up to 32 records contain shockable episode onsets marked by `[` and `]`, but without explicit VT vs. VF sub-classification.
* **4. Sampling Frequency**: 250 Hz, 12-bit resolution.
* **5. Number of Leads**: 1 lead (modified standard bipolar lead).
* **6. Signal Duration**: ~8.5 minutes (127,232 samples).
* **7. Annotation Format**: WFDB `.atr`. Records 1–3 contain `(VT`, `(VF`, and `(N` rhythm labels. Records 4–35 contain onset brackets `[` and `]` only; all beats are labeled `N`.
* **8. VT/VF Extractability**:
  * *Records 1–3*: High (directly extracted).
  * *Records 4–35*: **Low to Moderate**. Requires manual expert electrophysiologist re-adjudication or automated rhythm morphology verification to distinguish sustained monomorphic VT from coarse/fine VF within the bracketed episodes.
* **9. Patient ID Availability**: Yes (`cu01` through `cu35`).
* **10. Overlap with Existing Corpus**: Records 1–3 already used; records 4–35 are independent.
* **11. Licensing / Access**: Open Access (ODC-By v1.0).
* **12. Pipeline Compatibility**: High (native WFDB 250 Hz single-lead).
* **Expansion Verdict**: **High Priority for Expert Adjudication**. If records 4–35 can be reliably sub-classified into VT and VF, CUDB can immediately contribute up to ~30 additional independent tachyarrhythmia patients without acquiring new data files.

---

### Candidate 4: Sudden Cardiac Death Holter Database (`sddb`)
* **Status in Project**: Evaluated in Phase 6A Audit.
* **1. Independent Patients**: 23 patients (records 30 to 52).
* **2. VT Records**: 18 records.
* **3. VF Records**: 19 records.
* **4. Sampling Frequency**: 250 Hz.
* **5. Number of Leads**: 2 leads.
* **6. Signal Duration**: Up to 24 hours per tape.
* **7. Annotation Format**: WFDB `.atr` (audited) and `.ari` (unaudited beat annotations).
* **8. VT/VF Extractability**: High (exact elapsed onset time of VF documented in database metadata table).
* **9. Patient ID Availability**: Yes (records 30 to 52).
* **10. Overlap with Existing Corpus**: **CRITICAL FINDING: 100% PATIENT IDENTITY OVERLAP WITH VFDB**.
  * Official PhysioNet release documentation explicitly states:
    > *"We initiate this database with 23 complete Holter recordings (originally collected by Scott Greenwald while he was at MIT), from which half-hour excerpts have been available to researchers since 1989 as the MIT-BIH Malignant Ventricular Arrhythmia Database (vfdb)."*
  * Record 30 in SDDB is the complete 24-hour tape from which record 418/419 was cut.
  * Adding SDDB alongside VFDB in an expanded corpus would introduce **catastrophic duplicate patient leakage**.
* **11. Licensing / Access**: Open Access (ODC-By v1.0).
* **12. Pipeline Compatibility**: High (WFDB).
* **Expansion Verdict**: **REJECTED FOR EXPANSION**. Cannot be used as an independent dataset alongside VFDB. It may only serve as a long-term duration replacement for VFDB, not an expansion of patient diversity.

---

### Candidate 5: MIT-BIH Arrhythmia Database (`mitdb`)
* **Status in Project**: Evaluated in Phase 6A Audit (`task-351`).
* **1. Independent Patients**: 47 subjects (48 records: 100 to 234).
* **2. VT Records**: **13 independent patients** (`106`, `200`, `203`, `205`, `207`, `210`, `213`, `214`, `215`, `217`, `221`, `223`, `233`).
* **3. VF Records**: **0 chaotic VF records**; exactly **1 Ventricular Flutter (`(VFL`) record** (`207`).
* **4. Sampling Frequency**: 360 Hz.
* **5. Number of Leads**: 2 leads (modified Lead II and V1/V2/V4/V5).
* **6. Signal Duration**: 30 minutes each.
* **7. Annotation Format**: WFDB `.atr` reference annotations with explicit rhythm change labels.
* **8. VT/VF Extractability**: High for VT episodes.
* **9. Patient ID Availability**: Yes (documented mapping of subjects to records; record 201 and 202 are from the same subject).
* **10. Overlap with Existing Corpus**: Zero overlap with Challenge 2015, VFDB, or CUDB.
* **11. Licensing / Access**: Open Access (ODC-By v1.0).
* **12. Pipeline Compatibility**: Direct (native 360 Hz, zero resampling needed).
* **Expansion Verdict**: **Secondary Priority (VT Expansion Only)**. Can add 13 high-quality independent VT patients to diversify the VT cohort, but **fails to resolve the critical VF bottleneck** (adds zero chaotic VF patients).

---

### Candidate 6: American Heart Association (AHA) ECG Database
* **Status in Project**: Evaluated in Literature & Archive Audit.
* **1. Independent Patients**: **80 unique patients** (divided into 8 classes of 10 patients each).
* **2. VT Records**: **10 independent patients** (Class 7000: records 7001 to 7010).
* **3. VF Records**: **10 independent patients** (Class 8000: records 8001 to 8010).
* **4. Sampling Frequency**: 250 Hz.
* **5. Number of Leads**: 2 leads.
* **6. Signal Duration**: 35 minutes each (last 30 minutes annotated).
* **7. Annotation Format**: WFDB `.atr` with strict rhythm and beat annotations audited by an expert cardiologist panel.
* **8. VT/VF Extractability**: Excellent (gold standard clinical adjudications).
* **9. Patient ID Availability**: Yes (7001–7010 and 8001–8010).
* **10. Overlap with Existing Corpus**: Zero overlap (completely independent clinical collection).
* **11. Licensing / Access**: **Proprietary / Restricted**. Distributed commercially by the ECRI Institute under license from the American Heart Association ($2,000–$5,000 fee; restricted academic license). Not freely downloadable from PhysioNet.
* **12. Pipeline Compatibility**: High (standard WFDB 250 Hz).
* **Expansion Verdict**: **Highest Scientific Value, but Blocked by Access Restrictions**. If institutional access is acquired, AHA would provide 10 gold-standard, completely independent VF patients and 10 VT patients.

---

### Candidate 7: MIMIC-III / MIMIC-IV Waveform Database
* **Status in Project**: Evaluated for Scaled ICU Telemetry Expansion.
* **1. Independent Patients**: >10,000 intensive care unit patients.
* **2. VT Records**: Estimated >1,000 records with VT alarm annotations.
* **3. VF Records**: Estimated >200 records with VF/VFlutter alarm annotations.
* **4. Sampling Frequency**: 125 Hz (requires 125 Hz $\to$ 360 Hz upsampling via `resample_poly`).
* **5. Number of Leads**: Multi-lead bedside monitors (3 to 8 leads).
* **6. Signal Duration**: Multi-hour to multi-day continuous ICU telemetry.
* **7. Annotation Format**: WFDB `.hea`, `.dat`, plus numerical trend and alarm event logs.
* **8. VT/VF Extractability**: **Moderate**. While monitor alarms for "V-TACH" and "V-FIB" are logged, ICU monitor alarms suffer from an acknowledged **70%–85% false-alarm rate** (motion artifact, loose leads, muscle tremor). Unfiltered extraction would severely contaminate ground-truth labels with noise. Requires strict automated signal quality indexing (SQI) and expert alarm verification.
* **9. Patient ID Availability**: Yes (`subject_id` and `hadm_id` with de-identified timestamps).
* **10. Overlap with Existing Corpus**: Challenge 2015 drew a small subset of candidate alarms from early MIMIC archives, but exact record mappings are documented and easily segregated.
* **11. Licensing / Access**: **Credentialed Access Required**. Free of charge, but requires CITI Program Human Subjects Research certification and a signed PhysioNet Data Use Agreement (DUA). Cannot be downloaded via public scripts.
* **12. Pipeline Compatibility**: Moderate (requires 125 Hz upsampling and lead selection logic).
* **Expansion Verdict**: **Long-Term Scalability Target**. The only freely accessible resource capable of scaling independent VF patient counts beyond 50–100, pending credentialed access and false-alarm filtering.

---

### Candidate 8: St. Petersburg INCART 12-lead Arrhythmia Database (`incartdb`)
* **Status in Project**: Audited via `task-355`.
* **1. Independent Patients**: 32 patients (75 records).
* **2. VT Records**: **0 records** (verified via `task-355`).
* **3. VF Records**: **0 records** (verified via `task-355`).
* **4. Sampling Frequency**: 257 Hz.
* **5. Number of Leads**: 12 leads.
* **6. Signal Duration**: 32 minutes each.
* **7. Annotation Format**: WFDB `.atr` beat annotations.
* **8. VT/VF Extractability**: N/A (records contain PVCs, couplets, and bigeminy, but no sustained VT or VF).
* **Expansion Verdict**: **REJECTED**. Zero relevant arrhythmia episodes.

---

### Candidate 9: PTB-XL 12-Lead Electrocardiography Database
* **Status in Project**: Evaluated for Large-Scale Representation.
* **1. Independent Patients**: 18,885 patients (21,837 recordings).
* **2. VT Records**: ~200 records with diagnostic SCP-ECG statement `VTACH`.
* **3. VF Records**: **0 records**.
* **4. Sampling Frequency**: 500 Hz (and 100 Hz version).
* **5. Number of Leads**: 12 leads.
* **6. Signal Duration**: **10 seconds each**.
* **7. Annotation Format**: CSV metadata with SCP-ECG diagnostic codes.
* **8. VT/VF Extractability**: Incompatible. 10-second diagnostic resting ECGs cannot yield sliding 5-second windows with 2.5-second step for sustained arrhythmia analysis. Furthermore, conscious resting outpatients do not exhibit ventricular fibrillation.
* **Expansion Verdict**: **REJECTED**. Incompatible recording duration (10 seconds) and total absence of ventricular fibrillation.

---

### Candidate 10: European ST-T Database (`edb`)
* **Status in Project**: Evaluated.
* **1. Independent Patients**: 79 subjects (90 records).
* **2. VT / VF Records**: 0 sustained VT/VF. Focus is myocardial ischemia, ST depression, and T-wave changes.
* **Expansion Verdict**: **REJECTED**. Non-target pathology.

---

### Candidate 11: Real-World Out-of-Hospital Cardiac Arrest (OHCA) Defibrillator Registries
* **Status in Project**: Literature Assessment (EHU, Oslo EMS, Seattle EMS, Richmond EMS).
* **1. Independent Patients**: >1,000 true cardiac arrest events recorded by emergency medical services.
* **2. VT / VF Records**: Hundreds of true sudden cardiac arrest VF and pulseless VT episodes recorded through defibrillator pads.
* **3. Extractability & Quality**: Gold-standard clinical deployment data for AED / defibrillator algorithms.
* **4. Licensing & Access**: **Strictly Closed / Proprietary**. Governed by hospital institutional review boards (IRBs) and commercial defibrillator manufacturers (Zoll, Stryker, Philips). Not publicly accessible without funded clinical consortium agreements.
* **Expansion Verdict**: **Aspirational / Industry Benchmark Target**. Cannot be downloaded or integrated under open-source academic research terms.

---

## 3. Phase 6B: Candidate Dataset Ranking & Selection

### Quantitative Patient Diversity Matrix

| Dataset | Access Tier | Unique Patients | Independent VT Patients | Independent VF Patients | Overlap Risk | Pipeline Compatibility | Extraction Reliability | Recommended Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **AHA ECG Database** | Commercial / Restricted | 20 | 10 | **10** | **None** | High | High (Gold Standard) | **Priority 1 (Ideal Independent Benchmark)** |
| **CUDB (Records 4–35)** | Open Access (Local) | 32 | Unverified | **Up to 32** | **None** | High (Already in tree) | Moderate (Requires Re-adjudication) | **Priority 2 (Immediate In-Tree Target)** |
| **MITDB** | Open Access (Local) | 47 | **13** | 0 (1 VFL) | **None** | Direct (360 Hz) | High | **Priority 3 (VT Diversity Only)** |
| **MIMIC-III / IV** | Credentialed Access | 10,000+ | 1,000+ | **200+** | Minimal / Segregated | Moderate | Moderate (Requires SQI & Filtering) | **Priority 4 (Scalability Target)** |
| **SDDB** | Open Access | 23 | 18 | 19 | **100% with VFDB** | High | High | **DISQUALIFIED (Patient Leakage)** |
| **INCARTDB** | Open Access | 32 | 0 | 0 | None | High | Zero Episodes | **DISQUALIFIED (No VT/VF)** |
| **PTB-XL** | Open Access | 18,885 | ~200 | 0 | None | Incompatible | 10s recordings | **DISQUALIFIED (No VF, 10s clips)** |

### Strict Distinction of Sample Units

To eliminate statistical confusion between window counts and clinical observations, all future evaluations must report:

1. **Unique Physical Patients ($N_{\text{patient}}$)**: Independent biological human subjects. Zero patient overlap allowed across splits.
2. **Unique Physical Records ($N_{\text{record}}$)**: Separate recording sessions (some patients have multiple tapes).
3. **Rhythm Episodes ($N_{\text{episode}}$)**: Distinct clinical episodes of VT or VF within a recording.
4. **Window Count ($N_{\text{window}}$)**: Segmented 5-second analysis frames. Window-level sample sizes must **never** be used to claim statistical power.

---

## 4. Phase 6C: Standardized Harmonization Protocol Design

Before integrating any candidate dataset into Model 3, all raw ECG records must undergo identical standardized preprocessing:

```
[Raw Physical ECG] 
       │
       ▼
1. Lead Selection: Primary modified limb lead (Lead II or equivalent bipolar channel)
       │
       ▼
2. Calibration & Normalization: Convert raw ADCs to physical millivolts:
   V(mV) = (ADC - adc_zero) / gain + baseline
       │
       ▼
3. Bandpass Filtering: 0.5 Hz - 30.0 Hz 2nd-order zero-phase Butterworth filter
       │
       ▼
4. Resampling: scipy.signal.resample_poly to standard TARGET_FS = 360 Hz
       │
       ▼
5. Artifact & Quality Rejection: Exclude windows with NaN fraction > 0.05, 
   flatline amplitude < 0.05 mV, or saturation clipping > 10.0 mV
       │
       ▼
6. Segmentation: 5.0-second sliding window, 2.5-second step (WINDOW_SAMPLES = 1800, STEP_SAMPLES = 900)
       │
       ▼
7. Harmonized Labeling: Binary classification:
   Target = 0 (VT: Ventricular Tachycardia)
   Target = 1 (VF: Ventricular Fibrillation & Flutter)
```

---

## 5. Phase 6D & 6E: Evaluation Strategy & Generalization Decision Gate

### Baseline Feature Sets Evaluated (Zero New Feature Engineering)
1. `set0_baseline_old7`: The 7 reduced hand-crafted features from Phase 4.2.
2. `set1_full_physio`: All 21 raw-waveform physiological features.
3. `set2_domain_reduced`: 6 low-confounding consistent features (`lz_complexity`, `ac_decay_time`, `ac_zero_crossing_lag`, `ac_max_peak_ratio`, `sample_entropy`, `hjorth_mobility`).
4. `set3_best_low_conf`: Top 4 features by signal-to-confounder ratio.

### Split Design: Leave-One-Dataset-Out (LODO) Validation
With an expanded multi-database corpus (e.g. Challenge 2015, VFDB, Adjudicated CUDB, MITDB VT):
* **Fold 1**: Train on {VFDB, CUDB, MITDB} $\to$ Test on Challenge 2015.
* **Fold 2**: Train on {Challenge 2015, CUDB, MITDB} $\to$ Test on VFDB.
* **Fold 3**: Train on {Challenge 2015, VFDB, MITDB} $\to$ Test on CUDB.

### Phase 6E Generalization Decision Gate
The experiment will answer the core scientific question:
> *"Was the Phase 5 generalization failure primarily a feature-representation problem, a dataset-size problem, a domain-shift problem, or a combination of these?"*

* **Success Criteria for Passing Gate**:
  1. Average Cross-Database ROC-AUC across all held-out datasets $> \mathbf{0.75}$.
  2. Record-level balanced accuracy $> \mathbf{65\%}$ in both transfer directions.
  3. No class recall collapse (both VT and VF record recall $\ge \mathbf{70\%}$).
* If the expanded cohort passes the gate: Patient diversity was the bottleneck; proceed to edge quantization and ONNX deployment.
* If the expanded cohort still collapses: Domain shift and morphological divergence are fundamental; deep neural feature representations or domain adaptation are mathematically required.
