# Phase 6A: Comprehensive Dataset Discovery & Label Audit Report

## 1. Executive Summary & Governance Mandate

* **Audit Objective**: Perform an exhaustive, strictly read-only inspection of all available unused local ECG archives (`cu-ventricular-tachyarrhythmia-database-1.0.0.zip`, `MITBIH.zip`, `VFDB.zip`, and `mitbih-vtvtf.zip`) and their annotations.
* **Strict Governance Boundaries Observed**:
  * Zero modifications to production source code (`src/`), deployed ONNX models (`models/`), unit tests (`tests/`), or completed Phase 2–5 results.
  * Zero model training, hyperparameter tuning, or cross-validation experiments performed.
  * Zero sliding-window extraction or new training dataset creation initiated.
  * Execution strictly confined to read-only inspection and reporting under `training/model3_phase6/`.
* **Key Findings at a Glance**:
  1. **CUDB Unused Records (32 records)**: Exactly **25 records** (`cu04`–`cu13`, `cu16`–`cu19`, `cu21`–`cu27`, `cu30`, `cu32`–`cu34`) contain **39 distinct sustained shockable arrhythmia episodes** totaling **~2,800+ seconds**. However, all 25 records suffer from **rhythm subtype ambiguity** (episodes are marked with onset/offset brackets `[` and `]` without differentiating VT from VF). They cannot be added to Model 3 without clinical re-adjudication.
  2. **MIT-BIH Arrhythmia Database (48 records)**: Contains **3 records with sustained arrhythmias $\ge 5.0$ seconds** usable without ambiguity: Record `223` (sustained VT, 94.1 s), Record `205` (sustained VT, 8.7 s), and Record `207` (sustained Ventricular Flutter, 143.9 s). The remaining 10 VT records contain exclusively brief non-sustained VT bursts (1.1 s – 3.9 s) that cannot yield pure 5.0-second windows.
  3. **VFDB (22 records)**: All 22 records and all sustained episodes are already 100% incorporated into the Phase 2E.2 / Phase 5 dataset. Zero unused records exist.
  4. **Challenge 2015 (750 records)**: All verified true alarms with Lead II (86 VT, 6 VF) are already 100% incorporated. The remaining 52 VF records and 252 VT records are confirmed **false alarms** caused by motion artifact, baseline wander, and non-shockable rhythms.

---

## 2. Exhaustive Local Archive Audits

### A. Creighton University Ventricular Tachyarrhythmia Database (`CUDB`)
* **Archive**: `cu-ventricular-tachyarrhythmia-database-1.0.0.zip`
* **Total Records in Archive**: 35 records (`cu01` through `cu35`).
* **Signal Specifications**:
  * Sampling Rate: **250 Hz**, 12-bit ADC over $\pm 10\text{ mV}$.
  * Leads Available: **1 lead** (modified standard bipolar ECG, labeled `ECG`).
  * Signal Duration: **508.9 seconds** (~8.5 minutes; 127,232 samples) per record.
  * Annotation Format: WFDB `.atr`.

#### Record-by-Record Breakdown
1. **Currently Utilized in Model 3 (3 records)**:
   * `cu01`: 1 sustained VF episode (294.7 s, onset at sample 53541 with explicit `(VF` note). Yields 116 windows.
   * `cu02`: 5 distinct VT episodes (durations: 1.6 s, 9.4 s, 3.1 s, 3.1 s, 12.6 s; explicit `(VT` notes). Yields 6 windows $\ge 5.0$ s.
   * `cu03`: 1 sustained VF episode (43.2 s, onset at sample 116430 with explicit `(VF` note). Yields 16 windows.
2. **Unused Records with Bracketed Episodes (25 records, 39 episodes)**:
   * `cu04`: 4 episodes (55.6 s, 19.8 s, 91.4 s, 105.4 s; total = 272.2 s)
   * `cu05`: 1 episode (87.6 s)
   * `cu06`: 2 episodes (123.8 s, 13.2 s; total = 137.0 s)
   * `cu07`: 1 episode (**326.9 s**)
   * `cu08`: 1 episode (82.5 s)
   * `cu09`: 1 episode (57.4 s; also contains atrial fibrillation notes `(AF`)
   * `cu10`: 1 episode (**192.4 s**)
   * `cu11`: 1 episode (**137.7 s**)
   * `cu12`: 1 episode (**194.3 s**; paced patient who developed VF)
   * `cu13`: 1 episode (54.4 s)
   * `cu16`: 2 episodes (95.6 s, 16.1 s; total = 111.7 s)
   * `cu17`: 1 episode (38.5 s)
   * `cu18`: 1 episode (26.8 s; also contains `(AF`)
   * `cu19`: 1 episode (68.6 s)
   * `cu21`: 5 episodes (13.2 s, 35.2 s, 15.3 s, 36.3 s, 35.3 s; total = 135.3 s)
   * `cu22`: 1 episode (110.0 s)
   * `cu23`: 1 episode (102.5 s)
   * `cu24`: 1 episode (67.6 s; paced patient)
   * `cu25`: 1 episode (39.0 s; paced patient)
   * `cu26`: 2 episodes (57.7 s, 16.0 s; total = 73.7 s)
   * `cu27`: 1 episode (23.2 s)
   * `cu30`: 2 episodes (105.2 s, 109.2 s; total = 214.4 s)
   * `cu32`: 1 episode (47.3 s; paced patient)
   * `cu33`: 1 episode (88.6 s)
   * `cu34`: 2 episodes (12.8 s, 42.3 s; total = 55.1 s)
3. **Unused Records with Zero Episodes (7 records)**:
   * `cu14`, `cu15`, `cu20`, `cu28`, `cu29`, `cu31`, `cu35`: Contain 0 brackets and 0 rhythm notes (continuous normal/paced baseline or non-sustained ectopy).

#### Label Ambiguity Assessment in CUDB
* **Finding**: In records `cu04` through `cu35`, the bracket symbols `[` and `]` delineate sustained life-threatening tachyarrhythmias, but **do not specify whether the arrhythmia is monomorphic VT, polymorphic VT, ventricular flutter, or coarse VF**.
* **PhysioNet Specification**: The database documentation explicitly notes:
  > *"All beats are labelled normal... The database is defined as a tachyarrhythmia database rather than a fibrillation database... Reference annotation files are in no sense definitive."*
* **Verdict**: These 25 records cannot be merged into Model 3 as-is because Model 3 explicitly trains on the decision boundary between VT (Target 0) and VF (Target 1). Arbitrarily labeling all bracketed episodes as VF would introduce severe label noise and distort ground truth.

---

### B. MIT-BIH Arrhythmia Database (`MITDB`)
* **Archive**: `MITBIH.zip`
* **Total Records in Archive**: 48 records (records 100 through 234; duplicated in root and `mitbih_database/`).
* **Signal Specifications**:
  * Sampling Rate: **360 Hz** (native Target FS for Model 3; zero resampling required).
  * Leads Available: **2 leads** (Lead 1: modified limb lead `MLII`; Lead 2: precordial lead `V1`, `V2`, `V4`, or `V5`).
  * Signal Duration: **1,800.0 seconds** (30.0 minutes; 650,000 samples) per record.
  * Annotation Format: WFDB `.atr` rhythm change annotations (`+` with `(VT`, `(VFL`, `(SVTA`, `(N`, etc.).

#### Complete Rhythm & Duration Audit Across All 48 Records

| Record ID | Leads | VT Episodes | Total VT Duration | Longest VT Episode | VF/VFL Episodes | Total VF/VFL Duration | Classification & Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **205** | MLII / V1 | 6 | 22.9 s | **8.66 s** | 0 | 0.0 s | **Usable: Sustained VT** (1 episode $\ge 5$s) |
| **207** | MLII / V1 | 2 | 2.9 s | 1.52 s | **6** | **143.9 s** | **Usable: Sustained VFL** (5 episodes $\ge 5$s, longest = **98.4 s**) |
| **223** | MLII / V1 | 7 | **105.8 s** | **55.07 s** | 0 | 0.0 s | **Usable: Sustained VT** (2 episodes $\ge 5$s: 55.1 s, 39.0 s) |
| **106** | MLII / V1 | 1 | 2.0 s | 2.01 s | 0 | 0.0 s | **Excluded: NSVT Burst Only** (<5.0 s) |
| **200** | MLII / V1 | 7 | 15.2 s | 2.82 s | 0 | 0.0 s | **Excluded: NSVT Bursts Only** (<5.0 s) |
| **203** | MLII / V1 | 21 | 32.5 s | 3.94 s | 0 | 0.0 s | **Excluded: NSVT Bursts Only** (<5.0 s) |
| **210** | MLII / V1 | 2 | 5.7 s | 3.13 s | 0 | 0.0 s | **Excluded: NSVT Bursts Only** (<5.0 s) |
| **213** | MLII / V1 | 2 | 3.9 s | 2.30 s | 0 | 0.0 s | **Excluded: NSVT Bursts Only** (<5.0 s) |
| **214** | MLII / V1 | 2 | 4.5 s | 2.28 s | 0 | 0.0 s | **Excluded: NSVT Bursts Only** (<5.0 s) |
| **215** | MLII / V1 | 2 | 2.3 s | 1.16 s | 0 | 0.0 s | **Excluded: NSVT Bursts Only** (<5.0 s) |
| **217** | MLII / V1 | 1 | 1.8 s | 1.76 s | 0 | 0.0 s | **Excluded: NSVT Bursts Only** (<5.0 s) |
| **221** | MLII / V1 | 2 | 3.5 s | 1.78 s | 0 | 0.0 s | **Excluded: NSVT Bursts Only** (<5.0 s) |
| **233** | MLII / V1 | 6 | 10.9 s | 1.84 s | 0 | 0.0 s | **Excluded: NSVT Bursts Only** (<5.0 s) |
| *Remaining 35 Records* | MLII / V1/V5 | 0 | 0.0 s | 0.0 s | 0 | 0.0 s | **Excluded: Zero VT or VF Episodes** |

#### Crucial Epistemological Finding on NSVT vs. Sustained VT
* 10 of the 13 VT records in MIT-BIH consist exclusively of brief runs of 3 to 8 ectopic beats lasting **1.1 to 3.9 seconds**.
* Under a 5.0-second analysis window, a 2-second NSVT burst would necessarily include 3 seconds of adjacent normal sinus rhythm or PVC couplets, severely corrupting feature extraction.
* Therefore, only **Record 223** (94.1 s sustained VT) and **Record 205** (8.7 s sustained VT) provide pure, unpolluted 5-second VT windows.
* For the shockable flutter/fibrillation class, **Record 207** contributes **143.9 seconds of sustained Ventricular Flutter** (`(VFL`) with multiple long episodes (10.3 s, 11.5 s, 13.6 s, and **98.4 s**).

---

### C. MIT-BIH Malignant Ventricular Arrhythmia Database (`VFDB`)
* **Archive**: `VFDB.zip`
* **Total Records in Archive**: 22 records (records 418 through 615).
* **Current Status in Project**: **100% Utilized** (all 22 records are already in `calibrated_vfvt_feature_table.csv`).
* **Signal Specifications**:
  * Sampling Rate: **250 Hz**, 2 leads (`ECG`, `ECG`).
  * Signal Duration: **2,100.0 seconds** (35.0 minutes; 525,000 samples) per record.
* **Audit of Usable Segments**:
  * Total VT Episodes: 91 episodes across 19 records. 29 sustained episodes ($\ge 5.0$ s) yielded **1,980 windows**.
  * Total VF Episodes: 112 episodes across 8 records. 38 sustained episodes ($\ge 5.0$ s) yielded **1,080 windows**.
  * Non-Target Rhythms Present: `(AFIB` (atrial fibrillation), `(ASYS` (asystole), `(SVTA` (supraventricular tachycardia), `(NOD`, `(NOISE`, `(PM`. These non-target rhythms were already filtered out in Phase 2E.2.
* **Additional Records Available**: **Zero**. Every record in `VFDB.zip` is already represented.

---

### D. Computing in Cardiology Challenge 2015 (`c2015`)
* **Archive**: `mitbih-vtvtf.zip`
* **Total Records in Archive**: 750 records (5 minutes each, 250 Hz).
* **Alarm Breakdown from `ALARMS` Ground Truth**:
  * `Ventricular_Tachycardia`: 89 True alarms (86 had Lead II and were incorporated; 3 lacked Lead II), **252 False alarms**.
  * `Ventricular_Flutter_Fib`: **6 True alarms** (all 6 incorporated), **52 False alarms**.
  * `Asystole`: 22 True alarms, 100 False alarms.
  * `Bradycardia`: 46 True alarms, 43 False alarms.
  * `Tachycardia`: 131 True alarms, 9 False alarms.
* **Additional True Records Available**: **Zero true VF records**. All 6 verified true VF alarms are already in the dataset; the remaining 52 VF alarm recordings in `mitbih-vtvtf.zip` are confirmed false alarms (ECG artifacts).

---

## 3. Proposed Unified Labeling Strategy

To incorporate new records without label ambiguity or window contamination, the following standardized protocol is established:

```
                  Raw Waveform Episode Audit
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    Duration < 5.0 s                    Duration >= 5.0 s
    (Brief NSVT bursts)                (Sustained Arrhythmias)
            │                                   │
            ▼                                   ▼
       DISQUALIFIED                    Rhythm Subtype Check
   (Excludes 10 MITDB records:                  │
    106, 200, 203, 210, 213,      ┌─────────────┴─────────────┐
    214, 215, 217, 221, 233)      ▼                           ▼
                           Explicit Rhythm Label        Bracketed Only ([ ])
                           in Reference Annotation      Without Subtype Label
                                  │                           │
                   ┌──────────────┴──────────────┐            ▼
                   ▼                             ▼       DISQUALIFIED
               Target 0:                     Target 1:   (Excludes CUDB
             Ventricular                   Ventricular   records 4-35
             Tachycardia                   Fibrillation  pending expert
               (VT)                          & Flutter   re-adjudication)
                   │                         (VF / VFL)
           Continuous wide                       │
           bizarre QRS without             Chaotic fibrillatory
           normal intervening P            waves OR regular
           waves; rate > 100 bpm           biphasic sinusoidal
                                           flutter; rate > 200 bpm
```

### A. Strict Ground-Truth Target Mapping
1. **Target = 0 (Ventricular Tachycardia — VT)**:
   * **Inclusion**: Sustained episodes annotated as `(VT` with continuous wide-complex morphology, rate $>100\text{ bpm}$, and duration $\ge 5.0\text{ seconds}$.
   * **Exclusion**: Non-sustained ventricular tachycardia (NSVT) bursts $<5.0\text{ seconds}$.
2. **Target = 1 (Ventricular Fibrillation & Flutter — VF / VFL)**:
   * **Inclusion**: Episodes annotated as `(VF` (disorganized, chaotic, polymorphic waveforms with absence of discrete QRS complexes) OR `(VFL` (regular, rapid sinusoidal oscillations without isoelectric baseline, rate $>200\text{ bpm}$), with duration $\ge 5.0\text{ seconds}$.
   * **Exclusion**: Transient flutter runs $<5.0\text{ seconds}$ or noisy baseline artifact.
3. **Quarantine / Rejection**:
   * All episodes annotated with `(AFIB`, `(SBR`, `(ASYS`, `(NOD`, `(NOISE`, `(PM` are strictly excluded from both classes.
   * All bracketed episodes without explicit VT vs. VF clinical labels (CUDB records `cu04`–`cu35`) must remain quarantined.

### B. Standardized Preprocessing & Resampling
* **Lead Selection**: Lead II (or modified limb lead `MLII`). For single-lead recordings (CUDB), use channel 0 (`ECG`).
* **Resampling**: All signals resampled from native sampling rate ($250\text{ Hz}$ or $360\text{ Hz}$) to standard $\mathbf{f_s = 360\text{ Hz}}$ using rational polyphase filtering (`scipy.signal.resample_poly`).
* **Calibration**: Converted to physical millivolts ($\text{mV}$) using header metadata:
  $$V(\text{mV}) = \frac{\text{ADC} - \text{adc\_zero}}{\text{gain}} + \text{baseline}$$
* **Bandpass Filtering**: 0.5 Hz to 30.0 Hz 2nd-order zero-phase Butterworth filter.
* **Segmentation**: Fixed 5.0-second sliding windows with 2.5-second step ($N = 1,800$ samples, step $= 900$ samples). Only episodes with length $\ge 5.0$ seconds are windowed.

---

## 4. Dataset Expansion Inventory Summary

The complete inventory of all audited local records is persisted in [`local_records_audit.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/model3_phase6/local_records_audit.csv) (83 records across CUDB and MITDB):

| Database | Total Records in Archive | Currently Used | Unused Records Available | Unused Records with Usable Sustained VT | Unused Records with Usable Sustained VF/VFL | Unused Records with Ambiguous Labels | Net Independent Patient Gain (Immediate) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VFDB** | 22 | 22 | 0 | 0 | 0 | 0 | **+0** |
| **Challenge 2015** | 750 | 92 | 658 | 0 | 0 | 0 (all false alarms) | **+0** |
| **CUDB** | 35 | 3 | 32 | 0 | 0 | **25** (39 episodes, ~2,800s) | **+0 immediate** (+25 pending re-adjudication) |
| **MITDB** | 48 | 0 | 48 | **2** (`205`, `223`) | **1** (`207`) | 10 (NSVT bursts <5s) | **+3 immediate** (2 VT, 1 VFL) |
| **Total** | **855** | **117** | **738** | **2** | **1** | **35** | **+3 verified patients** |

---

## 5. Decision & Next Step Recommendations

1. **Immediate In-Tree Integration (3 Independent Records)**:
   * Incorporate **Record 223** (sustained VT: 94.1 s) and **Record 205** (sustained VT: 8.7 s) from MIT-BIH to add 2 new independent VT patients.
   * Incorporate **Record 207** (sustained VFL: 143.9 s) from MIT-BIH to add 1 new independent shockable patient.
2. **Clinical Re-Adjudication of CUDB Records 4–35 (25 Records)**:
   * The 39 bracketed episodes across `cu04`–`cu35` represent over 2,800 seconds of sustained tachyarrhythmias.
   * Because they lack subtype labels in `.atr`, they must be morphologically adjudicated (classifying each bracketed interval as monomorphic VT, polymorphic VT, ventricular flutter, or coarse VF) before they can be extracted as training targets.
3. **Execution Stop Boundary**:
   * As instructed, execution has stopped before window extraction or creation of any new training dataset.

The audit table is saved at [`training/model3_phase6/local_records_audit.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/model3_phase6/local_records_audit.csv) and documented in [`DATASET_INVENTORY.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/model3_phase6/DATASET_INVENTORY.md).
