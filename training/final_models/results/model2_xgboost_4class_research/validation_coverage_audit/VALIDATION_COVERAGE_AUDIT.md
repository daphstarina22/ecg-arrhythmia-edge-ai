# Research Audit: Evaluation Coverage and Data Overlap for Model 2 XGBoost

**Date**: 2026-10-25 23:03:15  
**Target Model**: Four-Class XGBoost Arrhythmia Classifier (`research_xgboost_4class_pipeline.joblib`)  
**Classes Under Audit**: `NSR`, `TACHY`, `BRADYCARDIA`, `VENTRICULAR`  
**Output Directory**: `training/final_models/results/model2_xgboost_4class_research/validation_coverage_audit/`  
**Safety Status**: Strictly read-only audit. Zero models retrained; zero features modified; zero thresholds changed; zero production code touched.

---

## 1. Executive Summary & Key Audit Findings

A thorough provenance and evaluation coverage audit was conducted on the four-class Model 2 research XGBoost pipeline. The audit uncovered **three fundamental findings** regarding existing evaluation claims:

1. **100% Training Overlap in Existing Offline Evaluations (Zero Independent Records)**:
   - All four records evaluated in the offline evaluation report (`mitdb_100`, `vfdb_425`, `mitdb_200`, and `vfdb_424`) were **explicitly present in the supervised 4-class training dataset** (`four_class_master_feature_dataset.csv`).
   - Because `research_xgboost_4class_pipeline.joblib` was fitted on the entire 7,493-row dataset prior to serialization, evaluating this serialized artifact offline on these four records constitutes **in-sample re-substitution testing**, not out-of-sample generalization.
   - **Zero completely unseen, independent records have been evaluated offline with `research_xgboost_4class_pipeline.joblib` to date.**
2. **Severe Class-Level Evaluation Blind Spots**:
   - **`BRADYCARDIA`**: **Zero windows** with official reference-annotated sinus bradycardia (`(SBR`) have been evaluated offline. `vfdb_425` was evaluated, but its ground-truth annotation is `(N` (sinus origin). `mitdb_232`, which provided **55.0% of all bradycardia training windows**, was never evaluated offline across its full 30-minute recording.
   - **`TACHY`**: **Zero windows** with true reference-annotated supraventricular tachycardia (`(SVTA` or `(ST`) have been evaluated offline. The 23 tachycardia predictions in `vfdb_424` and `mitdb_200` were misclassifications of flutter or noise.
   - **`VENTRICULAR`**: Sustained ventricular recall (97.14%) was demonstrated on **only 35 5-second windows across just two records** (`vfdb_424` and `vfdb_425`), both of which were in the training set.
3. **Massive Untapped Independent Validation Data in Workspace**:
   - The **Creighton University Ventricular Tachyarrhythmia Database (CUDB)** contains **35 records** in `cu-ventricular-tachyarrhythmia-database-1.0.0.zip` with clean WFDB rhythm annotations (`.atr`) for sustained Ventricular Fibrillation and Ventricular Tachycardia.
   - **Zero CUDB records were used in 4-class training**. CUDB represents a completely unpolluted, gold-standard independent benchmark ready for offline evaluation.

---

## 2. Provenance Audit: Training vs. Evaluation Record Overlap

The supervised 4-class training dataset contains **7,493 windows across 318 independent records**. The four records evaluated offline were audited against the training set:

| Record ID | Dataset Source | Windows in Training Set | Training Labels | Windows Evaluated Offline | True Ground Truth Rhythms | Independence Status |
| :--- | :---: | :---: | :--- | :---: | :--- | :---: |
| **`mitdb_100`** | MIT-BIH | 35 windows | `NSR` (35) | 361 windows | `(N` (100% Sinus Rhythm) | **TRAINING OVERLAP** |
| **`vfdb_425`** | VFDB | 286 windows | `NSR` (281), `VENTRICULAR` (5) | 420 windows | `(N` (Sinus Bradycardia), `(VT` (5 wins) | **TRAINING OVERLAP** |
| **`mitdb_200`** | MIT-BIH | 35 windows | `NSR` (35) | 361 windows | `(N` (53.1%), `(B` (46.1%), `(VT` (0.8%) | **TRAINING OVERLAP** |
| **`vfdb_424`** | VFDB | 343 windows | `NSR` (313), `VENTRICULAR` (30) | 420 windows | `(N` (59.9%), `(VFIB` (5.0%), `(VFL` (2.8%) | **TRAINING OVERLAP** |

> [!WARNING]
> **Data Leakage & In-Sample Bias Risk**  
> While fold-level cross-validation (`xgboost_depth5_oof_predictions.csv`) strictly separated training and validation records within each fold, the final deployed/offline model (`research_xgboost_4class_pipeline.joblib`) was fitted on **all 7,493 windows** (`X_full`).
> Therefore, `mitdb_100`, `vfdb_425`, `mitdb_200`, and `vfdb_424` were part of the training data for `research_xgboost_4class_pipeline.joblib`. Their offline evaluation results cannot be cited as proof of out-of-distribution generalization.

---

## 3. Evaluation Coverage Gaps by Rhythm Class

### 3.1 BRADYCARDIA: Severe Evaluation Gap
- **Training Foundation**: 220 windows across 39 records.
  - Record `mitdb_232` contributed **121 windows (55.0% of all training bradycardia)**.
  - 37 PhysioNet Challenge 2015 records contributed 95 windows (43.2%).
  - Record `vfdb_430` contributed 4 windows (1.8%).
- **Offline Evaluation Coverage**: **ZERO reference-annotated `(SBR` windows evaluated**.
  - The 12 windows classified as `BRADYCARDIA` in `vfdb_425` were rate-based discoveries within a record officially annotated as `(N`.
  - `mitdb_232` (which contains 30 continuous minutes of true `(SBR`) has **never been evaluated offline** with the serialized model.

### 3.2 VENTRICULAR: Narrow Coverage & In-Sample Bias
- **Training Foundation**: 1,838 windows across 119 records.
  - VFDB provided 1,512 windows (**82.3%**).
  - C2015 provided 285 windows (15.5%).
  - MITDB provided only 41 windows (2.2%).
- **Offline Evaluation Coverage**: Only **35 unambiguous 5-second windows** across 2 records (`vfdb_424`: 30 windows; `vfdb_425`: 5 windows).
  - Both records were in the training set.
  - In `mitdb_200`, all 7 VT bursts were brief non-sustained episodes (<2.82s) diluted by sinus conduction.
  - **Model 2 has zero evaluation on sustained ventricular fibrillation or flutter from an unseen database**.

### 3.3 TACHY: Total Evaluation Absence
- **Training Foundation**: 362 windows across 127 records.
  - C2015 terminal alarms provided 323 windows (**89.2%**).
  - MITDB provided 24 windows (6.6%).
  - VFDB provided 15 windows (4.1%).
- **Offline Evaluation Coverage**: **ZERO genuine supraventricular tachycardia (`(SVTA` or `(ST`) windows evaluated**.
  - The 23 TACHY predictions across `vfdb_424` (21) and `mitdb_200` (2) were misclassified regular ventricular flutter peaks or transitional noise.

### 3.4 NSR: Over-Represented, In-Sample Baseline
- **Training Foundation**: 5,073 windows across 56 records.
- **Offline Evaluation Coverage**: 1,064 unambiguous windows evaluated across 4 records.
  - 100% of these evaluated windows came from records present in the training set.

---

## 4. Benchmark Dataset Inventory & Unseen Data Discovery

| Benchmark Archive | File Path | Total Available Records | Records in 4-Class Training | Records Unseen / Excluded | Available Reference Annotations | Suitability for Independent Evaluation |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **CUDB** | `cu-ventricular-tachyarrhythmia-database-1.0.0.zip` | **35** | **0** | **35 (100%)** | WFDB ATR: episode markers `[` (start VF) and `]` (end VF) | **EXCELLENT (Gold standard independent test for sustained VF/VT)** |
| **VFDB** | `VFDB.zip` | 22 | 22 | 0 (0%) | WFDB ATR: `(VFIB`, `(VFL`, `(VT`, `(N`, etc. | Exhausted in training (No independent records remain) |
| **MIT-BIH** | `MITBIH.zip` | 48 | 41 | 7 (14.6%) | WFDB TXT: `(SBR`, `(SVTA`, `(VT`, `(B`, `(AFIB` | 7 records excluded (pacemaker / AFib). 41 records in training. |
| **Challenge 2015** | `mitbih-vtvtf.zip` | 750 | 255 | 495 (66.0%) | ALARMS CSV: True/False alarm labels | 495 records unseen (43 false Brady, 110 false VT, 8 true Brady). |

---

## 5. Prioritized List of Recommended Additional Records to Evaluate

To establish genuine out-of-distribution clinical validity for the unchanged 5-second Model 2 pipeline, we recommend evaluating the following **five prioritized benchmark records**:

### Rank 1: `cudb_cu01` — Unseen Sustained Ventricular Fibrillation Benchmark
- **Source**: CUDB (`cu-ventricular-tachyarrhythmia-database-1.0.0.zip`)
- **Independence**: **100% Independent (Zero rows in training set)**
- **Target Class**: `VENTRICULAR` (Coarse Ventricular Fibrillation)
- **Annotation Profile**: Clean normal sinus baseline from $t = 0	ext{s}$ to $214.2	ext{s}$, followed by **almost 5 continuous minutes (294.7 seconds, ~59 non-overlapping 5s windows) of sustained coarse VFIB** ($t = 214.2	ext{s}$ to $508.9	ext{s}$).
- **Clinical Value**: Proves whether Model 2 detects sustained, life-threatening ventricular fibrillation in an unseen hospital recording without in-sample training bias.

### Rank 2: `cudb_cu04` — Unseen Recurrent Ventricular Tachycardia & Flutter
- **Source**: CUDB (`cu-ventricular-tachyarrhythmia-database-1.0.0.zip`)
- **Independence**: **100% Independent (Zero rows in training set)**
- **Target Class**: `VENTRICULAR` (Recurrent Ventricular Tachycardia & Flutter)
- **Annotation Profile**: Multiple discrete paroxysms of sustained monomorphic VT/flutter ($t = 155	ext{--}211	ext{s}$, $224	ext{--}243	ext{s}$, $255	ext{--}346	ext{s}$, $370	ext{--}475	ext{s}$) totaling ~250 seconds of ventricular rhythm separated by sinus baseline.
- **Clinical Value**: Validates onset triggering latency and post-termination return to baseline across multiple recurrent ventricular tachyarrhythmia episodes.

### Rank 3: `mitdb_232` — Continuous Sinus Bradycardia Benchmark
- **Source**: MIT-BIH Arrhythmia Database (`MITBIH.zip`)
- **Independence**: Training Record (121 windows in training), but **NEVER evaluated offline across its full duration**.
- **Target Class**: `BRADYCARDIA` (Official `(SBR` Rhythm)
- **Annotation Profile**: **100% continuous Sinus Bradycardia (`(SBR`)** from sample 76 through the entire 30-minute recording (~360 windows @ 360 Hz).
- **Clinical Value**: Closes the severe Bradycardia evaluation gap by establishing continuous 30-minute precision, recall, and false-NSR leakage on an official `(SBR` recording.

### Rank 4: `mitdb_209` — Supraventricular Tachycardia Benchmark
- **Source**: MIT-BIH Arrhythmia Database (`MITBIH.zip`)
- **Independence**: Training Record (11 windows in training), but **NEVER evaluated offline across its full duration**.
- **Target Class**: `TACHY` (Supraventricular Tachycardia `(SVTA`)
- **Annotation Profile**: Contains confirmed paroxysms of sustained supraventricular tachycardia (`(SVTA`).
- **Clinical Value**: Resolves the total evaluation absence for TACHY by benchmarking genuine supraventricular tachyarrhythmia.

### Rank 5: `cudb_cu16` — Unseen Normal Sinus Rhythm Control
- **Source**: CUDB (`cu-ventricular-tachyarrhythmia-database-1.0.0.zip`)
- **Independence**: **100% Independent (Zero rows in training set)**
- **Target Class**: `NSR` (Independent Specificity Control)
- **Annotation Profile**: Predominant normal sinus rhythm control without sustained ventricular collapse.
- **Clinical Value**: Verifies that the model does not produce false ventricular or false tachycardia alarms on clean sinus rhythm in an unseen database.

---

## 6. Summary of Key Gaps & Single Best Next Validation Step

### Summary of Key Gaps:
1. **Zero Independent Record Validation**: All 4 previously evaluated records were in the training set.
2. **Zero Official `(SBR` Bradycardia Validation**: No record with an official `(SBR` annotation has been evaluated offline.
3. **Zero Genuine `(SVTA` Tachycardia Validation**: Model 2's performance on genuine supraventricular tachycardia is completely unvalidated offline.
4. **Sample Size Constraint on Ventricular Rhythms**: Previous ventricular sensitivity was demonstrated on only 35 5-second windows.

> [!TIP]
> **Single Best Next Validation Step: Offline Evaluation of `cudb_cu01` and `cudb_cu04`**
> Run the unchanged 5-second offline evaluation script on **`cudb_cu01`** and **`cudb_cu04`** (resampling from 250 Hz to 360 Hz).
> - Both records are **100% independent and unseen**.
> - They provide **over 100 non-overlapping windows of sustained Ventricular Fibrillation and Ventricular Flutter** alongside clean sinus baselines.
> - This will definitively prove whether the 97.14% ventricular recall observed in in-sample records generalizes to an unseen clinical population.