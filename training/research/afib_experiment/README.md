# Isolated Atrial Fibrillation (AFib) Research Pipeline

**Target Architecture:** Raspberry Pi 4B+ (ARM64 Debian Bookworm) + Texas Instruments ADS1292R (Dual-Channel @ 360 Hz)  
**Status:** Completed Preliminary Research Pipeline & Leak-Free Benchmark  
**Deliverable Directory:** `training/research/afib_experiment/`

---

## 1. Project Context & Strict Protection Rules

This experiment implements an **isolated, reproducible research pipeline** for a dedicated Atrial Fibrillation (AFib) rhythm classifier, strictly decoupled from the live AED production code.

### Mandatory Safety & Invariant Guarantees Observed:
1. **Strict PVC Exclusion:** No PVC classification or beat-segmentation logic was implemented or evaluated.
2. **Production Code Freeze:** `deploy/combined_ecg_app.py`, `deploy/ads1292r_driver.py`, signal-quality thresholds, and deployed ONNX models remain strictly identical to HEAD.
3. **No External Network Downloads:** No external packages were installed and no unapproved downloads were initiated.
4. **Isolated Work Directory:** All experiments, features, models, evaluations, and reports reside exclusively within `training/research/afib_experiment/`.
5. **Bradycardia & Asystole are Future Work Only:** Neither detector is implemented in this pipeline.

---

## 2. Directory Structure & Artifact Inventory

```text
training/research/afib_experiment/
├── README.md                              <- Experiment overview & reproduction guide
├── dataset_inventory.csv                  <- Verified 70-record audit (MIT-BIH + VFDB)
├── split_manifest.json                    <- Subject allocation manifest (zero leakage)
├── build_dataset_inventory.py             <- Record audit & inventory generation script
├── prepare_afib_data.py                   <- Feature extraction & guard-band windowing pipeline
├── train_afib_baseline.py                 <- Leak-free model training & threshold tuning script
├── evaluate_afib.py                       <- Test evaluation, per-subject audit, & confounder benchmark
├── features_train.csv                     <- 1,630 training windows (730 AFib, 900 Sinus)
├── features_val.csv                       <- 700 validation windows (340 AFib, 360 Sinus)
├── features_test.csv                      <- 2,444 held-out test windows (321 AFib, 2,123 Sinus)
├── features_confounders.csv               <- 253 confounder windows (AFL, Bigeminy, Paced, Brady)
├── afib_baseline_package.joblib           <- Serialized candidate baseline package (RF + Scaler)
├── afib_model_metadata.json               <- Training metadata, coefficients, & validation stats
├── evaluation_test_results.json           <- Held-out test performance, CIs, & per-subject metrics
├── evaluation_confounder_results.json     <- Slice-based error analysis across confounding rhythms
└── AFIB_EXPERIMENT_REPORT.md              <- Comprehensive technical research report
```

---

## 3. Step-by-Step Reproduction Guide

To reproduce the audit, dataset preparation, training, and evaluation from scratch within the project virtual environment:

### Step 1: Generate the Record-Level Dataset Inventory
Audits all 48 MIT-BIH records and 22 VFDB records, verifying sampling frequencies, channels, exact AFib/AFL durations, patient IDs, and exclusion reasons:
```bash
.\.venv\Scripts\python.exe training/research/afib_experiment/build_dataset_inventory.py
```
*Output:* `training/research/afib_experiment/dataset_inventory.csv`

### Step 2: Extract Features & Partition Cohorts (Zero Leakage)
Applies 2.5s guard bands, extracts 10-second windows with 5-second stride, computes 16 timing and spectral features, and partitions data by independent patient ID:
```bash
.\.venv\Scripts\python.exe training/research/afib_experiment/prepare_afib_data.py
```
*Outputs:* `features_train.csv`, `features_val.csv`, `features_test.csv`, `features_confounders.csv`, `split_manifest.json`

### Step 3: Train Interpretable Baselines & Tune Threshold
Fits `StandardScaler` strictly on training data, trains regularized Logistic Regression and Balanced Random Forest, evaluates on Validation set, and tunes the operating threshold:
```bash
.\.venv\Scripts\python.exe training/research/afib_experiment/train_afib_baseline.py
```
*Outputs:* `afib_baseline_package.joblib`, `afib_model_metadata.json`

### Step 4: Evaluate on Held-Out Test Cohort & Confounders
Evaluates the frozen model on unseen test subjects, computes bootstrap 95% confidence intervals, per-subject breakdown, false positives per hour, and confounder slices:
```bash
.\.venv\Scripts\python.exe training/research/afib_experiment/evaluate_afib.py
```
*Outputs:* `evaluation_test_results.json`, `evaluation_confounder_results.json`

---

## 4. Key Scientific Findings & Limitations

1. **Patient Leakage Prevention:**
   Records 201 and 202 originate from the **exact same 68-year-old male patient**. They are strictly grouped together in the Training partition. Zero overlap exists between Train, Validation, and Test cohorts.
2. **High AFib Sensitivity:**
   The Champion Random Forest model achieved **97.82% sensitivity** on the held-out test patient (`MIT_S221`, 314/321 windows detected), with a detection latency of **12.58 seconds**.
3. **The Confounder Reality (Why Irregular RR $\neq$ AFib):**
   - In pure Normal Sinus Rhythm subjects (Records 230, 231, 234), specificity was **100.00%** (0 false alarms).
   - In Record 233 (frequent ventricular bigeminy / PVC couplets), specificity collapsed to **1.10%** (269 false alarms). Ectopic compensatory pauses mimic AFib RR interval entropy.
   - In Atrial Flutter (`AFL`), the model exhibited a **100% false alarm rate**, proving that variable conduction flutter cannot be separated from AFib without dedicated atrial spectral peak purity features.
4. **Sample Size Constraint:**
   The local dataset hosts only **6 independent non-pacemaker AFib patients**. While sufficient for an exploratory feasibility benchmark, external sourcing (MIT-BIH AFDB, 23 patients, 250 hours) is required before production export.

---

## 5. External MIT-BIH AFDB Ingestion Support

The data pipeline is architected to seamlessly ingest the PhysioNet MIT-BIH Atrial Fibrillation Database (`afdb` v1.0.0) when downloaded by the user.

- **URL:** `https://physionet.org/content/afdb/1.0.0/`
- **Native $f_s$:** 250 Hz (converted to 360 Hz via polyphase resampling: `resample_poly(sig, up=36, down=25)`).
- **Curated Cohorts:** 23 validated recordings (`04015` through `08455`), excluding corrupted records `00735` and `03665`.
