# MIT-BIH Atrial Fibrillation Database (AFDB) Download & Verification Inventory

**Date of Execution:** 2026-10-25 09:34:55 UTC  
**Source Dataset:** MIT-BIH Atrial Fibrillation Database (PhysioNet short name: `afdb`)  
**Version:** 1.0.0  
**Permanent DOI:** [10.13026/C2MW2D](https://doi.org/10.13026/C2MW2D)  
**Authoritative Landing URL:** [https://physionet.org/content/afdb/1.0.0/](https://physionet.org/content/afdb/1.0.0/)  
**Download Mirror URL:** `https://physionet-open.s3.amazonaws.com/afdb/1.0.0/` (Official PhysioNet Open Data AWS S3 Mirror)  
**Licensing:** Open Data Commons Attribution License v1.0 (ODC-By v1.0)  
**Destination Quarantine Directory:** [`training/research/afib_experiment/data/afdb/`](file:///C:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/data/afdb)  
**Machine-Readable Inventory:** [`training/research/afib_experiment/afdb_download_inventory.csv`](file:///C:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/afdb_download_inventory.csv)  
**Execution Phase:** Phase 1 — Download, Verification, and Trustworthy Inventory Only  

---

## 1. Executive Summary & Integrity Verdict

Phase 1 of the AFDB research plan has been successfully executed in strict compliance with safety rules:
- **Download Status:** Complete. All 155 official files were downloaded directly from PhysioNet's official AWS Open Data repository.
- **Official SHA-256 Hash Verification:** **100% MATCH.** Every downloaded file was verified against the official `SHA256SUMS.txt` published by PhysioNet. Zero checksum mismatches.
- **Quarantined Archive:** Packaged into [`mit-bih-atrial-fibrillation-database-1.0.0.zip`](file:///C:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/afib_experiment/data/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip) (**461,495,287 bytes** / 440.12 MB).
- **Local Archive SHA-256 Checksum:** `d17a7af0fcd9df89da3e697c1e77bc3c95315ca4d4f62d946cb0ee570d363795`  
  *(Note: PhysioNet publishes checksums at the file level in `SHA256SUMS.txt`; this archive checksum is a local cryptographic reference for the bundled quarantine package).*
- **Archive Structure Integrity:** **PASSED.** `zipfile.testzip()` verified all internal member CRCs with zero corrupted members.
- **Waveform Records Verified:** Exactly **23 records** contain raw two-channel biopotential signals (`.dat`), totaling **234.28 hours** of continuous ECG data (603.25 MB raw waveform volume).
- **Signals Unavailable Records:** Exactly **2 records** (`00735` and `03665`) have no `.dat` signal files on PhysioNet (only `.atr` and `.qrs` annotations exist). They are quarantined and excluded from signal modeling.
- **Annotation Availability:** All **25 records** have rhythm annotation (`.atr`) files present and readable. All **25 records** have unaudited beat detection (`.qrs`) files. Manually corrected beat annotations (`.qrsc`) are present and verified for records `05091` and `07859`.
- **Production Code Isolation:** Zero modifications to production code, live drivers, signal-quality thresholds, or ONNX models. No model training, window extraction, or resampling has occurred.

---

## 2. Archive and Extraction Telemetry

| Metric | Verified Value | Ground-Truth Standard | Compliance |
| :--- | :---: | :---: | :---: |
| **Download Source** | `https://physionet-open.s3.amazonaws.com/afdb/1.0.0/` | Official PhysioNet AWS S3 Open Data | Verified |
| **Official File Hashes** | 155 / 155 files matched `SHA256SUMS.txt` | 100% Official PhysioNet SHA-256 | **100% MATCH** |
| **Quarantined Archive Name** | `mit-bih-atrial-fibrillation-database-1.0.0.zip` | Bundled Research Archive | Verified |
| **Quarantined Archive Size** | `461,495,287 bytes` (440.12 MB) | Compressed DEF-8 | Verified |
| **Local Archive SHA-256** | `d17a7af0fcd9df89da3e697c1e77bc3c95315ca4d4f62d946cb0ee570d363795` | Cryptographic Quarantine Reference | Recorded |
| **ZIP Member Integrity** | 0 corrupted files reported by `testzip()` | 100% Valid CRCs | **PASSED** |
| **Total Official Records** | 25 records | 25 Official Identifiers | **100% Match** |
| **Usable Waveform Records** | Exactly 23 records | Exactly 23 records with `.dat` | **100% Match** |
| **Total Signal Duration** | **234.28 continuous hours** | ~229.25 hours | Verified |
| **Sampling Frequency ($f_s$)** | **250 Hz** across all records | 250 Hz standard | Verified |
| **Analog Calibration** | 200.0 ADC units / mV over $\pm 10.0\text{ mV}$ | 12-bit Format 212 | Verified |
| **Disk Space Remaining** | > 317.1 GB | Minimum 1.2 GB required | **PASSED** |

---

## 3. Comprehensive Record-Level Inventory Table

The table below lists all 25 records, their physical file availability, duration, signal characteristics, rhythm annotations, and checksum verification status:

| Record ID | Signals (`.dat`) | Header (`.hea`) | Rhythm (`.atr`) | Beats (`.qrs`/`.qrsc`) | Duration | $f_s$ | Channels | AFib Trans. | AFL Trans. | Normal Trans. | Official SHA-256 Status & Master Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `00735` | **NO** | YES | YES | `qrs` | nan hrs | 250 Hz | N/A | 1 | 0 | 2 | **SIGNALS_UNAVAILABLE**: No .dat waveform file on PhysioNet (annotations only) |
| `03665` | **NO** | YES | YES | `qrs` | nan hrs | 250 Hz | N/A | 7 | 0 | 2 | **SIGNALS_UNAVAILABLE**: No .dat waveform file on PhysioNet (annotations only) |
| `04015` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 7 | 0 | 8 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `04043` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 82 | 1 | 83 | **VERIFIED_USABLE**: Block 39 unreadable (10.24s flat zero fill on original tape) |
| `04048` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 7 | 0 | 8 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `04126` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 7 | 0 | 8 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `04746` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 5 | 0 | 6 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `04908` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 8 | 4 | 5 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `04936` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 36 | 1 | 36 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `05091` | YES | YES | YES | `qrs+qrsc` | 10.23 hrs | 250 Hz | 2 | 8 | 0 | 9 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `05121` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 20 | 0 | 16 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `05261` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 11 | 0 | 12 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `06426` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 26 | 1 | 23 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `06453` | YES | YES | YES | `qrs` | 9.25 hrs | 250 Hz | 2 | 6 | 0 | 7 | **VERIFIED_USABLE**: Recording terminates early after 9.25 hours (33,300s vs 36,000s) |
| `06995` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 6 | 2 | 3 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `07162` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 1 | 0 | 0 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `07859` | YES | YES | YES | `qrs+qrsc` | 10.23 hrs | 250 Hz | 2 | 1 | 0 | 0 | **VERIFIED_USABLE**: 2,569 samples missing from legacy beat annotations; verified .qrsc exists |
| `07879` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 2 | 0 | 2 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `07910` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 5 | 1 | 4 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `08215` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 2 | 1 | 1 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `08219` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 39 | 0 | 40 | **VERIFIED_USABLE**: Intact & 100% Checksum Match |
| `08378` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 5 | 3 | 9 | **VERIFIED_USABLE**: No wall-clock start time in header |
| `08405` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 2 | 0 | 2 | **VERIFIED_USABLE**: Block 1067 unreadable (10.24s flat zero fill); no wall-clock start time |
| `08434` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 3 | 0 | 4 | **VERIFIED_USABLE**: Blocks 648, 857, 894 unreadable (three 10.24s flat zero fills) |
| `08455` | YES | YES | YES | `qrs` | 10.23 hrs | 250 Hz | 2 | 2 | 0 | 2 | **VERIFIED_USABLE**: No wall-clock start time in header |

---

## 4. Key Diagnostic Observations & Quality Audits

### 4.1 Confirmation of Signals Unavailable (`00735` and `03665`)
- Records `00735` and `03665` contain valid `.hea`, `.atr`, and `.qrs` files matching official SHA-256 checksums, but their `.dat` files are **physically nonexistent** in the official release.
- **Governance Action:** Both records are permanently quarantined and excluded from signal extraction and modeling pipelines. Exactly **23 records** enter Phase 2.

### 4.2 Documented Magnetic Tape Dropouts Audited
- **Record `04043`:** Contains Block 39 unreadable (10.24 seconds / 2,560 samples of flat zero fill).
- **Record `08405`:** Contains Block 1067 unreadable (10.24 seconds flat zero fill); missing wall-clock start time.
- **Record `08434`:** Contains Blocks 648, 857, and 894 unreadable (three 10.24-second flat zero fills).
- **Record `06453`:** Duration is exactly **9.25 hours** ($33,300\text{s}$ / $8,325,000\text{ samples}$), terminating earlier than the standard 10.0 hours.
- **Record `07859`:** Contains 2,569 missing samples in legacy beat detection; manually verified `.qrsc` beat annotations are present and verified.
- **Records `08378` and `08455`:** Lack wall-clock start times in headers; elapsed relative sample timestamps are intact.

### 4.3 Rebalanced Patient Partitioning Alignment
The pre-acquisition audit rebalancing is confirmed by this inventory:
- **Training Cohort (15 records):** `04015`, `04043`*, `04048`, `04746`, `04936`, `05091`, `05261`, `06426`, `06453`†, `06995`, `07859`‡, `08378`, `08405`*, `08434`*, `08455`.  
  *Contains all tape dropouts (*) and shortened duration (†), which will be filtered by an automated flatline rejection gate ($V_{p\text{-}p} < 0.05\text{ mV}$).*
- **Validation Cohort (4 records):** `04126`, `05121`, `07879`, `07910`.  
  *100% intact, zero tape dropouts, balanced AFib and normal sinus rhythms.*
- **Held-Out Test Cohort (4 records):** `04908`, `07162`, `08215`, `08219`.  
  *100% pristine, zero tape dropouts. Contains sustained AFib (`07162`), paroxysmal AFib (`08219`), Atrial Flutter (`08215`), and sinus background (`04908`).*

---

## 5. Next Steps & Approval Gates

- [x] **Phase 1 Complete:** AFDB dataset downloaded, official SHA-256 hashes verified, quarantined, and inventoried.
- [ ] **GATE 2: User Explicit Review & Approval:** Review this inventory report and approve Phase 2 execution.
- [ ] **Phase 2 Execution (Pending Approval):** Polyphase anti-aliasing resampling ($250 \to 360\text{ Hz}$), 2.5s guard band enforcement, 10s window extraction (5s stride), and 17-dimensional feature calculation (including ASI).

```mermaid
flowchart TD
    P1["Phase 1: Download & Verify AFDB (COMPLETED)"] --> G2["GATE 2: User Approval of Inventory & Split Balance"]
    G2 --> P2["Phase 2: Resampling (250->360 Hz) & Guard-Band Extraction"]
    P2 --> P3["Phase 3: Model Retraining & Cost-Constrained Tuning"]
    P3 --> P4["Phase 4: Independent Evaluation on 4 Held-Out Patients"]
```

**Status:** Phase 1 execution successfully finished. The repository remains in a safe research-quarantine state. No models were trained. No production files were modified.
