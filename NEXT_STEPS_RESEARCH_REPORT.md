# Comprehensive ECG Safety Audit and PVC Research Prototype Report

**Author:** Senior ECG Machine Learning Researcher & Edge-AI Deployment Engineer  
**Date:** October 25, 2026  
**Repository:** `ecg-arrhythmia-edge-ai`  
**Status:** Completed Investigation & Experimental Benchmark  
**Deliverable Document:** `NEXT_STEPS_RESEARCH_REPORT.md`  

---

## 1. Current Production Safety Findings (Phase 1 Deep Dive)

### 1.1 How Current Production Identifies Flat or Disconnected Signals
Inspection of the production signal quality evaluator in [`deploy/combined_ecg_app.py:149-164`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/combined_ecg_app.py#L149-L164) reveals:
```python
# 2. Physiological AC dynamic range & flatline analysis (de-mean to remove electrode DC bias)
ac_signal = window - np.median(window)
ac_std = float(np.std(ac_signal))
ac_ptp = float(np.ptp(ac_signal))

# Flatline detection: AC variance near zero or minimal movement
if ac_std < 0.015 or ac_ptp < 0.050:
    return SignalQualityReport(
        status="BAD_SIGNAL_FLAT",
        is_valid_for_ml=False,
        n_samples=n_samples,
        mean_mv=mean_val, std_mv=std_val, ptp_mv=ac_ptp, min_mv=min_val, max_mv=max_val,
        pct_nonfinite=0.0, pct_saturated=0.0,
        detail="Signal is flatline or disconnected"
    )
```
When `is_valid_for_ml` is `False`, [`deploy/combined_ecg_app.py:526-537`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/combined_ecg_app.py#L526-L537) executes:
```python
if not quality.is_valid_for_ml:
    with state_lock:
        system_state["signal_quality"] = quality.status
        system_state["signal_details"] = quality.detail
        system_state["final_decision"] = f"BAD SIGNAL ({quality.status})"
        system_state["triage_level"] = "BAD_SIGNAL"
        system_state["model1"] = {"label": "NOT RUN", "confidence": 0.0, "is_shockable": False}
        system_state["model2"] = {"rhythm": "NOT RUN", "confidence": 0.0}
        system_state["model3"] = {"subtype": "NOT RUN", "confidence": 0.0, "active": False}
    print(f"[{timestamp_str}] [SIGNAL DIAGNOSTIC] {quality.status}: {quality.detail} -> ML NOT RUN")
    continue
```
**Consequence:** Any window with $V_{\text{AC, ptp}} < 0.050\text{ mV}$ or $\text{std} < 0.015\text{ mV}$ is completely bypassed. Machine learning inference is halted.

### 1.2 Driver & Hardware Register Investigation (`deploy/ads1292r_driver.py`)
A rigorous examination of [`deploy/ads1292r_driver.py:145-175`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/ads1292r_driver.py#L145-L175) establishes the following critical facts:
1. **Lead-Off Registers Are Unconfigured:**
   In `_init_chip()` (lines 146–151), the driver writes to:
   - `REG_CONFIG1 = 0x02` (500 SPS)
   - `REG_CONFIG2 = 0xA0` (2.42V internal reference ON)
   - `REG_CH1SET = 0x00` (Gain 6, normal electrode input)
   - `REG_CH2SET = 0x00` (Gain 6, normal electrode input)
   - `REG_RLDSENS = 0x2C` (RLD sensed from CH1P/CH1N/CH2P)
   **`REG_LOFF` (address 0x03) and `REG_LOFFSENS` (address 0x07) are NEVER written.** By default, after reset, `LOFFSENS` is `0x00`, meaning lead-off sensing current comparators on Channel 1 and Channel 2 are **completely disabled**.
2. **Status Bytes in SPI Frames Are Discarded:**
   In `read_sample()` (lines 167–175):
   ```python
   frame = self.spi.xfer2([0x00] * 9)  # 3 status bytes + 3 bytes CH1 + 3 bytes CH2
   ch1_raw = self._to_signed24(frame[3:6])
   ch2_raw = self._to_signed24(frame[6:9])
   ```
   The first 3 bytes (`frame[0:3]`, containing the 24-bit Texas Instruments ADS1292R status word) are **dropped without being unpacked or inspected**.
3. **`LOFF_STAT` Register (0x08) Is Never Read at Runtime:**
   Because the chip runs in continuous read data mode (`RDATAC`), register reads (`RREG`) are locked out unless `SDATAC` is commanded. The runtime never queries `LOFF_STAT`.

### 1.3 Why `LOFF_STAT == 0` Does NOT Prove Electrode Contact
A naive proposal to map `(flatline AND LOFF_STAT == 0) => ASYSTOLE` is clinically hazardous due to the following failure modes:
1. **Electrode Short Circuit:** If lead wires touch each other (e.g. clipping leads together before applying to the patient), the inter-electrode impedance is $0\,\Omega$. The lead-off comparator registers zero open-circuit fault (`LOFF_STAT == 0`), while the differential voltage is flatline ($0.00\text{ mV}$). Naively mapping this to Asystole would sound high-priority emergency CPR alarms on an unattached device.
2. **Capacitive Coupling & High Impedance:** If electrodes are detached from skin but hanging near 50 Hz power cables or clothing, capacitive displacement currents can inject microvolts of noise that fluctuate above the flatline threshold without exceeding the DC lead-off current comparator.
3. **Electrolyte Gel Dehydration:** High contact impedance from dried pads can degrade signal-to-noise ratio without tripping the 6 nA DC comparator.
4. **Physiological Asystole vs. Ventricular Standstill:** True Asystole is characterized by intact skin impedance ($1\text{--}5\text{ k}\Omega$) combined with absent cardiac electrogenesis, confirmed across two orthogonal leads.

### 1.4 Production Design Recommendation
- **Leave production signal-quality thresholds strictly unchanged** in this software task.
- **Never infer asystole solely from a flatline signal.**
- **Hardware Requirement for Future Implementation:** Differentiating true Asystole from lead disconnection requires:
  1. Configuring `LOFF` (0x03) and `LOFFSENS` (0x07) with active AC or DC impedance excitation.
  2. Unpacking the 24-bit status frame in `read_sample()` in real-time.
  3. Physical validation on a calibrated hardware ECG simulator (e.g. Fluke PS420 / Rigel Uni-Sim) connected to the Raspberry Pi + ADS1292R benchtop setup to empirically verify comparator trip thresholds across $0\,\Omega$ short, $1\text{ k}\Omega$ normal skin, and $100\text{ k}\Omega$ open-circuit conditions.

---

## 2. Verified MIT-BIH Inventory and Annotation Counts

The local archive [`MITBIH.zip`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/MITBIH.zip) was programmatically audited via [`training/research/audit_mitbih.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/audit_mitbih.py).

### 2.1 File & Archive Structure
- **Physical Archive:** `c:\Users\LENOVO\Downloads\ecg-arrhythmia-edge-ai\MITBIH.zip` (302,308,888 bytes).
- **Total Recordings:** Exactly **48 records** (Record numbers: 100–124, 200–234).
- **CSV Signal Files:** 48 files (`100.csv` to `234.csv`), each containing 650,000 samples.
  - Native Sampling Rate ($f_s$): **360 Hz**.
  - Total Duration per Record: $\frac{650,000}{360} = 1,805.55\text{ s} \approx 30.09\text{ minutes}$.
  - Physical Calibration Formula: $\text{mV} = \frac{\text{raw} - 1024.0}{200.0}$.
- **Annotation Files:** 48 text files (`100annotations.txt` to `234annotations.txt`).
  - Columns: `Time`, `Sample #`, `Type`, `Sub`, `Chan`, `Num`, `Aux`.

### 2.2 Verified Annotation Symbols Across All 48 Records
*(Correcting previous duplicate-loop approximations)*:

| Annotation Symbol | Clinical Interpretation | Total Verified Occurrences |
| :---: | :--- | :---: |
| **`N`** | **Normal Sinus Beat** | **75,052** |
| `L` | Left Bundle Branch Block Beat | 8,075 |
| `R` | Right Bundle Branch Block Beat | 7,259 |
| **`V`** | **Premature Ventricular Contraction (PVC)** | **7,130** |
| `/` | Paced Beat | 7,028 |
| **`A`** | **Atrial Premature Beat (PAC)** | **2,546** |
| `+` | Rhythm Change Annotation Marker | 1,291 |
| `f` | Fusion of Paced and Normal Beat | 982 |
| `F` | Fusion of Ventricular and Normal Beat | 803 |
| `~` | Signal Quality Change / Edge Change | 616 |
| `!` | Ventricular Flutter Wave | 472 |
| `"` | Comment Marker | 437 |
| `j` | Nodal (Junctional) Escape Beat | 229 |
| `x` | Non-Conducted P-Wave (Blocked PAC) | 193 |
| `a` | Aberrated Atrial Premature Beat | 150 |
| `\|` | Isolated QRS-like Artifact | 132 |
| `E` | Ventricular Escape Beat | 106 |
| `J` | Nodal (Junctional) Premature Beat | 83 |
| `Q` | Unclassifiable Beat | 33 |
| `e` | Atrial Escape Beat | 16 |
| `[` / `]` | Start / End of Pacemaker Rhythm | 12 |
| `S` | Supraventricular Premature Beat | 2 |

### 2.3 Lead Configurations & Missing MLII Records
- **Records with Modified Limb Lead II (`MLII`):** **46 records** (40 with `MLII, V1`; 2 with `MLII, V5`; 2 with `MLII, V2`; 1 with `V5, MLII`; 1 with `MLII, V4`).
- **Records Lacking `MLII`:** Exactly **2 records** (`102` and `104` contain only `V5` and `V2`). These two records are excluded from single-lead Lead II processing.

### 2.4 Patient / Subject Mapping
- The 48 recordings derive from **47 unique individual patients**.
- In standard PhysioNet clinical records, **Record 201 and Record 202 originate from the same patient** (male, age 68). All other 46 records originate from separate individuals.

### 2.5 Beat-to-Sample Alignment Verification
Testing the `Sample #` index against raw ECG arrays confirmed that annotation indices correspond directly to the R-peak maxima within $\pm 1$ sample ($\pm 2.8\text{ ms}$).

---

## 3. PVC Prototype Design (`training/research/pvc_beat_prototype.py`)

An isolated research module was authored at [`training/research/pvc_beat_prototype.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/pvc_beat_prototype.py):

### 3.1 Architecture & Pipeline Isolation
- **Completely Decoupled:** Operates in `training/research/`; is **not imported** by `deploy/` or the production Raspberry Pi app.
- **Window Formulation:**
  - Pre-peak window: $200\text{ ms} = 72$ samples at 360 Hz.
  - Post-peak window: $300\text{ ms} = 108$ samples at 360 Hz.
  - Total beat window: **180 samples** ($500\text{ ms}$).
- **Boundary Handling:** If $s_{\text{idx}} < 72$ or $s_{\text{idx}} + 108 > N_{\text{samples}}$, the beat is marked as `BOUNDARY_OVERFLOW` and skipped. Across all 48 records, exactly 17 boundary beats were safely dropped.
- **Symbol Filtering:** Target classes are strictly Normal (`N` = 0) vs. Premature Ventricular Contraction (`V` = 1). All other 21 symbols (including atrial premature beats `A`, bundle branch blocks `L`/`R`, and paced beats `/`) are strictly excluded from the binary labels and tracked in an exclusion counter.

### 3.2 30-Dimensional Morphological & Timing Feature Representation
1. **Dynamic Amplitude Features (6):** `ptp_mv`, `std_mv`, `max_mv`, `min_mv`, `energy`, `zero_cross`.
2. **QRS Symmetry & Shape Features (6):** `peak_idx`, `trough_idx`, `skew`, `kurtosis`, `max_pos_slope`, `max_neg_slope`.
3. **RR Coupling & Timing Features (4):**
   - `pre_rr_sec`: Preceding RR interval (seconds).
   - `post_rr_sec`: Succeeding RR interval (seconds).
   - `rr_ratio`: $\frac{\text{pre\_rr}}{\text{post\_rr} + 10^{-4}}$ (Key physiological marker: PVCs exhibit short pre-RR followed by compensatory pause, yielding `rr_ratio` $< 0.7$).
   - `local_hr`: $\frac{60.0}{\text{pre\_rr}}$.
4. **Waveform Decimation Vector (14):** 14 points evenly subsampled across the 180-sample QRS complex to capture spatial trajectory.

---

## 4. Leakage Prevention and Evaluation Methodology

To prevent optimistic bias and patient memorization:
1. **Patient-Level Stratification:** The 47 subjects (including the combined subject `201_202`) were partitioned into three non-overlapping cohorts:
   - **Train Set (27 records):** `100`, `101`, `103`, `106`, `107`, `108`, `114`, `115`, `116`, `117`, `119`, `121`, `123`, `124`, `203`, `209`, `210`, `212`, `213`, `215`, `220`, `221`, `223`, `228`, `230`, `231`, `232`.
   - **Validation Set (9 records):** `109`, `111`, `113`, `122`, `201`, `202`, `208`, `233`, `234`.
   - **Test Set (10 records):** `105`, `112`, `118`, `200`, `205`, `207`, `214`, `217`, `219`, `222`.
2. **Leakage Verification:** Assertions verify zero intersection among train, val, and test record sets ($N_{\text{overlap}} = 0$).
3. **Encapsulated Preprocessing:** `StandardScaler` was fitted strictly on the training set (`fit_transform`) and subsequently applied to validation and test sets (`transform`).

---

## 5. Test & Experimental Results

### 5.1 Unit Test Suite (`training/research/test_pvc_prototype.py`)
A standalone test suite was authored and executed:
- `test_window_dimensions`: Passed (180 samples: 72 pre, 108 post).
- `test_feature_extraction_properties`: Passed (all 30 features finite on synthetic QRS).
- `test_patient_split_leak_free`: Passed (0 record overlap; records 201 and 202 strictly co-partitioned).
- `test_boundary_handling_logic`: Passed (correct underflow/overflow handling).
- `test_unsupported_symbol_classification`: Passed (`A`, `L`, `R`, `/` correctly rejected).
- **Result:** `Ran 5 tests in 0.004s. OK.`

### 5.2 Patient-Level Dataset Counts
- **Train Split (27 records):** 50,048 beats (Normal: 46,547, PVC: 3,501 = $7.0\%$).
- **Validation Split (9 records):** 16,543 beats (Normal: 14,463, PVC: 2,080 = $12.6\%$).
- **Held-Out Test Split (10 records):** 15,305 beats (Normal: 13,764, PVC: 1,541 = $10.1\%$).
- **Total Analyzed Beats:** **81,896 beats**.
- **Unsupported Symbols Excluded:** 21,578 annotations (e.g. `R`: 6,862, `L`: 5,952, `/`: 3,620, `+`: 1,239, `A`: 1,164, `F`: 803, etc.).

### 5.3 Held-Out Patient-Level Classifier Benchmark

```
+------------------------------------+------------+--------------+-----------------+---------------+---------------+
| Classifier Pipeline                | Accuracy   | Balanced Acc | Normal (N) F1   | PVC (V) Rec   | PVC (V) F1    |
+------------------------------------+------------+--------------+-----------------+---------------+---------------+
| Majority Class Baseline (Always N) | 89.93%     | 50.00%       | 0.9470          |  0.00%        | 0.0000        |
| Balanced Logistic Regression       | 97.68%     | 97.36%       | 0.9870          | 96.95%        | 0.8938        |
| Balanced Random Forest (100 trees) | 97.22%     | 96.00%       | 0.9844          | 94.48%        | 0.8724        |
+------------------------------------+------------+--------------+-----------------+---------------+---------------+
```

#### Confusion Matrices on Held-Out Test Patients (N=15,305 beats):
- **Logistic Regression:**
  ```
  [[13456 Normal Correct,   308 False PVCs ],
   [   47 Missed PVCs   ,  1494 PVC Correct]]
  ```
  *PVC Sensitivity:* **96.95%** (1,494 / 1,541). *PVC Precision:* **82.91%**.
- **Random Forest:**
  ```
  [[13423 Normal Correct,   341 False PVCs ],
   [   85 Missed PVCs   ,  1456 PVC Correct]]
  ```
  *PVC Sensitivity:* **94.48%** (1,456 / 1,541). *PVC Precision:* **81.02%**.

### 5.4 Per-Record Breakdown on Held-Out Test Set (Random Forest)

| Test Record ID | Total Beats | Normal Beats | True PVCs (`V`) | Record Accuracy | PVC Sensitivity (Recall) | PVC Precision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **105** | 2,567 | 2,526 | 41 | 92.4% | 95.1% | 16.8% (False pos from baseline drift) |
| **112** | 2,537 | 2,537 | 0 | 99.8% | N/A (0 PVCs) | N/A (Only 5 false alarms) |
| **118** | 16 | 0 | 16 | 62.5% | 62.5% (10/16) | 100.0% |
| **200** | 2,568 | 1,742 | 826 | 99.3% | **98.3%** (812/826) | **99.6%** |
| **205** | 2,642 | 2,571 | 71 | 100.0% | **100.0%** (71/71) | **100.0%** |
| **207** | 105 | 0 | 105 | 99.0% | **99.0%** (104/105) | 100.0% |
| **214** | 256 | 0 | 256 | 77.0% | 76.9% (197/256) | 100.0% |
| **217** | 406 | 244 | 162 | 89.9% | **98.8%** (160/162) | 80.4% |
| **219** | 2,146 | 2,082 | 64 | 96.8% | **98.4%** (63/64) | 48.5% |
| **222** | 2,062 | 2,062 | 0 | 98.4% | N/A (0 PVCs) | N/A (33 false alarms) |

---

## 6. Limitations and Unresolved Questions

1. **R-Peak Alignment Dependence:**
   The research prototype evaluates beats centered on reference annotations. In the physical Raspberry Pi pipeline, beats must be detected dynamically via Pan-Tompkins. If an ectopic QRS complex has an atypical derivative or wide biphasic morphology, Pan-Tompkins peak detection can jitter by $\pm 20\text{--}40\text{ ms}$, distorting the extracted 180-sample window.
2. **Confounding with Bundle Branch Blocks (LBBB / RBBB):**
   In this benchmark, `L` (8,075 beats) and `R` (7,259 beats) were excluded. In real clinical scenarios, LBBB produces wide QRS complexes ($>120\text{ ms}$) on every supraventricular beat, which can trigger false-positive PVC classifications unless patient-specific baselines are established.
3. **Hardware Lead-Off Tripping Thresholds:**
   Because the benchtop ADS1292R driver does not read status frame lead-off flags, the exact physical resistance trip points ($6\text{ nA}$ vs. $22\text{ nA}$ comparator thresholds) remain unverified on physical hardware.

---

## 7. Recommended Next Actions

1. **Hardware Bench Verification (Pi + ADS1292R):**
   Before changing production code in `deploy/`, connect the physical Raspberry Pi to an ADS1292R with test resistors ($100\,\Omega$, $1\text{ k}\Omega$, $10\text{ k}\Omega$, $100\text{ k}\Omega$, open circuit) and run a diagnostic probe script to verify how `LOFF_STAT` bits behave in continuous `RDATAC` mode.
2. **Maintain Production Pipeline Freeze:**
   Keep the production Method C temporal consensus, 10-second inference interval, and ONNX models untouched.
3. **Research Prototype Archiving:**
   Keep [`training/research/pvc_beat_prototype.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/research/pvc_beat_prototype.py) as an isolated research benchmark for beat-level ectopy tracking.

---

## 8. Repository File Status & Integrity Verification

### Git Status (`git status --short`)
```
?? DATASET_ARRHYTHMIA_FEASIBILITY_REPORT.md
?? NEXT_STEPS_RESEARCH_REPORT.md
?? training/research/audit_mitbih.py
?? training/research/pvc_beat_prototype.py
?? training/research/test_pvc_prototype.py
```

### Unchanged Production Integrity Confirmation
- [`deploy/ads1292r_driver.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/ads1292r_driver.py): **Strictly identical to HEAD.**
- [`deploy/combined_ecg_app.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/combined_ecg_app.py): **Strictly preserved.**
- [`deploy/models/`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/deploy/models/): **All ONNX models unchanged.**
- Signal-quality thresholds, SPI bus settings, and 3-model cascade logic remain **100% intact**.
- **No Git commits or pushes were executed.**
