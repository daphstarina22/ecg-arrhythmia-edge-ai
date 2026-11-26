# ECG Arrhythmia Edge AI: Read-Only Project Completion Audit
**System Safety, Architecture Verification, and AFib Integration Feasibility Analysis**

**Date:** October 26, 2026  
**Auditor:** Senior Edge-AI Systems & Biomedical ML Engineering Agent  
**Scope:** Read-Only Audit of Entire Repository (`c:\Users\LENOVO\Downloads\ecg-arrhythmia-edge-ai`)  
**Deployment Target Reality:** Raspberry Pi 4B+ (ARM Cortex-A72 Linux SBC) + TI ADS1292R SPI Analog Front-End  

---

## 1. Executive Summary

This project completion audit examines the technical maturity, hardware reality, runtime functionality, and safety status of the **ECG Arrhythmia Edge AI** codebase, with special focus on whether the experimental 5-second Atrial Fibrillation (AFib) classifier is safe and technically compatible for integration.

### High-Level Audit Findings:
1. **Target Hardware Reality vs. Misconceptions**:
   - The repository **contains zero STM32 Nucleo firmware, zero C/C++ HAL drivers, zero CMSIS-NN/DSP libraries, and zero microcontroller project files**.
   - The **actual, physical, tested hardware deployment target is a Raspberry Pi 4B+** running 64-bit Linux OS, Python 3, `spidev`, `RPi.GPIO`, `scipy`, `numpy`, and `onnxruntime`.
   - Hardware signal acquisition from the Texas Instruments ADS1292R 24-bit ECG Analog Front-End has been **physically executed, calibrated, and verified on Raspberry Pi hardware** (yielding 357.96 SPS, PTP 1.748 mV, zero non-finite samples).
2. **Core 3-Tier Production Cascade Maturity**:
   - The primary deployment pipeline in `deploy/` (`combined_ecg_app.py`, `edge_runtime.py`, `verify_deployment.py`) is **functional, tested, and ready for edge demonstration**.
   - It runs a 3-tier hierarchical cascade:
     - **Tier 1:** Shockable Screener (`shockable_classifier_candidate.onnx`, 10 features, 413 KB)
     - **Tier 2:** 4-Class Arrhythmia Classifier (`arrhythmia_multiclass_4class_candidate.onnx`, 12 features, 1.27 MB: NSR, Tachycardia, Bradycardia/Asystole, Ventricular)
     - **Tier 3:** Ventricular Specialist (`vf_vt_subtype_classifier_candidate.onnx`, 6 features, 199 KB: VT vs VF)
   - Benchmark execution time on Raspberry Pi is **$19.8\text{ ms}$ per 5-second window** ($>250\times$ real-time speedup, $<1\%$ CPU utilization).
3. **AFib Experimental Branch Integration Verdict**:
   - **Production Clinical Alerts:** **NO (REJECTED)**. Severe false alarm rate ($101.54\text{ FP/hr}$), poor generalization on outlier patients (ROC-AUC $0.5841$), and vulnerability to supraventricular/ectopic confounders make it clinically unsafe for autonomous user-facing alerts.
   - **Isolated Research / Demo Mode:** **CONDITIONAL YES**. Feasible as an opt-in, non-alerting, clearly disclaimed telemetry badge in the Flask dashboard.
   - **STM32 Nucleo Deployment:** **NO (INCOMPATIBLE)**. Zero embedded C firmware exists; massive software dependency gap.
   - **Host / Python App Execution:** **CONDITIONAL YES**. Requires handling a sampling rate mismatch ($250\text{ Hz}$ AFib vs. $360\text{ Hz}$ live acquisition) and non-blocking asynchronous execution.

---

## 2. Comprehensive Inventory of Repository Components

| Directory / File | Description | Execution Environment | Status / Hardware Validation |
| :--- | :--- | :--- | :--- |
| **`deploy/combined_ecg_app.py`** | Unified edge acquisition, signal quality diagnostics, 3-model cascade, and Flask web dashboard (`:5000`). | Raspberry Pi 4B+ Linux (Python 3.10+) | **EXECUTED ON HARDWARE**. Verified with live ADS1292R streaming and interactive browser UI. |
| **`deploy/ads1292r_driver.py`** | Physical SPI driver for TI ADS1292R (24-bit ADC, $V_{\text{ref}}=2.42\text{V}$, Gain=6, Mode 1, $360\text{ Hz}$). | Raspberry Pi 4B+ Linux (`spidev`, `RPi.GPIO`) | **EXECUTED ON HARDWARE**. Hardware verified via `deploy/test_ads1292r_live.py`. |
| **`deploy/edge_runtime.py`** | Lightweight, headless CLI edge runtime for the 3-model cascade (NumPy, SciPy, ONNX Runtime only; no Flask). | ARM Linux SBC or PC Host | **TESTED & FUNCTIONAL**. Executes inference on live or synthetic buffers. |
| **`deploy/verify_deployment.py`** | Automated self-test script verifying dependencies, model integrity, and synthetic inference. | Python 3.10+ (Any host) | **TESTED & VERIFIED**. Executed during audit; all checks and synthetic inferences passed. |
| **`deploy/models/`** | Exported ONNX candidate models for the 3-tier cascade (`shockable`, `arrhythmia_multiclass_4class`, `vf_vt_subtype`). | ONNX Runtime CPU | **VERIFIED**. Load cleanly with zero graph errors; inference latency $<1\text{ ms}$ each. |
| **`models/` (Root)** | Older/legacy exports (`arrhythmia_multiclass.onnx` is 3.75 MB vs 1.27 MB candidate). | ONNX Runtime | **LEGACY / SUPERSEDED**. Preserved for historical compatibility. |
| **`training/research/afib_5s_classifier/`** | Research pipeline for 5-second AFib detection (AFDB dataset, 250 Hz, 11 absolute features). | Offline Python Research | **RESEARCH ONLY**. Did not generalize on held-out test (ROC-AUC 0.6615). |
| **`training/research/.../patient_adaptive_baseline_study/`** | Causal EMA moving baseline tracker and combined 22-feature classifier (Config C). | Offline Python Research | **RESEARCH ONLY**. Improved held-out ROC-AUC to 0.9015, but produced 101.54 FP/hr. |
| **`training/research/.../temporal_persistence_study/`** | Causal consecutive-window persistence filtering study (`consecutive_2`, `majority_3_of_4`, etc.). | Offline Python Research | **RESEARCH ONLY**. Reduced false alarms on validation data by 55.7%, but unvalidated on new cohorts. |
| **`src/`** | Early project prototypes (`main.py`, `features.py`, `inference.py`, `preprocessing.py`). | PC Host | **SUPERSEDED** by `deploy/edge_runtime.py`. |
| **`training/model3_phase*/`** | Model 3 (VT vs VF) development phases and domain-generalization benchmarks. | Offline Research | **COMPLETED & FROZEN**. Candidates exported to `deploy/models/`. |

---

## 3. Detailed Hardware & Deployment Architecture Analysis

### 3.1 The Microcontroller / STM32 Nucleo Discrepancy
A critical question in this audit is whether components are intended for or runnable on the **STM32 Nucleo**:
1. **Source Code Inspection**:
   - A global regex search across all files revealed **zero instances** of STM32 code, STM32Cube configuration (`.ioc`), Keil/IAR projects, ARM GCC Makefiles, or C/C++ embedded drivers.
   - Mentions of "microcontroller" in the documentation appear strictly as aspirational theoretical targets (e.g., noting that compact 4-feature sets could fit on Cortex-M4).
2. **Actual Hardware Architecture**:
   - The entire embedded acquisition system is built around Linux user-space APIs:
     ```python
     import spidev
     import RPi.GPIO as GPIO
     ```
   - These libraries interact with the Linux kernel `/dev/spidev0.0` device node and Raspberry Pi BCM memory-mapped peripheral registers.
3. **Execution Reality**:
   - The existing code **CANNOT run on an STM32 Nucleo**.
   - Running on an STM32 Nucleo would require a complete from-scratch firmware implementation in C/C++:
     - Writing bare-metal or FreeRTOS drivers using STM32 HAL SPI and GPIO EXTI.
     - Porting complex digital signal processing (Welch PSD, Pan-Tompkins derivative/integration, Hjorth parameters, Lempel-Ziv complexity) to C/C++ using CMSIS-DSP.
     - Quantizing and compiling ONNX models to C code via STM32Cube.AI / X-CUBE-AI.
   - Therefore, the project's true edge deployment target is **Raspberry Pi 4B+**, not STM32.

### 3.2 Physical Hardware Execution Evidence
Unlike code that merely exists in a repository, the ADS1292R acquisition subsystem has been physically verified on hardware:
- **Verified SPI Configuration**: SPI0, Mode 1 ($\text{CPOL}=0, \text{CPHA}=1$), 1 MHz clock, Active-low `DRDY` interrupt on BCM GPIO 17, `START` on GPIO 27, `PWDN` on GPIO 22.
- **Calibrated Physical Transfer**: 24-bit 2's complement words decoded with $V_{\text{ref}}=2.42\text{V}$ and Gain=6 to calibrated physical millivolts ($\text{mV}$).
- **Hardware Telemetry Evidence**: `ads1292r_hardware_verification.csv` documents a 10.0-second live benchtop acquisition producing $3,580\text{ samples}$ at an empirical sample rate of $357.96\text{ SPS}$ (within $0.57\%$ of nominal $360\text{ Hz}$ target) with zero non-finite samples and active QRS complexes (PTP $1.748\text{ mV}$).

---

## 4. Explicit AFib Classifier Integration Decision

The experimental 5-second AFib classifier branch was evaluated through multiple research phases:
- **Phase 1 (Original Absolute Classifier)**: Held-out test ROC-AUC collapsed to **0.6615** (Validation was 0.9832).
- **Phase 2 (Causal Patient-Adaptive Classifier, Config C)**: Held-out test ROC-AUC recovered to **0.9015**, with window sensitivity $81.09\%$ and specificity $85.90\%$, but suffered **101.54 false alarms per eligible non-AFib hour** at threshold 0.560 ($122.95\text{ FP/hr}$ at threshold 0.500).
- **Phase 3 (Temporal Persistence Filtering)**: Validation evaluation of a 2-window consecutive rule (`consecutive_2`) reduced false alarm events by $55.7\%$ (from 940 to 416), but has **not yet been independently evaluated on a new patient cohort**.

### Decision Matrix Across the Four Scenarios:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             AFIB INTEGRATION DECISION                            │
├─────────────────────────────────────────────────┬────────────────────────────────┤
│ Target Scenario                                 │ Decision                       │
├─────────────────────────────────────────────────┼────────────────────────────────┤
│ A. Production detection & user clinical alerts  │ NO (UNSAFE / REJECTED)         │
│ B. Isolated research or demonstration mode      │ CONDITIONAL YES (RESEARCH-ONLY)│
│ C. Deployment on STM32 Nucleo                   │ NO (TECHNICALLY INCOMPATIBLE)  │
│ D. Execution in Python app / external host      │ CONDITIONAL YES (ASYNC STREAM) │
└─────────────────────────────────────────────────┴────────────────────────────────┘
```

---

### Detailed Rationale for Each Option:

### Option A: Production Detection & User-Facing Clinical Alerts
- **Verdict:** **NO (UNSAFE / REJECTED)**
- **Technical & Clinical Evidence**:
  1. **Severe Alarm Fatigue**: A false alarm rate of **$101.54\text{ false alarms per hour}$** corresponds to an alarm every **35.5 seconds** during normal sinus rhythm. In clinical and home-monitoring environments, false alarm rates of this magnitude lead to immediate user alarm fatigue, device abandonment, or dangerous suppression of alerts.
  2. **Inter-Patient Fragility**: On held-out patient `08215`, the adaptive classifier achieved an ROC-AUC of only **0.5841** and a window sensitivity of **59.45%**, failing to detect over $40\%$ of AFib windows.
  3. **Confounder Misclassification**: In held-out patient `04908`, **$90.9\%$ of Atrial Flutter (AFL) windows were misclassified as AFib**. In validation patient `05121`, ventricular ectopy triggered continuous false alarm bursts ($90.50\text{ FP/hr}$ even with persistence).
  4. **Unvalidated Persistence in Test Cohort**: While the `consecutive_2` filter showed a $55.7\%$ reduction on validation data, it has **not been tested on an unexposed external test cohort**. Deploying an unvalidated filter into production violates core safety and clinical engineering principles.

### Option B: Isolated Research or Demonstration Mode
- **Verdict:** **CONDITIONAL YES**
- **Permissible Scope & Mandatory Constraints**:
  1. **Strictly Opt-In Feature Flag**: Must be disabled by default. Accessible only via an explicit command-line flag (e.g. `--enable-experimental-afib`) or hidden developer settings menu.
  2. **Prominent Disclaimers**: The UI must display an unmissable banner:
     > `"CAUTION: EXPERIMENTAL AFIB RESEARCH PROTOCOL — NOT CLINICALLY VALIDATED — NOT FOR DIAGNOSTIC USE"`.
  3. **Zero Clinical Actionability**: The classifier must **never** sound alarms, change the primary emergency triage status (`triage_level`), or recommend medication/intervention. It should output only informational probability graphs and telemetry logs for edge computing demonstrations.

### Option C: Deployment on the STM32 Nucleo
- **Verdict:** **NO (TECHNICALLY INCOMPATIBLE)**
- **Technical Evidence**:
  1. **Total Absence of Embedded C Firmware**: The repository does not contain an STM32 project, HAL code, or embedded build toolchain.
  2. **Incompatible Software Stack**: The AFib pipeline relies on Python standard libraries, SciPy signal processing, and NumPy.
  3. **Implementation Cost**: Porting the 22-feature extraction pipeline, causal EMA tracking state machine, and inference engine to bare-metal C on a Cortex-M4/M33 Nucleo would require significant re-engineering that currently does not exist in the codebase.

### Option D: Execution in the Existing Python Application or External Host
- **Verdict:** **CONDITIONAL YES (AS A DECOUPLED RESEARCH STREAM)**
- **Required Architecture Adaptations**:
  1. **Sampling Rate Normalization**: The live acquisition pipeline in `deploy/combined_ecg_app.py` streams at **$360\text{ Hz}$** (geared for Model 1, Model 2, and Model 3). The AFib classifier was developed on **$250\text{ Hz}$** (from AFDB). To execute in the Python app, the 5-second buffer must either be downsampled via polyphase filtering from $360\text{ Hz} \to 250\text{ Hz}$ ($1,800 \to 1,250\text{ samples}$), or the AFib feature extractor must be retrained at $360\text{ Hz}$.
  2. **Non-Blocking Asynchronous Thread**: Feature extraction for 22 adaptive features (including sample entropy and autoregressive baseline tracking) must run in a separate low-priority worker thread so as not to block the **$19.8\text{ ms}$ real-time latency budget** of the primary life-critical shockable/arrhythmia cascade.
  3. **Isolated JSON Telemetry**: Outputs must be transmitted on a separate API endpoint (`/api/research/afib`) completely segregated from the primary `/api/status` endpoint used by the clinical dashboard.

---

## 5. Comprehensive Summary of Project Readiness

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PROJECT READINESS SCORECARD                               │
├───────────────────────────────────┬──────────────┬─────────────────────────────────────┤
│ Component                         │ Status       │ Operational Readiness               │
├───────────────────────────────────┼──────────────┼─────────────────────────────────────┤
│ ADS1292R SPI Hardware Acquisition │ VERIFIED     │ Production ready on Raspberry Pi 4B │
│ Signal Quality Diagnostic Layer   │ VERIFIED     │ Production ready (Flat/Sat checks)  │
│ Tier 1: Shockable Screener (ONNX) │ VERIFIED     │ Edge ready (< 1 ms latency)         │
│ Tier 2: 4-Class Arrhythmia (ONNX) │ VERIFIED     │ Edge ready (< 1 ms latency)         │
│ Tier 3: VT vs VF Specialist (ONNX)│ VERIFIED     │ Edge ready (< 1 ms latency)         │
│ Combined Flask Web Dashboard      │ VERIFIED     │ Edge ready (Port 5000)              │
│ Headless CLI Runtime              │ VERIFIED     │ Edge ready (`edge_runtime.py`)      │
│ STM32 Nucleo Firmware             │ NOT PRESENT  │ Not implemented / Incompatible      │
│ Experimental 5s AFib Classifier   │ RESEARCH ONLY│ Unsafe for clinical production      │
└───────────────────────────────────┴──────────────┴─────────────────────────────────────┘
```

---

## 6. Recommended Next Steps to Bring the Project to Clean Completion

1. **Maintain Strict Separation of Production and Research**:
   - Keep `deploy/` frozen with the verified 3-tier cascade (`shockable`, `arrhythmia_multiclass_4class`, `vf_vt_subtype`).
   - Keep `training/research/` designated as offline exploratory studies.
2. **Execute Preregistered Independent AFib Evaluation (Optional Future Phase)**:
   - If AFib research is pursued further, execute the preregistered evaluation plan on the unexposed **MIT-BIH Arrhythmia Database (MITDB)** cohort as specified in `PREREGISTERED_INDEPENDENT_EVALUATION_PLAN.md`.
   - Never evaluate or tune against the four historical AFDB test records (`04908`, `07162`, `08215`, `08219`).
3. **Hardware Documentation Realignment**:
   - Clarify project documentation to explicitly state that the **Raspberry Pi 4B+** is the validated deployment hardware, and that STM32 Nucleo represents an unexecuted future embedded target.

*Report finalized and approved for project completion audit.*
