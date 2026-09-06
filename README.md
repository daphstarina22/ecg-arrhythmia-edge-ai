# ECG Arrhythmia Edge AI

> **Real-Time 3-Model Hierarchical Arrhythmia Classification & Emergency Triage on Embedded Edge Devices (Raspberry Pi)**

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%20%7C%20Linux%20%7C%20macOS%20%7C%20Windows-blue)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-green)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%201.16+-orange)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

### Important Medical & Safety Disclaimer

> [!CAUTION]
> **This project is a research and educational prototype and is not a certified medical device. It must not be used for clinical diagnosis, patient monitoring in life-critical settings, or autonomous defibrillation decisions.**
> 
> All machine learning models, code, and thresholds provided herein are intended solely for academic research, algorithmic benchmarking, and edge computing education.

---

## 1. Project Overview

This repository provides an ultra-lightweight, real-time edge AI system for automated electrocardiogram (ECG) arrhythmia screening. Given raw **5-second ECG windows (1,800 samples at 360 Hz)**, the system performs zero-phase bandpass filtering, feature extraction, and hierarchical machine learning triage in **< 15 milliseconds** on a low-power single CPU core (Raspberry Pi).

The pipeline addresses real-world clinical constraints:
- **Zero Patient Data Leakage:** Evaluated using strict record-level group cross-validation (`StratifiedGroupKFold`).
- **Domain-Shift Aware:** Accounts for differences between bedside ICU false alarms (Computing in Cardiology Challenge 2015) and ambulatory Holters (MIT-BIH, VFDB).
- **Physical Feature Selection:** Prunes inverted and source-confounded features to retain only domain-stable complexity and autocorrelation dynamics.

---

## 2. 3-Model Hierarchical Cascade Architecture

Rather than forcing a single model to solve both broad triage and fine subtype differentiation, the architecture uses a 3-tier cascade:

```text
               Input 5-Second Raw ECG (1,800 samples @ 360 Hz)
                                     │
                 Bandpass Filter (0.5 – 40 Hz Butterworth)
                                     │
               Extract 14 Unified Spectral & Temporal Features
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
      Tier 1: Shockable Screener         Tier 2: Multi-Arrhythmia 4-Class
      (Model 1 - 10 features)            (Model 2 - 12 features)
      P(Shockable) vs Non-Shockable      [NSR, TACHY, BRADY_ASY, VENTRICULAR]
                     │                               │
                     └───────────────┬───────────────┘
                                     │
             Is Rhythm Shockable OR Class == "VENTRICULAR"?
                            ├── NO  ──► Finalize Non-Shockable Triage
                            │           (NSR / SVTA / Bradycardia)
                            └── YES
                                     │
                                     ▼
                     Tier 3: Ventricular Subtype Specialist
                     (Model 3 - 6 Domain-Stable Features)
                     VT (Tachycardia) vs. VF/VFL (Fibrillation / Flutter)
                                     │
                  Morphology & Artifact Guardrails Check
                     (Equivocal Subtype & Noise Warnings)
                                     │
                                     ▼
                     Integrated Clinical Triage Decision
                     (Emergency Priority, Shock Advisory, Action)
```

### The 3 Models
1. **Model 1: Shockable Screener (`shockable_classifier_candidate.onnx` - 413 KB)**
   - *Input:* 10 features (QRS-independent chaos & spectral power).
   - *Task:* Binary shockable screening ($0 = \text{Non-Shockable}$, $1 = \text{Shockable}$).
   - *Deployment Status:* **APPROVED FOR RESEARCH DEMO**.
2. **Model 2: Multi-Arrhythmia 4-Class (`arrhythmia_multiclass_4class_candidate.onnx` - 1.28 MB)**
   - *Input:* 12 features (timing, rhythm regularity, morphology, spectral bands).
   - *Task:* 4-class categorization: `NSR`, `TACHY` (Supraventricular Tachycardia), `BRADY_ASY` (Bradycardia/Asystole), `VENTRICULAR`.
   - *Deployment Status:* **APPROVED FOR RESEARCH DEMO**.
3. **Model 3: Ventricular Subtype Specialist (`vf_vt_subtype_classifier_candidate.onnx` - 199 KB)**
   - *Input:* 6 domain-stable features (`lz_complexity`, `ac_decay_time`, `ac_zero_crossing_lag`, `ac_max_peak_ratio`, `sample_entropy`, `hjorth_mobility`).
   - *Task:* Differentiates monomorphic Ventricular Tachycardia (`VT`) from chaotic Ventricular Fibrillation / Flutter (`VF/VFL`).
   - *Deployment Status:* **RESEARCH ONLY** (due to historical archive limit of $N=17$ physical VF patients).

---

## 3. Repository Structure

```text
ecg-arrhythmia-edge-ai/
├── deploy/                               # STANDALONE RASPBERRY PI DEPLOYMENT PACKAGE
│   ├── README_PI.md                      # Detailed hardware & OS guide for Raspberry Pi
│   ├── requirements-pi.txt               # Minimal edge dependencies (numpy, scipy, onnxruntime)
│   ├── verify_deployment.py              # Automated hardware & graph self-test script
│   ├── edge_runtime.py                   # Self-contained edge inference & benchmark runner
│   ├── run_demo.sh                       # One-click POSIX shell test script
│   └── models/                           # Final trained ONNX deployment models (~1.9 MB total)
│       ├── shockable_classifier_candidate.onnx
│       ├── arrhythmia_multiclass_4class_candidate.onnx
│       └── vf_vt_subtype_classifier_candidate.onnx
│
├── demo_edge_pipeline.py                 # Interactive clinical demonstration on real archives
├── requirements.txt                      # General Python dependencies
├── README.md                             # Main project documentation (this file)
│
├── training/                             # RESEARCH & TRAINING CODEBASE (Isolated)
│   ├── final_models/                     # Final controlled evaluation & dataset builders
│   │   ├── FINAL_MODEL_REPORT.md         # Comprehensive scientific evaluation report
│   │   ├── export_candidates/            # Canonical exported ONNX models
│   │   ├── results/                      # 5-fold CV & cross-database comparison tables
│   │   ├── model1_shockable/             # Model 1 training script
│   │   ├── model2_multiclass/            # Model 2 training script
│   │   └── model3_vt_vf/                 # Model 3 training script
│   ├── model3_phase6/                    # Phase 6 MIT-BIH & CUDB dataset integration audits
│   └── model3_phase5/                    # Phase 5 physiological feature extraction & audit
│
├── models/                               # Legacy baseline ONNX models
└── tests/                                # Unit & contract smoke tests
```

---

## 4. Quick Start on PC / Mac / Linux

### 1. Clone & Set Up Virtual Environment
```bash
git clone <REPOSITORY_URL>
cd ecg-arrhythmia-edge-ai
python3 -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Clinical Cascade Demonstration
Test the cascade on verified clinical episodes extracted from MIT-BIH, VFDB, and C2015:
```bash
python demo_edge_pipeline.py
```

### 4. Run the Edge Throughput Benchmark
```bash
python demo_edge_pipeline.py --benchmark
```

---

## 5. Raspberry Pi Deployment

The [`deploy/`](deploy/) folder is completely self-contained and ready to run on Raspberry Pi OS:

### 1. Install System Dependencies on the Pi
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv libopenblas-dev
```

### 2. Install Edge Python Packages
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r deploy/requirements-pi.txt
```
*(Only `numpy`, `scipy`, and `onnxruntime` are installed — no heavy training frameworks or compilers).*

### 3. Run One-Click Self-Test & Demo
```bash
chmod +x deploy/run_demo.sh
./deploy/run_demo.sh
```

For complete instructions on wiring an **AD8232 ECG sensor** + **ADS1115 ADC**, setting up a continuous ring-buffer, or enabling a `systemd` background service on boot, see [**`deploy/README_PI.md`**](deploy/README_PI.md).

---

## 6. Hardware Benchmarks & Performance Summary

Evaluated on single CPU core (AMD Ryzen 7 / ARM Cortex-A72 Raspberry Pi 4):

| Task / Stage | Input Dimension | Model Architecture | Average Latency | Peak RAM Footprint | Storage Size |
|---|---|---|---|---|---|
| **Feature Extraction** | 1,800 raw samples | Bandpass + Welch + LZ + Autocorr | ~9.5 – 12.0 ms | < 15 MB | — |
| **Tier 1: Shockable** | 10 features | Random Forest (100 trees, d=6) | ~0.20 ms | < 2.5 MB | 413 KB |
| **Tier 2: 4-Class** | 12 features | Random Forest (100 trees, d=7) | ~0.15 ms | < 5.0 MB | 1.28 MB |
| **Tier 3: VT vs VF** | 6 features | Random Forest (100 trees, d=5) | ~0.13 ms | < 1.8 MB | 199 KB |
| **TOTAL CASCADE** | **1,800 samples (5s)** | **3-Tier Hierarchical AI** | **~10.5 – 13.5 ms** | **< 60 MB** | **~1.9 MB** |

**Speedup:** Runs in ~11 ms for a 5,000 ms window (**>450x real-time speedup**), consuming < 3% of available single-core CPU budget.

---

## 7. Model Evaluation Summary & Clinical Limitations

For full statistical methodology, cross-database generalization matrices, and confusion matrices, refer to [**`training/final_models/FINAL_MODEL_REPORT.md`**](training/final_models/FINAL_MODEL_REPORT.md).

### Summary Table
| Model | Evaluated Task | Window Bal Acc | Record Bal Acc | Deployment Status | Key Clinical Limitation |
|---|---|---|---|---|---|
| **Model 1** | Shockable vs. Non-Shockable | **60.2%** | **55.9%** | **APPROVED FOR RESEARCH DEMO** | Asymmetric ICU alarm vs. Holter domain shift (requires noise-calibrated thresholds). |
| **Model 2** | 4-Class (NSR, Tachy, Brady, Ventricular) | **55.8%** | **57.9%** | **APPROVED FOR RESEARCH DEMO** | Borderline rate overlap between sinus tachycardia and normal sinus rhythm. |
| **Model 3** | VT vs. VF Subtype (6 Domain-Stable Features) | **81.5%** | **88.8%** | **RESEARCH ONLY** | Strict historical physical bottleneck ($N=17$ independent confirmed VF patients across archives). |

---

## 8. License & Acknowledgments

- **License:** MIT License.
- **Data Acknowledgments:** Datasets provided by PhysioNet (Computing in Cardiology Challenge 2015, MIT-BIH Arrhythmia Database, MIT-BIH Malignant Ventricular Ectopy Database, and Creighton University Database).
