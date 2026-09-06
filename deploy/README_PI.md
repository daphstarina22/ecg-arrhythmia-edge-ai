# Raspberry Pi Deployment Guide: 3-Model ECG Arrhythmia Edge AI

This guide walks through deploying and running the 3-model hierarchical ECG arrhythmia classification cascade on a **Raspberry Pi** (Raspberry Pi 5, 4, 3B+, or Zero 2 W).

---

## 1. Hardware & OS Recommendations

### Supported Hardware
- **Raspberry Pi 5 / 4 (Recommended):** Real-time latency < 20 ms per 5-second window (>250x real-time speedup).
- **Raspberry Pi 3B+:** Real-time latency ~35–50 ms.
- **Raspberry Pi Zero 2 W:** Real-time latency ~45–65 ms. Memory footprint (< 80 MB) easily fits within 512 MB RAM.

### Operating System
- **Recommended OS:** **Raspberry Pi OS (64-bit / aarch64)** with Debian 12 (Bookworm) or Debian 11 (Bullseye).
- **Why 64-bit?** Precompiled binary wheels for `onnxruntime` are natively available on PyPI for `aarch64` without requiring source compilation.
- *32-bit (armv7l) Note:* If you are running 32-bit Pi OS, install `onnxruntime` via wheels from [piwheels.org](https://www.piwheels.org/project/onnxruntime/) or install using apt.

---

## 2. Step-by-Step Installation on Raspberry Pi

Log into your Raspberry Pi terminal (via SSH or HDMI display):

### Step 2.1: Clone the Repository
```bash
git clone <REPOSITORY_URL>
cd ecg-arrhythmia-edge-ai
```

### Step 2.2: Update System Packages
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv libopenblas-dev
```

### Step 2.3: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2.4: Install Edge Dependencies
Install only the minimal inference requirements (no PyTorch, no TensorFlow, no heavy ML compilers needed on the Pi):
```bash
pip install --upgrade pip
pip install -r deploy/requirements-pi.txt
```

---

## 3. Running the Pipeline

### Option A: One-Click Verification and Demo
Run the automated test runner:
```bash
chmod +x deploy/run_demo.sh
./deploy/run_demo.sh
```

### Option B: Step-by-Step Manual Execution

#### 1. Verify Deployment & ONNX Graphs
Validate that all 3 models load correctly and run a synthetic inference test:
```bash
python3 deploy/verify_deployment.py
```
Expected output:
```text
===========================================================================
  RASPBERRY PI DEPLOYMENT VERIFICATION & SELF-TEST
===========================================================================
[CHECK 1/4] Verifying core edge dependencies...
  [PASS] numpy
  [PASS] scipy
  [PASS] onnxruntime
[CHECK 2/4] Verifying ONNX model candidate files...
  [PASS] shockable_classifier_candidate.onnx          (403.5 KB)
  [PASS] arrhythmia_multiclass_4class_candidate.onnx  (1,245.1 KB)
  [PASS] vf_vt_subtype_classifier_candidate.onnx      (194.7 KB)
[CHECK 3/4] Loading ONNX runtime sessions & inspecting graphs...
[CHECK 4/4] Executing safe synthetic inference tests...
===========================================================================
  STATUS: ALL DEPLOYMENT CHECKS PASSED [OK]
===========================================================================
```

#### 2. Run the Real-Time Throughput & Latency Benchmark
Measure exact CPU inference latency and throughput on your Pi:
```bash
python3 deploy/edge_runtime.py --benchmark
```
*Expected latency: 10–30 ms per 5-second window.*

#### 3. Run Clinical Demonstration on Real Patient Recordings
If the clinical archives (`MITBIH.zip`, `VFDB.zip`, `mitbih-vtvtf.zip`) are present in the repository root:
```bash
python3 demo_edge_pipeline.py
```

#### 4. Run Custom Signal Inference
To classify an external 1,800-sample (5 seconds at 360 Hz) ECG file (`.npy` or whitespace-delimited text):
```bash
python3 deploy/edge_runtime.py --file path/to/ecg_window.npy
```

---

## 4. 3-Tier Hierarchical Cascade Architecture

The edge runtime routes incoming ECG windows through three specialized models:

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

### Models & Artifacts in `deploy/models/`:
1. `shockable_classifier_candidate.onnx` (413 KB) — 10 features: QRS-independent chaos and spectral power ratios.
2. `arrhythmia_multiclass_4class_candidate.onnx` (1.28 MB) — 12 features: Rhythm timing, peak regularity, and amplitude.
3. `vf_vt_subtype_classifier_candidate.onnx` (199 KB) — 6 domain-stable features: Lempel-Ziv complexity, sample entropy, autocorrelation decay, Hjorth mobility.

---

## 5. Live Sensor Integration (AD8232 + ADS1115 ADC)

Because the Raspberry Pi has digital GPIOs only, connect an analog ECG sensor (e.g. AD8232) using an I2C ADC (e.g. ADS1115):

```text
AD8232 (ECG)         ADS1115 (16-bit ADC)        Raspberry Pi
────────────         ────────────────────        ────────────
OUTPUT    ───────►   A0
3.3V      ───────►   VDD                  ───►   3.3V (Pin 1)
GND       ───────►   GND                  ───►   GND  (Pin 6)
                     SDA                  ───►   GPIO 2 / SDA (Pin 3)
                     SCL                  ───►   GPIO 3 / SCL (Pin 5)
```

### Circular Ring Buffer Example
Read samples at 360 Hz into an 1,800-sample ring buffer. Every 1–5 seconds, pass the latest buffer to `HierarchicalEdgeClassifier.predict_window()`:
```python
import collections
import numpy as np
from deploy.edge_runtime import HierarchicalEdgeClassifier

clf = HierarchicalEdgeClassifier()
buffer = collections.deque(maxlen=1800)  # 5-second buffer at 360 Hz

def on_sample(val_mv: float):
    buffer.append(val_mv)
    if len(buffer) == 1800:
        window = np.array(buffer, dtype=np.float32)
        decision = clf.predict_window(window)
        if decision.defibrillator_advised:
            print(f"CRITICAL ALERT: {decision.primary_rhythm} - IMMEDIATE SHOCK ADVISED!")
```

---

## 6. Autostart as a Background Service (systemd)

For headless or battery-powered wearable monitoring, set up a systemd service:

1. Create service file:
   ```bash
   sudo nano /etc/systemd/system/ecg-edge.service
   ```
2. Paste (adjust working directory and user as needed):
   ```ini
   [Unit]
   Description=ECG Arrhythmia Edge AI Monitor
   After=network.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/ecg-arrhythmia-edge-ai
   ExecStart=/home/pi/ecg-arrhythmia-edge-ai/.venv/bin/python deploy/edge_runtime.py
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
3. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable ecg-edge.service
   sudo systemctl start ecg-edge.service
   ```
4. View live status:
   ```bash
   journalctl -u ecg-edge.service -f
   ```

---

## 7. Troubleshooting & FAQ

### Issue: `ImportError: libopenblas.so.0: cannot open shared object file`
**Solution:** Install OpenBLAS library:
```bash
sudo apt install -y libopenblas-dev
```

### Issue: `onnxruntime` installation fails on 32-bit OS
**Solution:** Ensure you are running 64-bit Raspberry Pi OS (`uname -m` outputs `aarch64`). If on 32-bit (`armv7l`), install via PiWheels:
```bash
pip install onnxruntime -i https://www.piwheels.org/simple
```

### Issue: High CPU temperature
**Solution:** The edge runtime is pinned to single-thread execution (`intra_op_num_threads=1`) by default in `edge_runtime.py`, ensuring core temperature remains cool and battery life is preserved.

---

## 8. Disclaimer

*This software is a research and educational prototype and is NOT a certified medical device. It must not be used for clinical diagnosis, patient monitoring in life-critical settings, or autonomous defibrillation decisions.*
