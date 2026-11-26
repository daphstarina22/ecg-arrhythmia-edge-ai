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

## 5. Live Sensor Integration

The Raspberry Pi does not possess onboard analog-to-digital converter (ADC) pins. Below are instructions for the two most common ECG acquisition hardware configurations.

### Option A: Texas Instruments ADS1292R (Recommended 24-bit SPI Medical AFE)

The **ADS1292R** is an integrated analog front-end with 24-bit delta-sigma ADCs, built-in Right Leg Drive (RLD), and lead-off detection. It is significantly higher fidelity than basic hobbyist sensors.

#### 1. Pin Connections (SPI)
| ADS1292R Pin | Raspberry Pi 40-Pin Header | Description |
|---|---|---|
| **VCC / 3.3V** | Pin 1 (3.3V Power) | Power supply |
| **GND** | Pin 6 (Ground) | Power ground |
| **SCLK** | Pin 23 (GPIO 11 / SPI_CLK) | SPI Clock |
| **MOSI / DIN** | Pin 19 (GPIO 10 / SPI_MOSI) | SPI Data In |
| **MISO / DOUT** | Pin 21 (GPIO 9 / SPI_MISO) | SPI Data Out |
| **CS / SS** | Pin 24 (GPIO 8 / SPI_CE0) | Chip Select |
| **DRDY** | Pin 22 (GPIO 25) | Data Ready (Active LOW) |
| **START** | Pin 1 (3.3V) or GPIO | High to start conversion |
| **PWDN / RESET**| Pin 1 (3.3V) | High via pull-up |

*Note: Ensure SPI is enabled on your Raspberry Pi:*
```bash
sudo raspi-config
# Navigate to: 3 Interface Options -> I4 SPI -> Enable -> Yes
```

#### 2. Sampling Rate & Resampling to 360 Hz
The models in this repository are trained on **360 Hz** ECG (1,800 samples per 5-second window). The ADS1292R natively samples at **500 SPS** (or 250 SPS). Use `scipy.signal.resample_poly` to resample the 5-second window from 500 Hz (2,500 samples) to 360 Hz (1,800 samples):

```python
import collections
import numpy as np
from scipy.signal import resample_poly
from deploy.edge_runtime import HierarchicalEdgeClassifier

# Initialize the 3-model edge classifier
clf = HierarchicalEdgeClassifier()

# 5-second raw buffer at 500 SPS = 2,500 samples
raw_buffer = collections.deque(maxlen=2500)

def parse_24bit_to_mv(b0, b1, b2, vref=2.42, gain=6.0):
    """Converts 3-byte 2's complement SPI reading to millivolts."""
    val = (b0 << 16) | (b1 << 8) | b2
    if val & 0x800000:
        val -= 0x1000000
    # Scale to millivolts (mV)
    return (val / 8388607.0) * (vref / gain) * 1000.0

def on_ads1292r_sample(mv_reading: float):
    raw_buffer.append(mv_reading)
    if len(raw_buffer) == 2500:
        raw_500hz = np.array(raw_buffer, dtype=np.float32)
        
        # Polyphase rational resample: 500 Hz * (18 / 25) = 360 Hz (exactly 1,800 samples)
        window_360hz = resample_poly(raw_500hz, up=18, down=25).astype(np.float32)
        
        # Run 3-model hierarchical classification
        decision = clf.predict_window(window_360hz)
        
        if decision.defibrillator_advised:
            print(f"CRITICAL ALERT: {decision.primary_rhythm} - IMMEDIATE SHOCK ADVISED!")
        else:
            print(f"Triage: {decision.primary_rhythm} | Confidence: {decision.confidence:.2f}")
```

---

### Option B: AD8232 (Analog) + ADS1115 (16-bit I2C ADC)

If using an analog sensor module such as the AD8232:

```text
AD8232 (ECG)         ADS1115 (16-bit ADC)        Raspberry Pi
────────────         ────────────────────        ────────────
OUTPUT    ───────►   A0
3.3V      ───────►   VDD                  ───►   3.3V (Pin 1)
GND       ───────►   GND                  ───►   GND  (Pin 6)
                     SDA                  ───►   GPIO 2 / SDA (Pin 3)
                     SCL                  ───►   GPIO 3 / SCL (Pin 5)
```

Read samples at 360 Hz into an 1,800-sample ring buffer. Every 1–5 seconds, pass the buffer to `clf.predict_window(window_360hz)`.

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
