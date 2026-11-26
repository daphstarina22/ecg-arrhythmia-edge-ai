"""Single source of truth for every constant used by the runtime."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
M1_PATH = MODELS_DIR / "shockable_classifier_candidate.onnx"
M2_PATH = MODELS_DIR / "arrhythmia_multiclass_4class_candidate.onnx"
M3_PATH = MODELS_DIR / "vf_vt_subtype_classifier_candidate.onnx"

# ---------------------------------------------------------------- acquisition
# ADS1292R register config is UNCHANGED from the validated firmware/driver:
# CONFIG1=0x02 means 500 SPS, so the hardware delivers 500 samples/s.
ADC_SPS = 500
CAPTURE_SEC = 5.0
RAW_SAMPLES = int(ADC_SPS * CAPTURE_SEC)          # 2500 raw samples = 5.000 s
SETTLE_DISCARD_SAMPLES = 50                        # 100 ms: discard after START (filter settling)
DRDY_TIMEOUT_SEC = 0.25                            # a stuck DRDY must fail, not hang the HMI
TIMING_WARN_FRAC = 0.005                           # >0.5 % off nominal duration -> warning
TIMING_REJECT_FRAC = 0.02                          # >2 % off (>=50 lost samples) -> reject capture

# Pins (BCM) and SPI
DRDY_PIN, START_PIN, PWDN_PIN = 17, 27, 22
SPI_BUS, SPI_DEVICE, SPI_HZ = 0, 0, 1_000_000
ADC_GAIN = 6.0
VREF = 2.42

# ---------------------------------------------------------------- model domain
# Models were trained on 360 Hz, 5 s (1800 sample) windows.
MODEL_FS = 360
MODEL_SAMPLES = int(MODEL_FS * CAPTURE_SEC)        # 1800
RESAMPLE_UP, RESAMPLE_DOWN = 18, 25                # 500 * 18/25 = 360 exactly

# Conditioning = training filter (Butterworth order 2, 0.5-40 Hz, zero phase)
# + 50 Hz mains notch (India mains; ADS1292R has no analog notch).
BAND_HZ = (0.5, 40.0)
BAND_ORDER = 2
MAINS_HZ = 50.0
NOTCH_Q = 30.0

# ---------------------------------------------------------------- quality gate
ADC_RAIL_MV = 380.0          # full scale at gain 6 / 2.42 V ref is +/-403 mV
RAIL_FRAC_MAX = 0.05
FLAT_STD_MV = 0.015
FLAT_PTP_MV = 0.050
MAX_PLAUSIBLE_PTP_MV = 10.0  # filtered peak-to-peak above this is not ECG (artifact)
MAINS_RATIO_WARN = 4.0       # warn only (the notch handles mains); tune on your hardware

# ---------------------------------------------------------------- decision
P_SHOCKABLE_THRESHOLD = 0.50     # unchanged from previous deployment
P_VF_THRESHOLD = 0.50            # unchanged from previous deployment
