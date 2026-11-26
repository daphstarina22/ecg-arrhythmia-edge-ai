# What was wrong and what changed

## Root causes found (each reproduced or read directly from the original code)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | **No classification / HMI never starts** | HMI POSTs `/control/start` and `/control/stop`; the backend (`combined_ecg_app.py`) defines neither route (404). Also the HMI's stitching depended on `total_samples`, which the backend only refreshed every 2 s. | HMI and acquisition are one process; no HTTP layer. Start button calls the pipeline directly. |
| 2 | **Wrong rhythm class** | ADS1292R is configured for 500 SPS (`CONFIG1=0x02`) but every sample was treated as 360 Hz. A true 72 bpm rhythm was read as 52 bpm (reproduced: classified BRADY_ASY). All frequency features, filters and the 50 Hz notch were mis-scaled by 1.39×. | Capture 2500 samples = 5.000 s, polyphase-resample 18/25 to the 1800-sample, 360 Hz window the models were trained on. Hardware registers untouched. |
| 3 | **Unreliable / dropped samples** | Filtering (`filtfilt`, `medfilt`), `np.roll` on 3600 floats, and 3-model inference ran *inside* the sample loop; DRDY wait had no timeout (HMI froze if DRDY stuck); lost samples were never detected. | Capture loop only reads+stores; processing happens after. DRDY timeout; capture duration is measured and rejected if >2 % off (warning >0.5 %). |
| 4 | **Noisy signal / shifted features** | Live path used a 0.05–100 Hz filter + median baseline; training used a 0.5–40 Hz order-2 band-pass. Live peak detector (Pan-Tompkins on raw, 200 ms refractory) ≠ training detector. | Same training filter, plus a 50 Hz notch (mains). Features are a 1:1 port of the training functions (parity-tested). |
| 5 | **Model 3 fed out-of-distribution input** | M3 was trained with a *different* Lempel-Ziv algorithm (LZ76, training mean 8.37 ± 0.68) than the one deployed (≈1.25, i.e. ≈ −10 σ). Sample-entropy edge case (`A=0`) also differed. | M3 gets its own training-exact features. |
| 6 | **AFib branch could never fire** | Needs 12 warm-up windows + 2-window persistence; one 5 s window per Start can't satisfy that, and "AFib unavailable" pushed normal rhythms toward UNCERTAIN. Not part of your decision table. | Removed from runtime; code kept in `research/afib/`. |
| 7 | **Silent fake data** | If spidev/GPIO import failed the app silently fell back to a mock and showed classifications. | Mock only via explicit `--mock`; red "SIMULATED SIGNAL" banner. |
| 8 | **Continuous background acquisition** | Acquisition thread ran forever. | ADC is configured at launch but conversions run only between `start()` and `stop()` (START pin low + SDATAC/STOP otherwise). |
| 9 | Contradictory shock logic | See `CLASSIFICATION_LOGIC.md` (open question). | Explicit CONFLICT. |

## Preserved unchanged
Three ONNX models (byte-identical, SHA256 in `models/SHA256SUMS`), SPI mode, pins, register values
(`CONFIG1 0x02, CONFIG2 0xA0, CH1SET/CH2SET 0x00, RLDSENS 0x2C`), 24-bit→mV scaling, chip-ID check,
M1/M2/M3 thresholds (0.5), voice prompts (espeak-ng/paplay, same sink default).

## Deliberately NOT changed (needs your data/decision)
* Model weights/thresholds. On synthetic ECG the probabilities are often near 0.5 (see README, "Limitations").
* Lead-off detection registers (LOFF) - not enabled in your validated config, so electrode problems are caught by the signal-quality gate only.
* Quality thresholds are initial values and untuned on real electrodes (`config.py`).

## Removed (all recoverable from git history, commit `2d463a2`)
`deploy/` (Flask backend, old HMI, old decision layer 9-label, edge_runtime, ads1292r_stream, live test scripts),
`src/`, `demo_edge_pipeline.py`, `test_enhanced_classification.py`, old-pipeline tests, old 10-feature `models/` set and `ecg_models.zip`.
Moved: AFib → `research/afib/`, notebook → `training/notebooks/`, old report/guide/README → `docs/legacy/`.
