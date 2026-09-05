# Model 3 Phase 2E.2

Research-only full calibrated reconstruction for the original Model 3 VF/VT
record set. This phase does not train Model 3, export ONNX, or modify the
production feature contract or prior phase outputs.

The runner reconstructs only VFDB and CUDB record IDs already present in the
original feature table, then combines them with the validated Phase 2E.1
Challenge2015 calibrated table. Additional raw records are not silently added.

WFDB `record.p_signal` is used as the calibrated physical signal. The runner
records the header sampling frequency, channel names, units, gain, baseline,
and ADC zero in `calibration_metadata.csv`. VFDB preserves the original first
ECG channel behavior; CUDB uses its single ECG channel. All signals are
resampled to 360 Hz with `resample_poly`, then the existing ten production
features are regenerated through the existing feature implementation.

Only auxiliary annotations containing `(VF` or `(VT` are retained. AFIB-like
annotations are ignored for labeling and any AFIB label in the output causes
validation to fail.

Run:

```powershell
.venv\Scripts\python.exe -m training.model3_phase2e2.model3_full_reconstruction vfvt_feature_table.csv VFDB.zip cu-ventricular-tachyarrhythmia-database-1.0.0.zip training/model3_phase2e/results/reconstructed_feature_table.csv --output-dir training/model3_phase2e2/results
```

The archive contents are extracted only into a temporary directory. No source
archive is modified.