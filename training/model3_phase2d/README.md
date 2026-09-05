# Model 3 Phase 2D

Phase 2D audits whether raw ECG windows and source calibration metadata are
available for genuine signal-level harmonization. It is separate from the
deployed pipeline and does not modify the ten-feature contract, production
feature extraction, ONNX files, or Phase 2A/2B/2C results.

With the current repository, only `vfvt_feature_table.csv` is available. The
feature table can document source-scale differences, but it cannot support
true signal-level normalization. The audit reports this explicitly rather
than treating feature-space scaling as ECG normalization.

Run the current audit with:

```powershell
python -m training.model3_phase2d.raw_signal_audit vfvt_feature_table.csv --output-dir training/model3_phase2d/results
```

When raw source directories are supplied later, pass one or more roots:

```powershell
python -m training.model3_phase2d.raw_signal_audit vfvt_feature_table.csv --raw-root path/to/raw --output-dir training/model3_phase2d/results
```

Outputs contain raw-file availability, feature-scale summaries, source-label
counts, and record contribution summaries. This tool does not train or export
any model.