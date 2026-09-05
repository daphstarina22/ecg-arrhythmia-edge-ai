# Phase 4.1: Record-Balanced Per-Record Evaluation

This research-only pipeline evaluates the existing reduced seven-feature
LogisticRegression candidate on the calibrated Phase 2E.2 table.

It compares:

- the existing Phase 3 baseline LogisticRegression;
- the identical model with training-only inverse-record-frequency weights.

Both use the exact source-held-out directions:

- VFDB + CUDB -> Challenge2015
- Challenge2015 + CUDB -> VFDB

Every test window is persisted. Equal-record metrics use record-level majority
predictions and per-record error summaries, so records with many windows do
not dominate the primary ranking.

Run:

```powershell
.venv\Scripts\python.exe -m training.model3_phase4.run_phase4 training/model3_phase3/artifacts/calibrated_vfvt_feature_table.csv --output-dir training/model3_phase4/results
```

No production source, ONNX file, prior phase result, or original feature table
is modified.