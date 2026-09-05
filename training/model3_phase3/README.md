# Model 3 Phase 3

Experimental retraining and evaluation on the calibrated Phase 2E.2 table.
This directory does not modify production code, ONNX models, the original
feature table, or prior phase results.

The source table is the validated raw-reconstructed artifact:

```text
training/model3_phase2e2/results/calibrated_model3_feature_table.csv
```

`reconstruct_dataset.py` validates and stages that artifact under Phase 3;
raw waveform reconstruction itself remains documented and reproducible in
Phase 2E.2.

The training runner evaluates full and reduced features, class-weighted
XGBoost, LogisticRegression, RandomForest, ExtraTrees, grouped random splits,
two source-held-out directions, and five repeatability seeds. The robustness
score is:

```text
0.4 * average held-out macro F1
+ 0.4 * average held-out VF F1
+ 0.2 * worst-direction held-out macro F1
```

No oversampling is used, no test rows are duplicated, and no ONNX artifact is
exported.

Run:

```powershell
.venv\Scripts\python.exe -m training.model3_phase3.reconstruct_dataset training/model3_phase2e2/results/calibrated_model3_feature_table.csv
.venv\Scripts\python.exe -m training.model3_phase3.train_models training/model3_phase3/artifacts/calibrated_vfvt_feature_table.csv
.venv\Scripts\python.exe -m training.model3_phase3.evaluate_models
```