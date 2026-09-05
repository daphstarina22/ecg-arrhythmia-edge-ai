# Model 3 Phase 2C

Phase 2C evaluates source-robust VF versus VT experiments without changing
the deployed models or the ten-feature inference contract.

The runner validates `vfvt_feature_table.csv`, rejects AFIB and unexpected
labels, reproduces the frozen record-group split, and writes only diagnostic
artifacts under `training/model3_phase2c/results/`.

Experiments include:

- raw ten-feature baseline;
- amplitude-free seven-feature variants;
- feature-space StandardScaler and RobustScaler LogisticRegression;
- training-only record-balanced XGBoost using inverse record-frequency sample
  weights;
- VFDB+CUDB to Challenge2015 and Challenge2015+CUDB to VFDB source holdouts.

Record balancing affects training weights only. Test rows are never sampled or
duplicated. The robustness ranking is:

```text
0.4 * mean(source-held-out macro F1)
+ 0.4 * mean(source-held-out VF F1)
+ 0.2 * worst-direction source-held-out macro F1
```

Raw ECG windows are not present in the feature CSV or repository, so the
normalization experiments are feature-space normalization only. They are not
signal-level ECG normalization and do not change `preprocessing.py`.

Run with:

```powershell
python -m training.model3_phase2c.model3_phase2c_experiments vfvt_feature_table.csv --output-dir training/model3_phase2c/results
```

No ONNX model is exported and no production source is imported or modified.