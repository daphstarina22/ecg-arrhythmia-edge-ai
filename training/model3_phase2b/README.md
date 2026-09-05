# Model 3 Phase 2B

Controlled VF/VT experiments using a feature table exported from the Kaggle
notebook. This area is experimental only.

The runner:

- reproduces the notebook's `GroupShuffleSplit(record_id, test_size=0.25,
  random_state=42)` split;
- evaluates a frozen baseline, class weighting, and training-only oversampling;
- evaluates VFDB+CUDB -> Challenge2015 and Challenge2015+CUDB -> VFDB;
- compares the full ten-feature contract with an experimental feature set that
  excludes `mean`, `std`, and `ptp`;
- reports source-label counts and feature-to-source diagnostics;
- writes CSV/JSON reports under the requested output directory only.

It does not import production code, modify `models/`, export ONNX, or change
the ten-feature inference contract. AFIB is rejected if present.

## Input

Export `vfvt_df` from the notebook without changing it:

```python
vfvt_df.to_csv("/kaggle/working/vfvt_feature_table.csv", index=False)
```

Then run:

```powershell
python -m training.model3_phase2b.model3_experiments vfvt_feature_table.csv
```

The environment must provide `xgboost`, as the saved notebook's Model 3
classifier is XGBoost. The runner never saves trained model binaries.

## Outputs

- `model3_comparison.csv`
- `model3_confusion_matrices.json`
- `model3_source_label_counts.csv`
- `model3_source_diagnostics.json`
- `model3_split_definition.json`
- `model3_config.json`

No numerical result is valid until the runner completes successfully against
the actual exported Kaggle feature table.