# Phase 2A Source-Bias Diagnostics

The local repository does not contain the Kaggle source records or persisted
feature tables, so this diagnostic cannot be executed locally without
inventing data. Run it in the Kaggle notebook after the existing feature-table
construction cells have completed.

The exact notebook tables should be assembled as follows:

```python
from training.baseline_analysis.phase2a_source_bias import run_phase2a

results = run_phase2a(
    model1_table=binary_df,
    model2_table=multiclass_df,
    model3_table=vfvt_df,
    combined_source_table=combined_feature_df,
)
```

Each table must contain the ten baseline features, `label`, `record_id`, and
`dataset_source`. The existing notebook uses source-prefixed record IDs such
as `vfdb_418`; do not replace them with window IDs.

The diagnostics print and return:

- grouped train/test counts and exact group intersections for Models 1-3
- source-by-label window counts, unique record counts, percent of source, and
  percent of class
- Model 1 source-only accuracy, balanced accuracy, precision, recall, F1, and
  ROC-AUC
- ten-feature source-classifier accuracy, macro F1, confusion matrix, and
  coefficient-based feature importance
- per-source summary statistics for all ten features

The diagnostic classifiers are not production models and produce no ONNX
files. No gain constant, hyperparameter, split rule, production model, or
Raspberry Pi baseline is changed.