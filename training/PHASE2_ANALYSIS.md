# Phase 2: Leakage and Generalization Analysis

This area is separate from the Raspberry Pi baseline. It does not overwrite
`models/`, the ONNX files, or `src/`.

## Current Notebook Findings

The latest effective training block uses `GroupShuffleSplit`:

- Model 1: one split, `test_size=0.2`, `random_state=42`
- Model 2: one split, `test_size=0.2`, `random_state=42`
- Model 3: one split, `test_size=0.25`, `random_state=42`

Groups are source-prefixed record IDs: `vfdb_<record>`, `cudb_<record>`,
`mitdb_<record>`, and `c2015_<record>`. The notebook asserts that no group is
present in both partitions. Therefore, windows from the same record do not
cross a model's train/test split, including overlapping windows.

Patient IDs are not available in the notebook data structures. Record-level
separation is therefore the strongest demonstrated protection, but patient-
level leakage cannot be ruled out if a source contains multiple records from
one patient.

## Leakage Risk Assessment

- Direct window-level leakage within each model: **not observed in the latest
  code**, because records are grouped before splitting.
- Same-record leakage: **guarded against within each model** by the group
  assertion.
- Patient-level leakage: **unknown**, because patient identifiers are absent.
- Cross-model split differences: possible. Each model creates its own split,
  so a record can be in training for one model and testing for another. This
  does not leak within an individual model, but it makes cross-model metrics
  incomparable as a shared validation benchmark.
- Repeated-window dependence: overlapping windows from one record remain
  correlated, but they stay on one side of the split under the current group
  key.

## Source-Bias Assessment

Model 1 positives come from VFDB and CUDB VF/VT windows. Its negatives come
from MIT-BIH windows filtered to exclude shockable annotations. Challenge2015
and AFDB are excluded from Model 1. This creates an unequal source distribution
by class: source identity is correlated with the target by construction.

Model 2 combines VFDB, CUDB, MIT-BIH, and true-alarm-only Challenge2015.
The latest notebook has five classes (`Asystole`, `Normal`, `Tachycardia`,
`VF`, `VT`); earlier notebook cells describe a six-class/AFIB version and are
stale. Model 3 combines VFDB, CUDB, and Challenge2015 VF/VT windows, so its VF
and VT source distributions must be checked separately.

Challenge2015 gain is marked unverified in the notebook. Because `mean`,
`std`, and `ptp` are amplitude-sensitive, this is a direct source-shift risk.

The raw datasets and a persisted feature table are not present in this
repository, so numerical per-source feature distribution comparisons and a
source-only diagnostic classifier cannot be honestly run here. The analysis
script in this directory performs those checks when supplied the Kaggle
feature table.

## Recorded Historical Metrics

These are preserved notebook outputs, not newly rerun results:

- Model 1: 0.99 accuracy, 0.9997337 ROC-AUC; 32,773 non-shockable and 3,198
  shockable windows.
- Model 2 latest five-class run: 0.85 accuracy, 0.74 macro F1. Per-class F1:
  Asystole 0.72, Normal 0.97, Tachycardia 0.87, VF 0.46, VT 0.66.
- Model 3: 0.74 accuracy, 0.7762 ROC-AUC; VT F1 0.84 and VF F1 0.29.

The notebook imports `confusion_matrix` but does not persist final confusion
matrices for these runs. The Phase 2 scripts require them for every new
experiment.

## Recommended Split

1. Resolve or record patient IDs for every source. Use patient ID as the group
   whenever it exists.
2. Otherwise use a canonical source-plus-record group and assert zero group
   intersection between train and test.
3. Keep all windows, including overlapping windows, within one group.
4. Use one fixed grouped test set per model and grouped cross-validation only
   on the training partition.
5. Report group counts and label/source counts for both partitions.

Do not tune hyperparameters until this split and audit are in place.