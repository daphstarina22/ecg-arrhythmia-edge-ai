# Controlled Experiments

The existing 10-feature ONNX models are the frozen baseline. Experiments here
must write new artifacts under `training/results/` and must not overwrite
`models/`.

## Shared Rules

- Use the ten deployment features in the exact deployed order for A and B.
- Use patient groups when available; otherwise use canonical
  `dataset_source + record_id` groups.
- Assert zero group intersection between train and test.
- Keep all overlapping windows from one group together.
- Do not tune hyperparameters in this phase.
- Report accuracy, balanced accuracy, precision, recall, F1, macro F1,
  ROC-AUC where applicable, PR-AUC for binary tasks, confusion matrix, and
  feature importance.

## Experiment A

Current 10 features with a fixed record-level split. This measures how much
the existing result changes when the split is made explicit and reproducible.

## Experiment B

Current 10 features, the same record-level split, and class balancing or class
weights. Change only the imbalance treatment; keep the classifier family and
hyperparameters fixed.

## Experiment C

Eleven or more features, including `rr_entropy` and additional justified
features, with the same record-level split. This requires retraining and is
not part of the current deployment baseline.

## Required Comparison Table

| Experiment | Model | Features | Split | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Macro F1 | ROC-AUC | PR-AUC | Results |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | Model 1 | 10 | record | | | | | | | | | |
| B | Model 1 | 10 + balancing | record | | | | | | | | | |
| C | Model 1 | 11+ | record | | | | | | | | | |

Use separate tables for Model 2 and Model 3 because their metrics and class
structures differ.