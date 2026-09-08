# Phase 4.1 Record-Balanced Per-Record Evaluation

The calibrated Phase 2E.2 table was evaluated without changing production
artifacts. Labels are VT=0 and VF=1; AFIB rows were rejected and none were
present. All source-held-out splits had zero record overlap.

## Method

Baseline: StandardScaler plus LogisticRegression(max_iter=1000,
class_weight="balanced", random_state=42).

Record-balanced: identical model, with training-only inverse-record-frequency
sample weights normalized to mean one.

Equal-record robustness formula:

```text
0.35 * average D1/D2 record macro F1
+ 0.35 * average D1/D2 VF-record recall
+ 0.30 * minimum D1/D2 record macro F1
```

Best candidate: `baseline_logistic_reduced` with score `0.421016`.

## Decision

This is an evaluation result only. No production model or ONNX artifact was
created or modified.
