# Phase 3 Model 3 Report

## Scope

This was an experimental retraining/evaluation run on the calibrated Phase 2E.2
raw-reconstructed table. No production model or ONNX file was changed.

Dataset: 14618 windows, 117 records, labels {'VT': 12650, 'VF': 1968}.
AFIB rows: 0. Group overlap: 0.

## Source Bias

Original source-classifier accuracy: 0.9290549688600054.
Calibrated source-classifier accuracy: 0.8410506363390198.

## Robustness Selection

Formula: `0.4 * mean held-out macro F1 + 0.4 * mean held-out VF F1 + 0.2 * worst held-out macro F1`.

Best ranked candidate: `D_logistic_reduced` with robustness score `0.49702971502666765`.

This ranking is based on source-held-out evaluation, not random-split accuracy.

## Production Decision

**NO MODEL APPROVED FOR PRODUCTION.** Source-held-out performance remains
asymmetric and no ONNX artifact was exported.
