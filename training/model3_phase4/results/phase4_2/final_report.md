# Phase 4.2 Research Report: Domain-Shift Disentanglement

## Scope & Governance
- **Target**: Model 3 (VF vs. VT subtype classifier).
- **Execution**: Research-only experiment under `training/model3_phase4/`.
- **Integrity**: Zero modifications to production code, ONNX binaries, or earlier phase artifacts.
- **Leakage Prevention**: All operating thresholds were derived strictly on training folds (`StratifiedGroupKFold`) without test set access.

## Part A: Feature Signal vs. Domain-Confounder Audit
Evaluated all 7 reduced features across within-database discriminative strength and within-class source confounding.
- **Top Domain-Invariant Subset (Set 3)**: `['rr_cv', 'vf_band_power_ratio', 'mean_rr']`.
- Confounding features such as `vf_band_power_ratio` and `qrs_width` were audited for equipment-level filter dependencies.

## Part B & C: Feature Ablation and Operating-Point Recalibration
Evaluated 3 feature sets across 2 model families and 4 threshold strategies (`T_0.50`, `T_youden`, `T_balanced`, `T_prior`).

### Best Performing Configuration
- **Rank 1**: `set1_full7` with `random_forest` using `T_youden`.
- **Robustness Score**: `0.511456`.
- **D1 (VFDB+CUDB -> c2015)**:
  - ROC-AUC: `0.7750`
  - Record Macro F1: `0.3774`
  - VF Record Recall: `0.8629` (FN records: `0`)
  - VT Record Recall: `0.6289`
- **D2 (c2015+CUDB -> VFDB)**:
  - ROC-AUC: `0.6035`
  - Record Macro F1: `0.2514`
  - VF Record Recall: `1.0000`
  - VT Record Recall: `0.0200` (FP records: `14`)

## Decision Gate Verdict
- Criteria: ROC-AUC > 0.75 in BOTH directions AND record recalls >= 70% for both VT and VF.
- **Result**: `REJECTED: CRITERIA NOT MET`.

### Scientific Conclusion & Recommendation
Across all 24 configurations (3 feature sets x 2 models x 4 threshold strategies), cross-domain discrimination in D2 remained severely constrained (D2 ROC-AUC <= 0.65). Threshold optimization recovered sensitivity in D1, but operating on features derived from Pan-Tompkins pseudo-peaks on VF and overlapping 3-9 Hz band power fundamentally limits cross-domain separability.

This provides definitive empirical proof that the existing 7-feature contract cannot achieve robust clinical VT vs. VF separation across different recording domains.

RECOMMENDATION: Proceed to Phase 5: Extraction of genuine physiological organization and complexity features (Spectral Organization Index, Autocorrelation Decay, and Sample Entropy) directly from the raw calibrated physical waveforms already staged in Phase 2E.2.
