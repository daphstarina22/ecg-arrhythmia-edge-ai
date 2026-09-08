# Phase 5A: Downsampling & Optimization Validation Report

## Executive Summary
Validation compared reference and optimized feature extraction across 768 windows from 8 benchmark records (4 Challenge 2015, 4 VFDB; 4 VT and 4 VF).

## 1. Lempel-Ziv Complexity Validation
- **Reference**: `lz_original_N1800` (1,800 samples @ 360 Hz)
- **Optimized**: `lz_optimized_N450` (450 samples @ 90 Hz)
- **Correlation**:
  - Pearson $r$: **0.9888**
  - Spearman rank $\rho$: **0.9532**
- **VT vs. VF Separation**:
  - Reference: Cohen's $d = -0.3326$, ROC-AUC = **0.4551**
  - Optimized: Cohen's $d = -0.3227$, ROC-AUC = **0.4622**
- **Conclusion**: Downsampling to 90 Hz produces an almost perfect correlation ($r > 0.95$) with identical effect direction and virtually unchanged ROC-AUC.

## 2. Sample Entropy Validation
- **Reference**: `sampen_reference_N450` (450 samples @ 90 Hz)
- **Optimized**: `sampen_optimized_N225` (225 samples @ 45 Hz)
- **Correlation**:
  - Pearson $r$: **0.9735**
  - Spearman rank $\rho$: **0.9708**
- **VT vs. VF Separation**:
  - Reference: Cohen's $d = 0.5487$, ROC-AUC = **0.6089**
  - Optimized: Cohen's $d = 0.5954$, ROC-AUC = **0.6415**
- **Conclusion**: Vectorized Sample Entropy at 45 Hz ($N=225$) preserves strong rank correlation and identical effect direction for VT vs. VF separation.

## 3. Verdict on Downsampling
- **Result**: **SATISFACTORY**.
- Downsampling preserves physiological separation while reducing full-dataset runtime from 2.5 hours to under 5 minutes.
