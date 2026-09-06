# Phase 4: Record-Balanced Evaluation & Domain-Shift Disentanglement

This research directory evaluates Model 3 (VF vs. VT subtype classifier) on the calibrated Phase 2E.2 / Phase 3 feature table. It does not modify production code, ONNX files, or prior phase results.

---

## Phase 4.1: Record-Balanced Per-Record Diagnostics

Phase 4.1 evaluated the 7-feature `StandardScaler + LogisticRegression` baseline with and without training-only inverse-record-frequency sample weights.

### Key Finding
- Inverse-record weighting changed virtually nothing (robustness score `0.421` vs `0.417`).
- Unmasked severe asymmetric failure:
  - **D1 (VFDB+CUDB -> Challenge2015)**: VF recall was only `24.5%` due to a fixed `0.5` threshold on a `6.6%` VF prevalence test set.
  - **D2 (Challenge2015+CUDB -> VFDB)**: VT recall collapsed to `16.6%` due to balanced weighting against a `14:1` training prevalence.

### Run Phase 4.1:
```powershell
.venv\Scripts\python.exe -m training.model3_phase4.run_phase4 training/model3_phase3/artifacts/calibrated_vfvt_feature_table.csv --output-dir training/model3_phase4/results
```

---

## Phase 4.2: Domain-Shift Disentanglement (Feature Audit, Ablation & Recalibration)

Phase 4.2 investigated whether the cross-domain collapse could be resolved by:
1. Auditing feature discriminative signal vs. domain confounding.
2. Ablating source-confounding features.
3. Calibrating operating thresholds strictly on training folds (`StratifiedGroupKFold`).

### Run Phase 4.2:
```powershell
.venv\Scripts\python.exe -m training.model3_phase4.run_phase4_2 training/model3_phase3/artifacts/calibrated_vfvt_feature_table.csv --output-dir training/model3_phase4/results/phase4_2
```

### Core Results:
- **Part A (Feature Audit)**:
  - `zero_crossings` and `dominant_freq` **inverted direction** across databases (Cohen's $d$ changed sign between Challenge 2015 and VFDB).
  - `qrs_width` had $3\times$ higher correlation with dataset source ($|d| = 1.467$) than with VT vs. VF ($|d| = 0.507$).
  - `peak_count` had near-zero discriminative effect size ($|d| = 0.052$).
- **Part B & C (Ablation & Calibration)**:
  - Evaluated 3 feature sets (`set1_full7`, `set2_ablation5`, `set3_objective`) across Logistic Regression and Random Forest using 4 training-derived threshold strategies (`T_0.50`, `T_youden`, `T_balanced`, `T_prior`).
  - D1 achieved ROC-AUC up to `0.775` and VF record recall of `86.3%` with calibrated thresholding.
  - **D2 ROC-AUC remained capped at 0.59–0.61 across all 24 configurations**.
  - In D2, achieving high VF recall caused VT recall to collapse to `1.4%–3.3%` (12–14 VT records misclassified as VF).

### Decision Gate Verdict:
**REJECTED: CRITERIA NOT MET.**
The existing 7-feature contract (derived from Pan-Tompkins peak detection and overlapping 3–9 Hz band power) lacks cross-domain physiological separability for VT vs. VF. Proceeding to Phase 5 (raw waveform complexity/organization features) is mathematically justified.