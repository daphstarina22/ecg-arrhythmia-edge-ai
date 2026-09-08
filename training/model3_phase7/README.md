# Phase 7: Final Constrained ML Decision Experiment

This directory contains the complete, self-contained implementation and results of the **Phase 7 Final Constrained ML Decision Experiment** for Model 3 (VT vs. VF/VFL subtype classification).

## Directory Structure

```
training/model3_phase7/
│
├── __init__.py
├── run_final_ml.py                 # Primary execution script
├── README.md                       # Phase 7 documentation and reproducibility guide
│
├── artifacts/                      # Experimental artifacts
│
└── results/
    ├── data_integrity_audit.json   # Step 1 Pre-ML audit (zero leakage, zero duplicates)
    ├── final_model_ranking.csv     # Official robustness rankings across D1 and D2
    ├── per_record_summary.csv      # Patient-level diagnostic predictions and probabilities
    ├── final_report.md             # Comprehensive 19-point final decision report
    │
    ├── predictions/                # Complete window-level predictions (all experiments, models, thresholds)
    │   ├── groupcv_setA_logreg_predictions.csv
    │   ├── groupcv_setA_rf_predictions.csv
    │   ├── ...
    │   ├── d1_setB_rf_predictions.csv
    │   └── d2_setC_hgb_predictions.csv
    │
    └── figures/                    # Diagnostic figures
        ├── model_robustness_ranking.png
        └── top_model_record_probability_distribution.png
```

## Experimental Design

1. **Pre-ML Data Integrity Audit**:
   - Zero record leakage across splits.
   - Zero missing or non-finite values.
   - 120 unique physical records across 4 clinical databases (`c2015`, `vfdb`, `cudb`, `mitdb`).

2. **Predefined Feature Sets**:
   - `setA_full_physio`: All 21 Phase 5 physiological features.
   - `setB_domain_stable`: 6 direction-consistent features (`lz_complexity`, `ac_decay_time`, `ac_first_secondary_peak`, `ac_zero_crossing_lag`, `ac_max_peak_ratio`, `sample_entropy`).
   - `setC_minimal_edge`: 4 ultra-compact features (`lz_complexity`, `ac_decay_time`, `sample_entropy`, `ac_first_secondary_peak`).

3. **Controlled Models**:
   - `logreg`: StandardScaler + LogisticRegression(class_weight="balanced", max_iter=1000)
   - `rf`: RandomForestClassifier(class_weight="balanced", max_depth=6, n_estimators=100)
   - `hgb`: HistGradientBoostingClassifier(class_weight="balanced", max_depth=5, max_iter=100, min_samples_leaf=20)

4. **Record-Balanced Sample Weighting**:
   - $\text{weight}(r) = \frac{1}{\text{windows}(r)}$ normalized to mean 1.0.
   - Strictly derived on training partitions; test data never influences weights.

5. **Evaluation Protocols**:
   - **Grouped CV**: 5-Fold StratifiedGroupKFold on `record_id`.
   - **Cross-Source D1**: Train on `VFDB + CUDB + MITDB` $\to$ Test on `Challenge 2015`.
   - **Cross-Source D2**: Train on `Challenge 2015 + CUDB + MITDB` $\to$ Test on `VFDB`.

6. **Threshold Policies**:
   - `T_fixed_0.50`: Standard 0.50 probability cutoff.
   - `T_train_calibrated`: Derived strictly from training data out-of-fold predictions via inner StratifiedGroupKFold maximizing Youden's $J$.

## How to Reproduce

Execute using the project virtual environment:

```powershell
.venv\Scripts\python.exe training/model3_phase7/run_final_ml.py
```
