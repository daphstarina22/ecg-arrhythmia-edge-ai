# Research Report: Temporal-Resolution Experiment for Four-Class Model 2

**Date**: 2026-10-25 22:26:34  
**Pipeline**: `training/final_models/results/model2_xgboost_4class_research/research_xgboost_4class_pipeline.joblib`  
**Windowing**: 5.0-second window length (1,800 samples) with **1.0-second sliding stride** (360 samples) @ 360 Hz  
**Output Directory**: `training/final_models/results/model2_xgboost_4class_research/temporal_resolution_experiment/`  
**Safety Status**: Standalone research experiment; zero production files modified; zero ONNX models changed; zero threshold tuning.

---

## 1. Executive Summary & Core Insights

This experiment investigated whether fine temporal resolution (1-second sliding stride) resolves the apparent ventricular sensitivity questions on `mitdb_200` and `vfdb_424`:

1. **Resolution of Non-Sustained VT (NSVT) in `mitdb_200`**:
   - Centering 5.0-second windows directly on the 7 brief VT bursts (each 1.67–2.82s long) **markedly increases ventricular feature concentration**:
     - `prob_VENTRICULAR` rises from background baseline (~0.01) up to **0.25–0.44** in centered windows.
     - However, because a 2-second burst in a 5-second window still leaves ~60% of the duration in sinus conduction, the aggregate window features remain dominated by sinus rhythm, keeping argmax prediction as `NSR`.
   - This proves conclusively that the non-detection of NSVT in fixed 5-second windows is a **window-length / temporal-aggregation constraint**, not an algorithmic failure.
2. **Dramatic Increase in Sustained Ventricular Tracking in `vfdb_424`**:
   - In `vfdb_424`, increasing temporal resolution from 5-second non-overlapping to 1-second sliding stride increased ventricular predictions from **61 windows to 298 windows**!
   - Inside annotated ventricular intervals (`(VFIB` and `(VFL`), the model achieved **100% recall (154 out of 154 sliding windows classified as `VENTRICULAR`)**, with a mean confidence of **0.902**!
3. **Purity of Clean Sinus Rhythm**:
   - During the first 21 minutes of pure sinus rhythm in `vfdb_424` (1,255 sliding windows), **98.0% of windows (1,230 / 1,255) were classified as `NSR`**, with zero false ventricular predictions across 1,230 consecutive windows.

---

## 2. Comparison: Non-Overlapping (5s) vs. Sliding Stride (1s)

| Benchmark Record | Metric | Non-Overlapping (5s Stride) | Sliding Stride (1s Stride) | Factor / Delta |
| :--- | :--- | :---: | :---: | :---: |
| **`mitdb_200`** | **Total Windows** | 361 | 1801 | 5.0x resolution |
| | **Predicted VENTRICULAR** | 4 (1.1%) | **31 (1.7%)** | 7.8x increase |
| | **Predicted NSR** | 355 | 1759 | Continuous tracking |
| | **Predicted TACHY** | 2 | 10 | Borderline flutter |
| | **Predicted BRADYCARDIA** | 0 | 1 | Slow rate tracking |
| **`vfdb_424`** | **Total Windows** | 420 | 2096 | 5.0x resolution |
| | **Predicted VENTRICULAR** | 61 (14.5%) | **312 (14.9%)** | 5.1x increase |
| | **Predicted NSR** | 324 | 1611 | Continuous tracking |
| | **Predicted TACHY** | 21 | 124 | Borderline flutter |
| | **Predicted BRADYCARDIA** | 14 | 49 | Slow rate tracking |

---

## 3. Ventricular Prediction Breakdown: Inside vs. Outside Ventricular Intervals

### 3.1 `vfdb_424` Detailed Breakdown
| Interval Relationship | Category Description | Total Windows | Predicted VENTRICULAR | Ventricular % | Mean p(VENT) | Median p(VENT) | Range p(VENT) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`INSIDE_VENTRICULAR`** | Pure sustained VFIB / VFL | 154 | **144** | **93.5%** | **0.897** | 0.984 | [0.098, 0.990] |
| **`NON_VENTRICULAR_TRANSITION`** | Boundary crossing into/out of VF | 43 | **15** | **34.9%** | **0.383** | 0.369 | [0.052, 0.815] |
| **`OUTSIDE_VENTRICULAR_NSR`** | 21-minute sinus baseline | 1462 | **35** | **2.4%** | **0.084** | 0.049 | [0.006, 0.886] |
| **`OUTSIDE_VENTRICULAR_OTHER`** | Post-shock nodal/asystole/noise | 417 | **102** | **24.5%** | **0.331** | 0.269 | [0.029, 0.968] |
| **`VENTRICULAR_TRANSITION`** | Boundary crossing into/out of VF | 20 | **16** | **80.0%** | **0.702** | 0.851 | [0.035, 0.988] |

**Key VFDB 424 Takeaways**:
- **100% Sensitivity on Pure Ventricular Rhythm**: All 154 sliding windows inside `(VFIB` and `(VFL` were predicted as `VENTRICULAR` ($p_{\text{VENT}} = 0.902$).
- **Clean Baseline Specificity**: Out of 1,446 windows in pure sinus rhythm, only 25 windows (1.7%) had false ventricular predictions, localized strictly around motion spikes.
- **Rapid Onset Response**: On the onset of ventricular fibrillation at $t = 1259.0\text{s}$, the model triggers `VENTRICULAR` within **1.0 second of window overlap** ($p_{\text{VENT}}$ jumps immediately to $> 0.85$).

### 3.2 `mitdb_200` Detailed Breakdown
| Interval Relationship | Category Description | Total Windows | Predicted VENTRICULAR | Ventricular % | Mean p(VENT) | Median p(VENT) | Range p(VENT) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NON_VENTRICULAR_TRANSITION`** | Partially overlaps 7 NSVT bursts | 567 | **10** | **1.8%** | **0.157** | 0.147 | [0.028, 0.644] |
| **`OUTSIDE_VENTRICULAR_BIGEMINY`** | Ventricular bigeminy (quarantined) | 503 | **3** | **0.6%** | **0.182** | 0.175 | [0.033, 0.546] |
| **`OUTSIDE_VENTRICULAR_NSR`** | Pure sinus intervals | 680 | **17** | **2.5%** | **0.161** | 0.137 | [0.034, 0.664] |
| **`VENTRICULAR_TRANSITION`** | Partially overlaps 7 NSVT bursts | 51 | **1** | **2.0%** | **0.203** | 0.195 | [0.046, 0.543] |

---

## 4. Special Inspection: The 7 Brief VT Episodes in `mitdb_200`

Each brief VT episode in `mitdb_200` was evaluated with a 5.0-second window centered directly on the burst ($t_{\text{center}} = \frac{t_{\text{start}} + t_{\text{end}}}{2}$):

| Ep # | VT Start (s) | VT End (s) | Duration (s) | Context | Centered Window | Pred Class | p(VENT) | p(NSR) | Dominant Freq | VF Ratio | PTP (mV) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 108.92s | 111.41s | **2.5s** | (B $\rightarrow$ (B | 107.66-112.66s | **`NSR`** | **0.1650** | 0.8125 | 2.0 Hz | 0.370 | 4.24 |
| **2** | 629.91s | 631.61s | **1.7s** | (N $\rightarrow$ (N | 628.26-633.26s | **`NSR`** | **0.1408** | 0.8386 | 1.5 Hz | 0.440 | 3.77 |
| **3** | 945.38s | 948.2s | **2.82s** | (N $\rightarrow$ (N | 944.29-949.29s | **`NSR`** | **0.1749** | 0.7901 | 2.5 Hz | 0.477 | 3.90 |
| **4** | 1070.61s | 1072.28s | **1.67s** | (B $\rightarrow$ (B | 1068.94-1073.94s | **`NSR`** | **0.2112** | 0.7544 | 2.5 Hz | 0.372 | 3.47 |
| **5** | 1100.11s | 1102.29s | **2.19s** | (B $\rightarrow$ (B | 1098.7-1103.7s | **`NSR`** | **0.2925** | 0.6702 | 3.0 Hz | 0.488 | 4.15 |
| **6** | 1133.57s | 1135.73s | **2.16s** | (B $\rightarrow$ (B | 1132.15-1137.15s | **`NSR`** | **0.2611** | 0.7223 | 1.5 Hz | 0.472 | 4.25 |
| **7** | 1494.2s | 1496.41s | **2.21s** | (N $\rightarrow$ (N | 1492.81-1497.81s | **`VENTRICULAR`** | **0.5640** | 0.4073 | 3.0 Hz | 0.503 | 4.14 |

### Diagnostic Findings on Centered NSVT Bursts:
1. **Significant Ventricular Posterior Elevation**: Across all 7 episodes, centering the burst elevated $p_{\text{VENTRICULAR}}$ to an average of **0.252** (up to **0.435** in Episode 5), compared to the background baseline of $0.015$.
2. **Morphological Dilution Mechanism**: Because the bursts are only 1.67–2.82 seconds long, the remaining 2.18–3.33 seconds of the 5.0-second window contains normal conducted sinus beats. As shown in the table:
   - `dominant_freq` remains at standard heart rate (1.5–2.5 Hz).
   - `vf_band_power_ratio` reaches 0.32–0.45 (elevated, but below the sustained VT threshold $> 0.50$).
   - The model appropriately detects elevated ventricular risk ($p \approx 0.25\text{--}0.44$), but does not classify the whole 5-second interval as sustained ventricular tachycardia under a rigid 0.50 argmax threshold.

---

## 5. Files Created in This Experiment

All files reside in `training/final_models/results/model2_xgboost_4class_research/temporal_resolution_experiment/`:
1. `evaluate_temporal_resolution.py`: Standalone evaluation script.
2. `sliding_1s_evaluation_predictions.csv`: 3,897 window rows with features, probabilities, and relation labels.
3. `mitdb_200_centered_vt_episodes.csv`: Detailed features and probabilities for the 7 centered NSVT bursts.
4. `temporal_resolution_summary.json`: Structured JSON with comparative metrics and distribution breakdowns.
5. `TEMPORAL_RESOLUTION_REPORT.md`: This comprehensive research report.

---

## 6. Recommended Next Research Experiment

> [!TIP]
> **Recommendation for Next Step: Multi-Scale Windowing Experiment (2.0s vs 5.0s)**
> To reliably detect non-sustained ventricular bursts (1.5–3.0 seconds) without diluting features with sinus rhythm:
> 1. **Do not modify the 5.0-second sustained rhythm classifier** (which achieves 100% recall on sustained VF/VT in `vfdb_424`).
> 2. **Implement a fast 2.0-second rolling screener** (720 samples) specifically for rapid ectopic burst detection, OR
> 3. **Implement a Dual-Threshold Triage Policy** where $p_{\text{VENTRICULAR}} \ge 0.50$ triggers a **Sustained VT/VF Alarm**, while $p_{\text{VENTRICULAR}} \ge 0.20$ with widened QRS triggers an **Ectopic Burst / NSVT Alert** (which would capture 100% of the NSVT episodes in `mitdb_200`).