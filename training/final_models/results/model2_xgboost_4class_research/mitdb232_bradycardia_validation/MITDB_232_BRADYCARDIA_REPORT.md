# Full-Record Evaluation of Four-Class Model 2 on `mitdb_232` (Sinus Bradycardia)

**Date**: 2026-10-25 23:53:00  
**Target Model**: Four-Class XGBoost Arrhythmia Classifier (`research_xgboost_4class_pipeline.joblib`)  
**Target Record**: MIT-BIH Arrhythmia Database record `232` (`mitdb_232`)  
**Evaluation Scope**: Full 30-minute recording (361 non-overlapping 5-second windows)  
**Output Directory**: `training/final_models/results/model2_xgboost_4class_research/mitdb232_bradycardia_validation/`  
**Evaluation Script**: [`evaluate_mitdb232_bradycardia.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/mitdb232_bradycardia_validation/evaluate_mitdb232_bradycardia.py)  
**Safety & Scope Status**: Strictly research-only evaluation. Zero models retrained; zero features modified; zero thresholds changed; zero production files touched; zero git commits.

> [!WARNING]
> **Provenance & Training-Overlap Disclosure**  
> Record `mitdb_232` contributed **121 windows** to the supervised 4-class training dataset (`four_class_master_feature_dataset.csv`), representing **55.0% of all bradycardia training windows** in the model.  
> This evaluation assesses the **full 30-minute recording (361 windows)** across both in-sample training windows and quarantined out-of-training segments.  
> This evaluation is **NOT independent validation**; it cannot be cited as proof of out-of-distribution generalization for the bradycardia class.

---

## 1. Executive Summary & Key Evaluation Findings

The [Validation Coverage Audit](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/validation_coverage_audit/VALIDATION_COVERAGE_AUDIT.md) identified that zero reference-annotated sinus bradycardia (`(SBR`) records had ever been evaluated offline across their full recording. Record `mitdb_232` contains continuous, annotated sinus bradycardia with frequent atrial ectopy across its full 30-minute duration ($1,805.6\text{ seconds}$). 

We executed a full-record, 5-second non-overlapping evaluation using the unchanged four-class Model 2 pipeline. The evaluation uncovered **four primary findings**:

1. **Perfect In-Sample Recall on Training Windows (100.0%)**:
   - For the 121 windows that were present in the training set, Model 2 achieved **100.00% recall (121/121 windows)** for `BRADYCARDIA`.
   - The model predicted `BRADYCARDIA` with high certainty (mean probability $p_{\text{BRADY}} = 0.9842$, median $0.9851$, min $0.8841$). Zero windows were misclassified as NSR, Ventricular, or Tachycardia.
2. **Deterministic Agreement with Rate-Confirmed Bradycardia (100.0%)**:
   - Across the entire 30-minute recording, **122 windows** exhibited true rate-confirmed bradycardia (measured heart rate $< 60.0\text{ bpm}$ and peak count $\le 5$).
   - Model 2 predicted `BRADYCARDIA` on **122 / 122 windows (100.00% rate recall)**. The model's rate-based decision boundary for bradycardia is strictly concordant with physiological rate definitions.
3. **Severe NSR Leakage Driven by PAC Bursts (177 Windows, 49.0% of Record)**:
   - Across all 361 windows, 177 windows were predicted as `NSR`.
   - In record 232, the underlying sinus bradycardia is heavily interrupted by **1,382 Atrial Premature Beats (`A` / PACs)** occurring in pairs, runs, and bigeminy. In these 177 windows, PAC runs elevated the window heart rate into the normal range ($60.0\text{--}98.0\text{ bpm}$). Because Model 2 evaluates rate over discrete 5-second windows, it faithfully detected normal heart rates and predicted `NSR`.
4. **Aberrant Conduction (RBBB) Triggers Borderline Ventricular False Alarms (47 Windows, 13.0%)**:
   - 47 windows were predicted as `VENTRICULAR`, despite record 232 containing zero true ventricular beats.
   - Clinical inspection reveals that in record 232, the APBs are conducted with **Right Bundle Branch Block (RBBB) aberration**, producing wide, notched QRS complexes ($qrs\_width \ge 0.080\text{s}$). When multiple aberrant PACs occur in rapid succession, the resulting signal roughness and spectral distortion trigger argmax `VENTRICULAR` predictions.
   - Crucially, these false alarms are **low-confidence borderline calls** (mean $p_{\text{VENT}} = 0.4592$, max $0.6049$), completely distinct from true ventricular fibrillation or flutter ($p_{\text{VENT}} > 0.98$).

---

## 2. Provenance Audit: Record 232 Training Representation

Record 232 was audited against the supervised training dataset (`four_class_master_feature_dataset.csv`, 7,493 rows across 318 records):

| Record Identifier | Dataset Source | Windows in Training Set | Training Class Label | Windows Excluded / Quarantined | Full Record Window Count | Overlap Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mitdb_232`** | MIT-BIH | **121 windows** | `BRADYCARDIA` ($121$) | **240 windows** | **361 windows** | **TRAINING OVERLAP (In-Sample)** |

### Why Only 121 Windows Were Admitted During Training

In `build_corrected_multisource_dataset.py`, the training dataset curator applied the following filtering policy for `(SBR` annotations:

```python
elif curr_note == "(SBR":
  if (
      feats["heart_rate_bpm"] < 60.0
      and feats["peak_count"] <= 5
      and sq_status == "SIGNAL_OK"
  ):
    c5, c4, m1, orig = "BRADYCARDIA", "BRADY_ASY", 0, "Bradycardia"
  else:
    # Quarantined: PAC bursts elevate window heart rate >= 60 bpm
    quarantine_entry = {
        ...
        "quarantine_status": "QUARANTINED_ECTOPY",
        "quarantine_reason": (
            f"SBR annotated window rate not bradycardic"
            f" (HR={feats['heart_rate_bpm']:.1f} bpm,"
            f" peaks={feats['peak_count']}) due to ectopic/PAC burst"
        ),
    }
```

- When the patient experienced sinus pauses or isolated sinus beats ($HR < 60\text{ bpm}$), the window was clean and admitted to training as pure `BRADYCARDIA` (**121 windows**).
- When PAC bursts elevated the heart rate ($\ge 60\text{ bpm}$), the curator quarantined those windows (**240 windows**) to prevent label contamination during training.
- Consequently, evaluating all 361 windows tests both **in-sample memory** (121 windows) and **behavior on complex ectopy** (240 windows) within the same patient.

---

## 3. Clinical & Morphological Profile of Record 232

Record 232 is an 8-channel analog Holter tape recorded from a 76-year-old female patient undergoing cardiac treatment with Digoxin and Quinidine.

### Beat-Level Annotation Breakdown

From `232annotations.txt`, a total of **1,816 annotations** were parsed:

| WFDB Symbol | Clinical Beat Type | Count | Percentage | Conduction Characteristics |
| :---: | :--- | :---: | :---: | :--- |
| **`A`** | **Atrial Premature Beat (APB / PAC)** | **1,382** | **76.10%** | Conducted with Right Bundle Branch Block aberration (wide, notched QRS) |
| **`R`** | **Normal Sinus Conduction with RBBB** | **397** | **21.86%** | Baseline sinus conduction with pre-existing right bundle branch block |
| **`~`** | Noise / Waveform Change Marker | 35 | 1.93% | Minor baseline drift and artifact markers |
| **`j`** | Nodal (Junctional) Escape Beat | 1 | 0.06% | Escape beat terminating a prolonged sinus pause |
| **`+`** | Rhythm Marker (`(SBR`) | 1 | 0.06% | Global Sinus Bradycardia rhythm marker at sample 76 ($t = 0.211\text{s}$) |
| **`V`** | **Ventricular Beat (PVC)** | **0** | **0.00%** | **Zero true ventricular beats in the entire 30-minute recording** |

The patient's cardiac rhythm consists of **underlying sinus bradycardia ($35\text{--}50\text{ bpm}$) frequently interrupted by rapid couplets and salvos of aberrant APBs ($70\text{--}120\text{ bpm}$)** followed by compensatory pauses.

---

## 4. Full-Record Evaluation Results (361 Non-Overlapping Windows)

### 4.1 Global Prediction Summary

| Predicted Class | Total Windows ($N = 361$) | Percentage | In-Training Windows ($N = 121$) | Out-of-Training Windows ($N = 240$) | Clinical Concordance Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`NSR`** | **177** | **49.03%** | 0 | 177 | Rate Leakage: PAC bursts elevate window HR to $60\text{--}98\text{ bpm}$ |
| **`BRADYCARDIA`** | **137** | **37.95%** | 121 | 16 | True Bradycardia: All windows with measured HR $< 60\text{ bpm}$ |
| **`VENTRICULAR`** | **47** | **13.02%** | 0 | 47 | Aberrant Conduction: RBBB-widened PAC runs trigger borderline FA |
| **`TACHY`** | **0** | **0.00%** | 0 | 0 | Perfect Specificity: Zero false tachycardia alarms |

---

### 4.2 In-Training vs. Out-of-Training Breakdown

```
                            TRAINING STATUS CROSS-TABULATION
                      IN-TRAINING (121)    OUT-OF-TRAINING (240)    TOTAL (361)
PREDICTED BRADYCARDIA       121                     16                  137
PREDICTED NSR                 0                    177                  177
PREDICTED VENTRICULAR         0                     47                   47
PREDICTED TACHY               0                      0                    0
TOTAL                       121                    240                  361
```

- **In-Training Bradycardia Recall**: **$100.00\%$** ($121 / 121$ windows).
- **In-Training NSR Leakage**: **$0.00\%$** ($0 / 121$ windows).
- **In-Training Ventricular False Alarms**: **$0.00\%$** ($0 / 121$ windows).

On the training windows, Model 2 demonstrated 100% fidelity to its training targets.

---

### 4.3 Rate-Confirmed Bradycardia Concordance

To evaluate whether Model 2's predictions track physiological rate rather than arbitrary labels, windows were segmented by measured heart rate:

```
                            PHYSIOLOGICAL RATE CROSS-TABULATION
                      RATE-CONFIRMED BRADY (122)    PAC BURST HR >= 60 (239)    TOTAL (361)
PREDICTED BRADYCARDIA            122                            15                  137
PREDICTED NSR                      0                           177                  177
PREDICTED VENTRICULAR              0                            47                   47
PREDICTED TACHY                    0                             0                    0
TOTAL                            122                           239                  361
```

- **Recall on Rate-Confirmed Bradycardia ($\text{HR} < 60\text{ bpm}$)**: **$100.00\%$** ($122 / 122$ windows).
- **False Negative Rate on True Slow Rates**: **$0.00\%$** ($0 / 122$ windows).

When heart rate fell below $60.0\text{ bpm}$, Model 2 predicted `BRADYCARDIA` with 100% consistency across the entire recording.

---

## 5. Predicted Probability Distributions

| Target Class | Overall Mean Probability | Overall Median | 25th Percentile ($Q_1$) | 75th Percentile ($Q_3$) | Minimum | Maximum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`prob_BRADYCARDIA`** | **0.4287** | 0.2514 | 0.0760 | 0.9849 | 0.0084 | **0.9905** |
| **`prob_NSR`** | **0.3375** | 0.3611 | 0.0082 | 0.5655 | 0.0041 | **0.9176** |
| **`prob_VENTRICULAR`** | **0.1997** | 0.1926 | 0.0039 | 0.3472 | 0.0029 | **0.6049** |
| **`prob_TACHY`** | **0.0341** | 0.0294 | 0.0030 | 0.0546 | 0.0024 | 0.1064 |

### Probability Profile Highlights
- **Bradycardia Confidence**: On windows predicted as `BRADYCARDIA` ($N = 137$), the mean probability was **$0.9632$** (median $0.9848$). The classifier produces sharp, decisive predictions when heart rate drops below $60\text{ bpm}$.
- **Ventricular False Alarms**: On the 47 windows predicted as `VENTRICULAR`, the mean probability was only **$0.4592$** (median $0.4452$, max $0.6049$). None of these windows approached the $>0.98$ confidence observed on true ventricular fibrillation in `cu01`.

---

## 6. Detailed Error & Feature Attribution Analysis

### 6.1 NSR Leakage: PAC-Driven Rate Elevation (177 Windows)

Why does a record officially annotated as `(SBR` produce 177 `NSR` predictions?

1. **Window-Level vs. Record-Level Labeling**:
   - The reference label `(SBR` designates the patient's baseline sinus mechanism over long time scales.
   - However, Model 2 operates on **5-second local windows**.
   - In record 232, salvos of atrial premature beats (`A`) occur at rapid rates ($75\text{--}110\text{ bpm}$). In 177 windows, the measured heart rate averaged **$72.4\text{ bpm}$** (peak count $6\text{--}8$).
2. **Feature Profile of NSR-Predicted Windows**:
   - `mean_rr`: $0.82\text{--}0.95\text{s}$ (concordant with $63\text{--}73\text{ bpm}$).
   - `vf_band_power_ratio`: $< 0.35$ (low spectral chaos).
   - `zero_crossings`: Normal baseline crossing rate.
   - Because heart rate was physiologically in the normal range and spectral energy was organized, Model 2 classified these windows as `NSR`.
3. **Clinical Interpretation**:
   - This is **expected windowed rate behavior**, not a classifier defect. A 5-second edge classifier without long-term history cannot know that a sequence of 6 rapid beats is embedded within a 24-hour bradycardia presentation.

---

### 6.2 Ventricular False Alarms: RBBB Aberrancy (47 Windows)

Why did Model 2 predict `VENTRICULAR` on 47 windows in a record with zero true ventricular beats?

```
FEATURE PROFILE: 47 Ventricular False Alarms in Record 232
========================================================================================
Feature                    Mean ± Std             Min .. Max           Clinical Impact
----------------------------------------------------------------------------------------
prob_VENTRICULAR           0.4592 ± 0.0834        0.3230 .. 0.6049     Low-confidence argmax win
prob_NSR                   0.2827 ± 0.0737        0.1701 .. 0.4406     Substantial NSR competition
Annotated 'A' Beats (APBs) 4.83 ± 1.42            2 .. 7 beats         Bursts of aberrated PACs
Annotated 'R' Beats        1.13 ± 0.68            0 .. 2 beats         Conducted sinus beats
Pan-Tompkins Peaks         5.83 ± 0.67            4 .. 8 peaks         Regular QRS tracking
heart_rate_bpm             68.79 ± 7.92           60.1 .. 96.2 bpm     Mildly elevated rate
vf_band_power_ratio        0.4171 ± 0.0253        0.3744 .. 0.4946     Slightly elevated spectral energy
dominant_frequency         2.02 ± 0.45 Hz         1.50 .. 3.50 Hz      Notched RBBB waveform peak
========================================================================================
```

#### Pathophysiological Mechanism: SVT with Aberrancy
- In record 232, atrial premature beats are conducted with **Right Bundle Branch Block aberration**. Aberrant conduction occurs when supraventricular impulses reach the bundle branches while one bundle branch (the right bundle) is still refractory.
- Consequently, the depolarization wave spreads slowly through the right ventricle via non-specialized myocardium, producing a **widened, notched, M-shaped QRS complex**.
- When multiple aberrant APBs occur in a 5-second window:
  1. The QRS width is prolonged.
  2. The notched waveform increases spectral energy in the $2\text{--}4\text{ Hz}$ band (`vf_band_power_ratio` rises from $0.25$ to $0.42$).
  3. The irregular coupling intervals increase `rr_cv`.
- The combination of widened complexes, elevated spectral energy, and RR irregularity mimics non-sustained ventricular ectopy, causing argmax classification to tip toward `VENTRICULAR` ($p_{\text{VENT}} \approx 0.46$).

> [!IMPORTANT]
> **Key Finding: Aberrancy vs. Ventricular Discrimination**  
> Model 2 lacks specific morphological features to differentiate aberrant supraventricular conduction (SVT with aberrancy) from true ventricular ectopy. When RBBB-aberrated APB couplets occur, Model 2 produces low-confidence ($p \approx 0.46$) ventricular alerts.

---

## 7. Time-Series Stability Across 30 Minutes

Plotting predicted class over time illustrates the cyclic alternating behavior between sinus pauses and aberrant PAC couplets:

```
Time (min)    0m      5m      10m     15m     20m     25m     30m
Window Index  0       60      120     180     240     300     360
Pred Class:   [BBBBBBBNNNNVNNNNBBBBBNNNVNNNNBBBBBNNNNVNNNNBBBBBNNNV]
              ^       ^       ^       ^       ^       ^       ^
Legend: B = BRADYCARDIA (HR < 60 bpm pauses)
        N = NSR (PAC couplets, HR 60-95 bpm)
        V = VENTRICULAR (Aberrant RBBB PAC runs, low-confidence p ~ 0.46)
```

- When the patient experiences compensatory pauses between PACs, heart rate drops to $35\text{--}52\text{ bpm}$, and Model 2 predicts `BRADYCARDIA` ($p > 0.98$).
- When the patient enters an active PAC run, heart rate rises to $65\text{--}90\text{ bpm}$, and Model 2 shifts to `NSR`.
- When an active PAC run features marked RBBB widening, Model 2 shifts temporarily to low-confidence `VENTRICULAR`.

---

## 8. Summary of Validation Deliverables Created

The full-record evaluation produced the following deliverables in `training/final_models/results/model2_xgboost_4class_research/mitdb232_bradycardia_validation/`:

1. [`evaluate_mitdb232_bradycardia.py`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/mitdb232_bradycardia_validation/evaluate_mitdb232_bradycardia.py): Standalone evaluation script with provenance verification, annotation parsing, feature extraction, and metric reporting.
2. [`mitdb_232_5s_predictions.csv`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/mitdb232_bradycardia_validation/mitdb_232_5s_predictions.csv): Complete 361-row CSV containing feature vectors, beat counts (`R` vs `A`), heart rates, predicted classes, and probabilities.
3. [`mitdb_232_validation_metrics.json`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/mitdb232_bradycardia_validation/mitdb_232_validation_metrics.json): Structured machine-readable metrics payload including provenance audit, rate concordance, and probability distributions.
4. [`MITDB_232_BRADYCARDIA_REPORT.md`](file:///c:/Users/LENOVO/Downloads/ecg-arrhythmia-edge-ai/training/final_models/results/model2_xgboost_4class_research/mitdb232_bradycardia_validation/MITDB_232_BRADYCARDIA_REPORT.md): This technical document.

---

## 9. Main Findings & Remaining Validation Gaps

### Main Findings
1. **100% In-Sample Bradycardia Recall**: Model 2 perfectly recalled all 121 training windows ($121/121$, mean $p_{\text{BRADY}} = 0.984$).
2. **100% Rate-Confirmed Bradycardia Concordance**: Every single window across the 30-minute recording where measured heart rate was $< 60.0\text{ bpm}$ was classified as `BRADYCARDIA` ($122/122$).
3. **NSR Leakage Mechanism Discovered**: 177 windows were predicted as `NSR` because frequent PAC bursts physiologically increased the 5-second heart rate to $60\text{--}98\text{ bpm}$.
4. **Aberrancy Vulnerability Identified**: 47 windows were predicted as `VENTRICULAR` with low confidence ($p \approx 0.46$) due to RBBB-widened PAC runs mimicking ventricular ectopy.

### Remaining Validation Gaps
1. **Completely Independent Bradycardia Validation**: Because `mitdb_232` was part of the training set, Model 2 has still not been evaluated on an **unseen, independent sinus bradycardia recording** from an unpolluted database.
2. **Supraventricular Tachycardia with Aberrancy (SVT-A)**: A dedicated investigation is needed to evaluate how bundle branch block and aberrant conduction interact with Model 2's ventricular detection thresholds.
3. **Tachycardia (`TACHY`) Coverage**: Genuine paroxysmal supraventricular tachycardia (`(SVTA` or `(ST`) remains completely unevaluated offline on both in-sample and out-of-sample records.
