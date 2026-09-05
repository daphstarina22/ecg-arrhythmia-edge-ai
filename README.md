# ECG Arrhythmia Detection Pipeline

Trains three XGBoost models from multi-source ECG data, exported to ONNX for deployment on a Raspberry Pi. All three models share one 10-feature vector computed per 5-second window, so a single feature-extraction pass on-device feeds all three.

## Models

| Model | Task | Classes | File |
|---|---|---|---|
| 1 | Shockable vs. non-shockable (primary decision) | binary | `shockable_classifier.onnx` |
| 2 | Rhythm type (informational, does not gate the shock decision) | Normal, Tachycardia, VT, Asystole, VF | `arrhythmia_multiclass.onnx` |
| 3 | VF vs. VT subtype (shown only when Model 1 says shockable) | VT, VF | `vf_vt_subtype_classifier.onnx` |

## Data sources

| Source | Contributes to | Notes |
|---|---|---|
| VFDB (MIT-BIH Malignant Ventricular Arrhythmia) | Models 1, 2, 3 | 22 records |
| CUDB (CU Ventricular Tachyarrhythmia) | Models 1, 2, 3 | 35 records. `.atr` mixes rhythm markers with incidental beat annotations — must filter to non-empty `aux_note` before treating entries as segment boundaries. Has genuine sensor dropout (NaN) on some records (traced to `cu02`) — interpolated if <10% of a segment, skipped otherwise |
| mitdb (MIT-BIH Arrhythmia) | Models 1, 2 | 48 records, used both for multiclass labels and as the clean non-shockable set for Model 1 |
| Challenge2015 | Model 2 only | 750 records. **Excluded from Model 1** — its ADC gain (`CHALLENGE2015_GAIN=200`) is unverified (no header file in this CSV mirror), and including it dropped Model 1 precision from 99% to 38% by corrupting the amplitude-based features |

AFDB (MIT-BIH Atrial Fibrillation) was trialed and removed — see "AFIB was dropped" below.

## Key pipeline decisions

**Beat detection is `pan_tompkins_detect`, applied uniformly to every source.** Standard Pan-Tompkins stages (bandpass → derivative → squaring → moving-window integration → adaptive peak detection with a refractory period), validated against mitdb ground truth at 0.98–0.99 recall/precision. This replaced an earlier setup that used real `.atr`/`.qrs` annotations for sources that had them (mitdb, AFDB) and a cruder fallback for sources that didn't (VFDB, CUDB, Challenge2015).

That earlier setup was a real bug, not just a simplification: **the Raspberry Pi has no ground-truth annotations at inference** — every window it sees gets whatever the live detector finds. Training some sources on annotation-derived features taught the model patterns that can't occur live. This was confirmed directly with a since-removed AFIB-discrimination feature (RR-interval entropy): it separated AFIB from Normal cleanly on annotation-fed sources but showed almost no separation — and inverted — on detector-fed sources, meaning it was partly learning "came from an annotated source" rather than real rhythm structure.

**AFIB was dropped from Model 2 entirely.** AFDB was added specifically to give AFIB real volume (its own dataset with real onset/offset). Two rounds of feature engineering followed:
1. Fixing the peak-detector mismatch (see above)
2. Adding an RR-interval-entropy feature specifically for AFIB

Both were validated with real numbers, and both failed: AFIB still didn't separate from Normal on VFDB/CUDB (the entropy feature actively inverted on VFDB), while the detector swap cost Models 1 and 3 a small amount of accuracy (Model 1 shockable F1 0.99→0.98; Model 3 accuracy 0.76→0.73). AFIB was dropped rather than continuing to trade the two safety-relevant models' quality for a class that isn't separable with this feature set on this signal quality. AFDB and the entropy feature were removed along with it, since neither has any remaining purpose.

**AFDB-style volume dominance is worth watching if new sources are added.** When AFDB was still in the pipeline, its 23 records produced 334,015 windows (5-second windows, 50% overlap over long recordings) — about 89% of Model 2's entire training set, which measurably degraded VT/VF/Asystole classes that AFDB has nothing to do with. Any future high-frequency source should be capped per-record before merging (a `subsample_per_record` helper existed for this — removed with AFDB, but the pattern is worth reusing if needed).

## Feature vector (10 features, same order feeds all three models)

```
f0 = mean
f1 = std
f2 = ptp
f3 = zero_crossings
f4 = peak_count
f5 = mean_rr
f6 = rr_cv
f7 = qrs_width
f8 = dominant_freq
f9 = vf_band_power_ratio
```

## Diagnostics built into the pipeline

- **CUDB/NaN check** — confirms sensor-dropout interpolation worked (`0/897` expected)
- **Amplitude scale comparison** — flags if any source's gain assumption is off by a large factor from the others (this is how the Challenge2015 exclusion from Model 1 was caught)
- **Cross-source feature consistency check** (`feature_summary_by_source`) — computes `mean_rr`/`rr_cv`/`peak_count`/`qrs_width` for a given label across all sources; run this any time a new source is added, before trusting any model trained on it. This is what caught both the original threshold-detector mismatch and the AFIB entropy failure
- **Zero-test-support warning** (Model 2) — flags any class that landed entirely in the training split under `GroupShuffleSplit`, so a class's precision/recall being reported as near-zero can be distinguished from it simply having no test data at all

## Known open items

- Challenge2015's ADC gain is still unverified — revisit if a header file surfaces, since it's currently excluded from Model 1 on that basis alone
- Model 2's VF/VT/Asystole classes are moderate (F1 0.4–0.45 range) rather than strong — Models 1 and 3 are the ones actually gating the shock decision, so this is lower priority, but worth revisiting if Model 2's output is ever surfaced to a user directly

## Running

Delete all cells in the notebook and paste in `arrhythmia_pipeline.py` fresh, then Run All. Partial cell-by-cell edits have repeatedly caused stale-code execution during development — always run the whole file fresh after a change.
