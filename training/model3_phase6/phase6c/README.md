# Model 3 — Phase 6C: CUDB Quarantined Record Re-Adjudication & Rhythm Audit

## 1. Overview & Objective

Phase 6C performs an evidence-based, strictly read-only audit of the quarantined Creighton University Ventricular Tachyarrhythmia Database (`CUDB`) records (`cu04` through `cu35`).

### Scientific Background
In Phase 2E.2 and Phase 5, Model 3 (VF vs. VT subtype classifier) utilized only 3 CUDB records (`cu01`, `cu02`, `cu03`) because their WFDB annotations contained explicit subtype strings (`(VF` or `(VT`). The remaining 32 records (`cu04`–`cu35`) were quarantined because their reference annotations delineate life-threatening arrhythmias with onset and offset bracket markers (`[` and `]`) without differentiating between:
- Monomorphic Ventricular Tachycardia (MVT)
- Polymorphic Ventricular Tachycardia (PVT)
- Ventricular Flutter (VFL)
- Coarse Ventricular Fibrillation (CVF)
- Fine Ventricular Fibrillation (FVF)

### Core Mandate
* **Do NOT assume all CUDB episodes are VF**. PhysioNet explicitly documents that episodes often begin as ventricular tachycardia before degenerating into fibrillation.
* **Do NOT invent clinical labels**. Preserve source ambiguity whenever definitive subtype evidence is absent.
* **Do NOT treat multiple windows as independent patients**. Focus strictly on patient/record-level diversity ($N$).
* **Do NOT rebuild the ML dataset** or train models in Phase 6C. This phase is diagnostic auditing and evidence synthesis only.

---

## 2. Directory Structure

```
training/model3_phase6/phase6c/
├── README.md                          # Methodology, governance, and audit documentation
├── smoke_test.py                      # Pre-flight smoke test verifying waveform loading & diagnostics
├── run_cudb_audit.py                  # Full audit runner for records cu04 through cu35
└── results/
    ├── cudb_episode_audit.csv         # Machine-readable audit table across all candidate episodes
    ├── final_report.md                # Synthesis report, patient diversity impact, and decision gate
    └── plots/                         # 4-panel diagnostic visualizations per candidate episode
        ├── cu04_ep01.png
        ├── cu04_ep02.png
        └── ...
```

---

## 3. Rhythm Classification Governance Categories

Every candidate episode is placed into exactly ONE of the following categories:
1. `MONOMORPHIC_VT`: Regular, discrete wide-complex tachycardia ($f_{\text{dom}} \approx 2.0 - 4.0\text{ Hz}$), high autocorrelation periodicity, low/moderate complexity.
2. `POLYMORPHIC_VT`: Irregular wide-complex tachycardia with beat-to-beat axis/morphology variations.
3. `VENTRICULAR_FLUTTER`: Rapid, regular sinusoidal oscillations without isoelectric baseline ($f_{\text{dom}} \approx 3.5 - 6.0\text{ Hz}$), very high spectral concentration, high autocorrelation periodicity.
4. `COARSE_VF`: Chaotic, disorganized fibrillatory waves ($f_{\text{dom}} \approx 3.0 - 7.0\text{ Hz}$), high spectral entropy, high Sample Entropy, high Lempel-Ziv complexity.
5. `FINE_VF`: Low-amplitude fibrillatory activity ($\text{peak-to-peak} < 0.3\text{ mV}, \text{RMS} < 0.08\text{ mV}$), high entropy, absence of discrete QRS.
6. `NON_TARGET_RHYTHM`: Paced baseline, motion/lead artifact, supraventricular rhythms (e.g. `cu09` AFib).
7. `AMBIGUOUS`: Mixed, transitional, or unresolvable morphology where VT, flutter, and fibrillation coexist or rapidly degenerate.
8. `INSUFFICIENT_EVIDENCE`: Episodes failing duration or signal quality thresholds.

Inclusion Recommendations:
- `INCLUDE_VT`: Verified sustained VT.
- `INCLUDE_VF`: Verified sustained VF or Ventricular Flutter.
- `EXCLUDE`: Non-target, artifact, or paced non-shockable rhythm.
- `REQUIRES_MANUAL_REVIEW`: Plausible target rhythm, but source annotation lacks physician-signed subtype confirmation.
- `EXCLUDE_TOO_SHORT`: Episode duration $< 5.0\text{ s}$.

---

## 4. Waveform Diagnostics Computed

For each candidate episode, 10 descriptive physical and mathematical diagnostics are extracted on a representative 5.0-second window:
1. **Dominant Frequency ($f_{\text{dom}}$)**: Peak frequency in the 0.5–30 Hz band via Welch PSD.
2. **Spectral Entropy**: Normalized Shannon entropy of the power spectrum.
3. **Spectral Concentration**: Fractional power concentrated in $[f_{\text{dom}} - 1, f_{\text{dom}} + 1]\text{ Hz}$.
4. **Spectral Flatness**: Ratio of geometric mean to arithmetic mean of spectral power.
5. **Autocorrelation Periodicity**: Height of the primary secondary peak above preceding trough.
6. **Autocorrelation Decay Time**: Time lag at which ACF drops below $1/e$.
7. **Sample Entropy**: Downsampled (45 Hz, $N=225, m=2, r=0.2\sigma$) conditional probability of pattern repetition.
8. **Lempel-Ziv Complexity**: 90 Hz downsampled binary sequence complexity rate.
9. **Hjorth Mobility & Complexity**: Variance ratio of first and second derivatives.
10. **Signal Amplitude Statistics**: Min, max, mean, std, peak-to-peak, and RMS amplitude in mV.
