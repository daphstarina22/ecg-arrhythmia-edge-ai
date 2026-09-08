# Model 3 — Phase 6: Dataset Expansion & Generalization Validation

## 1. Objective

Phase 6 investigates whether the cross-database generalization collapse observed in Model 3 (VF vs. VT classifier) is primarily caused by **insufficient independent patient diversity** in the ventricular fibrillation class.

### Key Conclusions Leading to Phase 6
* **Phase 4.2**: Hand-crafted 7-feature models collapsed across datasets (D2 ROC-AUC ~0.59–0.61) due to spectral direction inversion and Pan-Tompkins QRS peak breakdown on fibrillatory waves.
* **Phase 5**: Raw-waveform physiological features (autocorrelation decay/lag, downsampled Lempel-Ziv complexity, and Sample Entropy) eliminated sign inversion and improved D2 ROC-AUC to a maximum of **0.6825**, but failed the generalization gate (ROC-AUC > 0.75, balanced record recall > 70%).
* **Root Bottleneck**: The entire dataset contains only **16 independent physical patient records with ventricular fibrillation** (6 in Challenge 2015, 8 in VFDB, 2 in CUDB). The 14,618 sliding windows are pseudo-replicates of these 16 patients.

---

## 2. Governance & Strict Boundaries

1. **No New Feature Engineering or Deep Models Yet**: Do not introduce convolutional networks, complex ensemble models, or new feature definitions until the dataset expansion analysis is completed.
2. **Directory Isolation**: All Phase 6 code, documentation, and artifacts must reside strictly under `training/model3_phase6/`.
3. **Immutability of Prior Phases**:
   * Do NOT modify `src/`, `models/`, `tests/`, or production ONNX runtimes.
   * Do NOT modify Phase 2, Phase 3, Phase 4, or Phase 5 artifacts.
4. **Leakage Prevention**: Never mix sliding windows from the same patient across training and testing partitions.

---

## 3. Directory Structure

```
training/model3_phase6/
├── README.md                                  # Overview, boundaries, and workflow instructions
├── DATASET_INVENTORY.md                       # Comprehensive 12-point audit of 11 candidate ECG databases
├── PHASE6A_AUDIT_REPORT.md                    # Phase 6A local archives audit report
├── dataset_inventory.csv                      # Machine-readable inventory of audited databases
├── local_records_audit.csv                    # Audit of all 83 local records in CUDB & MIT-BIH
├── extract_mitbih_records.py                  # Phase 6B MIT-BIH extraction and 12-point integrity audit runner
├── artifacts/
│   └── expanded_physiological_feature_table.csv # 14,704 windows (Phase 5 + MIT-BIH 205, 207, 223)
└── results/
    ├── segment_audit_table.csv                # Audit of extracted MIT-BIH rhythm segments
    ├── expansion_audit.json                   # 12-point programmatic integrity audit output
    └── final_report.md                        # Phase 6B final synthesis, patient audit, and decision gate
```

---

## 4. Phase 6 Status & Roadmap

* **Phase 6A — Dataset Inventory & Expansion Audit** [COMPLETED]:
  * Full audit of 11 public/restricted ECG databases across 12 criteria.
  * Discovered critical **100% patient identity overlap between SDDB and VFDB** (prevented catastrophic data leakage).
  * Audited 83 local records across CUDB and MIT-BIH: quarantined 25 ambiguous CUDB records and identified 3 usable MIT-BIH records (205, 207, 223).
* **Phase 6B — MIT-BIH Dataset Expansion & Patient-Level Limitation Audit** [COMPLETED]:
  * Extracted verified sustained segments: Record 205 (VT), Record 207 (VFL $\to$ VF), Record 223 (VT).
  * Extracted 21 Strategy B physiological features across 86 new windows.
  * Created `expanded_physiological_feature_table.csv` ($14,704$ windows across 120 records).
  * 12/12 integrity audit passed (zero missing values, zero duplicates, zero ID collisions, original data unmodified).
  * Patient-level limitation audit: VF patient cohort increased from 16 to 17 records (+1 record, +6.25%).
  * Concluded with Decision Gate:
    $$\text{\textbf{DATASET EXPANSION INSUFFICIENT — ACQUIRE ADDITIONAL INDEPENDENT VF DATA}}$$

