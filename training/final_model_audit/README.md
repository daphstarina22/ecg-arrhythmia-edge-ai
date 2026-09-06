# Final Model & System Architecture Audit

**Date**: September 6, 2026  
**Status**: Completed (Read-Only Audit)  
**Scope**: Model 1 (Shockable), Model 2 (Multiclass), Model 3 (VF vs. VT), ONNX Deployment, and System Architecture  
**Strict Safety Mode**: No production code, deployed models, or historical artifacts were modified.

---

## Executive Summary

Following the completion of the Phase 2–7 Model 3 research investigation, this audit examined the complete 3-model system deployed in the repository.

The audit verified 100% byte-for-byte provenance linking the deployed ONNX models (`models/*.onnx`) directly to `notebook346a8689af (1).ipynb` (Cell 102). However, structural investigation revealed that **Model 1 and Model 2 suffer from severe dataset design vulnerabilities and source confounding**, while the system-level inference architecture contains critical clinical contradictions.

### Key Audit Findings

1. **Model 1 (Shockable vs. Non-Shockable: 99% Accuracy Illusion)**:
   - **Critical Confounding**: All 32,773 Non-Shockable training windows originated exclusively from MIT-BIH Arrhythmia Database, while all 3,198 Shockable windows originated exclusively from VFDB (95.7%) and CUDB (4.3%).
   - The reported 0.99 ROC-AUC reflects the classifier learning the digitization, filtering, and amplitude characteristics of MIT-BIH vs. VFDB hardware rather than physiological rhythm features.
   - Non-shockable rhythms from VFDB, CUDB, and Challenge 2015 were deliberately discarded during notebook training to protect training accuracy.

2. **Model 2 (Multiclass Arrhythmia Classifier)**:
   - **Severe Class-to-Source Entanglement**: 
     - `Tachycardia` ($N=15,210$) originated **100% from Challenge 2015** (0% from other sources).
     - `Normal` ($N=31,933$) originated **0% from Challenge 2015** (all from MIT-BIH, VFDB, and CUDB).
   - Challenge 2015 exhibits an 8x–20x amplitude scale discrepancy (`ptp` mean = 20.88 vs. 1.61–2.28 in Holter databases), meaning the model distinguishes Normal from Tachycardia primarily via signal voltage scaling rather than heart rate or rhythm morphology.
   - Poor discrimination for critical classes: VF precision is only **36.0%**.

3. **System Architecture & Inference Conflicts**:
   - **Contradictory Routing**: Model 2 independently predicts `VF` and `VT`, competing with both Model 1 and Model 3. If Model 1 predicts non-shockable, Model 3 is bypassed, yet Model 2 can simultaneously output a VF alarm.
   - **No Human-Readable Class Mappings**: `src/inference.py` returns raw integer arrays (`0, 1, 2, 3, 4`) with zero internal label mapping dictionaries.
   - **Runtime Inefficiency**: `src/inference.py` re-instantiates new `OnnxClassifier` sessions and re-reads 4.3 MB of models on every single 5-second window.

---

## Audit Reports Directory

| Document | Focus & Content |
| :--- | :--- |
| [`MODEL1_AUDIT.md`](./MODEL1_AUDIT.md) | Deep-dive into Model 1 dataset composition, source co-extensiveness, amplitude leakage, evaluation quality, and verdict. |
| [`MODEL2_AUDIT.md`](./MODEL2_AUDIT.md) | Deep-dive into Model 2 class taxonomy, imbalance, source entanglement, confusion patterns, and verdict. |
| [`ONNX_DEPLOYMENT_AUDIT.md`](./ONNX_DEPLOYMENT_AUDIT.md) | Byte-level ONNX inspection, graph inputs/outputs, node attributes, and contract verification against `src/`. |
| [`SYSTEM_ARCHITECTURE_AUDIT.md`](./SYSTEM_ARCHITECTURE_AUDIT.md) | Interaction logic across all 3 models, routing deadlocks, contradictory alerts, and edge engine performance. |
| [`FINAL_DECISION.md`](./FINAL_DECISION.md) | Comprehensive decision table, minimum required pre-training actions, and prioritized execution roadmap. |

---

## Supporting Data Artifacts

- [`results/model1_source_label_matrix.csv`](./results/model1_source_label_matrix.csv): Exact window and record counts by source for Model 1.
- [`results/model2_taxonomy_source_matrix.csv`](./results/model2_taxonomy_source_matrix.csv): 5-class distribution, source contributions, and risk levels for Model 2.
- [`results/onnx_model_contract_audit.csv`](./results/onnx_model_contract_audit.csv): Verified tensor dimensions, data types, node types, and file sizes.
