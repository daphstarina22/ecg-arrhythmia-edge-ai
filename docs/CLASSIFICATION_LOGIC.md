# Classification logic (Policy B)

Models (in `models/`) and verified label maps (from `training/final_models/common/data_loader.py` and `train_model2.py`):

| Model | Output id → label |
|---|---|
| M1 shockable | 0 `NON_SHOCKABLE`, 1 `SHOCKABLE` |
| M2 rhythm    | 0 `NSR`, 1 `TACHY`, 2 `BRADY_ASY`, 3 `VENTRICULAR` |

Model 1 and Model 2 run exactly once per Start. Model 3 is removed from the runtime decision path.

## HMI output (`ecg_edge/decision.py`)

Exhaustive M1 $\times$ M2 truth table (8 combinations):

| Rhythm (M2) | Shock (M1) | HMI banner shows | Speech / Voice prompt |
|---|---|---|---|
| NSR | SHOCKABLE | `CONFLICT — NOT CLASSIFIED` | "Model disagreement. No classification. Re-check electrodes and repeat." |
| TACHY | SHOCKABLE | `VF/VT SHOCKABLE` | "Shockable rhythm detected. VF or VT." |
| BRADY_ASY | SHOCKABLE | `CONFLICT — NOT CLASSIFIED` | "Model disagreement. No classification. Re-check electrodes and repeat." |
| VENTRICULAR | SHOCKABLE | `VF/VT SHOCKABLE` | "Shockable rhythm detected. VF or VT." |
| NSR | NON_SHOCKABLE | `NORMAL SINUS RHYTHM` | "Normal sinus rhythm. Non shockable." |
| TACHY | NON_SHOCKABLE | `TACHYCARDIA` | "Tachycardia. Non shockable." |
| BRADY_ASY | NON_SHOCKABLE | `OTHER — NOT CLASSIFIED` | "Other rhythm. Not classified." |
| VENTRICULAR | NON_SHOCKABLE | `VENTRICULAR (NON-SHOCKABLE)` | "Ventricular rhythm, non shockable." |

Signal rejected by the quality gate → `NO RESULT — POOR SIGNAL`. Hardware/timing fault → `NO RESULT — ACQUISITION ERROR`.

## Policy notes

1. **Combined VF/VT Output**: Defibrillator emergency response relies on shockability triage (`VF/VT SHOCKABLE`) rather than unreliable surface-subtyping between coarse VF and rapid VT.
2. **Model 3 Removal**: Model 3 (VF vs VT) is not loaded or executed at runtime.
3. **Bradycardia Handling**: `BRADY_ASY` is excluded from standalone diagnostic display and mapped safely to `OTHER — NOT CLASSIFIED`.
4. **Deterministic Contradiction**: Any discordant prediction between shockability and baseline non-ventricular rhythms (`SHOCKABLE` with `NSR` or `BRADY_ASY`) yields `CONFLICT — NOT CLASSIFIED`.
