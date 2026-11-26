# Classification logic

Models (unchanged, in `models/`) and verified label maps (from `training/final_models/common/data_loader.py`,
`train_model2.py`, `training/model3_phase5/extract_physiological_features.py`):

| Model | Output id → label |
|---|---|
| M1 shockable | 0 `NON_SHOCKABLE`, 1 `SHOCKABLE` |
| M2 rhythm    | 0 `NSR`, 1 `TACHY`, 2 `BRADY_ASY`, 3 `VENTRICULAR` |
| M3 subtype   | 0 `VT`, 1 `VF` |

M1 and M2 run once per Start. M3 runs once, only if M1=`SHOCKABLE` or M2=`VENTRICULAR`.

## HMI output (`ecg_edge/decision.py`)

| M2 rhythm | M1 | M3 | HMI shows |
|---|---|---|---|
| NSR | NON_SHOCKABLE | – | `NSR — NON-SHOCKABLE` |
| TACHY | NON_SHOCKABLE | – | `TACHY — NON-SHOCKABLE` |
| BRADY_ASY | NON_SHOCKABLE | – | `BRADY_ASY — NON-SHOCKABLE` |
| VENTRICULAR | SHOCKABLE | VF | `VF — SHOCKABLE` |
| VENTRICULAR | SHOCKABLE | VT | `VT — SHOCKABLE` |
| **anything else** | | | `CONFLICT — NOT CLASSIFIED` (raw M1/M2/M3 outputs listed, no shock statement, no shock voice prompt) |

Signal rejected by the quality gate → `NO RESULT — POOR SIGNAL`. Hardware/timing fault → `NO RESULT — ACQUISITION ERROR`.

## OPEN QUESTION — needs your decision

`VENTRICULAR + NON_SHOCKABLE + VF/VT` is contradictory: M1 and M2 were trained from the same labels
(`VENTRICULAR` ⇔ `SHOCKABLE` for every training row), so this combination can only be an error of one model.
What the **old** code did with it (executed from git history, not assumed):

* `+ VF` → displayed **SHOCKABLE / VENTRICULAR FIBRILLATION** and spoke "defibrillation shock advised" — M1's NON_SHOCKABLE was silently overridden.
* `+ VT` → displayed **NOT SHOCKABLE / VENTRICULAR TACHYCARDIA**.

The new code shows `CONFLICT` for both. Options to choose from: (a) keep CONFLICT and require a repeat capture;
(b) trust M2+M3 (show shockable); (c) trust M1 (show non-shockable); (d) require agreement over 2 consecutive Starts.
It is a one-function change in `decision.py`.

**Second undefined case you did not list:** `SHOCKABLE + (NSR | TACHY | BRADY_ASY)`. It is the same kind of
model disagreement and is also shown as CONFLICT. On synthetic ECG it occurred far more often (27/150) than
the case above (0/150) - synthetic data is not a valid accuracy estimate, but it suggests this is the case to decide on.
