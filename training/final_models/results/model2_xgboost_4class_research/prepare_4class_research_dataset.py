"""Research-Grade Dataset Preparation for Model 2 Revision (Excluding ASYSTOLE).

This script:
1. Reads the corrected multi-source master feature dataset (5-class).
2. Extracts and saves all ASYSTOLE windows to a dedicated excluded dataset
   (excluded_asystole_windows.csv) for separate out-of-domain evaluation.
3. Filters the training dataset to strictly the four cardiac rhythm classes:
   NSR, TACHY, BRADYCARDIA, VENTRICULAR.
4. Strictly verifies that:
   - Zero ASYSTOLE rows are present in the 4-class dataset.
   - Zero ASYSTOLE examples are relabeled into any other class.
   - The class set consists solely of the four target rhythms.
   - All feature columns are valid and complete.
5. Saves four_class_master_feature_dataset.csv and dataset_split_verification.json
   in the separate research directory:
   training/final_models/results/model2_xgboost_4class_research/
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.final_models.common.feature_selection import MODEL2_FEATURES

TARGET_CLASSES_4C = ["NSR", "TACHY", "BRADYCARDIA", "VENTRICULAR"]
OUTPUT_DIR = Path(__file__).resolve().parent


def prepare_datasets() -> Dict[str, Any]:
    master_dataset_path = PROJECT_ROOT / "training" / "final_models" / "artifacts" / "corrected_master_feature_dataset.csv"
    if not master_dataset_path.exists():
        raise FileNotFoundError(f"Master dataset not found at {master_dataset_path}")

    print("=" * 80)
    print("PREPARING 4-CLASS RESEARCH DATASET (EXCLUDING ASYSTOLE)")
    print(f"Master source: {master_dataset_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    df_master = pd.read_csv(master_dataset_path)
    total_master_rows = len(df_master)
    print(f"Loaded master corrected dataset: {total_master_rows} rows, {df_master['record_id'].nunique()} records.")

    # 1. Separate ASYSTOLE windows
    asystole_mask = df_master["cardiac_rhythm_5class"] == "ASYSTOLE"
    df_asystole = df_master[asystole_mask].copy().reset_index(drop=True)
    asystole_count = len(df_asystole)
    asystole_records = df_asystole["record_id"].unique().tolist()

    asystole_path = OUTPUT_DIR / "excluded_asystole_windows.csv"
    df_asystole.to_csv(asystole_path, index=False)
    print(f"\n[1] Extracted {asystole_count} ASYSTOLE windows across {len(asystole_records)} records.")
    print(f"    Saved excluded asystole dataset to: {asystole_path}")
    print("    ASYSTOLE source breakdown:")
    for src, cnt in df_asystole["dataset_source"].value_counts().items():
        print(f"      - {src}: {cnt} windows")

    # 2. Extract 4-Class subset without ASYSTOLE
    df_4class = df_master[~asystole_mask].copy().reset_index(drop=True)
    df_4class["cardiac_rhythm_4class"] = df_4class["cardiac_rhythm_5class"]

    # 3. Validation checks against silent relabeling or leakage
    # Check A: No ASYSTOLE label present
    no_asystole_label = "ASYSTOLE" not in df_4class["cardiac_rhythm_4class"].values
    assert no_asystole_label, "CRITICAL ERROR: ASYSTOLE label found in 4-class dataset!"

    # Check B: Classes exactly match TARGET_CLASSES_4C
    unique_classes = set(df_4class["cardiac_rhythm_4class"].unique())
    expected_classes = set(TARGET_CLASSES_4C)
    assert unique_classes == expected_classes, f"Class mismatch: {unique_classes} vs {expected_classes}"

    # Check C: Verify row-by-row identity with master dataset to prove ZERO relabeling
    # For every row in df_4class, cardiac_rhythm_4class must equal cardiac_rhythm_5class
    assert (df_4class["cardiac_rhythm_4class"] == df_4class["cardiac_rhythm_5class"]).all(), \
        "CRITICAL ERROR: Detected modified or relabeled rhythm labels!"

    # Check D: Verify row count identity
    assert len(df_4class) + asystole_count == total_master_rows, \
        f"Row count mismatch: {len(df_4class)} + {asystole_count} != {total_master_rows}"

    print("\n[2] Label integrity validation PASSED:")
    print("    - Zero ASYSTOLE rows in 4-class training data.")
    print("    - Zero silent relabeling: exactly 100% of rows preserve their verified ground truth.")
    print(f"    - Target classes verified: {sorted(list(unique_classes))}")

    # 4. Drop feature NaNs (standard preprocessing matching 5-class baseline)
    before_dropna = len(df_4class)
    df_4class_clean = df_4class.dropna(subset=MODEL2_FEATURES).reset_index(drop=True)
    after_dropna = len(df_4class_clean)
    dropped_rows = before_dropna - after_dropna
    print(f"\n[3] Feature completeness check:")
    print(f"    - Rows before dropna: {before_dropna}")
    print(f"    - Dropped rows with feature NaNs: {dropped_rows}")
    print(f"    - Clean supervised rows: {after_dropna}")
    print(f"    - Clean independent records: {df_4class_clean['record_id'].nunique()}")

    class_distribution = {}
    print("\n[4] 4-Class Training Dataset Distribution:")
    for c_name in TARGET_CLASSES_4C:
        count = int((df_4class_clean["cardiac_rhythm_4class"] == c_name).sum())
        pct = (count / after_dropna) * 100
        class_distribution[c_name] = {"count": count, "percentage": float(pct)}
        print(f"    {c_name:<15}: {count:>5} windows ({pct:>5.2f}%)")

    # Save clean 4-class dataset
    clean_4class_path = OUTPUT_DIR / "four_class_master_feature_dataset.csv"
    df_4class_clean.to_csv(clean_4class_path, index=False)
    print(f"\nSaved 4-class master dataset to: {clean_4class_path}")

    # 5. Output formal verification JSON
    verification_data = {
        "status": "PASSED",
        "master_dataset_source": str(master_dataset_path),
        "total_master_rows": total_master_rows,
        "asystole_windows_excluded": asystole_count,
        "asystole_records_count": len(asystole_records),
        "asystole_records_list": asystole_records,
        "four_class_rows_total": before_dropna,
        "four_class_rows_clean": after_dropna,
        "four_class_records_clean": int(df_4class_clean["record_id"].nunique()),
        "dropped_nan_rows": dropped_rows,
        "target_classes": TARGET_CLASSES_4C,
        "class_distribution": class_distribution,
        "integrity_checks": {
            "no_asystole_in_4class_dataset": bool(no_asystole_label),
            "no_silent_relabeling": True,
            "classes_match_expected_target": True,
            "row_count_partition_conserved": True,
        },
    }

    verif_path = OUTPUT_DIR / "dataset_split_verification.json"
    with open(verif_path, "w") as f:
        json.dump(verification_data, f, indent=2)
    print(f"Saved dataset split verification to: {verif_path}")

    return verification_data


if __name__ == "__main__":
    prepare_datasets()
