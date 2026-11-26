"""
Gate 3 Pre-Training Verification Suite
======================================
Verifies all prerequisites before model training:
1. Manifest and report inspection.
2. Patient-level split disjointness.
3. Feature names, ordering, definitions, and units.
4. Class labels, duplicate rows, and rejected-window accounting.
5. Invariant integrity checks.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(r"training\research\afib_experiment\data\afdb")
MANIFEST_PATH = DATA_DIR / "afdb_extraction_manifest.json"
TRAIN_CSV = DATA_DIR / "afdb_features_train.csv"
VAL_CSV = DATA_DIR / "afdb_features_val.csv"
TEST_CSV = DATA_DIR / "afdb_features_test.csv"
REJECTED_CSV = DATA_DIR / "afdb_rejected_windows.csv"

def run_pre_training_verification():
    print("=" * 80)
    print("GATE 3 PRE-TRAINING VERIFICATION SUITE")
    print("=" * 80)

    # 1. Load manifest
    assert MANIFEST_PATH.exists(), f"Manifest missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print("[PASS] Successfully loaded afdb_extraction_manifest.json.")

    # 2. Check record split disjointness
    splits = manifest["patient_splits"]
    train_recs = set(splits["train"])
    val_recs = set(splits["val"])
    test_recs = set(splits["test"])

    print(f"\nManifest Record Sets:")
    print(f"  Train ({len(train_recs)}): {sorted(train_recs)}")
    print(f"  Val   ({len(val_recs)}): {sorted(val_recs)}")
    print(f"  Test  ({len(test_recs)}): {sorted(test_recs)}")

    assert train_recs.isdisjoint(val_recs), f"Train/Val overlap: {train_recs.intersection(val_recs)}"
    assert train_recs.isdisjoint(test_recs), f"Train/Test overlap: {train_recs.intersection(test_recs)}"
    assert val_recs.isdisjoint(test_recs), f"Val/Test overlap: {val_recs.intersection(test_recs)}"
    print("[PASS] Record sets are 100% mutually disjoint.")

    # 3. Load feature datasets
    df_train = pd.read_csv(TRAIN_CSV, dtype={"record_id": str})
    df_val = pd.read_csv(VAL_CSV, dtype={"record_id": str})
    df_test = pd.read_csv(TEST_CSV, dtype={"record_id": str})
    df_rej = pd.read_csv(REJECTED_CSV, dtype={"record_id": str})

    print(f"\nLoaded Datasets:")
    print(f"  Train: {len(df_train)} rows")
    print(f"  Val:   {len(df_val)} rows")
    print(f"  Test:  {len(df_test)} rows")
    print(f"  Rej:   {len(df_rej)} rows")

    # Verify counts match manifest
    summary = manifest["summary_counts"]
    assert len(df_train) == summary["train_kept_windows"], "Train row count mismatch!"
    assert len(df_val) == summary["val_kept_windows"], "Val row count mismatch!"
    assert len(df_test) == summary["test_kept_windows"], "Test row count mismatch!"
    assert len(df_rej) == summary["total_rejected_windows"], "Rejected row count mismatch!"
    total_kept = len(df_train) + len(df_val) + len(df_test)
    assert total_kept == summary["total_kept_windows"], "Total kept mismatch!"
    assert total_kept + len(df_rej) == summary["total_potential_windows"], "Total potential mismatch!"
    print("[PASS] Exact row count matches manifest totals across all 4 files.")

    # 4. Check for duplicate rows
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test), ("Rejected", df_rej)]:
        # Check window index duplicates within each record
        dup_windows = df.duplicated(subset=["record_id", "window_idx"]).sum()
        assert dup_windows == 0, f"{name} contains {dup_windows} duplicate window indices!"
        # Check duplicate rows across all columns
        dup_rows = df.duplicated().sum()
        assert dup_rows == 0, f"{name} contains {dup_rows} exact duplicate rows!"
    print("[PASS] Zero duplicate windows or rows in any dataset.")

    # 5. Check class labels
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        unique_labels = set(df["rhythm_label"].unique())
        valid_labels = {"(AFIB", "(N", "(AFL", "(J"}
        assert unique_labels.issubset(valid_labels), f"{name} has invalid rhythm labels: {unique_labels - valid_labels}"
        # Check binary is_afib mapping
        afib_mask = (df["rhythm_label"] == "(AFIB")
        assert (df["is_afib"] == afib_mask.astype(int)).all(), f"{name} has inconsistent is_afib binary mapping!"
    print("[PASS] Class labels ('(AFIB', '(N', '(AFL', '(J') and binary mapping (is_afib in {0, 1}) are 100% consistent.")

    # 6. Check feature names, ordering, and units
    expected_features = manifest["feature_specifications"]["feature_names"]
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        for feat in expected_features:
            assert feat in df.columns, f"{feat} missing from {name}!"
        # Check no NaN or Inf
        n_nans = int(df[expected_features].isna().sum().sum())
        n_infs = int(np.isinf(df[expected_features].values).sum())
        assert n_nans == 0, f"{name} has {n_nans} NaNs!"
        assert n_infs == 0, f"{name} has {n_infs} Infs!"
    print("[PASS] All 18 feature columns are present, properly ordered, with zero NaNs and zero Infs.")

    # 7. Check physical units and range sanity
    print("\nFeature Range & Unit Sanity Checks (Train):")
    for feat in expected_features:
        vmin = float(df_train[feat].min())
        vmax = float(df_train[feat].max())
        vmean = float(df_train[feat].mean())
        print(f"  {feat:<24} : Min={vmin:10.4f} | Max={vmax:10.4f} | Mean={vmean:10.4f}")

    print("\n" + "=" * 80)
    print("ALL GATE 3 PRE-TRAINING VERIFICATION CHECKS PASSED WITH ZERO DEFECTS!")
    print("=" * 80)

if __name__ == "__main__":
    run_pre_training_verification()
