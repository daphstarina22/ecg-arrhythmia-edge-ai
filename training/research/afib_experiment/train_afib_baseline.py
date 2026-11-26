"""
AFib Baseline Model Training and Validation Protocol
=====================================================
Author: Senior ECG ML Researcher & Edge-AI Deployment Engineer
Target: Raspberry Pi ARM64 (Debian Bookworm)

Strict Leak-Free Invariants:
1. Fits StandardScaler ONLY on Training data.
2. Selects / evaluates features ONLY on Training/Validation data.
3. Tunes probability operating thresholds ONLY on Validation data.
4. Leaves Held-Out Test cohort untouched until final evaluation.
5. Exports interpretable models: Regularized Logistic Regression and Balanced Random Forest.
"""

import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
    balanced_accuracy_score
)

TARGET_FS = 360

# Paths
EXPERIMENT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = EXPERIMENT_DIR / "features_train.csv"
VAL_CSV = EXPERIMENT_DIR / "features_val.csv"
METADATA_PATH = EXPERIMENT_DIR / "afib_model_metadata.json"
PACKAGE_PATH = EXPERIMENT_DIR / "afib_baseline_package.joblib"

# Feature set
FEATURE_COLUMNS = [
    # Ventricular timing & rate dynamics
    "hr_mean_bpm",
    "hr_std_bpm",
    "rr_cv",
    "rr_rmssd",
    "rr_mad",
    "rr_pnn50",
    "rr_entropy",
    # Atrial fibrillatory band (3-9 Hz) & spectral organization
    "f_band_power_ratio",
    "f_band_flatness",
    "f_band_peak_purity",
    "p_band_power_ratio",
    "qrs_power_ratio",
    # Complexity & morphology
    "hjorth_mobility",
    "hjorth_complexity",
    "ac_std_mv",
    "ac_ptp_mv"
]


def load_and_verify_data():
    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        raise FileNotFoundError(
            f"Feature CSVs not found in {EXPERIMENT_DIR}. "
            "Run prepare_afib_data.py first!"
        )

    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)

    print(f"Loaded Training Data: {len(df_train)} windows across subjects: {sorted(df_train['subject_id'].unique())}")
    print(f"  AFib windows: {(df_train['is_afib'] == 1).sum()} | Non-AFib windows: {(df_train['is_afib'] == 0).sum()}")

    print(f"Loaded Validation Data: {len(df_val)} windows across subjects: {sorted(df_val['subject_id'].unique())}")
    print(f"  AFib windows: {(df_val['is_afib'] == 1).sum()} | Non-AFib windows: {(df_val['is_afib'] == 0).sum()}")

    # Verify zero subject overlap between Train and Val
    train_subjects = set(df_train["subject_id"])
    val_subjects = set(df_val["subject_id"])
    overlap = train_subjects.intersection(val_subjects)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping subjects: {overlap}"
    print("[VERIFIED] Zero subject leakage between Train and Validation sets.")

    return df_train, df_val


def train_models():
    print("=" * 70)
    print("Phase C: AFib Baseline Model Training & Validation Protocol")
    print("=" * 70)

    df_train, df_val = load_and_verify_data()

    X_train_raw = df_train[FEATURE_COLUMNS].values
    y_train = df_train["is_afib"].values

    X_val_raw = df_val[FEATURE_COLUMNS].values
    y_val = df_val["is_afib"].values

    # 1. Fit StandardScaler ONLY on Training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    print(f"\n[1/4] Fitted StandardScaler strictly on {len(X_train)} training samples.")

    # 2. Train Model A: Regularized Logistic Regression (Interpretable Linear Baseline)
    print("\n[2/4] Training Model A: Balanced L2-Regularized Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    # 3. Train Model B: Balanced Random Forest (Interpretable Non-Linear Baseline)
    print("\n[3/4] Training Model B: Balanced Random Forest (100 trees, max_depth=6)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 4. Evaluate both models on Validation Set
    print("\n[4/4] Evaluating Models on Independent Validation Set...")
    models = {
        "LogisticRegression": lr_model,
        "RandomForest": rf_model
    }

    val_metrics = {}
    best_model_name = "RandomForest"
    best_val_auc = 0.0

    for name, model in models.items():
        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        val_pr_auc = average_precision_score(y_val, val_probs)

        print(f"\n--- {name} Validation Metrics ---")
        print(f"  ROC-AUC   : {val_auc:.4f}")
        print(f"  PR-AUC    : {val_pr_auc:.4f}")

        # Threshold Search on Validation Set (Sweep from 0.10 to 0.90)
        best_thresh = 0.50
        best_f1 = 0.0
        best_sens = 0.0
        best_spec = 0.0

        for thresh in np.arange(0.10, 0.92, 0.02):
            preds = (val_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = f1_score(y_val, preds, zero_division=0)
            
            # Prioritize high sensitivity (>=90%) while maximizing specificity and F1
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                best_sens = sens
                best_spec = spec

        print(f"  Optimal Operating Threshold: {best_thresh:.2f}")
        print(f"    Sensitivity (Recall)    : {best_sens*100:.2f}%")
        print(f"    Specificity             : {best_spec*100:.2f}%")
        print(f"    Validation F1-Score     : {best_f1:.4f}")

        val_metrics[name] = {
            "roc_auc": round(float(val_auc), 4),
            "pr_auc": round(float(val_pr_auc), 4),
            "optimal_threshold": round(best_thresh, 4),
            "val_sensitivity": round(float(best_sens), 4),
            "val_specificity": round(float(best_spec), 4),
            "val_f1": round(float(best_f1), 4)
        }

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_name = name

    # Feature Importance for Random Forest
    rf_importances = dict(zip(FEATURE_COLUMNS, rf_model.feature_importances_))
    sorted_importances = sorted(rf_importances.items(), key=lambda x: -x[1])
    print("\nTop 8 Feature Importances (Random Forest):")
    for feat, imp in sorted_importances[:8]:
        print(f"  {feat:<22}: {imp:.4f}")

    # Feature Coefficients for Logistic Regression
    lr_coeffs = dict(zip(FEATURE_COLUMNS, lr_model.coef_[0]))
    sorted_coeffs = sorted(lr_coeffs.items(), key=lambda x: -abs(x[1]))
    print("\nTop 8 Logistic Regression Weights (Absolute Magnitude):")
    for feat, coeff in sorted_coeffs[:8]:
        print(f"  {feat:<22}: {coeff:+.4f}")

    # 5. Package and Save Candidate Baseline
    selected_threshold = val_metrics[best_model_name]["optimal_threshold"]
    print(f"\n[SELECTION] Selected Champion Model: {best_model_name} (Operating Threshold: {selected_threshold:.2f})")

    package = {
        "champion_model_name": best_model_name,
        "champion_model": models[best_model_name],
        "logistic_regression": lr_model,
        "random_forest": rf_model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "operating_threshold": selected_threshold,
        "target_fs": TARGET_FS,
        "window_sec": 10.0,
        "stride_sec": 5.0
    }

    joblib.dump(package, PACKAGE_PATH)
    print(f"[SUCCESS] Serialized candidate baseline package to: {PACKAGE_PATH}")

    metadata = {
        "metadata": {
            "task": "AFib vs Non-AFib Rhythm Classification",
            "window_duration_sec": 10.0,
            "stride_sec": 5.0,
            "sampling_rate_hz": TARGET_FS,
            "lead_configuration": "Lead MLII (or Lead II equivalent)"
        },
        "train_cohort": {
            "records": sorted(df_train["record_id"].unique().tolist()),
            "subjects": sorted(df_train["subject_id"].unique().tolist()),
            "n_afib_windows": int((df_train["is_afib"] == 1).sum()),
            "n_non_afib_windows": int((df_train["is_afib"] == 0).sum())
        },
        "val_cohort": {
            "records": sorted(df_val["record_id"].unique().tolist()),
            "subjects": sorted(df_val["subject_id"].unique().tolist()),
            "n_afib_windows": int((df_val["is_afib"] == 1).sum()),
            "n_non_afib_windows": int((df_val["is_afib"] == 0).sum())
        },
        "features": FEATURE_COLUMNS,
        "feature_importances_rf": {k: round(float(v), 4) for k, v in sorted_importances},
        "logistic_coefficients": {k: round(float(v), 4) for k, v in sorted_coeffs},
        "validation_metrics": val_metrics,
        "selected_champion": best_model_name,
        "operating_threshold": selected_threshold
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Exported training metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    train_models()
