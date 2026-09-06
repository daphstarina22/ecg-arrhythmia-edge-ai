"""Controlled training and evaluation of Model 2 (Multi-Arrhythmia Classifier)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import to_onnx

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.final_models.common.evaluation import (
    compute_multiclass_metrics,
    compute_record_level_metrics,
)
from training.final_models.common.feature_selection import MODEL2_FEATURES
from sklearn.impute import SimpleImputer
SEED = 42


def get_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=7,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    max_depth=5,
                    learning_rate=0.05,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]),
    }


def evaluate_taxonomy(
    df: pd.DataFrame,
    target_col: str,
    class_names: list[str],
    taxonomy_name: str,
    results_dir: Path,
) -> tuple[list[dict], str, Pipeline]:
    print(f"\n{'='*75}\nEvaluating Model 2 Taxonomy: {taxonomy_name.upper()}\n{'='*75}")
    print("Class counts:\n", df[target_col].value_counts())

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = df[target_col].map(class_to_idx).values.astype(int)
    X = df[MODEL2_FEATURES].values
    groups = df["record_id"].values

    models = get_models()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []

    best_name = None
    best_bal_acc = -1.0
    best_pipe = None

    for model_name, pipeline in models.items():
        print(f"\n--- Model: {model_name} on {taxonomy_name} ---")
        fold_metrics = []
        oof_preds = np.zeros(len(df), dtype=int)
        oof_probs = np.zeros((len(df), len(class_names)), dtype=float)

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            pipeline.fit(X_train, y_train)
            val_preds = pipeline.predict(X_val)
            val_probs = pipeline.predict_proba(X_val)

            oof_preds[val_idx] = val_preds
            oof_probs[val_idx] = val_probs

            m = compute_multiclass_metrics(y_val, val_preds, val_probs, class_names=class_names)
            fold_metrics.append(m)

        avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
        avg_bal_acc = float(np.mean([m["balanced_accuracy"] for m in fold_metrics]))
        avg_macro_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))
        avg_weighted_f1 = float(np.mean([m["weighted_f1"] for m in fold_metrics]))

        print(
            f"Window Metrics -> Accuracy: {avg_acc:.4f}, Balanced Acc: {avg_bal_acc:.4f}, "
            f"Macro F1: {avg_macro_f1:.4f}, Weighted F1: {avg_weighted_f1:.4f}"
        )

        # Print per-class metrics
        overall_m = compute_multiclass_metrics(y, oof_preds, oof_probs, class_names=class_names)
        print("Per-class performance:")
        for c_name, scores in overall_m["per_class"].items():
            print(f"  {c_name:<15}: Prec = {scores['precision']:.3f}, Rec = {scores['recall']:.3f}, F1 = {scores['f1']:.3f} (support={scores['support']})")

        # Save predictions
        pred_df = df[["record_id", "dataset_source", "original_label"]].copy()
        pred_df["true_label"] = y
        pred_df["prediction"] = oof_preds
        for idx, c_name in enumerate(class_names):
            pred_df[f"prob_{c_name}"] = oof_probs[:, idx]

        pred_csv = results_dir / f"model2_{taxonomy_name}_{model_name}_predictions.csv"
        pred_df.to_csv(pred_csv, index=False)

        # Record level aggregation
        rec_m = compute_record_level_metrics(pred_df, is_binary=False)
        print(f"Record-Level Metrics -> Balanced Acc: {rec_m['balanced_accuracy']:.4f}, Macro F1: {rec_m['macro_f1']:.4f}")

        rec_csv = results_dir / f"model2_{taxonomy_name}_{model_name}_record_summary.csv"
        rec_m["per_record_df"].to_csv(rec_csv, index=False)

        # Save confusion matrix
        cm_path = results_dir / f"model2_{taxonomy_name}_{model_name}_confusion_matrix.json"
        cm_data = {
            "classes": class_names,
            "confusion_matrix": overall_m["confusion_matrix"],
            "per_class": overall_m["per_class"],
        }
        with open(cm_path, "w") as f:
            json.dump(cm_data, f, indent=2)

        res_entry = {
            "taxonomy": taxonomy_name,
            "model": model_name,
            "evaluation": "5fold_stratified_groupcv",
            "window_acc": avg_acc,
            "window_balanced_acc": avg_bal_acc,
            "window_macro_f1": avg_macro_f1,
            "window_weighted_f1": avg_weighted_f1,
            "record_balanced_acc": rec_m["balanced_accuracy"],
            "record_macro_f1": rec_m["macro_f1"],
        }
        for c_name, scores in overall_m["per_class"].items():
            res_entry[f"{c_name}_f1"] = scores["f1"]
            res_entry[f"{c_name}_rec"] = scores["recall"]

        results.append(res_entry)

        if avg_bal_acc > best_bal_acc:
            best_bal_acc = avg_bal_acc
            best_name = model_name
            best_pipe = pipeline

    return results, best_name, best_pipe


def run_model2_experiments():
    art_dir = PROJECT_ROOT / "training" / "final_models" / "artifacts"
    results_dir = PROJECT_ROOT / "training" / "final_models" / "results"
    export_dir = PROJECT_ROOT / "training" / "final_models" / "export_candidates"
    results_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    master_df = pd.read_csv(art_dir / "master_multisource_feature_dataset.csv")
    master_df = master_df.dropna(subset=MODEL2_FEATURES).reset_index(drop=True)

    all_comparison_rows = []

    # 1. Primary 4-Class Taxonomy
    class_names_4c = ["NSR", "TACHY", "BRADY_ASY", "VENTRICULAR"]
    res_4c, best_4c_name, best_4c_pipe = evaluate_taxonomy(
        master_df,
        target_col="model2_4class",
        class_names=class_names_4c,
        taxonomy_name="primary_4class",
        results_dir=results_dir,
    )
    all_comparison_rows.extend(res_4c)

    # 2. Secondary 5-Class Taxonomy
    class_names_5c = ["NSR", "TACHY", "BRADY_ASY", "VT", "VF_VFL"]
    res_5c, best_5c_name, best_5c_pipe = evaluate_taxonomy(
        master_df,
        target_col="model2_5class",
        class_names=class_names_5c,
        taxonomy_name="secondary_5class",
        results_dir=results_dir,
    )
    all_comparison_rows.extend(res_5c)

    # Save summary table
    summary_df = pd.DataFrame(all_comparison_rows)
    summary_csv = results_dir / "model2_comparison_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nModel 2 comparison summary saved to: {summary_csv}")

    # Export Primary 4-class ONNX Candidate
    print(f"\nExporting primary 4-class ONNX candidate ({best_4c_name})...")
    class_to_idx = {name: idx for idx, name in enumerate(class_names_4c)}
    y_4c = master_df["model2_4class"].map(class_to_idx).values.astype(int)
    X = master_df[MODEL2_FEATURES].values
    best_4c_pipe.fit(X, y_4c)

    dummy_input = np.zeros((1, len(MODEL2_FEATURES)), dtype=np.float32)
    onx_4c = to_onnx(best_4c_pipe, dummy_input, target_opset=15)
    onnx_path_4c = export_dir / "arrhythmia_multiclass_4class_candidate.onnx"
    with open(onnx_path_4c, "wb") as f:
        f.write(onx_4c.SerializeToString())
    print(f"Exported Model 2 4-class ONNX candidate to: {onnx_path_4c} ({onnx_path_4c.stat().st_size:,} bytes)")

    # Export Secondary 5-class ONNX Candidate
    print(f"\nExporting secondary 5-class ONNX candidate ({best_5c_name})...")
    class_to_idx_5c = {name: idx for idx, name in enumerate(class_names_5c)}
    y_5c = master_df["model2_5class"].map(class_to_idx_5c).values.astype(int)
    onnx_path_5c = export_dir / "arrhythmia_multiclass_5class_candidate.onnx"
    try:
        best_5c_pipe.fit(X, y_5c)
        onx_5c = to_onnx(best_5c_pipe, dummy_input, target_opset=15)
        with open(onnx_path_5c, "wb") as f:
            f.write(onx_5c.SerializeToString())
        print(f"Exported Model 2 5-class ONNX candidate to: {onnx_path_5c} ({onnx_path_5c.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"Direct ONNX export for {best_5c_name} failed ({e}). Falling back to Random Forest for 5-class ONNX candidate...")
        rf_pipe = get_models()["random_forest"]
        rf_pipe.fit(X, y_5c)
        onx_5c = to_onnx(rf_pipe, dummy_input, target_opset=15)
        with open(onnx_path_5c, "wb") as f:
            f.write(onx_5c.SerializeToString())
        print(f"Exported Model 2 5-class Random Forest ONNX candidate to: {onnx_path_5c} ({onnx_path_5c.stat().st_size:,} bytes)")


if __name__ == "__main__":
    run_model2_experiments()
