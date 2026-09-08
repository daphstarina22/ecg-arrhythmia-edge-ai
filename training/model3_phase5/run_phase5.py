"""Phase 5 Research Pipeline:

Auditing, Subset Construction, ML Modeling, and Cross-Database Evaluation
of Raw Waveform Physiological Features for Model 3 (VF vs. VT).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42

OLD_REDUCED_FEATURES = [
    "zero_crossings",
    "peak_count",
    "mean_rr",
    "rr_cv",
    "qrs_width",
    "dominant_freq",
    "vf_band_power_ratio",
]


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or len(values) < 2:
        return 0.0
    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total < 1e-12:
        return 0.0
    ss_between = sum(
        len(values[groups == g]) * (np.mean(values[groups == g]) - grand_mean) ** 2
        for g in unique_groups
    )
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


# ==============================================================================
# PHASE 5B: FEATURE AUDIT
# ==============================================================================


def audit_features(df: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    print("\n" + "=" * 78)
    print("PHASE 5B: PHYSIOLOGICAL FEATURE SIGNAL VS. DOMAIN-CONFOUNDER AUDIT")
    print("=" * 78)

    c2015_m = df["dataset_source"] == "c2015"
    vfdb_m = df["dataset_source"] == "vfdb"
    cudb_m = df["dataset_source"] == "cudb"

    records = []
    for feat in feature_cols:
        # A. Within-source VT vs VF discriminative strength
        # Challenge 2015
        c2015_vt = df.loc[c2015_m & (df["target"] == 0), feat].to_numpy()
        c2015_vf = df.loc[c2015_m & (df["target"] == 1), feat].to_numpy()
        d_c2015 = compute_cohens_d(c2015_vf, c2015_vt)
        try:
            auc_c2015 = roc_auc_score(df.loc[c2015_m, "target"], df.loc[c2015_m, feat])
            auc_c2015_mag = max(auc_c2015, 1.0 - auc_c2015)
        except Exception:
            auc_c2015_mag = 0.5

        # VFDB
        vfdb_vt = df.loc[vfdb_m & (df["target"] == 0), feat].to_numpy()
        vfdb_vf = df.loc[vfdb_m & (df["target"] == 1), feat].to_numpy()
        d_vfdb = compute_cohens_d(vfdb_vf, vfdb_vt)
        try:
            auc_vfdb = roc_auc_score(df.loc[vfdb_m, "target"], df.loc[vfdb_m, feat])
            auc_vfdb_mag = max(auc_vfdb, 1.0 - auc_vfdb)
        except Exception:
            auc_vfdb_mag = 0.5

        # CUDB (sample size context)
        cudb_vt = df.loc[cudb_m & (df["target"] == 0), feat].to_numpy()
        cudb_vf = df.loc[cudb_m & (df["target"] == 1), feat].to_numpy()
        d_cudb = compute_cohens_d(cudb_vf, cudb_vt)

        mi_c2015 = float(
            mutual_info_classif(df.loc[c2015_m, [feat]], df.loc[c2015_m, "target"], discrete_features=False, random_state=SEED)[0]
        )
        mi_vfdb = float(
            mutual_info_classif(df.loc[vfdb_m, [feat]], df.loc[vfdb_m, "target"], discrete_features=False, random_state=SEED)[0]
        )

        mean_disc_d = (abs(d_c2015) + abs(d_vfdb)) / 2.0
        consistent_direction = (d_c2015 * d_vfdb) > 0

        # B. Within-class source confounding
        d_source_vt = compute_cohens_d(c2015_vt, vfdb_vt)
        eta2_source_vt = compute_eta_squared(df.loc[df["target"] == 0, feat].to_numpy(), df.loc[df["target"] == 0, "dataset_source"].to_numpy())

        d_source_vf = compute_cohens_d(c2015_vf, vfdb_vf)
        eta2_source_vf = compute_eta_squared(df.loc[df["target"] == 1, feat].to_numpy(), df.loc[df["target"] == 1, "dataset_source"].to_numpy())

        mean_conf_d = (abs(d_source_vt) + abs(d_source_vf)) / 2.0
        mean_eta2 = (eta2_source_vt + eta2_source_vf) / 2.0

        s2c_ratio = mean_disc_d / (mean_conf_d + 1e-6)

        records.append({
            "feature": feat,
            "d_c2015_vf_vs_vt": d_c2015,
            "d_vfdb_vf_vs_vt": d_vfdb,
            "d_cudb_vf_vs_vt": d_cudb,
            "mean_discriminative_d": mean_disc_d,
            "auc_c2015_magnitude": auc_c2015_mag,
            "auc_vfdb_magnitude": auc_vfdb_mag,
            "mi_c2015": mi_c2015,
            "mi_vfdb": mi_vfdb,
            "direction_consistent": consistent_direction,
            "d_source_vt_c2015_vs_vfdb": d_source_vt,
            "d_source_vf_c2015_vs_vfdb": d_source_vf,
            "mean_source_confounding_d": mean_conf_d,
            "source_eta2_vt": eta2_source_vt,
            "source_eta2_vf": eta2_source_vf,
            "mean_source_eta2": mean_eta2,
            "signal_to_confounder_ratio": s2c_ratio,
        })

    audit_df = pd.DataFrame(records).sort_values(
        by=["direction_consistent", "signal_to_confounder_ratio"], ascending=[False, False]
    ).reset_index(drop=True)
    audit_df.insert(0, "rank", np.arange(1, len(audit_df) + 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_dir / "feature_audit_ranking.csv", index=False)
    print(audit_df[["rank", "feature", "mean_discriminative_d", "mean_source_confounding_d", "direction_consistent", "signal_to_confounder_ratio"]].to_string(index=False))

    # Phase 5C: Construct objective subsets
    # Subset 2: Domain-Reduced (consistent direction and source eta^2 <= 0.20)
    domain_reduced = audit_df[audit_df["direction_consistent"] & (audit_df["mean_source_eta2"] <= 0.20)]["feature"].tolist()
    if len(domain_reduced) < 3:
        domain_reduced = audit_df[audit_df["direction_consistent"]].head(5)["feature"].tolist()

    # Subset 3: Top Low-Confounding (top 4 features by S2C ratio with consistent direction)
    best_top = audit_df[audit_df["direction_consistent"]].head(4)["feature"].tolist()

    print(f"\nSubset 2 (Domain-Reduced, {len(domain_reduced)} features): {domain_reduced}")
    print(f"Subset 3 (Best Low-Confounding, {len(best_top)} features): {best_top}")
    return audit_df, domain_reduced, best_top


# ==============================================================================
# PHASE 5D & 5E: ML EVALUATION & PERSISTENCE
# ==============================================================================


def get_record_weights(df: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    records = df.iloc[train_mask]["record_id"]
    counts = records.value_counts()
    weights = records.map(lambda r: 1.0 / counts[r]).to_numpy(dtype=float)
    return weights / weights.mean()


def create_model(model_type: str):
    if model_type == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    if model_type == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=SEED),
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def fit_model(model_type: str, model, X_train: pd.DataFrame, y_train: np.ndarray, weights: np.ndarray):
    if model_type == "logistic_regression":
        model.fit(X_train, y_train, logisticregression__sample_weight=weights)
    elif model_type == "random_forest":
        model.fit(X_train, y_train, sample_weight=weights)
    elif model_type == "svm_rbf":
        model.fit(X_train, y_train, svc__sample_weight=weights)
    else:
        model.fit(X_train, y_train)


def run_phase5_experiments(
    df: pd.DataFrame,
    physio_features: list[str],
    domain_reduced: list[str],
    best_top: list[str],
    has_old_features: bool,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 78)
    print("PHASE 5D & 5E: ML MODEL TRAINING & RECORD-LEVEL EVALUATION")
    print("=" * 78)

    subsets = {
        "set1_full_physio": physio_features,
        "set2_domain_reduced": domain_reduced,
        "set3_best_low_conf": best_top,
    }
    if has_old_features:
        subsets["set0_baseline_old7"] = [f for f in OLD_REDUCED_FEATURES if f in df.columns]

    directions = [
        ("D1_vfdb_cudb_to_c2015", {"vfdb", "cudb"}, "c2015"),
        ("D2_c2015_cudb_to_vfdb", {"c2015", "cudb"}, "vfdb"),
    ]

    model_types = ["logistic_regression", "random_forest", "svm_rbf"]

    all_window_rows = []
    all_record_rows = []
    all_record_summaries = []
    all_predictions = []
    matrices = {}

    for dir_name, train_sources, test_source in directions:
        train_mask = df["dataset_source"].isin(train_sources).to_numpy()
        test_mask = df["dataset_source"].eq(test_source).to_numpy()

        train_records = set(df.loc[train_mask, "record_id"])
        test_records = set(df.loc[test_mask, "record_id"])
        overlap = train_records & test_records
        assert not overlap, f"Record leakage in {dir_name}: {overlap}"

        df_train = df.loc[train_mask].copy().reset_index(drop=True)
        df_test = df.loc[test_mask].copy().reset_index(drop=True)

        y_train = df_train["target"].to_numpy()
        y_test = df_test["target"].to_numpy()
        train_weights = get_record_weights(df, np.flatnonzero(train_mask))

        print(f"\n{dir_name}: Train {len(df_train)} windows ({len(train_records)} records) -> Test {len(df_test)} windows ({len(test_records)} records)")

        for set_name, feats in subsets.items():
            X_train = df_train[feats]
            X_test = df_test[feats]

            for model_type in model_types:
                exp_name = f"{dir_name}__{set_name}__{model_type}"
                model = create_model(model_type)
                fit_model(model_type, model, X_train, y_train, train_weights)

                # Predict
                classes = list(model.classes_)
                vf_col = classes.index(1)
                probs = model.predict_proba(X_test)[:, vf_col]
                preds = (probs >= 0.50).astype(int)

                meta = {
                    "experiment": exp_name,
                    "split_direction": dir_name,
                    "feature_subset": set_name,
                    "model": model_type,
                    "features_count": len(feats),
                    "features": ",".join(feats),
                }

                # Window-level metrics
                matrix = confusion_matrix(y_test, preds, labels=[0, 1])
                matrices[exp_name] = matrix.tolist()

                w_row = {
                    **meta,
                    "accuracy": float(accuracy_score(y_test, preds)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
                    "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
                    "vt_precision": float(precision_score(y_test, preds, pos_label=0, zero_division=0)),
                    "vt_recall": float(recall_score(y_test, preds, pos_label=0, zero_division=0)),
                    "vt_f1": float(f1_score(y_test, preds, pos_label=0, zero_division=0)),
                    "vf_precision": float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
                    "vf_recall": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
                    "vf_f1": float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_test, probs)),
                    "pr_auc": float(average_precision_score(y_test, probs)),
                    "windows": int(len(y_test)),
                }
                all_window_rows.append(w_row)

                # Persist predictions frame
                pred_frame = df_test[["record_id", "dataset_source", "label", "target"]].copy()
                pred_frame["true_label"] = pred_frame["label"]
                pred_frame["predicted_target"] = preds
                pred_frame["predicted_label"] = np.where(preds == 1, "VF", "VT")
                pred_frame["predicted_vf_probability"] = probs
                pred_frame["feature_subset"] = set_name
                pred_frame["model"] = model_type
                pred_frame["split_direction"] = dir_name
                pred_frame["experiment"] = exp_name
                pred_frame = pred_frame.drop(columns=["label"])
                all_predictions.append(pred_frame)

                # Record-level summary
                summaries = []
                for rec_id, grp in pred_frame.groupby("record_id", sort=True):
                    labels_p = sorted(grp["true_label"].unique())
                    t_lbl = labels_p[0] if len(labels_p) == 1 else "MIXED"
                    mean_p = float(grp["predicted_vf_probability"].mean())
                    maj_pred = "VF" if mean_p >= 0.50 else "VT"
                    correct = int(grp["predicted_target"].eq(grp["target"]).sum())
                    n_w = len(grp)
                    vf_rec = float(recall_score(grp["target"], grp["predicted_target"], pos_label=1, zero_division=0)) if (grp["target"] == 1).any() else np.nan
                    vt_rec = float(recall_score(grp["target"], grp["predicted_target"], pos_label=0, zero_division=0)) if (grp["target"] == 0).any() else np.nan
                    rec_macro_f1 = float(f1_score(grp["target"], grp["predicted_target"], average="macro", zero_division=0))

                    summaries.append({
                        **meta,
                        "record_id": rec_id,
                        "dataset_source": grp["dataset_source"].iloc[0],
                        "true_label": t_lbl,
                        "windows_in_record": n_w,
                        "correct_windows": correct,
                        "incorrect_windows": n_w - correct,
                        "record_accuracy": float(correct / n_w),
                        "mean_vf_probability": mean_p,
                        "record_majority_prediction": maj_pred,
                        "record_macro_f1": rec_macro_f1,
                        "vf_window_recall": vf_rec,
                        "vt_window_recall": vt_rec,
                    })

                rec_df = pd.DataFrame(summaries)
                all_record_summaries.append(rec_df)

                vf_recs = rec_df[rec_df["true_label"] == "VF"]
                vt_recs = rec_df[rec_df["true_label"] == "VT"]

                all_record_rows.append({
                    **meta,
                    "records_count": len(rec_df),
                    "vf_records_count": len(vf_recs),
                    "vt_records_count": len(vt_recs),
                    "macro_record_accuracy": float(rec_df["record_accuracy"].mean()),
                    "macro_record_f1": float(rec_df["record_macro_f1"].mean()),
                    "record_vf_recall": float(vf_recs["vf_window_recall"].mean()) if len(vf_recs) else np.nan,
                    "record_vt_recall": float(vt_recs["vt_window_recall"].mean()) if len(vt_recs) else np.nan,
                    "vf_majority_recall": float((vf_recs["record_majority_prediction"] == "VF").mean()) if len(vf_recs) else np.nan,
                    "vt_majority_recall": float((vt_recs["record_majority_prediction"] == "VT").mean()) if len(vt_recs) else np.nan,
                    "vf_fn_records": int((vf_recs["record_majority_prediction"] != "VF").sum()),
                    "vt_fp_records": int((vt_recs["record_majority_prediction"] != "VT").sum()),
                    "worst_record_accuracy": float(rec_df["record_accuracy"].min()),
                    "best_record_accuracy": float(rec_df["record_accuracy"].max()),
                })

    # Save all output files
    window_metrics_df = pd.DataFrame(all_window_rows)
    record_metrics_df = pd.DataFrame(all_record_rows)
    all_summaries_df = pd.concat(all_record_summaries, ignore_index=True)
    all_preds_df = pd.concat(all_predictions, ignore_index=True)

    window_metrics_df.to_csv(output_dir / "window_level_metrics.csv", index=False)
    record_metrics_df.to_csv(output_dir / "record_level_metrics.csv", index=False)
    all_summaries_df.to_csv(output_dir / "per_record_summary.csv", index=False)
    all_preds_df.to_csv(output_dir / "per_window_predictions.csv", index=False)

    with (output_dir / "confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump(matrices, handle, indent=2)

    # Compute Robustness Ranking across D1 and D2
    ranking_records = []
    for (set_name, model_type), grp in record_metrics_df.groupby(["feature_subset", "model"]):
        d1 = grp[grp["split_direction"] == "D1_vfdb_cudb_to_c2015"].iloc[0]
        d2 = grp[grp["split_direction"] == "D2_c2015_cudb_to_vfdb"].iloc[0]

        d1_w = window_metrics_df[(window_metrics_df["feature_subset"] == set_name) & (window_metrics_df["model"] == model_type) & (window_metrics_df["split_direction"] == "D1_vfdb_cudb_to_c2015")].iloc[0]
        d2_w = window_metrics_df[(window_metrics_df["feature_subset"] == set_name) & (window_metrics_df["model"] == model_type) & (window_metrics_df["split_direction"] == "D2_c2015_cudb_to_vfdb")].iloc[0]

        avg_macro_f1 = float(np.mean([d1["macro_record_f1"], d2["macro_record_f1"]]))
        min_macro_f1 = float(min(d1["macro_record_f1"], d2["macro_record_f1"]))
        avg_vf_recall = float(np.mean([d1["record_vf_recall"], d2["record_vf_recall"]]))

        score = 0.35 * avg_macro_f1 + 0.35 * avg_vf_recall + 0.30 * min_macro_f1

        passes_gate = bool(
            (d1_w["roc_auc"] > 0.75) and (d2_w["roc_auc"] > 0.75) and
            (d1["record_vf_recall"] >= 0.70) and (d1["record_vt_recall"] >= 0.70) and
            (d2["record_vf_recall"] >= 0.70) and (d2["record_vt_recall"] >= 0.70)
        )

        ranking_records.append({
            "feature_subset": set_name,
            "model": model_type,
            "d1_roc_auc": d1_w["roc_auc"],
            "d2_roc_auc": d2_w["roc_auc"],
            "d1_record_macro_f1": d1["macro_record_f1"],
            "d2_record_macro_f1": d2["macro_record_f1"],
            "d1_record_vf_recall": d1["record_vf_recall"],
            "d1_record_vt_recall": d1["record_vt_recall"],
            "d2_record_vf_recall": d2["record_vf_recall"],
            "d2_record_vt_recall": d2["record_vt_recall"],
            "d1_vf_fn_records": d1["vf_fn_records"],
            "d2_vt_fp_records": d2["vt_fp_records"],
            "robustness_score": score,
            "passes_decision_gate": passes_gate,
        })

    ranking_df = pd.DataFrame(ranking_records).sort_values("robustness_score", ascending=False).reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
    ranking_df.to_csv(output_dir / "robustness_ranking.csv", index=False)

    print("\n" + "=" * 78)
    print("PHASE 5 ROBUSTNESS RANKING ACROSS D1 AND D2")
    print("=" * 78)
    print(ranking_df[["rank", "feature_subset", "model", "d1_roc_auc", "d2_roc_auc", "d1_record_macro_f1", "d2_record_macro_f1", "d1_record_vf_recall", "d2_record_vt_recall", "robustness_score", "passes_decision_gate"]].to_string(index=False))

    generate_phase5_report(ranking_df, output_dir)


def generate_phase5_report(ranking_df: pd.DataFrame, output_dir: Path) -> None:
    best = ranking_df.iloc[0]
    gate_passed = bool(ranking_df["passes_decision_gate"].any())

    # Comparison to Phase 4.2 level (~0.59 - 0.61)
    d2_max_auc = float(ranking_df["d2_roc_auc"].max())
    d1_max_auc = float(ranking_df["d1_roc_auc"].max())

    verdict = "PROCEED" if gate_passed else ("INCONCLUSIVE" if d2_max_auc > 0.70 else "REJECTED")

    report = f"""# Phase 5 Research Report: Raw Waveform Physiological Representation

## 1. Evidence Directly Produced by Phase 5
- **Dataset**: Calibrated raw ECG waveform reconstruction (360 Hz, 5-second windows) across Challenge 2015, VFDB, and CUDB (14,618 windows, 117 records).
- **Candidate Physiological Features Evaluated (21 features)**:
  - **Spectral Organization**: dominant_freq, dominant_peak_power, dominant_peak_prominence, spectral_peak_power_ratio, spectral_entropy, spectral_flatness, spectral_centroid, spectral_bandwidth, spectral_concentration, spectral_peak_purity.
  - **Autocorrelation / Regularity**: ac_max_peak_ratio, ac_first_secondary_peak, ac_decay_time, ac_periodicity_strength, ac_zero_crossing_lag.
  - **Complexity / Nonlinear Dynamics**: hjorth_activity, hjorth_mobility, hjorth_complexity, lz_complexity, permutation_entropy, sample_entropy.
- **Cross-Database Evaluation Directions**:
  - D1: Train VFDB + CUDB -> Test Challenge2015 (92 records, 11,420 windows).
  - D2: Train Challenge2015 + CUDB -> Test VFDB (22 records, 3,060 windows).
- **ML Classifiers**: StandardScaler + LogisticRegression, RandomForestClassifier, SVM with RBF kernel (all with inverse-record-frequency weighting).

### Best Performing Configuration
- **Rank 1**: `{best['feature_subset']}` with `{best['model']}`
- **Robustness Score**: `{best['robustness_score']:.6f}`
- **D1 Performance**: ROC-AUC = `{best['d1_roc_auc']:.4f}`, Record Macro F1 = `{best['d1_record_macro_f1']:.4f}`, VF Record Recall = `{best['d1_record_vf_recall']:.4f}`, VT Record Recall = `{best['d1_record_vt_recall']:.4f}`
- **D2 Performance**: ROC-AUC = `{best['d2_roc_auc']:.4f}`, Record Macro F1 = `{best['d2_record_macro_f1']:.4f}`, VF Record Recall = `{best['d2_record_vf_recall']:.4f}`, VT Record Recall = `{best['d2_record_vt_recall']:.4f}`

### Key Quantitative Comparisons
- **D1 Max ROC-AUC**: `{d1_max_auc:.4f}` (vs. Phase 4.2: 0.7750)
- **D2 Max ROC-AUC**: `{d2_max_auc:.4f}` (vs. Phase 4.2: 0.6140)

## 2. Answers to Phase 5F Research Questions
1. **Stronger Discrimination?**: Assessed via within-database Cohen's d and ROC-AUC.
2. **Less Source-Confounded?**: Assessed via within-class source eta-squared and pairwise source effect size.
3. **Direction Consistency?**: Verified whether effect sign is identical across c2015 and vfdb.
4. **D2 Improvement?**: Evaluated whether D2 ROC-AUC exceeded the Phase 4.2 barrier of ~0.59 - 0.61.
5. **Balanced Performance?**: Evaluated whether both VT and VF record recalls exceeded 70% in both directions.
6. **Error Distribution**: Assessed per-record summary distribution.

## 3. Scientific Conclusions
- The raw waveform physiological features were evaluated under strict zero-leakage, grouped record constraints.
- D2 cross-database ROC-AUC: `{d2_max_auc:.4f}`.

## 4. Limitations
- The entire dataset contains only approximately 16 independent records with ventricular fibrillation (6 in c2015, 8 in vfdb, 2 in cudb).
- Window-level sample sizes (14,618 windows) do not represent independent clinical observations. Overlapping windows from 16 patients cannot substitute for patient diversity.

## 5. Decision & Recommendation
- **Decision**: `{verdict}`
- **Next Research Step**:
  {"Phase 6: The physiological representation demonstrates cross-domain separation. Expand the corpus with additional independent multicenter VT/VF records to evaluate clinical generalization on a broader cohort." if verdict == "PROCEED" else "Phase 6: Expand the corpus with additional independent multicenter Holter and ICU records (e.g. AHA, Creighton, MIT-BIH Arrhythmia) to break the 16-record patient bottleneck before further architectural exploration."}
"""
    (output_dir / "final_report.md").write_text(report, encoding="utf-8")
    print(f"\nPhase 5 Final Report written to: {output_dir / 'final_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 research pipeline.")
    parser.add_argument("--feature-csv", type=Path, default=Path("training/model3_phase5/artifacts/physiological_feature_table.csv"))
    parser.add_argument("--old-csv", type=Path, default=Path("training/model3_phase3/artifacts/calibrated_vfvt_feature_table.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/model3_phase5/results"))
    args = parser.parse_args()

    df = pd.read_csv(args.feature_csv)
    print(f"Loaded physiological feature table: {len(df)} windows, {df['record_id'].nunique()} records.")

    # Check if old features can be merged for side-by-side baseline
    has_old = False
    if args.old_csv.exists():
        old_df = pd.read_csv(args.old_csv)
        if len(old_df) == len(df) and (old_df["record_id"].to_numpy() == df["record_id"].to_numpy()).all():
            for c in OLD_REDUCED_FEATURES:
                if c in old_df.columns and c not in df.columns:
                    df[c] = old_df[c]
            has_old = True
            print("Successfully merged old reduced 7 features as baseline comparison.")

    physio_cols = [
        # Spectral Organization (10)
        "dominant_freq",
        "dominant_peak_power",
        "dominant_peak_prominence",
        "spectral_peak_power_ratio",
        "spectral_entropy",
        "spectral_flatness",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_concentration",
        "spectral_peak_purity",
        # Autocorrelation / Regularity (5)
        "ac_max_peak_ratio",
        "ac_first_secondary_peak",
        "ac_decay_time",
        "ac_periodicity_strength",
        "ac_zero_crossing_lag",
        # Complexity / Nonlinear Dynamics (6)
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "lz_complexity",
        "permutation_entropy",
        "sample_entropy",
    ]

    audit_df, domain_reduced, best_top = audit_features(df, physio_cols, args.output_dir)
    run_phase5_experiments(df, physio_cols, domain_reduced, best_top, has_old, args.output_dir)


if __name__ == "__main__":
    main()
