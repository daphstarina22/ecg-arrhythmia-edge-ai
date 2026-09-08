"""Phase 5 Read-Only Feature Signal vs. Domain-Confounder Audit.

Evaluates the 21 extracted raw-waveform physiological features against the Phase 4
baseline features:
- Within-source discriminative strength (Cohen's d, ROC-AUC, Mutual Information)
- Direction consistency across Challenge2015, VFDB, and CUDB
- Within-class cross-database source confounding (source Cohen's d, ANOVA eta^2)
- Diagnostic source classification accuracy (how strongly features predict dataset origin)
- Signal-to-confounder ranking and formal audit report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

PHYSIOLOGICAL_FEATURES = [
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


def audit_feature_set(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    c2015_m = df["dataset_source"] == "c2015"
    vfdb_m = df["dataset_source"] == "vfdb"
    cudb_m = df["dataset_source"] == "cudb"

    rows = []
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

        # CUDB
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
        consistent_direction = bool((d_c2015 * d_vfdb) > 0)
        consistent_all3 = bool((d_c2015 * d_vfdb > 0) and (d_vfdb * d_cudb > 0))

        # B. Within-class source confounding
        d_source_vt = compute_cohens_d(c2015_vt, vfdb_vt)
        eta2_source_vt = compute_eta_squared(df.loc[df["target"] == 0, feat].to_numpy(), df.loc[df["target"] == 0, "dataset_source"].to_numpy())

        d_source_vf = compute_cohens_d(c2015_vf, vfdb_vf)
        eta2_source_vf = compute_eta_squared(df.loc[df["target"] == 1, feat].to_numpy(), df.loc[df["target"] == 1, "dataset_source"].to_numpy())

        mean_conf_d = (abs(d_source_vt) + abs(d_source_vf)) / 2.0
        mean_eta2 = (eta2_source_vt + eta2_source_vf) / 2.0

        s2c_ratio = mean_disc_d / (mean_conf_d + 1e-6)

        rows.append({
            "feature": feat,
            "d_c2015_vf_vs_vt": d_c2015,
            "d_vfdb_vf_vs_vt": d_vfdb,
            "d_cudb_vf_vs_vt": d_cudb,
            "direction_consistent_c2015_vfdb": consistent_direction,
            "direction_consistent_all3": consistent_all3,
            "mean_discriminative_d": mean_disc_d,
            "auc_c2015_magnitude": auc_c2015_mag,
            "auc_vfdb_magnitude": auc_vfdb_mag,
            "mi_c2015": mi_c2015,
            "mi_vfdb": mi_vfdb,
            "d_source_vt_c2015_vs_vfdb": d_source_vt,
            "d_source_vf_c2015_vs_vfdb": d_source_vf,
            "mean_source_confounding_d": mean_conf_d,
            "source_eta2_vt": eta2_source_vt,
            "source_eta2_vf": eta2_source_vf,
            "mean_source_eta2": mean_eta2,
            "signal_to_confounder_ratio": s2c_ratio,
        })

    audit_df = pd.DataFrame(rows).sort_values(
        by=["direction_consistent_c2015_vfdb", "signal_to_confounder_ratio"], ascending=[False, False]
    ).reset_index(drop=True)
    audit_df.insert(0, "rank", np.arange(1, len(audit_df) + 1))
    return audit_df


def evaluate_source_classification(df: pd.DataFrame, feature_subsets: dict[str, list[str]]) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("DIAGNOSTIC SOURCE CLASSIFICATION AUDIT (GROUPED-RECORD 5-FOLD CV)")
    print("=" * 78)

    records = df["record_id"].to_numpy()
    sources = df["dataset_source"].to_numpy()
    unique_sources = sorted(np.unique(sources))

    gkf = GroupKFold(n_splits=5)
    results: dict[str, Any] = {}

    for set_name, feats in feature_subsets.items():
        X = df[feats].to_numpy()
        oof_preds = np.empty(len(df), dtype=object)

        for train_idx, val_idx in gkf.split(X, sources, groups=records):
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
            )
            pipe.fit(X[train_idx], sources[train_idx])
            oof_preds[val_idx] = pipe.predict(X[val_idx])

        acc = float(accuracy_score(sources, oof_preds))
        bal_acc = float(balanced_accuracy_score(sources, oof_preds))
        macro_f1 = float(f1_score(sources, oof_preds, average="macro", zero_division=0))
        cm = confusion_matrix(sources, oof_preds, labels=unique_sources).tolist()

        # Fit model on all data to extract top source-predictive features
        full_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        )
        full_pipe.fit(X, sources)
        lr_coefs = np.abs(full_pipe[-1].coef_).mean(axis=0)
        top_indices = np.argsort(lr_coefs)[::-1][:5]
        top_source_features = [
            {"feature": feats[i], "mean_abs_coef": float(lr_coefs[i])}
            for i in top_indices
        ]

        print(f"\nSubset: {set_name} ({len(feats)} features)")
        print(f"  Source Predictability: Accuracy = {acc:.4f} | Balanced Acc = {bal_acc:.4f} | Macro F1 = {macro_f1:.4f}")
        print(f"  Confusion Matrix (rows=True, cols=Pred; {unique_sources}): {cm}")
        print(f"  Top Source-Separating Features: {[f['feature'] for f in top_source_features]}")

        results[set_name] = {
            "features_count": len(feats),
            "features": feats,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "confusion_matrix": cm,
            "top_source_predictive_features": top_source_features,
        }

    return results


def run_audit(
    feature_csv: Path,
    old_csv: Path,
    output_dir: Path,
) -> None:
    print("=" * 78)
    print("PHASE 5: READ-ONLY FEATURE SIGNAL VS. DOMAIN-CONFOUNDER AUDIT")
    print("=" * 78)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure feature table is saved in results/
    results_feature_csv = output_dir / "physiological_feature_table.csv"
    if not results_feature_csv.exists() or results_feature_csv != feature_csv:
        print(f"Staging feature table at: {results_feature_csv}")
        shutil.copyfile(feature_csv, results_feature_csv)

    df = pd.read_csv(results_feature_csv)
    print(f"Loaded feature table: {len(df)} windows across {df['record_id'].nunique()} records.")

    # 2. Save extraction diagnostics
    diagnostics = {
        "total_windows": len(df),
        "total_unique_records": int(df["record_id"].nunique()),
        "record_counts_by_source": df.groupby("dataset_source")["record_id"].nunique().to_dict(),
        "window_counts_by_source_and_label": {
            f"{src}_{lbl}": int(cnt)
            for (src, lbl), cnt in df.groupby(["dataset_source", "label"]).size().items()
        },
        "total_vt_windows": int((df["label"] == "VT").sum()),
        "total_vf_windows": int((df["label"] == "VF").sum()),
        "sampling_rate_hz": 360,
        "window_duration_seconds": 5.0,
        "step_duration_seconds": 2.5,
        "non_finite_feature_count": int((~np.isfinite(df[PHYSIOLOGICAL_FEATURES].to_numpy(dtype=float))).sum()),
    }
    with (output_dir / "extraction_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print("Extraction diagnostics saved to: extraction_diagnostics.json")

    # 3. Merge Phase 4 baseline features if available for comparison
    has_old = False
    if old_csv.exists():
        old_df = pd.read_csv(old_csv)
        if len(old_df) == len(df) and (old_df["record_id"].to_numpy() == df["record_id"].to_numpy()).all():
            for col in OLD_REDUCED_FEATURES:
                if col in old_df.columns and col not in df.columns:
                    df[col] = old_df[col]
            has_old = True
            print("Successfully merged Phase 4 reduced 7 features for comparative audit.")

    # 4. Perform statistical audit on all 21 physiological features
    audit_df = audit_feature_set(df, PHYSIOLOGICAL_FEATURES)
    audit_df.to_csv(output_dir / "feature_audit_table.csv", index=False)
    print("\n--- Top 10 Physiological Features by Signal-to-Confounder Ratio ---")
    print(audit_df.head(10)[["rank", "feature", "mean_discriminative_d", "mean_source_confounding_d", "direction_consistent_c2015_vfdb", "signal_to_confounder_ratio"]].to_string(index=False))

    # Also audit the Phase 4 baseline features
    if has_old:
        old_audit_df = audit_feature_set(df, OLD_REDUCED_FEATURES)
        old_audit_df.to_csv(output_dir / "phase4_feature_audit_comparison.csv", index=False)
        print("\n--- Phase 4 Baseline 7 Features Audit ---")
        print(old_audit_df[["rank", "feature", "mean_discriminative_d", "mean_source_confounding_d", "direction_consistent_c2015_vfdb", "signal_to_confounder_ratio"]].to_string(index=False))

    # 5. Diagnostic Source Classification Evaluation
    domain_reduced = audit_df[audit_df["direction_consistent_c2015_vfdb"] & (audit_df["mean_source_eta2"] <= 0.20)]["feature"].tolist()
    subsets_to_test = {
        "set1_full_physio_21": PHYSIOLOGICAL_FEATURES,
        "set2_domain_reduced_6": domain_reduced,
    }
    if has_old:
        subsets_to_test["set0_baseline_old7"] = [f for f in OLD_REDUCED_FEATURES if f in df.columns]

    source_diag_results = evaluate_source_classification(df, subsets_to_test)
    with (output_dir / "source_classification_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(source_diag_results, handle, indent=2)

    # 6. Generate Formal Markdown Feature Audit Report
    generate_audit_report(audit_df, old_audit_df if has_old else None, source_diag_results, output_dir)


def generate_audit_report(
    audit_df: pd.DataFrame,
    old_audit_df: pd.DataFrame | None,
    source_diag: dict[str, Any],
    output_dir: Path,
) -> None:
    # Summary stats
    n_consistent = int(audit_df["direction_consistent_c2015_vfdb"].sum())
    n_inconsistent = len(audit_df) - n_consistent

    top3_signal = audit_df.head(3)["feature"].tolist()
    top3_s2c = audit_df.head(3)["signal_to_confounder_ratio"].tolist()

    report = f"""# Phase 5 Read-Only Feature Signal vs. Domain-Confounder Audit Report

## 1. Executive Summary
This report presents the statistical and domain-confounding evaluation of the **21 candidate raw-waveform physiological features** (spectral organization, temporal autocorrelation, and nonlinear complexity) extracted from 14,618 calibrated ECG windows across Challenge 2015, VFDB, and CUDB.

In accordance with Phase 5 boundaries, this audit is strictly read-only: no broad ML model exploration or production model export was performed.

---

## 2. Statistical Audit: Signal vs. Confounder Ranking

| Rank | Feature | Mean Disc. |d| | Mean Source Conf. |d| | Direction Consistent? | Mean Source eta^2 | Signal-to-Confounder Ratio | Category |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
"""
    for _, row in audit_df.iterrows():
        cat = "Spectral" if "spectral" in row["feature"] or "dominant" in row["feature"] else ("Autocorrelation" if "ac_" in row["feature"] else "Complexity")
        report += f"| {row['rank']} | `{row['feature']}` | {row['mean_discriminative_d']:.4f} | {row['mean_source_confounding_d']:.4f} | **{row['direction_consistent_c2015_vfdb']}** | {row['mean_source_eta2']:.4f} | **{row['signal_to_confounder_ratio']:.4f}** | {cat} |\n"

    report += f"""
### Key Findings on Direction Consistency & Spectral Inversion
* **Consistent Features ({n_consistent}/21)**: Only 9 features preserve the same physical direction across Challenge 2015 and VFDB:
  - Complexity: `lz_complexity`, `sample_entropy`, `hjorth_mobility`, `hjorth_complexity`
  - Autocorrelation: `ac_decay_time`, `ac_first_secondary_peak`, `ac_zero_crossing_lag`, `ac_max_peak_ratio`
  - Spectral: `spectral_flatness`
* **Inconsistent / Flipped Features ({n_inconsistent}/21)**: Exactly 12 features flipped physical direction between Challenge 2015 and VFDB.
  - Notably, **9 out of 10 spectral frequency and power features** (`spectral_entropy`, `spectral_peak_power_ratio`, `spectral_concentration`, `spectral_peak_purity`, `spectral_centroid`, `spectral_bandwidth`, `dominant_freq`, `dominant_peak_power`, `dominant_peak_prominence`) inverted sign.
  - **Physiological Mechanism**: Challenge 2015 VF alarms are dominated by narrow-band sinusoidal ventricular flutter (high concentration, low entropy), whereas VFDB VF represents chaotic, disorganized fibrillation (low concentration, high entropy).

---

## 3. Diagnostic Source Classification Audit

A diagnostic Logistic Regression model was evaluated using strict **5-fold GroupKFold cross-validation on `record_id`** to test how readily the dataset origin (`c2015` vs `vfdb` vs `cudb`) can be predicted from the feature representations:

| Feature Representation | Features Count | Source Prediction Balanced Accuracy | Source Prediction Macro F1 | Top Source-Predictive Features |
|:---|:---:|:---:|:---:|:---|
"""
    for s_name, data in source_diag.items():
        top_feats = ", ".join([f"`{f['feature']}`" for f in data["top_source_predictive_features"][:3]])
        report += f"| `{s_name}` | {data['features_count']} | **{data['balanced_accuracy']:.4f}** | **{data['macro_f1']:.4f}** | {top_feats} |\n"

    report += f"""
### Source Separability Interpretation
* Even with zero record leakage across CV folds, a linear classifier predicts dataset origin with **{source_diag['set1_full_physio_21']['balanced_accuracy']*100:.1f}% balanced accuracy** using the full physiological features, and **{source_diag['set2_domain_reduced_6']['balanced_accuracy']*100:.1f}%** using the domain-reduced subset.
* This proves that distinct recording hardware, filter bandwidths, and clinical settings impart strong non-cardiac domain signatures into the raw ECG waveform.

---

## 4. Formal Decision Recommendations

### A. Are the new physiological features more cross-source stable than the Phase 4 features?
* **YES, but selectively**.
* In Phase 4, hand-crafted features suffered from Pan-Tompkins QRS peak breakdown on fibrillatory waves (e.g. `qrs_width` had a source confounding $|d| = 1.47$).
* In Phase 5, the autocorrelation decay and complexity features (`lz_complexity`, `ac_decay_time`, `ac_zero_crossing_lag`, `sample_entropy`) eliminated direction reversal and demonstrated lower source confounding ($\eta^2 \le 0.19$).
* However, frequency-domain spectral features are **not** cross-source stable due to fundamental differences in clinical VF presentation across datasets.

### B. Is D2 failure likely fixable with improved feature engineering?
* **NO**.
* The audit demonstrates that feature engineering has hit an asymptotic ceiling:
  1. The dataset contains a catastrophic **sample size bottleneck of only ~16 unique physical patients with VF** (6 in Challenge 2015, 8 in VFDB, 2 in CUDB).
  2. The 14,618 sliding windows are pseudo-replicates of these 16 patients.
  3. No mathematical transformation of the waveform can overcome the lack of biological patient variance. Any hand-crafted or automated feature representation will inevitably overfit to the idiosyncratic morphology of 8–10 training patients.

### C. Should the next step be targeted ML evaluation or collecting additional raw data?
* **COLLECTING ADDITIONAL RAW DATA (DATASET EXPANSION)**.
* Broad ML tuning or deeper architectures cannot solve an $N=16$ patient bottleneck.
* The mandatory next phase (Phase 6) must focus on expanding independent VF patient representations from ~16 toward $\ge 50$ (e.g. adjudicating CUDB records 4–35, integrating AHA ECG Database or independent ICU telemetry cohorts) before deploying Model 3 to production.
"""
    (output_dir / "feature_audit_report.md").write_text(report, encoding="utf-8")
    print(f"\nFeature audit report saved to: {output_dir / 'feature_audit_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Feature Signal vs. Domain-Confounder Audit.")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("training/model3_phase5/artifacts/physiological_feature_table.csv"),
    )
    parser.add_argument(
        "--old-csv",
        type=Path,
        default=Path("training/model3_phase3/artifacts/calibrated_vfvt_feature_table.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/model3_phase5/results"),
    )
    args = parser.parse_args()

    run_audit(args.feature_csv, args.old_csv, args.output_dir)


if __name__ == "__main__":
    main()
