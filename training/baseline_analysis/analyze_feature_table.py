"""Audit leakage, source bias, and feature distributions from a feature table.

Expected columns are the ten deployment features plus ``label``,
``record_id``, and ``dataset_source``. This script is intentionally separate
from model training and does not modify or retrain any deployed model.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_NAMES = [
    "mean", "std", "ptp", "zero_crossings", "peak_count", "mean_rr",
    "rr_cv", "qrs_width", "dominant_freq", "vf_band_power_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_table", type=Path)
    parser.add_argument("--label", default="label")
    args = parser.parse_args()

    table = pd.read_csv(args.feature_table)
    required = set(FEATURE_NAMES + [args.label, "record_id", "dataset_source"])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("rows:", len(table))
    print("records:", table["record_id"].nunique())
    print("class_counts_by_source:")
    print(table.groupby(["dataset_source", args.label]).size().to_string())
    print("feature_summary_by_source:")
    print(table.groupby("dataset_source")[FEATURE_NAMES].agg(["count", "mean", "std", "median"]).to_string())

    groups = table["record_id"].astype(str).to_numpy()
    labels = table[args.label].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(table, labels, groups))
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    print("record_overlap:", sorted(train_groups & test_groups))
    if train_groups & test_groups:
        raise RuntimeError("Record leakage detected")

    source = table[["dataset_source"]]
    transformer = ColumnTransformer(
        [("source", OneHotEncoder(handle_unknown="ignore"), ["dataset_source"])],
        remainder="drop",
    )
    diagnostic = make_pipeline(
        transformer,
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    diagnostic.fit(source.iloc[train_idx], labels[train_idx])
    probabilities = diagnostic.predict_proba(source.iloc[test_idx])[:, 1]
    predictions = diagnostic.predict(source.iloc[test_idx])
    print("source_only_balanced_accuracy:", balanced_accuracy_score(labels[test_idx], predictions))
    print("source_only_roc_auc:", roc_auc_score(labels[test_idx], probabilities))
    print("source_only_pr_auc:", average_precision_score(labels[test_idx], probabilities))


if __name__ == "__main__":
    main()