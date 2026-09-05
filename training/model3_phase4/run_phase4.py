"""Phase 4.1 calibrated record-balanced per-record evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "zero_crossings", "peak_count", "mean_rr", "rr_cv",
    "qrs_width", "dominant_freq", "vf_band_power_ratio",
]
LABELS = ["VT", "VF"]
SEED = 42


def validate(table: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURES + ["label", "record_id", "dataset_source"])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if set(table["label"].astype(str)) != set(LABELS):
        raise ValueError("Labels must be exactly VT and VF")
    if table["label"].astype(str).str.upper().eq("AFIB").any():
        raise ValueError("AFIB is present")
    if not np.isfinite(table[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite reduced features are present")
    result = table.copy()
    result["record_id"] = result["record_id"].astype(str)
    result["dataset_source"] = result["dataset_source"].astype(str)
    result["target"] = result["label"].map({"VT": 0, "VF": 1}).astype(int)
    return result


def make_split(table: pd.DataFrame, train_sources: set[str], test_source: str):
    train = np.flatnonzero(table["dataset_source"].isin(train_sources).to_numpy())
    test = np.flatnonzero(table["dataset_source"].eq(test_source).to_numpy())
    train_records = set(table.iloc[train]["record_id"])
    test_records = set(table.iloc[test]["record_id"])
    overlap = sorted(train_records & test_records)
    assert not overlap, f"Record overlap: {overlap}"
    for partition, indices in (("train", train), ("test", test)):
        if set(table.iloc[indices]["label"]) != set(LABELS):
            raise ValueError(f"{partition} lacks both labels")
    info = {
        "direction": f"train={sorted(train_sources)}, test={test_source}",
        "train_windows": int(len(train)),
        "test_windows": int(len(test)),
        "train_records": int(len(train_records)),
        "test_records": int(len(test_records)),
        "overlap": overlap,
        "train_record_ids": sorted(train_records),
        "test_record_ids": sorted(test_records),
    }
    return train, test, info


def record_sample_weights(table: pd.DataFrame, train: np.ndarray) -> np.ndarray:
    records = table.iloc[train]["record_id"]
    counts = records.value_counts()
    weights = records.map(lambda record: 1.0 / counts[record]).to_numpy(dtype=float)
    return weights / weights.mean()


def fit_predict(table: pd.DataFrame, train: np.ndarray, test: np.ndarray, balanced: bool):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )
    X_train = table.iloc[train][FEATURES]
    y_train = table.iloc[train]["target"].to_numpy()
    X_test = table.iloc[test][FEATURES]
    if balanced:
        model.fit(
            X_train,
            y_train,
            logisticregression__sample_weight=record_sample_weights(table, train),
        )
    else:
        model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    return predictions, probabilities


def window_metrics(frame: pd.DataFrame, experiment: str, split_name: str) -> tuple[dict, list[list[int]]]:
    truth = frame["target"].to_numpy()
    prediction = frame["predicted_target"].to_numpy()
    probability = frame["predicted_vf_probability"].to_numpy()
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    return {
        "experiment": experiment,
        "split_direction": split_name,
        "evaluation_level": "window",
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
        "vt_precision": precision_score(truth, prediction, pos_label=0, zero_division=0),
        "vt_recall": recall_score(truth, prediction, pos_label=0, zero_division=0),
        "vt_f1": f1_score(truth, prediction, pos_label=0, zero_division=0),
        "vf_precision": precision_score(truth, prediction, pos_label=1, zero_division=0),
        "vf_recall": recall_score(truth, prediction, pos_label=1, zero_division=0),
        "vf_f1": f1_score(truth, prediction, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(truth, probability),
        "windows": len(frame),
        "records": frame["record_id"].nunique(),
        "record_overlap": 0,
    }, matrix.tolist()


def per_record_summary(frame: pd.DataFrame, experiment: str, split_name: str) -> pd.DataFrame:
    rows = []
    for record_id, group in frame.groupby("record_id", sort=True):
        labels_present = sorted(group["true_label"].unique())
        true_label = labels_present[0] if len(labels_present) == 1 else "MIXED"
        true_target = int(group["target"].iloc[0])
        predicted_fraction = float(group["predicted_target"].mean())
        majority_target = int(predicted_fraction >= 0.5)
        correct = group["predicted_target"].eq(group["target"])
        record_accuracy = float(correct.mean())
        record_f1 = float(majority_target == true_target)
        vf_present = (group["target"] == 1).any()
        vt_present = (group["target"] == 0).any()
        vf_recall = recall_score(group["target"], group["predicted_target"], labels=[0, 1], pos_label=1, zero_division=0) if vf_present else np.nan
        vt_recall = recall_score(group["target"], group["predicted_target"], labels=[0, 1], pos_label=0, zero_division=0) if vt_present else np.nan
        record_macro_f1 = f1_score(group["target"], group["predicted_target"], labels=[0, 1], average="macro", zero_division=0)
        record_precision = precision_score(group["target"], group["predicted_target"], labels=[0, 1], average="macro", zero_division=0)
        record_recall = recall_score(group["target"], group["predicted_target"], labels=[0, 1], average="macro", zero_division=0)
        false_positive_rate = float((group["predicted_target"] == 1).mean()) if not vf_present else np.nan
        if true_label == "VF":
            status = "predominantly_correct" if predicted_fraction >= 0.5 else "predominantly_false_negative"
        elif true_label == "VT":
            status = "predominantly_correct" if predicted_fraction < 0.5 else "predominantly_false_positive"
        else:
            status = "mixed_prediction_behavior"
        if true_label != "MIXED" and 0.2 < predicted_fraction < 0.8:
            status = "mixed_prediction_behavior"
        rows.append({
            "experiment": experiment,
            "split_direction": split_name,
            "record_id": record_id,
            "dataset_source": group["dataset_source"].iloc[0],
            "true_label": true_label,
            "windows_in_record": len(group),
            "correct_windows": int(correct.sum()),
            "incorrect_windows": int((~correct).sum()),
            "record_accuracy": record_accuracy,
            "record_majority_prediction": "VF" if majority_target else "VT",
            "record_f1": record_f1,
            "record_macro_f1": record_macro_f1,
            "record_precision": record_precision,
            "record_recall": record_recall,
            "vf_window_recall": vf_recall,
            "vt_window_recall": vt_recall,
            "labels_present": "|".join(labels_present),
            "vf_probability_mean": group["predicted_vf_probability"].mean(),
            "vf_prediction_fraction": predicted_fraction,
            "false_positive_rate": false_positive_rate,
            "status": status,
        })
    return pd.DataFrame(rows)


def record_metrics(summary: pd.DataFrame, experiment: str, split_name: str) -> dict:
    vf = summary[summary.labels_present.str.contains("VF")]
    vt = summary[summary.labels_present.str.contains("VT")]
    pure = summary[summary.true_label.isin(["VT", "VF"])]
    majority_truth = (pure["true_label"] == "VF").astype(int)
    majority_pred = (pure["record_majority_prediction"] == "VF").astype(int)
    return {
        "experiment": experiment,
        "split_direction": split_name,
        "evaluation_level": "equal_record",
        "records": len(summary),
        "macro_record_accuracy": summary.record_accuracy.mean(),
        "macro_record_f1": summary.record_macro_f1.mean(),
        "record_vf_recall": vf.vf_window_recall.mean() if len(vf) else np.nan,
        "record_vt_recall": vt.vt_window_recall.mean() if len(vt) else np.nan,
        "record_precision": pure["record_precision"].mean() if len(pure) else np.nan,
        "record_recall": pure["record_recall"].mean() if len(pure) else np.nan,
        "record_f1": pure["record_macro_f1"].mean() if len(pure) else np.nan,
        "worst_record_accuracy": summary.record_accuracy.min(),
        "best_record_accuracy": summary.record_accuracy.max(),
        "median_record_accuracy": summary.record_accuracy.median(),
        "p25_record_accuracy": summary.record_accuracy.quantile(0.25),
        "p75_record_accuracy": summary.record_accuracy.quantile(0.75),
        "worst_record_f1": summary.record_f1.min(),
        "best_record_f1": summary.record_f1.max(),
        "vf_false_negative_records": int((vf.vf_window_recall < 0.5).sum()),
        "vt_false_positive_records": int((vt.vt_window_recall < 0.5).sum()),
    }


def source_label_counts(table: pd.DataFrame, train: np.ndarray, test: np.ndarray, split_name: str) -> pd.DataFrame:
    rows = []
    for partition, indices in (("train", train), ("test", test)):
        rows.append(table.iloc[indices].groupby(["dataset_source", "label"], observed=True).agg(
            windows=("label", "size"), records=("record_id", "nunique")
        ).reset_index().assign(split_direction=split_name, partition=partition))
    return pd.concat(rows, ignore_index=True)


def plot_outputs(record_table: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for column, title, filename in [
        ("record_f1", "Per-record F1", "per_record_f1.png"),
        ("record_recall", "Per-record recall", "per_record_recall.png"),
        ("windows_in_record", "Windows per record", "windows_per_record.png"),
    ]:
        axis = record_table.boxplot(column=column, by="split_direction", figsize=(8, 4))
        axis.set_title(title); axis.set_xlabel("Split direction"); axis.figure.suptitle("")
        axis.figure.tight_layout(); axis.figure.savefig(plots / filename, dpi=140); plt.close(axis.figure)
    predictions["true_class"] = predictions["true_label"]
    axis = predictions.boxplot(column="predicted_vf_probability", by=["dataset_source", "true_class"], figsize=(10, 5))
    axis.set_title("VF probability by source and true class"); axis.figure.suptitle("")
    axis.figure.tight_layout(); axis.figure.savefig(plots / "vf_probability_by_source_class.png", dpi=140); plt.close(axis.figure)
    errors = record_table[record_table["incorrect_windows"] > 0].sort_values("incorrect_windows", ascending=False).head(15)
    if not errors.empty:
        axis = errors.plot.barh(x="record_id", y="incorrect_windows", figsize=(8, 5), title="Top error records")
        axis.figure.tight_layout(); axis.figure.savefig(plots / "top_error_records.png", dpi=140); plt.close(axis.figure)


def run(table: pd.DataFrame, output_dir: Path) -> None:
    table = validate(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions = []
    all_record_summaries = []
    window_rows = []
    record_rows = []
    matrices = {}
    splits = {}
    split_counts = []
    for split_name, train_sources, test_source in [
        ("D1_vfdb_cudb_to_c2015", {"vfdb", "cudb"}, "c2015"),
        ("D2_c2015_cudb_to_vfdb", {"c2015", "cudb"}, "vfdb"),
    ]:
        train, test, info = make_split(table, train_sources, test_source)
        splits[split_name] = info
        split_counts.append(source_label_counts(table, train, test, split_name))
        for experiment, balanced in [("baseline_logistic_reduced", False), ("record_balanced_logistic_reduced", True)]:
            predicted, probabilities = fit_predict(table, train, test, balanced)
            frame = table.iloc[test][["record_id", "dataset_source", "label", "target"]].copy()
            frame["true_label"] = frame["label"]
            frame["predicted_target"] = predicted
            frame["predicted_label"] = np.where(predicted == 1, "VF", "VT")
            frame["predicted_vf_probability"] = probabilities
            frame["experiment"] = experiment
            frame["split_direction"] = split_name
            frame["windows_in_record"] = frame["record_id"].map(frame["record_id"].value_counts())
            frame = frame.drop(columns=["label"])
            all_predictions.append(frame)
            metrics, matrix = window_metrics(frame, experiment, split_name)
            window_rows.append(metrics); matrices[f"{split_name}_{experiment}"] = matrix
            summary = per_record_summary(frame, experiment, split_name)
            all_record_summaries.append(summary)
            record_rows.append(record_metrics(summary, experiment, split_name))

    prediction_table = pd.concat(all_predictions, ignore_index=True)
    record_table = pd.concat(all_record_summaries, ignore_index=True)
    window_metrics_table = pd.DataFrame(window_rows)
    record_metrics_table = pd.DataFrame(record_rows)
    prediction_table.to_csv(output_dir / "per_window_predictions.csv", index=False)
    record_table.to_csv(output_dir / "per_record_summary.csv", index=False)
    window_metrics_table.to_csv(output_dir / "window_level_metrics.csv", index=False)
    record_metrics_table.to_csv(output_dir / "record_level_metrics.csv", index=False)
    pd.concat(split_counts, ignore_index=True).to_csv(output_dir / "split_source_label_counts.csv", index=False)
    table.groupby(["dataset_source", "label", "record_id"], observed=True).size().reset_index(name="windows").to_csv(output_dir / "record_distribution.csv", index=False)
    with (output_dir / "confusion_matrices.json").open("w", encoding="utf-8") as handle: json.dump(matrices, handle, indent=2)
    with (output_dir / "split_definitions.json").open("w", encoding="utf-8") as handle: json.dump(splits, handle, indent=2)

    ranking_rows = []
    for experiment, group in record_metrics_table.groupby("experiment"):
        d1 = group[group.split_direction == "D1_vfdb_cudb_to_c2015"].iloc[0]
        d2 = group[group.split_direction == "D2_c2015_cudb_to_vfdb"].iloc[0]
        score = 0.35 * np.mean([d1.macro_record_f1, d2.macro_record_f1]) + 0.35 * np.mean([d1.record_vf_recall, d2.record_vf_recall]) + 0.30 * min(d1.macro_record_f1, d2.macro_record_f1)
        ranking_rows.append({"experiment": experiment, "d1_record_macro_f1": d1.macro_record_f1, "d2_record_macro_f1": d2.macro_record_f1, "d1_vf_record_recall": d1.record_vf_recall, "d2_vf_record_recall": d2.record_vf_recall, "robustness_score": score})
    ranking = pd.DataFrame(ranking_rows).sort_values("robustness_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(output_dir / "robustness_ranking.csv", index=False)
    plot_outputs(record_table, prediction_table, output_dir)

    best = ranking.iloc[0]
    report = f"""# Phase 4.1 Record-Balanced Per-Record Evaluation

The calibrated Phase 2E.2 table was evaluated without changing production
artifacts. Labels are VT=0 and VF=1; AFIB rows were rejected and none were
present. All source-held-out splits had zero record overlap.

## Method

Baseline: StandardScaler plus LogisticRegression(max_iter=1000,
class_weight=\"balanced\", random_state=42).

Record-balanced: identical model, with training-only inverse-record-frequency
sample weights normalized to mean one.

Equal-record robustness formula:

```text
0.35 * average D1/D2 record macro F1
+ 0.35 * average D1/D2 VF-record recall
+ 0.30 * minimum D1/D2 record macro F1
```

Best candidate: `{best['experiment']}` with score `{best['robustness_score']:.6f}`.

## Decision

This is an evaluation result only. No production model or ONNX artifact was
created or modified.
"""
    (output_dir / "final_report.md").write_text(report, encoding="utf-8")
    print(window_metrics_table.to_string(index=False))
    print("\n" + record_metrics_table.to_string(index=False))
    print("\n" + ranking.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4.1 record-balanced evaluation.")
    parser.add_argument("feature_table", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/model3_phase4/results"))
    args = parser.parse_args()
    run(pd.read_csv(args.feature_table), args.output_dir)


if __name__ == "__main__":
    main()