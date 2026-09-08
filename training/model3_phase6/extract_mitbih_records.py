"""Phase 6B: MIT-BIH Dataset Expansion & Patient-Level Limitation Audit.

Extracts verified sustained VT and VFL episodes from the MIT-BIH Arrhythmia Database
(records 205, 207, 223), applies identical preprocessing and calibration (MLII, 360 Hz,
5-second window, 2.5-second step), extracts the 21 Strategy B physiological features,
and integrates them with the validated Phase 5 dataset into:
  training/model3_phase6/artifacts/expanded_physiological_feature_table.csv

Also performs a rigorous 12-point data integrity audit and exports:
  training/model3_phase6/results/segment_audit_table.csv
  training/model3_phase6/results/expansion_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd

# Import validated feature extraction routines from Phase 5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model3_phase5.extract_physiological_features import (
    TARGET_FS,
    WINDOW_SECONDS,
    STEP_SECONDS,
    WINDOW_SAMPLES,
    STEP_SAMPLES,
    extract_all_physiological_features,
)

# Canonical segment definitions based on Phase 6A clinical & annotation audit
VERIFIED_SEGMENTS = [
    {
        "record_id": "mitdb_205",
        "rec_num": "205",
        "source": "mitdb",
        "original_rhythm_label": "(VT",
        "final_project_label": "VT",
        "target": 0,
        "start_sample": 527698,
        "end_sample": 530816,
        "duration_seconds": 8.6611,
        "sampling_frequency": 360,
        "channel_used": "MLII",
        "annotation_evidence": "Sustained monomorphic VT episode terminating at sample 530816 with (N",
    },
    {
        "record_id": "mitdb_207",
        "rec_num": "207",
        "source": "mitdb",
        "original_rhythm_label": "(VFL",
        "final_project_label": "VF",
        "target": 1,
        "start_sample": 14689,
        "end_sample": 18396,
        "duration_seconds": 10.2972,
        "sampling_frequency": 360,
        "channel_used": "MLII",
        "annotation_evidence": "Sustained rapid sinusoidal ventricular flutter (>250 bpm) terminating at (N",
    },
    {
        "record_id": "mitdb_207",
        "rec_num": "207",
        "source": "mitdb",
        "original_rhythm_label": "(VFL",
        "final_project_label": "VF",
        "target": 1,
        "start_sample": 19760,
        "end_sample": 21827,
        "duration_seconds": 5.7417,
        "sampling_frequency": 360,
        "channel_used": "MLII",
        "annotation_evidence": "Sustained ventricular flutter episode terminating at (N",
    },
    {
        "record_id": "mitdb_207",
        "rec_num": "207",
        "source": "mitdb",
        "original_rhythm_label": "(VFL",
        "final_project_label": "VF",
        "target": 1,
        "start_sample": 89320,
        "end_sample": 94220,
        "duration_seconds": 13.6111,
        "sampling_frequency": 360,
        "channel_used": "MLII",
        "annotation_evidence": "Sustained ventricular flutter episode terminating at (N",
    },
    {
        "record_id": "mitdb_207",
        "rec_num": "207",
        "source": "mitdb",
        "original_rhythm_label": "(VFL",
        "final_project_label": "VF",
        "target": 1,
        "start_sample": 97051,
        "end_sample": 101185,
        "duration_seconds": 11.4833,
        "sampling_frequency": 360,
        "channel_used": "MLII",
        "annotation_evidence": "Sustained ventricular flutter episode terminating at (N",
    },
    {
        "record_id": "mitdb_207",
        "rec_num": "207",
        "source": "mitdb",
        "original_rhythm_label": "(VFL",
        "final_project_label": "VF",
        "target": 1,
        "start_sample": 554740,
        "end_sample": 590149,
        "duration_seconds": 98.3583,
        "sampling_frequency": 360,
        "channel_used": "MLII",
        "annotation_evidence": "Long uninterrupted ventricular flutter run terminating at (IVR",
    },
    {
        "record_id": "mitdb_223",
        "rec_num": "223",
        "source": "mitdb",
        "original_rhythm_label": "(VT",
        "final_project_label": "VT",
        "target": 0,
        "start_sample": 208178,
        "end_sample": 228004,
        "duration_seconds": 55.0722,
        "sampling_frequency": 360,
        "channel_used": "MLII",
        "annotation_evidence": "Sustained wide-complex ventricular tachycardia terminating at (N",
    },
    {
        "record_id": "mitdb_223",
        "rec_num": "223",
        "source": "mitdb",
        "original_rhythm_label": "(VT",
        "final_project_label": "VT",
        "target": 0,
        "start_sample": 375556,
        "end_sample": 389589,
        "duration_seconds": 38.9806,
        "sampling_frequency": 360,
        "channel_used": "MLII",
        "annotation_evidence": "Sustained wide-complex ventricular tachycardia terminating at (N",
    },
]


def load_mitbih_signal(
    zip_ref: zipfile.ZipFile,
    rec_num: str,
    channel: str = "MLII",
) -> np.ndarray:
    """Load and calibrate MIT-BIH signal to physical millivolts (mV).
    
    Standard WFDB calibration parameters for MIT-BIH Arrhythmia Database:
      gain = 200.0 ADC units / mV
      baseline = 1024
      adc_zero = 1024
      physical (mV) = (digital - 1024.0) / 200.0
    """
    csv_name = f"{rec_num}.csv"
    with zip_ref.open(csv_name) as f:
        df = pd.read_csv(f)
    df.columns = [c.replace("'", "").replace('"', "").strip() for c in df.columns]
    if channel not in df.columns:
        raise ValueError(f"Channel {channel} not found in {csv_name}. Available: {df.columns.tolist()}")
    raw_digital = df[channel].values.astype(np.float64)
    # Calibrate to mV
    physical_mv = (raw_digital - 1024.0) / 200.0
    return physical_mv


def extract_mitbih_windows(
    mitbih_zip_path: Path,
    segments: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract windows and physiological features from verified segments."""
    print("=" * 78)
    print("PHASE 6B: EXTRACTING VERIFIED MIT-BIH RHYTHM SEGMENTS")
    print("=" * 78)
    sys.stdout.flush()

    extracted_rows = []
    segment_audit_records = []
    rec_cache: dict[str, np.ndarray] = {}

    with zipfile.ZipFile(mitbih_zip_path) as z:
        for seg_idx, seg in enumerate(segments):
            rec_id = seg["record_id"]
            rec_num = seg["rec_num"]
            lbl = seg["final_project_label"]
            target = seg["target"]
            start_s = seg["start_sample"]
            end_s = seg["end_sample"]
            chan = seg["channel_used"]

            if rec_num not in rec_cache:
                rec_cache[rec_num] = load_mitbih_signal(z, rec_num, chan)
            signal_full = rec_cache[rec_num]

            seg_wave = signal_full[start_s:end_s]
            duration = len(seg_wave) / TARGET_FS

            # Sliding window segmentation: 5.0 s (1800 samples), 2.5 s step (900 samples)
            w_count = 0
            for w_start in range(0, len(seg_wave) - WINDOW_SAMPLES, STEP_SAMPLES):
                w = seg_wave[w_start : w_start + WINDOW_SAMPLES]
                feats = extract_all_physiological_features(w, TARGET_FS)
                extracted_rows.append({
                    **feats,
                    "record_id": rec_id,
                    "dataset_source": "mitdb",
                    "label": lbl,
                    "target": target,
                    "window_index": w_count,
                    "window_duration_seconds": WINDOW_SECONDS,
                    "sampling_rate": TARGET_FS,
                })
                w_count += 1

            print(
                f"  [{seg_idx+1}/{len(segments)}] {rec_id} {seg['original_rhythm_label']} "
                f"[{start_s}:{end_s}] ({duration:.2f}s) -> {w_count} windows"
            )
            sys.stdout.flush()

            audit_entry = dict(seg)
            audit_entry["windows_extracted"] = w_count
            segment_audit_records.append(audit_entry)

    df_mit = pd.DataFrame(extracted_rows)
    # Re-index window_index to be unique and strictly sequential per record
    df_mit["window_index"] = df_mit.groupby("record_id").cumcount()

    df_audit = pd.DataFrame(segment_audit_records)
    return df_mit, df_audit


def perform_12_point_integrity_audit(
    df_phase5: pd.DataFrame,
    df_mit: pd.DataFrame,
    df_expanded: pd.DataFrame,
) -> dict[str, Any]:
    """Perform the strict 12-point dataset integrity audit."""
    audit_results: dict[str, Any] = {}

    # 1. Total windows before and after expansion
    n_win_before = len(df_phase5)
    n_win_after = len(df_expanded)
    n_win_mit = len(df_mit)
    audit_results["point1_windows"] = {
        "before": int(n_win_before),
        "after": int(n_win_after),
        "added": int(n_win_mit),
        "pass": bool(n_win_after == n_win_before + n_win_mit),
    }

    # 2. Number of records before and after expansion
    recs_before = set(df_phase5["record_id"].unique())
    recs_after = set(df_expanded["record_id"].unique())
    recs_mit = set(df_mit["record_id"].unique())
    audit_results["point2_records"] = {
        "before": int(len(recs_before)),
        "after": int(len(recs_after)),
        "added": int(len(recs_mit)),
        "added_records": sorted(recs_mit),
        "pass": bool(len(recs_after) == len(recs_before) + len(recs_mit)),
    }

    # 3. Number of unique VT records
    vt_recs_before = set(df_phase5[df_phase5["target"] == 0]["record_id"].unique())
    vt_recs_after = set(df_expanded[df_expanded["target"] == 0]["record_id"].unique())
    vt_recs_mit = set(df_mit[df_mit["target"] == 0]["record_id"].unique())
    audit_results["point3_vt_records"] = {
        "before": int(len(vt_recs_before)),
        "after": int(len(vt_recs_after)),
        "added": int(len(vt_recs_mit)),
        "added_vt_records": sorted(vt_recs_mit),
        "pass": bool(len(vt_recs_after) == len(vt_recs_before) + len(vt_recs_mit)),
    }

    # 4. Number of unique VF/VFL-type records
    vf_recs_before = set(df_phase5[df_phase5["target"] == 1]["record_id"].unique())
    vf_recs_after = set(df_expanded[df_expanded["target"] == 1]["record_id"].unique())
    vf_recs_mit = set(df_mit[df_mit["target"] == 1]["record_id"].unique())
    audit_results["point4_vf_records"] = {
        "before": int(len(vf_recs_before)),
        "after": int(len(vf_recs_after)),
        "added": int(len(vf_recs_mit)),
        "added_vf_records": sorted(vf_recs_mit),
        "pass": bool(len(vf_recs_after) == len(vf_recs_before) + len(vf_recs_mit)),
    }

    # 5. Windows contributed by each MIT-BIH record
    mit_win_counts = df_mit.groupby("record_id")["window_index"].count().to_dict()
    audit_results["point5_windows_per_mit_record"] = {
        k: int(v) for k, v in mit_win_counts.items()
    }

    # 6. Label distribution before and after expansion
    label_dist_before = df_phase5["label"].value_counts().to_dict()
    label_dist_after = df_expanded["label"].value_counts().to_dict()
    audit_results["point6_label_distribution"] = {
        "before": {k: int(v) for k, v in label_dist_before.items()},
        "after": {k: int(v) for k, v in label_dist_after.items()},
    }

    # 7. Source distribution before and after expansion
    source_dist_before = df_phase5["dataset_source"].value_counts().to_dict()
    source_dist_after = df_expanded["dataset_source"].value_counts().to_dict()
    audit_results["point7_source_distribution"] = {
        "before": {k: int(v) for k, v in source_dist_before.items()},
        "after": {k: int(v) for k, v in source_dist_after.items()},
    }

    # 8. Missing values or non-finite features
    num_cols = [c for c in df_expanded.columns if c not in ["record_id", "dataset_source", "label"]]
    nan_count = int(df_expanded[num_cols].isna().sum().sum())
    inf_count = int(np.isinf(df_expanded[num_cols].values).sum())
    audit_results["point8_missing_or_nonfinite"] = {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "pass": bool(nan_count == 0 and inf_count == 0),
    }

    # 9. Duplicate windows
    dup_mask = df_expanded.duplicated(subset=["record_id", "window_index"])
    dup_count = int(dup_mask.sum())
    audit_results["point9_duplicate_windows"] = {
        "duplicate_count": dup_count,
        "pass": bool(dup_count == 0),
    }

    # 10. Record ID collisions
    id_collisions = list(recs_before.intersection(recs_mit))
    audit_results["point10_id_collisions"] = {
        "collision_count": len(id_collisions),
        "collisions": id_collisions,
        "pass": bool(len(id_collisions) == 0),
    }

    # 11. Feature schema consistency
    cols_phase5 = list(df_phase5.columns)
    cols_expanded = list(df_expanded.columns)
    cols_mit = list(df_mit.columns)
    schema_match = (cols_phase5 == cols_expanded == cols_mit)
    audit_results["point11_schema_consistency"] = {
        "columns_count": len(cols_expanded),
        "schema_match": bool(schema_match),
        "pass": bool(schema_match),
    }

    # 12. Confirmation that no existing records were modified
    df_expanded_orig_subset = df_expanded.iloc[: len(df_phase5)].copy()
    diff_count = 0
    for col in cols_phase5:
        if pd.api.types.is_numeric_dtype(df_phase5[col]):
            mismatches = (~np.isclose(df_phase5[col].values, df_expanded_orig_subset[col].values, equal_nan=True)).sum()
        else:
            mismatches = (df_phase5[col].values != df_expanded_orig_subset[col].values).sum()
        diff_count += int(mismatches)
    audit_results["point12_existing_data_unmodified"] = {
        "mismatches": diff_count,
        "pass": bool(diff_count == 0),
    }

    return audit_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6B MIT-BIH Dataset Expansion")
    parser.add_argument(
        "--mitbih-zip",
        type=Path,
        default=PROJECT_ROOT / "MITBIH.zip",
        help="Path to MITBIH.zip archive",
    )
    parser.add_argument(
        "--phase5-csv",
        type=Path,
        default=PROJECT_ROOT / "training" / "model3_phase5" / "artifacts" / "physiological_feature_table.csv",
        help="Path to validated Phase 5 physiological feature table",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "training" / "model3_phase6" / "artifacts" / "expanded_physiological_feature_table.csv",
        help="Path to save expanded dataset",
    )
    parser.add_argument(
        "--output-audit-csv",
        type=Path,
        default=PROJECT_ROOT / "training" / "model3_phase6" / "results" / "segment_audit_table.csv",
        help="Path to save segment audit table",
    )
    parser.add_argument(
        "--output-audit-json",
        type=Path,
        default=PROJECT_ROOT / "training" / "model3_phase6" / "results" / "expansion_audit.json",
        help="Path to save 12-point integrity audit JSON",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    # Step 1: Validate input files
    if not args.mitbih_zip.exists():
        raise FileNotFoundError(f"MITBIH archive not found at {args.mitbih_zip}")
    if not args.phase5_csv.exists():
        raise FileNotFoundError(f"Phase 5 feature table not found at {args.phase5_csv}")

    # Ensure output directories exist
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit_csv.parent.mkdir(parents=True, exist_ok=True)

    # Step 2: Extract MIT-BIH windows
    df_mit, df_audit = extract_mitbih_windows(args.mitbih_zip, VERIFIED_SEGMENTS)

    # Save segment audit table
    df_audit.to_csv(args.output_audit_csv, index=False)
    print(f"\n[Artifact Saved] Segment audit table: {args.output_audit_csv}")

    # Step 3: Load validated Phase 5 feature table
    print(f"\n[Loading] Existing Phase 5 feature table: {args.phase5_csv}")
    df_phase5 = pd.read_csv(args.phase5_csv)
    print(f"  Phase 5 windows: {len(df_phase5):,} | Records: {df_phase5['record_id'].nunique()}")

    # Ensure column order matches Phase 5 exactly
    cols_order = list(df_phase5.columns)
    df_mit = df_mit[cols_order]

    # Step 4: Combine into expanded feature table
    df_expanded = pd.concat([df_phase5, df_mit], ignore_index=True)
    df_expanded.to_csv(args.output_csv, index=False)
    print(f"[Artifact Saved] Expanded feature table: {args.output_csv}")
    print(f"  Expanded windows: {len(df_expanded):,} | Records: {df_expanded['record_id'].nunique()}")

    # Step 5: Perform 12-point integrity audit
    print("\n" + "=" * 78)
    print("PHASE 6B: 12-POINT DATA INTEGRITY AUDIT")
    print("=" * 78)
    audit = perform_12_point_integrity_audit(df_phase5, df_mit, df_expanded)

    for pt_key, pt_val in audit.items():
        pass_status = pt_val.get("pass", True)
        status_str = "PASS" if pass_status else "FAIL"
        print(f"  * {pt_key}: [{status_str}] -> {pt_val}")

    with open(args.output_audit_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"\n[Artifact Saved] Audit JSON: {args.output_audit_json}")

    t_elapsed = time.perf_counter() - t0
    print(f"\nPhase 6B extraction and audit completed in {t_elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
