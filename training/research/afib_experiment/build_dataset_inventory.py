"""
Comprehensive Dataset Inventory and Verification Script for AFib Research
========================================================================
Audits all records in local MIT-BIH and VFDB archives.
Distinguishes verified data from assumptions.
Calculates exact usable window counts under 10s window, 5s stride, 2.5s guard band.
Outputs training/research/afib_experiment/dataset_inventory.csv.
"""
import os
import zipfile
import collections
from pathlib import Path
import pandas as pd
import numpy as np

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "training" / "research" / "afib_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard PhysioNet MIT-BIH patient demographic mapping
# Verified from PhysioNet MIT-BIH Arrhythmia Database directory notes
MITBIH_PATIENT_METADATA = {
    "100": {"subject_id": "MIT_S100", "age": 69, "sex": "M", "paced": False},
    "101": {"subject_id": "MIT_S101", "age": 75, "sex": "F", "paced": False},
    "102": {"subject_id": "MIT_S102", "age": 84, "sex": "F", "paced": False},
    "103": {"subject_id": "MIT_S103", "age": 70, "sex": "M", "paced": False},
    "104": {"subject_id": "MIT_S104", "age": 66, "sex": "F", "paced": True},
    "105": {"subject_id": "MIT_S105", "age": 73, "sex": "M", "paced": False},
    "106": {"subject_id": "MIT_S106", "age": 24, "sex": "F", "paced": False},
    "107": {"subject_id": "MIT_S107", "age": 63, "sex": "M", "paced": True},
    "108": {"subject_id": "MIT_S108", "age": 87, "sex": "F", "paced": False},
    "109": {"subject_id": "MIT_S109", "age": 64, "sex": "M", "paced": False},
    "111": {"subject_id": "MIT_S111", "age": 47, "sex": "F", "paced": False},
    "112": {"subject_id": "MIT_S112", "age": 54, "sex": "M", "paced": False},
    "113": {"subject_id": "MIT_S113", "age": 24, "sex": "F", "paced": False},
    "114": {"subject_id": "MIT_S114", "age": 72, "sex": "F", "paced": False},
    "115": {"subject_id": "MIT_S115", "age": 39, "sex": "F", "paced": False},
    "116": {"subject_id": "MIT_S116", "age": 68, "sex": "M", "paced": False},
    "117": {"subject_id": "MIT_S117", "age": 69, "sex": "M", "paced": False},
    "118": {"subject_id": "MIT_S118", "age": 69, "sex": "M", "paced": False},
    "119": {"subject_id": "MIT_S119", "age": 51, "sex": "F", "paced": False},
    "121": {"subject_id": "MIT_S121", "age": 89, "sex": "F", "paced": False},
    "122": {"subject_id": "MIT_S122", "age": 51, "sex": "M", "paced": False},
    "123": {"subject_id": "MIT_S123", "age": 63, "sex": "F", "paced": False},
    "124": {"subject_id": "MIT_S124", "age": 77, "sex": "M", "paced": False},
    "200": {"subject_id": "MIT_S200", "age": 64, "sex": "M", "paced": False},
    "201": {"subject_id": "MIT_S201_202", "age": 68, "sex": "M", "paced": False},  # SAME PATIENT AS 202
    "202": {"subject_id": "MIT_S201_202", "age": 68, "sex": "M", "paced": False},  # SAME PATIENT AS 201
    "203": {"subject_id": "MIT_S203", "age": 43, "sex": "M", "paced": False},
    "205": {"subject_id": "MIT_S205", "age": 59, "sex": "M", "paced": False},
    "207": {"subject_id": "MIT_S207", "age": 89, "sex": "F", "paced": False},
    "208": {"subject_id": "MIT_S208", "age": 63, "sex": "F", "paced": False},
    "209": {"subject_id": "MIT_S209", "age": 62, "sex": "M", "paced": False},
    "210": {"subject_id": "MIT_S210", "age": 89, "sex": "M", "paced": False},
    "212": {"subject_id": "MIT_S212", "age": 32, "sex": "F", "paced": False},
    "213": {"subject_id": "MIT_S213", "age": 61, "sex": "M", "paced": False},
    "214": {"subject_id": "MIT_S214", "age": 53, "sex": "M", "paced": False},
    "215": {"subject_id": "MIT_S215", "age": 81, "sex": "M", "paced": False},
    "217": {"subject_id": "MIT_S217", "age": 65, "sex": "M", "paced": True},
    "219": {"subject_id": "MIT_S219", "age": 74, "sex": "M", "paced": False},
    "220": {"subject_id": "MIT_S220", "age": 87, "sex": "F", "paced": False},
    "221": {"subject_id": "MIT_S221", "age": 83, "sex": "M", "paced": False},
    "222": {"subject_id": "MIT_S222", "age": 84, "sex": "F", "paced": False},
    "223": {"subject_id": "MIT_S223", "age": 73, "sex": "M", "paced": False},
    "228": {"subject_id": "MIT_S228", "age": 80, "sex": "F", "paced": False},
    "230": {"subject_id": "MIT_S230", "age": 32, "sex": "M", "paced": False},
    "231": {"subject_id": "MIT_S231", "age": 72, "sex": "F", "paced": False},
    "232": {"subject_id": "MIT_S232", "age": 76, "sex": "F", "paced": False},
    "233": {"subject_id": "MIT_S233", "age": 57, "sex": "M", "paced": False},
    "234": {"subject_id": "MIT_S234", "age": 56, "sex": "F", "paced": False},
}


def audit_mitbih(window_sec=10.0, stride_sec=5.0, guard_sec=2.5):
    zip_path = PROJECT_ROOT / "MITBIH.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"MITBIH.zip missing at {zip_path}")

    records_data = []
    fs = 360
    win_len = int(window_sec * fs)
    step = int(stride_sec * fs)
    guard = int(guard_sec * fs)

    with zipfile.ZipFile(zip_path) as z:
        records = sorted({f.replace(".csv", "") for f in z.namelist() if f.endswith(".csv") and "/" not in f})

        for rec in records:
            # 1. Parse CSV header for leads and sample count
            csv_name = f"{rec}.csv"
            with z.open(csv_name) as f:
                header = f.readline().decode("utf-8").strip()
                cols = [c.replace("'", "").strip() for c in header.split(",")]
                leads = cols[1:]
                # Count lines without loading full dataframe into memory
                line_count = 1 + sum(1 for _ in f)

            total_samples = line_count - 1
            total_duration_sec = total_samples / fs

            # 2. Parse annotations
            ann_name = f"{rec}annotations.txt"
            lines = z.read(ann_name).decode("utf-8", errors="replace").splitlines()
            
            rhythms = []
            beat_types = collections.Counter()
            for l in lines[1:]:
                p = l.split()
                if len(p) >= 3:
                    try:
                        s_idx = int(p[1])
                        b_type = p[2]
                        beat_types[b_type] += 1
                        for part in p[2:]:
                            if part.startswith("("):
                                rhythms.append((s_idx, part))
                    except ValueError:
                        pass

            # Interval calculation
            intervals = []
            for i, (s_idx, tag) in enumerate(rhythms):
                e_idx = rhythms[i+1][0] if i+1 < len(rhythms) else total_samples
                intervals.append((s_idx, e_idx, (e_idx - s_idx) / fs, tag))

            afib_intervals = [x for x in intervals if x[3] == "(AFIB"]
            afl_intervals = [x for x in intervals if x[3] == "(AFL"]
            other_intervals = [x for x in intervals if x[3] not in ("(AFIB", "(AFL")]

            afib_duration_sec = sum(x[2] for x in afib_intervals)
            afl_duration_sec = sum(x[2] for x in afl_intervals)
            non_afib_duration_sec = total_duration_sec - afib_duration_sec

            other_tags = sorted({x[3] for x in other_intervals})
            if afl_intervals:
                other_tags = sorted(set(other_tags + ["(AFL"]))

            # Usable window counts with guard bands
            usable_afib_windows = 0
            usable_non_afib_windows = 0
            usable_afl_windows = 0
            usable_other_windows = 0

            for s_idx, e_idx, dur_sec, tag in intervals:
                eff_start = s_idx + guard
                eff_end = e_idx - guard
                if eff_end - eff_start >= win_len:
                    n_w = 1 + (eff_end - eff_start - win_len) // step
                    if tag == "(AFIB":
                        usable_afib_windows += n_w
                    elif tag == "(AFL":
                        usable_afl_windows += n_w
                    elif tag == "(N":
                        usable_non_afib_windows += n_w
                    else:
                        usable_other_windows += n_w

            # Determine exclusion reasons
            exclusion_reasons = []
            if "MLII" not in leads:
                exclusion_reasons.append("Missing Lead MLII (only V5/V2)")
            
            meta = MITBIH_PATIENT_METADATA.get(rec, {})
            if meta.get("paced", False):
                exclusion_reasons.append("Pacemaker rhythm (artificial pacing spikes distort intrinsic dynamics)")

            # Format record dictionary
            records_data.append({
                "archive": "MITBIH",
                "record_id": rec,
                "subject_id": meta.get("subject_id", f"UNKNOWN_{rec}"),
                "patient_age": meta.get("age", "Unknown"),
                "patient_sex": meta.get("sex", "Unknown"),
                "sampling_rate_hz": fs,
                "available_channels": "/".join(leads),
                "lead_ii_available": "MLII" in leads,
                "total_duration_sec": round(total_duration_sec, 2),
                "afib_episodes_count": len(afib_intervals),
                "afib_duration_sec": round(afib_duration_sec, 2),
                "non_afib_duration_sec": round(non_afib_duration_sec, 2),
                "afl_duration_sec": round(afl_duration_sec, 2),
                "other_rhythm_labels": ", ".join(other_tags) if other_tags else "None",
                "usable_afib_windows": usable_afib_windows,
                "usable_non_afib_pure_sinus_windows": usable_non_afib_windows,
                "usable_afl_windows": usable_afl_windows,
                "usable_other_windows": usable_other_windows,
                "total_usable_windows": usable_afib_windows + usable_non_afib_windows + usable_afl_windows + usable_other_windows,
                "is_usable_for_afib_training": (usable_afib_windows > 0 and len(exclusion_reasons) == 0),
                "is_usable_for_control_cohort": (usable_afib_windows == 0 and len(exclusion_reasons) == 0),
                "exclusion_reasons": "; ".join(exclusion_reasons) if exclusion_reasons else "None (Valid)"
            })

    return records_data


def audit_vfdb(window_sec=10.0, stride_sec=5.0, guard_sec=2.5):
    zip_path = PROJECT_ROOT / "VFDB.zip"
    if not zip_path.exists():
        print("VFDB.zip not found, skipping VFDB audit.")
        return []

    import tempfile, wfdb
    records_data = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)
        recs = sorted({f.split(".")[0] for f in os.listdir(tmpdir) if f.endswith(".dat")})

        for rec in recs:
            rec_path = os.path.join(tmpdir, rec)
            header = wfdb.rdheader(rec_path)
            fs = header.fs
            win_len = int(window_sec * fs)
            step = int(stride_sec * fs)
            guard = int(guard_sec * fs)

            total_samples = header.sig_len
            total_duration_sec = total_samples / fs

            ann = wfdb.rdann(rec_path, "atr")
            rhythms = []
            for i, note in enumerate(ann.aux_note):
                cleaned = note.strip().rstrip("\x00")
                if cleaned.startswith("("):
                    rhythms.append((ann.sample[i], cleaned))

            intervals = []
            for i, (s_idx, tag) in enumerate(rhythms):
                e_idx = rhythms[i+1][0] if i+1 < len(rhythms) else total_samples
                intervals.append((s_idx, e_idx, (e_idx - s_idx) / fs, tag))

            afib_intervals = [x for x in intervals if x[3] == "(AFIB"]
            afl_intervals = [x for x in intervals if x[3] in ("(AFL", "(VFL")]
            other_intervals = [x for x in intervals if x[3] not in ("(AFIB", "(AFL", "(VFL")]

            afib_duration_sec = sum(x[2] for x in afib_intervals)
            non_afib_duration_sec = total_duration_sec - afib_duration_sec

            other_tags = sorted({x[3] for x in intervals if x[3] != "(AFIB"})

            usable_afib_windows = 0
            usable_non_afib_windows = 0
            usable_other_windows = 0

            for s_idx, e_idx, dur_sec, tag in intervals:
                eff_start = s_idx + guard
                eff_end = e_idx - guard
                if eff_end - eff_start >= win_len:
                    n_w = 1 + (eff_end - eff_start - win_len) // step
                    if tag == "(AFIB":
                        usable_afib_windows += n_w
                    elif tag == "(N":
                        usable_non_afib_windows += n_w
                    else:
                        usable_other_windows += n_w

            exclusion_reasons = []
            # VFDB recordings have 250 Hz sampling rate, uncalibrated/generic lead vectors, and high malignant ventricular ectopy
            if fs != 360:
                exclusion_reasons.append(f"Non-target sampling rate ({fs} Hz requires resampling)")
            if any(t in ["(VFIB", "(VT", "(VFL", "(ASYS"] for t in other_tags):
                exclusion_reasons.append("Severe malignant ventricular/asystole confounders present")

            records_data.append({
                "archive": "VFDB",
                "record_id": f"vfdb_{rec}",
                "subject_id": f"VFDB_S{rec}",
                "patient_age": "Unknown",
                "patient_sex": "Unknown",
                "sampling_rate_hz": fs,
                "available_channels": "/".join(header.sig_name),
                "lead_ii_available": False,  # VFDB leads are modified bipolar, not standard Lead II
                "total_duration_sec": round(total_duration_sec, 2),
                "afib_episodes_count": len(afib_intervals),
                "afib_duration_sec": round(afib_duration_sec, 2),
                "non_afib_duration_sec": round(non_afib_duration_sec, 2),
                "afl_duration_sec": 0.0,
                "other_rhythm_labels": ", ".join(other_tags) if other_tags else "None",
                "usable_afib_windows": usable_afib_windows,
                "usable_non_afib_pure_sinus_windows": usable_non_afib_windows,
                "usable_afl_windows": 0,
                "usable_other_windows": usable_other_windows,
                "total_usable_windows": usable_afib_windows + usable_non_afib_windows + usable_other_windows,
                "is_usable_for_afib_training": False,  # Reserved for future cross-database stress testing
                "is_usable_for_control_cohort": False,
                "exclusion_reasons": "; ".join(exclusion_reasons) if exclusion_reasons else "Valid"
            })

    return records_data


def generate_inventory():
    print("=== Phase A: Auditing Local Datasets for AFib Research ===")
    mit_data = audit_mitbih()
    vf_data = audit_vfdb()
    all_data = mit_data + vf_data

    df = pd.DataFrame(all_data)
    csv_path = OUTPUT_DIR / "dataset_inventory.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Exported record-level dataset inventory to: {csv_path}")
    print(f"Total records audited: {len(df)} (MIT-BIH: {len(mit_data)}, VFDB: {len(vf_data)})")

    # Print summary highlights
    mit_df = df[df["archive"] == "MITBIH"]
    afib_mit = mit_df[mit_df["afib_episodes_count"] > 0]
    print(f"\nMIT-BIH Records with AFib: {len(afib_mit)} records")
    print(afib_mit[["record_id", "subject_id", "afib_episodes_count", "afib_duration_sec", "usable_afib_windows", "exclusion_reasons"]])

    print("\nSummary of Usable Windows in MIT-BIH:")
    print(f"  Total Usable AFib Windows: {mit_df['usable_afib_windows'].sum()}")
    print(f"  Total Usable Pure Sinus (N) Windows: {mit_df['usable_non_afib_pure_sinus_windows'].sum()}")
    print(f"  Total Usable AFL (Flutter) Windows: {mit_df['usable_afl_windows'].sum()}")
    print(f"  Total Usable Other Windows: {mit_df['usable_other_windows'].sum()}")


if __name__ == "__main__":
    generate_inventory()
