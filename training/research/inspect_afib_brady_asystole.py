"""
Read-Only Dataset Inspection Script: AFib, Bradycardia, and Asystole
====================================================================
Gathers exact local annotation counts, records, sample spans, and lead availability.
"""
import zipfile
import io
import collections
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def inspect_mitbih():
    print("--- 1. MIT-BIH Arrhythmia Database (MITBIH.zip) ---")
    zip_path = PROJECT_ROOT / "MITBIH.zip"
    if not zip_path.exists():
        print("MITBIH.zip not found.")
        return

    afib_records = collections.defaultdict(list)
    afl_records = collections.defaultdict(list)
    sbr_records = collections.defaultdict(list)

    with zipfile.ZipFile(zip_path) as z:
        ann_files = sorted([f for f in z.namelist() if f.endswith("annotations.txt") and "/" not in f])
        for f in ann_files:
            rec = f.replace("annotations.txt", "")
            lines = z.read(f).decode("utf-8", errors="replace").splitlines()
            for l in lines[1:]:
                parts = l.split()
                if len(parts) >= 3:
                    try:
                        s_idx = int(parts[1])
                        for p in parts[2:]:
                            if p == "(AFIB":
                                afib_records[rec].append(s_idx)
                            elif p == "(AFL":
                                afl_records[rec].append(s_idx)
                            elif p == "(SBR":
                                sbr_records[rec].append(s_idx)
                    except ValueError:
                        pass

    print(f"Records with (AFIB: {len(afib_records)} records -> {list(afib_records.keys())}")
    for r, s_list in afib_records.items():
        print(f"  Record {r}: {len(s_list)} AFIB marker(s)")

    print(f"\nRecords with (AFL (Atrial Flutter): {len(afl_records)} records -> {list(afl_records.keys())}")
    for r, s_list in afl_records.items():
        print(f"  Record {r}: {len(s_list)} AFL marker(s)")

    print(f"\nRecords with (SBR (Sinus Bradycardia): {len(sbr_records)} records -> {list(sbr_records.keys())}")
    for r, s_list in sbr_records.items():
        print(f"  Record {r}: {len(s_list)} SBR marker(s)")


def inspect_vfdb():
    print("\n--- 2. MIT-BIH Malignant Ventricular Ectopy (VFDB.zip) ---")
    zip_path = PROJECT_ROOT / "VFDB.zip"
    if not zip_path.exists():
        print("VFDB.zip not found.")
        return

    import tempfile, os, wfdb
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)
        records = sorted({f.split(".")[0] for f in os.listdir(tmpdir) if f.endswith(".dat")})
        
        afib_recs = collections.defaultdict(int)
        sbr_recs = collections.defaultdict(int)
        asys_recs = collections.defaultdict(int)

        for rec in records:
            try:
                ann = wfdb.rdann(os.path.join(tmpdir, rec), "atr")
                for note in ann.aux_note:
                    n = note.strip().rstrip("\x00")
                    if n == "(AFIB":
                        afib_recs[rec] += 1
                    elif n == "(SBR":
                        sbr_recs[rec] += 1
                    elif n == "(ASYS":
                        asys_recs[rec] += 1
            except Exception:
                pass

    print(f"VFDB records with (AFIB: {len(afib_recs)} -> {dict(afib_recs)}")
    print(f"VFDB records with (SBR : {len(sbr_recs)} -> {dict(sbr_recs)}")
    print(f"VFDB records with (ASYS: {len(asys_recs)} -> {dict(asys_recs)}")


def inspect_c2015():
    print("\n--- 3. PhysioNet Challenge 2015 (mitbih-vtvtf.zip) ---")
    zip_path = PROJECT_ROOT / "mitbih-vtvtf.zip"
    if not zip_path.exists():
        print("mitbih-vtvtf.zip not found.")
        return

    with zipfile.ZipFile(zip_path) as outer:
        inner_info = next(i for i in outer.infolist() if i.filename.endswith("training.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_info))) as inner:
            alarms_text = inner.read("training/ALARMS").decode("utf-8")
            c2015_true = collections.Counter()
            c2015_false = collections.Counter()
            for line in alarms_text.splitlines():
                if line.strip():
                    p = line.strip().split(",")
                    if len(p) == 3:
                        if p[2] == "1":
                            c2015_true[p[1]] += 1
                        else:
                            c2015_false[p[1]] += 1

    print(f"C2015 True Alarms : {dict(c2015_true)}")
    print(f"C2015 False Alarms: {dict(c2015_false)}")


if __name__ == "__main__":
    inspect_mitbih()
    inspect_vfdb()
    inspect_c2015()
