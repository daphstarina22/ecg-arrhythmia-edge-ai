"""Audit all MIT-BIH records, leads, annotation symbols, and sample mappings."""
import zipfile
import collections
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = PROJECT_ROOT / "MITBIH.zip"

def audit():
    print(f"Reading: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        all_files = z.namelist()
        records = sorted({f.replace(".csv", "") for f in all_files if f.endswith(".csv") and "/" not in f})
        print(f"Total unique records discovered: {len(records)}")

        total_symbols = collections.Counter()
        record_symbols = collections.defaultdict(collections.Counter)
        leads_per_record = {}
        sample_counts = {}

        for rec in records:
            csv_name = f"{rec}.csv"
            ann_name = f"{rec}annotations.txt"

            # Check CSV header
            with z.open(csv_name) as f:
                header = f.readline().decode("utf-8").strip()
                cols = [c.replace("'", "").strip() for c in header.split(",")]
                leads_per_record[rec] = cols[1:]

            # Check annotations
            with z.open(ann_name) as f:
                lines = f.read().decode("utf-8", errors="replace").splitlines()
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            s_idx = int(parts[1])
                            sym = parts[2]
                            total_symbols[sym] += 1
                            record_symbols[rec][sym] += 1
                        except ValueError:
                            pass

    print("\n=== TOTAL ANNOTATION SYMBOLS ACROSS ALL 48 RECORDS ===")
    for sym, count in total_symbols.most_common():
        print(f"  {sym:>4}: {count:>8}")

    print("\n=== LEAD CONFIGURATIONS ===")
    lead_dist = collections.Counter([tuple(v) for v in leads_per_record.values()])
    for l_tuple, cnt in lead_dist.items():
        print(f"  Leads {l_tuple}: {cnt} records")

    # Specific focus on MLII availability
    non_mlii = [rec for rec, leads in leads_per_record.items() if not any("MLII" in l for l in leads)]
    print(f"\nRecords WITHOUT 'MLII': {len(non_mlii)} -> {non_mlii}")

    print("\n=== PVC ('V') BREAKDOWN PER RECORD ===")
    v_records = {rec: counts["V"] for rec, counts in record_symbols.items() if counts["V"] > 0}
    print(f"Records with >= 1 'V' beat: {len(v_records)} / {len(records)}")
    for rec, cnt in sorted(v_records.items(), key=lambda x: -x[1]):
        total_rec_beats = sum(record_symbols[rec].values())
        pct = (cnt / total_rec_beats) * 100.0
        n_cnt = record_symbols[rec]["N"]
        print(f"  Record {rec}: {cnt:>5} V beats ({pct:>5.1f}%) | N beats: {n_cnt:>5} | Total: {total_rec_beats:>5}")

    print("\n=== ATRIAL PREMATURE ('A') BREAKDOWN PER RECORD ===")
    a_records = {rec: counts["A"] for rec, counts in record_symbols.items() if counts["A"] > 0}
    print(f"Records with >= 1 'A' beat: {len(a_records)} / {len(records)}")

    # Check mapping between record numbers and patient/subject IDs in MIT-BIH standard documentation
    # Standard MIT-BIH Arrhythmia Database subject mapping:
    # 48 records: 23 records randomly chosen from 4000 ambulatory ECGs (100-124),
    # 25 records chosen from same set to include less common but clinically significant arrhythmias (200-234).
    # Note records 201 and 202 are from the same patient (male 68).
    # All other records are from distinct patients!
    print("\nPatient / Subject mapping note:")
    print("In PhysioNet MIT-BIH Arrhythmia Database, records 201 and 202 are from the same male subject (age 68).")
    print("All other 46 records correspond to 46 distinct individual patients (total 47 subjects).")

if __name__ == "__main__":
    audit()
