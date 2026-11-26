"""
Detailed AFib Episode and Window Capacity Audit
================================================
Calculates exact start/end samples, durations (sec), and valid non-transition
window capacities for AFib and non-AFib segments in local MIT-BIH and VFDB.
"""
import zipfile
import collections
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def audit_mitbih_afib_episodes():
    print("=== LOCAL MIT-BIH ARRHYTHMIA DATABASE (MITBIH.zip) AFIB EPISODES ===")
    zip_path = PROJECT_ROOT / "MITBIH.zip"
    if not zip_path.exists():
        print("MITBIH.zip missing.")
        return

    records_with_afib = ["201", "202", "203", "210", "217", "219", "221", "222"]
    
    with zipfile.ZipFile(zip_path) as z:
        for rec in records_with_afib:
            ann_name = f"{rec}annotations.txt"
            lines = z.read(ann_name).decode("utf-8", errors="replace").splitlines()
            
            rhythm_events = []
            for l in lines[1:]:
                parts = l.split()
                if len(parts) >= 3:
                    try:
                        s_idx = int(parts[1])
                        for p in parts[2:]:
                            if p.startswith("("):
                                rhythm_events.append((s_idx, p))
                    except ValueError:
                        pass
            
            # Map intervals
            total_samples = 650000
            fs = 360
            intervals = []
            for i, (start_idx, r_tag) in enumerate(rhythm_events):
                end_idx = rhythm_events[i+1][0] if i+1 < len(rhythm_events) else total_samples
                dur_sec = (end_idx - start_idx) / fs
                intervals.append((start_idx, end_idx, dur_sec, r_tag))
                
            afib_intervals = [x for x in intervals if x[3] == "(AFIB"]
            afl_intervals = [x for x in intervals if x[3] == "(AFL"]
            other_intervals = [x for x in intervals if x[3] not in ("(AFIB", "(AFL")]
            
            tot_afib_sec = sum(x[2] for x in afib_intervals)
            tot_afl_sec = sum(x[2] for x in afl_intervals)
            tot_other_sec = sum(x[2] for x in other_intervals)
            
            print(f"\nRecord {rec} (Total duration: {total_samples/fs:.1f}s):")
            print(f"  AFIB: {len(afib_intervals):>2} episode(s), Total time: {tot_afib_sec:>6.1f}s ({tot_afib_sec/60:>4.1f} min)")
            print(f"  AFL : {len(afl_intervals):>2} episode(s), Total time: {tot_afl_sec:>6.1f}s ({tot_afl_sec/60:>4.1f} min)")
            print(f"  Other:{len(other_intervals):>2} episode(s), Total time: {tot_other_sec:>6.1f}s ({tot_other_sec/60:>4.1f} min)")
            
            # Show individual AFIB episodes if <= 5
            if len(afib_intervals) <= 5:
                for idx, ep in enumerate(afib_intervals):
                    print(f"    Episode {idx+1}: samples {ep[0]:>6} -> {ep[1]:>6} ({ep[2]:>5.1f}s)")
            else:
                shortest = min(x[2] for x in afib_intervals)
                longest = max(x[2] for x in afib_intervals)
                print(f"    Durations range from {shortest:.1f}s to {longest:.1f}s")


if __name__ == "__main__":
    audit_mitbih_afib_episodes()
