"""
Phase 2 Data Quality Audit and Window Count Simulator for AFDB (High-Performance Vectorized)
Author: Antigravity IDE
Date: October 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import wfdb

DATA_DIR = r"training\research\afib_experiment\data\afdb\raw"
RECORD_IDS = [
    "04015", "04043", "04048", "04126", "04746", "04908", "04936",
    "05091", "05121", "05261", "06426", "06453", "06995", "07162",
    "07859", "07879", "07910", "08215", "08219", "08378", "08405",
    "08434", "08455"
]

SPLITS = {
    "train": ["04015", "04043", "04048", "04746", "04936", "05091", "05261",
              "06426", "06453", "06995", "07859", "08378", "08405", "08434", "08455"],
    "val": ["04126", "05121", "07879", "07910"],
    "test": ["04908", "07162", "08215", "08219"]
}

def audit_record(rec_id):
    rec_path = os.path.join(DATA_DIR, rec_id)
    header = wfdb.rdheader(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    
    # Read signal in physical units (mV)
    record = wfdb.rdrecord(rec_path, physical=True)
    sig = record.p_signal
    
    fs = header.fs
    n_samples = header.sig_len
    duration_s = n_samples / fs
    duration_h = duration_s / 3600.0
    
    # 1. RHYTHM ANNOTATION AUDIT
    t_samples = []
    t_rhythms = []
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        clean_aux = aux.strip('\x00').strip()
        t_samples.append(int(s))
        t_rhythms.append(clean_aux)
        
    t_samples = np.array(t_samples, dtype=np.int64)
    first_start = t_samples[0]
    first_rhythm = t_rhythms[0]
    pre_annotation_gap_s = first_start / fs
    
    # Check Channel 1 (primary Lead II analog)
    ch1 = sig[:, 1]
    
    n_nans = int(np.isnan(sig).sum())
    n_infs = int(np.isinf(sig).sum())
    v_min_ch0, v_max_ch0 = float(np.min(sig[:, 0])), float(np.max(sig[:, 0]))
    v_min_ch1, v_max_ch1 = float(np.min(sig[:, 1])), float(np.max(sig[:, 1]))
    
    # Dropout / Flatline Detection:
    # Identify sample regions with continuous consecutive zeros >= 1.0 second (250 samples)
    # or extreme constant values
    from itertools import groupby
    is_zero = (ch1 == 0.0)
    
    # Build mask of samples belonging to long zero runs (>= 250 samples = 1.0s)
    long_zero_mask = np.zeros(n_samples, dtype=bool)
    idx = 0
    for k, g in groupby(is_zero):
        run_len = sum(1 for _ in g)
        if k and run_len >= int(1.0 * fs):
            long_zero_mask[idx:idx+run_len] = True
        idx += run_len
        
    cumsum_long_zero = np.pad(np.cumsum(long_zero_mask), (1, 0))
    
    # 2. WINDOW VECTORIZATION
    # Window: 10.0 seconds (2,500 samples at 250 Hz)
    # Stride: 5.0 seconds (1,250 samples at 250 Hz)
    # Guard band: 2.5 seconds (625 samples at 250 Hz)
    w_len = int(10.0 * fs)
    w_step = int(5.0 * fs)
    guard = int(2.5 * fs)
    
    w_starts = np.arange(0, n_samples - w_len + 1, w_step)
    w_ends = w_starts + w_len
    n_total_windows = len(w_starts)
    
    # Vectorized Guard Band & Transition Intersection Check
    reject_transition = np.zeros(n_total_windows, dtype=bool)
    for t in t_samples:
        reject_transition |= ((w_starts - guard < t) & (t < w_ends + guard))
        
    # Check pre-annotation unknown region: if w_starts < first_start
    reject_unknown = (w_starts < first_start)
    
    # Check if window contains samples from a long dropout zero run (>= 1.0s)
    w_dropout_samples = cumsum_long_zero[w_ends] - cumsum_long_zero[w_starts]
    reject_dropout = (w_dropout_samples > 0)
    
    # Determine rhythm segment for each window
    seg_idx = np.searchsorted(t_samples, w_starts, side='right') - 1
    
    # Candidate surviving windows
    candidate_mask = (~reject_transition) & (~reject_unknown) & (~reject_dropout) & (seg_idx >= 0)
    candidate_indices = np.where(candidate_mask)[0]
    
    # Check physiological amplitude: Vpp >= 0.05 mV
    reject_flatline = np.zeros(n_total_windows, dtype=bool)
    for idx in candidate_indices:
        ws = w_starts[idx]
        we = w_ends[idx]
        w_sig = ch1[ws:we]
        v_pp = np.ptp(w_sig)
        if v_pp < 0.05:
            reject_flatline[idx] = True
            
    final_kept_mask = candidate_mask & (~reject_flatline)
    
    kept_rhythms = [t_rhythms[seg_idx[i]] for i in np.where(final_kept_mask)[0]]
    from collections import Counter
    rhythm_counts = dict(Counter(kept_rhythms))
    
    n_afib = rhythm_counts.get("(AFIB", 0)
    n_afl = rhythm_counts.get("(AFL", 0)
    n_junc = rhythm_counts.get("(J", 0)
    n_norm = rhythm_counts.get("(N", 0)
    n_non_afib = len(kept_rhythms) - n_afib
    
    total_rejected_quality = int(np.sum(reject_dropout) + np.sum(reject_flatline))
    
    return {
        "record_id": rec_id,
        "fs": fs,
        "n_samples": int(n_samples),
        "duration_hours": round(duration_h, 4),
        "n_transitions": len(t_samples),
        "pre_annotation_gap_s": round(pre_annotation_gap_s, 4),
        "first_annotation_sample": int(first_start),
        "first_annotation_rhythm": first_rhythm,
        "unique_rhythms": list(set(t_rhythms)),
        "n_nans": n_nans,
        "n_infs": n_infs,
        "v_range_ch0": [round(v_min_ch0, 4), round(v_max_ch0, 4)],
        "v_range_ch1": [round(v_min_ch1, 4), round(v_max_ch1, 4)],
        "total_possible_windows": int(n_total_windows),
        "n_rejected_unknown": int(np.sum(reject_unknown)),
        "n_rejected_transition": int(np.sum(reject_transition)),
        "n_rejected_dropout": int(np.sum(reject_dropout)),
        "n_rejected_low_amplitude": int(np.sum(reject_flatline)),
        "total_rejected_quality": total_rejected_quality,
        "n_kept_windows": int(np.sum(final_kept_mask)),
        "n_afib_windows": int(n_afib),
        "n_non_afib_windows": int(n_non_afib),
        "n_afl_windows": int(n_afl),
        "n_junctional_windows": int(n_junc),
        "n_normal_windows": int(n_norm),
        "rhythm_window_counts": rhythm_counts
    }

def main():
    results = []
    print("Starting refined vectorized AFDB audit across 23 records...", flush=True)
    for rec_id in RECORD_IDS:
        res = audit_record(rec_id)
        results.append(res)
        print(f"[{rec_id}] Total: {res['total_possible_windows']} | Kept: {res['n_kept_windows']} (AFIB: {res['n_afib_windows']}, Non-AFIB: {res['n_non_afib_windows']}, AFL: {res['n_afl_windows']}) | Rej Trans: {res['n_rejected_transition']}, Rej Quality: {res['total_rejected_quality']}", flush=True)
        
    df_res = pd.DataFrame(results)
    
    # Assign cohort
    cohort_map = {}
    for c, recs in SPLITS.items():
        for r in recs:
            cohort_map[r] = c
    df_res["cohort"] = df_res["record_id"].map(cohort_map)
    
    print("\n" + "="*80, flush=True)
    print("COHORT BREAKDOWN (TRAIN / VAL / TEST)", flush=True)
    print("="*80, flush=True)
    cohort_agg = df_res.groupby("cohort")[[
        "n_samples", "duration_hours", "total_possible_windows", "n_rejected_transition", "total_rejected_quality",
        "n_kept_windows", "n_afib_windows", "n_non_afib_windows", "n_afl_windows", "n_junctional_windows", "n_normal_windows"
    ]].sum()
    print(cohort_agg.to_string(), flush=True)
    
    print("\n" + "="*80, flush=True)
    print("OVERALL TOTALS ACROSS ENTIRE AFDB", flush=True)
    print("="*80, flush=True)
    overall = df_res[[
        "total_possible_windows", "n_rejected_transition", "total_rejected_quality",
        "n_kept_windows", "n_afib_windows", "n_non_afib_windows", "n_afl_windows", "n_junctional_windows", "n_normal_windows"
    ]].sum()
    print(overall.to_string(), flush=True)
    
    # Save detailed JSON and CSV
    out_json = r"training\research\afib_experiment\data\afdb\phase2_audit_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    out_csv = r"training\research\afib_experiment\data\afdb\phase2_record_window_counts.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved metrics to:\n  JSON: {out_json}\n  CSV:  {out_csv}", flush=True)

if __name__ == "__main__":
    main()
