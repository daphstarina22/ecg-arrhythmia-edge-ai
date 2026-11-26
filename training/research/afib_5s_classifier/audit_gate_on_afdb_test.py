"""
Audit Safety Gate on Held-Out Test Set (True AFib Suppression Check)
===================================================================
Tests whether Tier 1 (Shockable) or Tier 2 (Ventricular) erroneously intercept
and suppress true Atrial Fibrillation in the 4 held-out test patients.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import scipy.signal as signal
import pandas as pd
import onnxruntime as ort
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_DIR = PROJECT_ROOT / "training" / "research" / "afib_5s_classifier"
RAW_AFDB_DIR = PROJECT_ROOT / "training" / "research" / "afib_experiment" / "data" / "afdb" / "raw"
MODELS_DIR = PROJECT_ROOT / "training" / "final_models" / "export_candidates"

M1_PATH = MODELS_DIR / "shockable_classifier_candidate.onnx"
M2_PATH = MODELS_DIR / "arrhythmia_multiclass_4class_candidate.onnx"

from evaluate_safety_gated_cascade import extract_cascade_features_from_sig, MODEL1_FEATURES, MODEL2_FEATURES, MODEL2_CLASSES

TARGET_FS = 360
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_FS)

TEST_RECORDS = ["04908", "07162", "08215", "08219"]


def main():
    print("=" * 80)
    print("AUDITING SAFETY GATE ON HELD-OUT TEST AFIB WINDOWS")
    print("=" * 80)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess_m1 = ort.InferenceSession(str(M1_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    sess_m2 = ort.InferenceSession(str(M2_PATH), sess_options=opts, providers=["CPUExecutionProvider"])

    total_afib_windows_audited = 0
    t1_false_shockable_on_afib = 0
    t2_false_ventricular_on_afib = 0

    per_record_audit = {}

    for rec_id in TEST_RECORDS:
        rec_path = str(RAW_AFDB_DIR / rec_id)
        ann = wfdb.rdann(rec_path, 'atr')
        rec = wfdb.rdrecord(rec_path, physical=True)
        sig_250 = rec.p_signal[:, 1]
        sig_360 = signal.resample_poly(sig_250, up=36, down=25).astype(np.float32)

        # Find pure AFib segments
        t_samples = np.array(ann.sample)
        t_rhythms = [note.strip('\x00').strip() for note in ann.aux_note]

        # Sample up to 200 non-overlapping AFib windows per test record
        rec_afib_audited = 0
        rec_t1_shockable = 0
        rec_t2_vent = 0

        n_total_win = int(len(sig_360) / WINDOW_SAMPLES)
        for w_i in range(n_total_win):
            ws = w_i * WINDOW_SAMPLES
            we = ws + WINDOW_SAMPLES
            ws_250 = int(ws * 250 / TARGET_FS)
            
            # Check rhythm at ws_250
            idx = np.searchsorted(t_samples, ws_250, side='right') - 1
            if idx < 0 or idx >= len(t_rhythms) or t_rhythms[idx] != "(AFIB":
                continue

            w_sig = sig_360[ws:we]
            if len(w_sig) < WINDOW_SAMPLES or np.ptp(w_sig) < 0.05:
                continue

            c_feats = extract_cascade_features_from_sig(w_sig, fs=TARGET_FS)
            vec_m1 = np.array([[c_feats[f] for f in MODEL1_FEATURES]], dtype=np.float32)
            vec_m2 = np.array([[c_feats[f] for f in MODEL2_FEATURES]], dtype=np.float32)

            out_m1 = sess_m1.run(None, {sess_m1.get_inputs()[0].name: vec_m1})
            p_shockable = float(out_m1[1][0].get(1, 0.0) if isinstance(out_m1[1], list) else out_m1[1][0][1])

            out_m2 = sess_m2.run(None, {sess_m2.get_inputs()[0].name: vec_m2})
            pred_class_m2 = MODEL2_CLASSES[int(out_m2[0][0])]

            is_shock = (p_shockable >= 0.50)
            is_vent = (pred_class_m2 == "VENTRICULAR")

            rec_afib_audited += 1
            if is_shock:
                rec_t1_shockable += 1
            if is_vent:
                rec_t2_vent += 1

            if rec_afib_audited >= 250:
                break

        total_afib_windows_audited += rec_afib_audited
        t1_false_shockable_on_afib += rec_t1_shockable
        t2_false_ventricular_on_afib += rec_t2_vent

        per_record_audit[rec_id] = {
            "afib_windows_audited": rec_afib_audited,
            "false_shockable_intercepts": rec_t1_shockable,
            "false_ventricular_intercepts": rec_t2_vent,
            "afib_pass_through_rate_pct": round((rec_afib_audited - rec_t1_shockable - rec_t2_vent) / rec_afib_audited * 100.0, 2)
        }
        print(f"Record {rec_id}: Audited {rec_afib_audited} AFib windows -> T1 False Shockable: {rec_t1_shockable} ({rec_t1_shockable/rec_afib_audited*100:.1f}%), T2 False Vent: {rec_t2_vent} ({rec_t2_vent/rec_afib_audited*100:.1f}%) | Gate Pass-Through: {per_record_audit[rec_id]['afib_pass_through_rate_pct']:.1f}%")

    global_pass_through = (total_afib_windows_audited - t1_false_shockable_on_afib - t2_false_ventricular_on_afib) / total_afib_windows_audited * 100.0
    print("\n" + "=" * 80)
    print(f"TOTAL AFIB WINDOWS AUDITED: {total_afib_windows_audited}")
    print(f"Tier 1 (Shockable) False Positives on AFib: {t1_false_shockable_on_afib} ({t1_false_shockable_on_afib/total_afib_windows_audited*100:.2f}%)")
    print(f"Tier 2 (Ventricular) False Positives on AFib: {t2_false_ventricular_on_afib} ({t2_false_ventricular_on_afib/total_afib_windows_audited*100:.2f}%)")
    print(f"True AFib Allowed Through to Tier 3: {global_pass_through:.2f}%")
    print("=" * 80)

    audit_results = {
        "total_afib_windows_audited": total_afib_windows_audited,
        "tier1_false_shockable_intercepts": t1_false_shockable_on_afib,
        "tier2_false_ventricular_intercepts": t2_false_ventricular_on_afib,
        "global_pass_through_rate_pct": round(global_pass_through, 2),
        "per_record_audit": per_record_audit
    }

    out_json = RESEARCH_DIR / "safety_gate_true_afib_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)


if __name__ == "__main__":
    main()
