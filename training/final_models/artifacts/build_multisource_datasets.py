"""Constructs balanced, multi-source, physically calibrated feature datasets for Model 1 and Model 2."""

import io
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly, welch
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TARGET_FS = 360
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)
STEP_SEC = 5.0  # Non-overlapping 5-second windows to maximize record independence
STEP_SAMPLES = int(TARGET_FS * STEP_SEC)


def butter_bandpass(sig: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: int = TARGET_FS) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(2, [max(0.001, lowcut / nyq), min(0.999, highcut / nyq)], btype="band")
    return filtfilt(b, a, sig)


def extract_features_single_window(sig: np.ndarray, fs: int = TARGET_FS) -> dict[str, float]:
    ptp_val = float(np.ptp(sig))
    std_val = float(np.std(sig))
    zero_cross = int(np.sum(np.diff(sig > 0) != 0))

    nperseg = min(len(sig), int(fs * 2))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    valid_mask = (freqs >= 0.5) & (freqs <= 30.0)
    freqs_band = freqs[valid_mask]
    psd_band = psd[valid_mask]

    total_power = np.sum(psd_band) + 1e-12
    vf_mask = (freqs_band >= 3.0) & (freqs_band <= 9.0)
    vf_power = np.sum(psd_band[vf_mask])
    vf_ratio = float(vf_power / total_power)

    dom_idx = np.argmax(psd_band)
    dom_freq = float(freqs_band[dom_idx])

    psd_norm = psd_band / total_power
    psd_norm = psd_norm[psd_norm > 0]
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm) + 1e-12))
    spec_purity = float(psd_band[dom_idx] / (np.median(psd_band) + 1e-12))

    diff1 = np.diff(sig)
    diff2 = np.diff(diff1)
    var_zero = std_val**2 + 1e-12
    var_d1 = float(np.var(diff1)) + 1e-12
    var_d2 = float(np.var(diff2)) + 1e-12

    mobility = float(np.sqrt(var_d1 / var_zero))
    complexity = float(np.sqrt(var_d2 / var_d1) / mobility)

    # LZ complexity (downsampled to 90 Hz, N=450)
    step = max(1, int(round(fs / 90.0)))
    sig_down = sig[::step]
    med = np.median(sig_down)
    binary_seq = (sig_down > med).astype(np.int8)

    words = set()
    w = ""
    lz_count = 0
    for b in binary_seq:
        w += str(b)
        if w not in words:
            words.add(w)
            lz_count += 1
            w = ""
    if w:
        lz_count += 1
    n_seq = len(binary_seq)
    lz_norm = float(lz_count * np.log2(n_seq) / n_seq) if n_seq > 0 else 0.0

    # Pan-Tompkins Peak Detection
    diff_sq = diff1**2
    int_len = int(fs * 0.150)
    integrated = np.convolve(diff_sq, np.ones(int_len) / int_len, mode="same")
    threshold = 0.3 * np.max(integrated) if np.max(integrated) > 0 else 1.0

    refractory = int(fs * 0.300)
    peaks = []
    last_p = -refractory
    for p in range(1, len(integrated) - 1):
        if integrated[p] > integrated[p - 1] and integrated[p] > integrated[p + 1] and integrated[p] > threshold:
            if p - last_p >= refractory:
                peaks.append(p)
                last_p = p

    peak_count = len(peaks)
    if peak_count >= 2:
        rr_intervals = np.diff(peaks) / fs
        mean_rr = float(np.mean(rr_intervals))
        rr_cv = float(np.std(rr_intervals) / (mean_rr + 1e-12))
        qrs_width = 0.080
    else:
        mean_rr = 5.0
        rr_cv = 0.0
        qrs_width = 0.0

    return {
        "calibrated_ptp": ptp_val,
        "calibrated_std": std_val,
        "zero_crossings": zero_cross,
        "vf_band_power_ratio": vf_ratio,
        "dominant_freq": dom_freq,
        "spectral_entropy": spec_entropy,
        "spectral_peak_purity": spec_purity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "lz_complexity": lz_norm,
        "peak_count": peak_count,
        "mean_rr": mean_rr,
        "rr_cv": rr_cv,
        "qrs_width": qrs_width,
    }


def parse_header_metadata(header_text: str) -> dict[str, dict]:
    channels = {}
    lines = header_text.splitlines()
    if not lines:
        return channels
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 9:
            ch_name = parts[8]
            gain_str = parts[2].split("/")[0]
            gain = float(gain_str) if gain_str != "0" else 200.0
            baseline = float(parts[4]) if len(parts) > 4 else 0.0
            adc_zero = float(parts[2].split("(")[1].split(")")[0]) if "(" in parts[2] else 0.0
            channels[ch_name] = {"gain": gain, "baseline": baseline, "adc_zero": adc_zero}
    return channels


def extract_all() -> pd.DataFrame:
    all_rows = []
    t0 = time.time()
    print("Starting multi-source extraction...")

    # -------------------------------------------------------------
    # 1. Challenge 2015
    # -------------------------------------------------------------
    c2015_path = PROJECT_ROOT / "mitbih-vtvtf.zip"
    with zipfile.ZipFile(c2015_path) as outer:
        inner_info = next(i for i in outer.infolist() if i.filename.endswith("training.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_info))) as inner:
            alarms_text = inner.read("training/ALARMS").decode("utf-8")
            true_alarms = {}
            for line in alarms_text.splitlines():
                if line.strip():
                    p = line.strip().split(",")
                    if len(p) == 3 and p[2] == "1":
                        true_alarms[p[0]] = p[1]

            print(f"Challenge 2015: Found {len(true_alarms)} true alarm records to process.")
            for rec_id, alarm_type in true_alarms.items():
                hea_name = f"training/{rec_id}.hea"
                mat_name = f"training/{rec_id}.mat"
                if hea_name not in inner.namelist() or mat_name not in inner.namelist():
                    continue

                header_text = inner.read(hea_name).decode("utf-8")
                ch_meta = parse_header_metadata(header_text)
                target_ch = "II" if "II" in ch_meta else list(ch_meta.keys())[0] if ch_meta else None
                if not target_ch:
                    continue

                mat = loadmat(io.BytesIO(inner.read(mat_name)))["val"]
                # Find channel index
                ch_idx = 0
                lines = header_text.splitlines()
                for idx, line in enumerate(lines[1:]):
                    if len(line.split()) >= 9 and line.split()[8] == target_ch:
                        ch_idx = idx
                        break

                raw_sig = mat[ch_idx].astype(np.float64)
                gain = ch_meta[target_ch]["gain"]
                base = ch_meta[target_ch]["baseline"]
                adc_z = ch_meta[target_ch]["adc_zero"]

                # Physical millivolts calibration
                calib_sig = ((raw_sig - adc_z) / gain + base).astype(np.float32)
                # Resample from 250 to 360 Hz
                sig_360 = resample_poly(calib_sig, TARGET_FS, 250).astype(np.float32)
                sig_filt = butter_bandpass(sig_360)

                # Map labels
                if alarm_type == "Ventricular_Tachycardia":
                    m1_label, m2_4c, m2_5c, orig_rhythm = 1, "VENTRICULAR", "VT", "VT"
                elif alarm_type == "Ventricular_Flutter_Fib":
                    m1_label, m2_4c, m2_5c, orig_rhythm = 1, "VENTRICULAR", "VF_VFL", "VF"
                elif alarm_type == "Tachycardia":
                    m1_label, m2_4c, m2_5c, orig_rhythm = 0, "TACHY", "TACHY", "TACHY"
                elif alarm_type in ["Bradycardia", "Asystole"]:
                    m1_label, m2_4c, m2_5c, orig_rhythm = 0, "BRADY_ASY", "BRADY_ASY", alarm_type
                else:
                    continue

                win_idx = 0
                for start in range(0, len(sig_filt) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                    w = sig_filt[start : start + WINDOW_SAMPLES]
                    feats = extract_features_single_window(w)
                    feats.update({
                        "record_id": f"c2015_{rec_id}",
                        "dataset_source": "c2015",
                        "window_index": win_idx,
                        "original_label": orig_rhythm,
                        "model1_shockable": m1_label,
                        "model2_4class": m2_4c,
                        "model2_5class": m2_5c,
                    })
                    all_rows.append(feats)
                    win_idx += 1

    print(f"Challenge 2015 extracted. Total windows so far: {len(all_rows)} in {time.time() - t0:.1f} s")

    # -------------------------------------------------------------
    # 2. VFDB (Malignant Ventricular Ectopy Database)
    # -------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(PROJECT_ROOT / "VFDB.zip") as z:
            z.extractall(tmpdir)
        records = sorted({f.split(".")[0] for f in os.listdir(tmpdir) if f.endswith(".dat")})
        print(f"VFDB: Processing {len(records)} records...")
        for rec in records:
            try:
                rec_obj = wfdb.rdrecord(os.path.join(tmpdir, rec))
                ann = wfdb.rdann(os.path.join(tmpdir, rec), "atr")
            except Exception:
                continue

            sig_calib = rec_obj.p_signal[:, 0].astype(np.float32)
            sig_360 = resample_poly(sig_calib, TARGET_FS, int(rec_obj.fs)).astype(np.float32)
            sig_filt = butter_bandpass(sig_360)

            # Map sample boundaries to annotations
            ratio = TARGET_FS / rec_obj.fs
            boundaries = [(int(s * ratio), note.strip().rstrip("\x00")) for s, note in zip(ann.sample, ann.aux_note)]

            for start in range(0, len(sig_filt) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                end = start + WINDOW_SAMPLES
                # Determine active rhythm
                active_notes = [note for s, note in boundaries if s <= start]
                curr_note = active_notes[-1] if active_notes else ""

                if not curr_note or "(NOISE" in curr_note:
                    continue

                if "(VT" in curr_note:
                    m1_label, m2_4c, m2_5c, orig_rhythm = 1, "VENTRICULAR", "VT", "VT"
                elif any(x in curr_note for x in ["(VF", "(VFIB", "(VFL"]):
                    m1_label, m2_4c, m2_5c, orig_rhythm = 1, "VENTRICULAR", "VF_VFL", "VF"
                elif any(x in curr_note for x in ["(N", "(NSR"]):
                    m1_label, m2_4c, m2_5c, orig_rhythm = 0, "NSR", "NSR", "NSR"
                elif "(ASYS" in curr_note:
                    m1_label, m2_4c, m2_5c, orig_rhythm = 0, "BRADY_ASY", "BRADY_ASY", "Asystole"
                elif "(SVTA" in curr_note or "(AFIB" in curr_note:
                    m1_label, m2_4c, m2_5c, orig_rhythm = 0, "TACHY", "TACHY", "SVTA"
                else:
                    continue

                w = sig_filt[start:end]
                feats = extract_features_single_window(w)
                feats.update({
                    "record_id": f"vfdb_{rec}",
                    "dataset_source": "vfdb",
                    "window_index": start // STEP_SAMPLES,
                    "original_label": orig_rhythm,
                    "model1_shockable": m1_label,
                    "model2_4class": m2_4c,
                    "model2_5class": m2_5c,
                })
                all_rows.append(feats)

    print(f"VFDB extracted. Total windows so far: {len(all_rows)} in {time.time() - t0:.1f} s")

    # -------------------------------------------------------------
    # 3. Creighton University Database (CUDB - cu01, cu02, cu03)
    # -------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(PROJECT_ROOT / "cu-ventricular-tachyarrhythmia-database-1.0.0.zip") as z:
            z.extractall(tmpdir)
        base = os.path.join(tmpdir, "cu-ventricular-tachyarrhythmia-database-1.0.0")
        for rec in ["cu01", "cu02", "cu03"]:
            try:
                rec_obj = wfdb.rdrecord(os.path.join(base, rec))
                ann = wfdb.rdann(os.path.join(base, rec), "atr")
            except Exception:
                continue

            sig_calib = rec_obj.p_signal[:, 0].astype(np.float32)
            sig_360 = resample_poly(sig_calib, TARGET_FS, int(rec_obj.fs)).astype(np.float32)
            sig_filt = butter_bandpass(sig_360)

            ratio = TARGET_FS / rec_obj.fs
            boundaries = [(int(s * ratio), note.strip().rstrip("\x00")) for s, note in zip(ann.sample, ann.aux_note)]

            for start in range(0, len(sig_filt) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                active_notes = [note for s, note in boundaries if s <= start]
                curr_note = active_notes[-1] if active_notes else ""

                if not curr_note:
                    continue

                if "(VT" in curr_note:
                    m1_label, m2_4c, m2_5c, orig_rhythm = 1, "VENTRICULAR", "VT", "VT"
                elif any(x in curr_note for x in ["(VF", "(VFL"]):
                    m1_label, m2_4c, m2_5c, orig_rhythm = 1, "VENTRICULAR", "VF_VFL", "VF"
                elif "(N" in curr_note:
                    m1_label, m2_4c, m2_5c, orig_rhythm = 0, "NSR", "NSR", "NSR"
                else:
                    continue

                w = sig_filt[start : start + WINDOW_SAMPLES]
                feats = extract_features_single_window(w)
                feats.update({
                    "record_id": f"cudb_{rec}",
                    "dataset_source": "cudb",
                    "window_index": start // STEP_SAMPLES,
                    "original_label": orig_rhythm,
                    "model1_shockable": m1_label,
                    "model2_4class": m2_4c,
                    "model2_5class": m2_5c,
                })
                all_rows.append(feats)

    print(f"CUDB extracted. Total windows so far: {len(all_rows)} in {time.time() - t0:.1f} s")

    # -------------------------------------------------------------
    # 4. MIT-BIH Arrhythmia Database
    # -------------------------------------------------------------
    with zipfile.ZipFile(PROJECT_ROOT / "MITBIH.zip") as z:
        csv_files = sorted([f for f in z.namelist() if f.endswith(".csv") and "/" not in f])
        print(f"MIT-BIH: Processing {len(csv_files)} records...")

        for fname in csv_files:
            rec_id = fname.replace(".csv", "")
            ann_name = f"{rec_id}annotations.txt"
            if ann_name not in z.namelist():
                continue

            with z.open(fname) as f:
                df_csv = pd.read_csv(f)
            mlii_cols = [c for c in df_csv.columns if "MLII" in c]
            if not mlii_cols:
                continue

            raw_sig = df_csv[mlii_cols[0]].values.astype(np.float32)
            # Physical calibration (gain = 200, baseline = 1024)
            sig_calib = ((raw_sig - 1024.0) / 200.0).astype(np.float32)
            sig_filt = butter_bandpass(sig_calib)

            ann_content = z.read(ann_name).decode("utf-8", errors="replace")
            rhythm_markers = []
            for line in ann_content.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        sample_idx = int(parts[1])
                        for p in parts[2:]:
                            if p.startswith("("):
                                rhythm_markers.append((sample_idx, p))
                    except ValueError:
                        pass

            win_count = 0
            # To avoid MIT-BIH overwhelming the dataset with 30,000 NSR windows,
            # take non-overlapping 5s windows (up to 40 windows per record for NSR)
            # while capturing ALL arrhythmia windows (VT, VFL, SVTA, SBR)
            for start in range(0, len(sig_filt) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                active_notes = [m for s, m in rhythm_markers if s <= start]
                curr_note = active_notes[-1] if active_notes else "(N"

                if "(NOISE" in curr_note:
                    continue

                if curr_note == "(VT":
                    m1_label, m2_4c, m2_5c, orig_rhythm = 1, "VENTRICULAR", "VT", "VT"
                elif curr_note == "(VFL":
                    m1_label, m2_4c, m2_5c, orig_rhythm = 1, "VENTRICULAR", "VF_VFL", "VF"
                elif curr_note in ["(N", "(B", "(T"]:
                    # Cap NSR/Ectopy to 35 windows per record to prevent class imbalance
                    if win_count >= 35:
                        continue
                    m1_label, m2_4c, m2_5c, orig_rhythm = 0, "NSR", "NSR", "NSR"
                elif curr_note in ["(SVTA", "(AFIB", "(AFL"]:
                    m1_label, m2_4c, m2_5c, orig_rhythm = 0, "TACHY", "TACHY", "SVTA"
                elif curr_note == "(SBR":
                    m1_label, m2_4c, m2_5c, orig_rhythm = 0, "BRADY_ASY", "BRADY_ASY", "Bradycardia"
                else:
                    continue

                w = sig_filt[start : start + WINDOW_SAMPLES]
                feats = extract_features_single_window(w)
                feats.update({
                    "record_id": f"mitdb_{rec_id}",
                    "dataset_source": "mitdb",
                    "window_index": start // STEP_SAMPLES,
                    "original_label": orig_rhythm,
                    "model1_shockable": m1_label,
                    "model2_4class": m2_4c,
                    "model2_5class": m2_5c,
                })
                all_rows.append(feats)
                win_count += 1

    df_master = pd.DataFrame(all_rows)
    print(f"\nExtraction complete in {time.time() - t0:.1f} s! Total windows: {len(df_master)}, unique records: {df_master['record_id'].nunique()}")
    print("\nDataset Source Distribution:")
    print(df_master["dataset_source"].value_counts())
    print("\nModel 1 Label Distribution:")
    print(df_master["model1_shockable"].value_counts())
    print("\nModel 2 (4-class) Label Distribution:")
    print(df_master["model2_4class"].value_counts())
    print("\nModel 2 (5-class) Label Distribution:")
    print(df_master["model2_5class"].value_counts())

    # Save artifacts
    art_dir = PROJECT_ROOT / "training" / "final_models" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    master_path = art_dir / "master_multisource_feature_dataset.csv"
    df_master.to_csv(master_path, index=False)
    print(f"\nSaved master dataset to: {master_path}")

    # Save Model 1 dataset
    m1_path = art_dir / "model1_shockable_dataset.csv"
    df_master.to_csv(m1_path, index=False)
    print(f"Saved Model 1 dataset to: {m1_path}")

    # Save Model 2 4-class dataset
    m2_4c_path = art_dir / "model2_multiclass_4class_dataset.csv"
    df_master.to_csv(m2_4c_path, index=False)
    print(f"Saved Model 2 4-class dataset to: {m2_4c_path}")

    # Save Model 2 5-class dataset
    m2_5c_path = art_dir / "model2_multiclass_5class_dataset.csv"
    df_master.to_csv(m2_5c_path, index=False)
    print(f"Saved Model 2 5-class dataset to: {m2_5c_path}")

    return df_master


if __name__ == "__main__":
    extract_all()
