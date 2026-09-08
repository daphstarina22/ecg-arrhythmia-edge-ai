"""Phase 5B: Raw Waveform Physiological Feature Extraction (Strategy B).

Extracts spectral organization, temporal autocorrelation, and complexity/nonlinear
features directly from calibrated raw ECG waveforms (360 Hz, 5-second windows)
for Challenge 2015, VFDB, and CUDB using validated Strategy B optimizations.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.signal as signal
from scipy.signal import resample_poly
import wfdb

TARGET_FS = 360
WINDOW_SECONDS = 5.0
STEP_SECONDS = 2.5
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SECONDS)
STEP_SAMPLES = int(TARGET_FS * STEP_SECONDS)

LABEL_MAP_C2015 = {
    "Ventricular_Tachycardia": "VT",
    "Ventricular_Flutter_Fib": "VF",
}


def classify_aux_wfdb(note: str) -> str | None:
    normalized = note.strip().rstrip("\x00")
    if "(VF" in normalized:
        return "VF"
    if "(VT" in normalized:
        return "VT"
    return None


# ==============================================================================
# OPTIMIZED PHYSIOLOGICAL FEATURE CALCULATORS (STRATEGY B)
# ==============================================================================


def compute_spectral_features(x: np.ndarray, fs: int = TARGET_FS) -> dict[str, float]:
    """Extract spectral organization features from 0.5 - 30 Hz band."""
    nperseg = min(256, len(x))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    mask = (freqs >= 0.5) & (freqs <= 30.0)
    f_band = freqs[mask]
    p_band = psd[mask]
    total_power = float(np.sum(p_band)) + 1e-12

    dom_idx = int(np.argmax(p_band))
    dom_freq = float(f_band[dom_idx])
    dom_power = float(p_band[dom_idx])

    peaks, props = signal.find_peaks(p_band, prominence=0.01 * dom_power)
    prominences = props.get("prominences", [])
    dom_prominence = float(np.max(prominences)) if len(prominences) > 0 else 0.0

    peak_power_ratio = float(dom_power / total_power)

    p_norm = p_band / total_power
    spec_entropy = float(-np.sum(p_norm * np.log2(p_norm + 1e-12)) / np.log2(len(p_norm)))

    geo_mean = float(np.exp(np.mean(np.log(p_band + 1e-12))))
    arith_mean = float(np.mean(p_band)) + 1e-12
    spec_flatness = float(geo_mean / arith_mean)

    spec_centroid = float(np.sum(f_band * p_band) / total_power)
    spec_bandwidth = float(np.sqrt(np.sum(((f_band - spec_centroid) ** 2) * p_band) / total_power))

    conc_mask = (f_band >= dom_freq - 1.0) & (f_band <= dom_freq + 1.0)
    spec_concentration = float(np.sum(p_band[conc_mask]) / total_power)

    if len(peaks) > 0:
        top_peak_power = float(np.sum(np.sort(p_band[peaks])[-3:]))
        spec_peak_purity = float(top_peak_power / total_power)
    else:
        spec_peak_purity = peak_power_ratio

    return {
        "dominant_freq": dom_freq,
        "dominant_peak_power": dom_power,
        "dominant_peak_prominence": dom_prominence,
        "spectral_peak_power_ratio": peak_power_ratio,
        "spectral_entropy": spec_entropy,
        "spectral_flatness": spec_flatness,
        "spectral_centroid": spec_centroid,
        "spectral_bandwidth": spec_bandwidth,
        "spectral_concentration": spec_concentration,
        "spectral_peak_purity": spec_peak_purity,
    }


def compute_autocorrelation_features(x: np.ndarray, fs: int = TARGET_FS) -> dict[str, float]:
    """Extract temporal autocorrelation and periodicity features."""
    x_centered = x - np.mean(x)
    var = float(np.var(x_centered))
    if var < 1e-12:
        return {
            "ac_max_peak_ratio": 0.0,
            "ac_first_secondary_peak": 0.0,
            "ac_decay_time": 0.0,
            "ac_periodicity_strength": 0.0,
            "ac_zero_crossing_lag": 0.0,
        }
    n = len(x_centered)
    r = signal.fftconvolve(x_centered, x_centered[::-1], mode="full")[n - 1:]
    r_norm = r / r[0]

    min_lag = int(0.2 * fs)
    max_lag = min(int(1.0 * fs), len(r_norm) - 1)
    cardiac_r = r_norm[min_lag:max_lag]

    ac_max_peak = float(np.max(cardiac_r)) if len(cardiac_r) > 0 else 0.0

    peaks, _ = signal.find_peaks(r_norm[:max_lag], distance=max(min_lag // 2, 1))
    valid_peaks = [p for p in peaks if p >= min_lag]
    first_sec_peak = float(r_norm[valid_peaks[0]]) if len(valid_peaks) > 0 else ac_max_peak

    decay_idx = np.where(r_norm < (1.0 / np.e))[0]
    decay_time = float(decay_idx[0] / fs) if len(decay_idx) > 0 else float(max_lag / fs)

    zero_idx = np.where(r_norm < 0.0)[0]
    zero_lag = float(zero_idx[0] / fs) if len(zero_idx) > 0 else float(max_lag / fs)

    if len(valid_peaks) > 0:
        p_idx = valid_peaks[0]
        trough_val = float(np.min(r_norm[:p_idx]))
        periodicity_strength = float(r_norm[p_idx] - trough_val)
    else:
        periodicity_strength = 0.0

    return {
        "ac_max_peak_ratio": ac_max_peak,
        "ac_first_secondary_peak": first_sec_peak,
        "ac_decay_time": decay_time,
        "ac_periodicity_strength": periodicity_strength,
        "ac_zero_crossing_lag": zero_lag,
    }


def compute_complexity_features_strategy_b(x: np.ndarray) -> dict[str, float]:
    """Extract complexity & entropy using validated Strategy B downsampling."""
    x_centered = x - np.mean(x)
    var_0 = float(np.var(x_centered))
    dx = np.diff(x_centered)
    var_1 = float(np.var(dx))
    ddx = np.diff(dx)
    var_2 = float(np.var(ddx))

    hjorth_activity = var_0
    hjorth_mobility = float(np.sqrt(var_1 / (var_0 + 1e-12)))
    mobility_dx = float(np.sqrt(var_2 / (var_1 + 1e-12)))
    hjorth_complexity = float(mobility_dx / (hjorth_mobility + 1e-12))

    # Lempel-Ziv Complexity on 90 Hz downsampled binary sequence (N=450)
    x_sub = x[::4]
    binary = (x_sub > np.median(x_sub)).tobytes()
    n = len(binary)
    i, k, l = 0, 1, 1
    c = 1
    while l + k <= n:
        if binary[i + k - 1] == binary[l + k - 1]:
            k += 1
        else:
            if k > 1:
                i += 1
                if i == l:
                    c += 1
                    l += k
                    i = 0
                    k = 1
                else:
                    k = 1
            else:
                c += 1
                l += 1
                i = 0
                k = 1
    b = n / (math.log2(n) if n > 1 else 1)
    lz_complexity = float(c / b)

    # Permutation entropy (m=3, tau=2)
    m = 3
    tau = 2
    n_patterns = len(x) - (m - 1) * tau
    if n_patterns > 0:
        sub_indices = np.arange(n_patterns)
        matrix = np.column_stack([x[sub_indices + j * tau] for j in range(m)])
        ranks = np.argsort(matrix, axis=1)
        hash_codes = ranks[:, 0] * 9 + ranks[:, 1] * 3 + ranks[:, 2]
        _, counts = np.unique(hash_codes, return_counts=True)
        probs = counts / n_patterns
        perm_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)) / math.log2(math.factorial(m)))
    else:
        perm_entropy = 0.0

    # Sample entropy on 45 Hz downsampled signal (N=225, m=2, r=0.2*std)
    x_down = x[::8].astype(np.float32)
    std_down = float(np.std(x_down))
    if std_down > 1e-8:
        r_thresh = 0.2 * std_down
        X2 = np.lib.stride_tricks.sliding_window_view(x_down, 2)
        X3 = np.lib.stride_tricks.sliding_window_view(x_down, 3)
        d2 = np.max(np.abs(X2[:, None, :] - X2[None, :, :]), axis=2)
        d3 = np.max(np.abs(X3[:, None, :] - X3[None, :, :]), axis=2)
        np.fill_diagonal(d2, np.inf)
        np.fill_diagonal(d3, np.inf)
        B = np.sum(d2 < r_thresh)
        A = np.sum(d3 < r_thresh)
        samp_entropy = float(-np.log(A / B)) if A > 0 and B > 0 else 0.0
    else:
        samp_entropy = 0.0

    return {
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
        "lz_complexity": lz_complexity,
        "permutation_entropy": perm_entropy,
        "sample_entropy": samp_entropy,
    }


def extract_all_physiological_features(x: np.ndarray, fs: int = TARGET_FS) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(compute_spectral_features(x, fs))
    feats.update(compute_autocorrelation_features(x, fs))
    feats.update(compute_complexity_features_strategy_b(x))
    return feats


# ==============================================================================
# PIPELINE WITH FULL DATASET EXTRACTION & STREAMING PROGRESS
# ==============================================================================


def process_c2015_extracted(
    c2015_zip_path: Path,
    reference_records: set[str],
    total_target_windows: int,
    current_count: int,
    t_start_global: float,
) -> tuple[list[dict[str, Any]], int]:
    print("\n[Source 1/3] Extracting Challenge 2015 raw records...")
    sys.stdout.flush()
    rows = []
    total_windows_done = current_count

    with zipfile.ZipFile(c2015_zip_path) as outer:
        training_info = next(item for item in outer.infolist() if item.filename.endswith("/training.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(training_info))) as archive:
            alarms = {}
            for line in archive.read("training/ALARMS").decode(errors="replace").splitlines():
                fields = line.strip().split(",")
                if len(fields) == 3 and fields[2] == "1" and fields[1] in LABEL_MAP_C2015:
                    alarms[fields[0]] = LABEL_MAP_C2015[fields[1]]

            records_list = [
                line for line in archive.read("training/RECORDS").decode(errors="replace").splitlines() if line
            ]
            valid_records = [r for r in records_list if r in alarms and f"c2015_{r}" in reference_records]

            for rec_idx, rec_id in enumerate(valid_records):
                t_rec_start = time.perf_counter()
                label = alarms[rec_id]
                full_rec_id = f"c2015_{rec_id}"

                hea_text = archive.read(f"training/{rec_id}.hea").decode(errors="replace")
                lines = hea_text.splitlines()
                fs_orig = float(lines[0].split()[2])

                chan_idx = None
                gain, baseline, adc_zero = 200.0, 0.0, 0.0
                for idx, line in enumerate(lines[1:]):
                    if not line.strip() or line.startswith("#"):
                        continue
                    fields = line.split()
                    if fields[-1] == "II":
                        chan_idx = idx
                        gain = float(fields[2].split("/")[0])
                        baseline = float(fields[4])
                        adc_zero = float(fields[5])
                        break
                if chan_idx is None:
                    continue

                val = loadmat(io.BytesIO(archive.read(f"training/{rec_id}.mat")))["val"]
                digital = val[chan_idx].astype(np.float64)
                physical = (digital - adc_zero) / gain + baseline
                if fs_orig != TARGET_FS:
                    physical = resample_poly(physical, TARGET_FS, int(fs_orig))

                rec_windows = 0
                for w_idx, start in enumerate(range(0, len(physical) - WINDOW_SAMPLES, STEP_SAMPLES)):
                    seg = physical[start:start + WINDOW_SAMPLES]
                    feats = extract_all_physiological_features(seg, TARGET_FS)
                    rec_windows += 1
                    total_windows_done += 1
                    rows.append({
                        **feats,
                        "record_id": full_rec_id,
                        "dataset_source": "c2015",
                        "label": label,
                        "target": 0 if label == "VT" else 1,
                        "window_index": w_idx,
                        "window_duration_seconds": WINDOW_SECONDS,
                        "sampling_rate": TARGET_FS,
                    })

                t_rec_elapsed = time.perf_counter() - t_rec_start
                t_global_elapsed = time.perf_counter() - t_start_global
                rate = total_windows_done / max(t_global_elapsed, 0.001)
                rem_windows = total_target_windows - total_windows_done
                eta_sec = rem_windows / max(rate, 0.001)

                if (rec_idx + 1) % 5 == 0 or (rec_idx + 1) == len(valid_records):
                    print(
                        f"  [c2015] [{rec_idx+1}/{len(valid_records)}] {full_rec_id} ({label}): "
                        f"{rec_windows} win | Total: {total_windows_done}/{total_target_windows} "
                        f"({100.0*total_windows_done/total_target_windows:.1f}%) | "
                        f"Elapsed: {t_global_elapsed/60.0:.2f}m | ETA: {eta_sec/60.0:.2f}m"
                    )
                    sys.stdout.flush()

    return rows, total_windows_done


def process_wfdb_extracted(
    archive_path: Path,
    source: str,
    reference_records: set[str],
    total_target_windows: int,
    current_count: int,
    t_start_global: float,
) -> tuple[list[dict[str, Any]], int]:
    print(f"\n[Source] Extracting {source.upper()} raw records...")
    sys.stdout.flush()
    rows = []
    total_windows_done = current_count

    with tempfile.TemporaryDirectory(prefix=f"phase5_{source}_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(tmp_path)

        rec_root = tmp_path
        if source == "cudb":
            rec_root = tmp_path / "cu-ventricular-tachyarrhythmia-database-1.0.0"

        prefix = f"{source}_"
        allowed = sorted({r[len(prefix):] for r in reference_records if r.startswith(prefix)})

        for rec_idx, rec_id in enumerate(allowed):
            t_rec_start = time.perf_counter()
            rec_path = rec_root / rec_id
            full_rec_id = f"{source}_{rec_id}"
            try:
                record = wfdb.rdrecord(str(rec_path))
                if record.p_signal is None:
                    continue
                ecg_indices = [
                    idx for idx, name in enumerate(record.sig_name or [])
                    if "ECG" in str(name).upper() or str(name).upper() == "II"
                ]
                if not ecg_indices:
                    continue
                chan_idx = 0 if source == "vfdb" else ecg_indices[0]
                signal_raw = record.p_signal[:, chan_idx].astype(np.float64)

                if np.isnan(signal_raw).mean() > 0.1:
                    continue
                if np.isnan(signal_raw).any():
                    signal_raw = pd.Series(signal_raw).interpolate(limit_direction="both").to_numpy()

                annotation = wfdb.rdann(str(rec_path), "atr")
                boundaries = list(zip(annotation.sample, annotation.aux_note))
                if source == "cudb":
                    boundaries = [(s, n) for s, n in boundaries if str(n).strip().rstrip("\x00")]

                rec_windows = 0
                w_global_idx = 0
                for idx, (start_s, note) in enumerate(boundaries):
                    lbl = classify_aux_wfdb(str(note))
                    if lbl is None:
                        continue
                    end_s = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(signal_raw)
                    seg = signal_raw[start_s:end_s]
                    if len(seg) < record.fs:
                        continue
                    if record.fs != TARGET_FS:
                        seg = resample_poly(seg, TARGET_FS, int(record.fs))

                    for w_start in range(0, len(seg) - WINDOW_SAMPLES, STEP_SAMPLES):
                        w = seg[w_start:w_start + WINDOW_SAMPLES]
                        feats = extract_all_physiological_features(w, TARGET_FS)
                        rec_windows += 1
                        total_windows_done += 1
                        rows.append({
                            **feats,
                            "record_id": full_rec_id,
                            "dataset_source": source,
                            "label": lbl,
                            "target": 0 if lbl == "VT" else 1,
                            "window_index": w_global_idx,
                            "window_duration_seconds": WINDOW_SECONDS,
                            "sampling_rate": TARGET_FS,
                        })
                        w_global_idx += 1

                t_global_elapsed = time.perf_counter() - t_start_global
                rate = total_windows_done / max(t_global_elapsed, 0.001)
                rem_windows = total_target_windows - total_windows_done
                eta_sec = rem_windows / max(rate, 0.001)

                print(
                    f"  [{source}] [{rec_idx+1}/{len(allowed)}] {full_rec_id}: "
                    f"{rec_windows} win | Total: {total_windows_done}/{total_target_windows} "
                    f"({100.0*total_windows_done/total_target_windows:.1f}%) | "
                    f"Elapsed: {t_global_elapsed/60.0:.2f}m | ETA: {eta_sec/60.0:.2f}m"
                )
                sys.stdout.flush()

            except Exception as exc:
                print(f"  Warning: failed on {source}_{rec_id}: {exc}")
                sys.stdout.flush()
                continue

    return rows, total_windows_done


def run_full_extraction(
    ref_csv: Path,
    c2015_zip: Path,
    vfdb_zip: Path,
    cudb_zip: Path,
    output_csv: Path,
) -> pd.DataFrame:
    print("=" * 78)
    print("PHASE 5B: FULL PHYSIOLOGICAL FEATURE EXTRACTION (STRATEGY B)")
    print("=" * 78)
    sys.stdout.flush()

    ref_df = pd.read_csv(ref_csv)
    total_target = len(ref_df)
    ref_records = set(ref_df["record_id"].unique())
    print(f"Target extraction: exactly {total_target} windows across {len(ref_records)} unique records.")
    sys.stdout.flush()

    t_start_global = time.perf_counter()
    count = 0

    c2015_rows, count = process_c2015_extracted(c2015_zip, ref_records, total_target, count, t_start_global)
    vfdb_rows, count = process_wfdb_extracted(vfdb_zip, "vfdb", ref_records, total_target, count, t_start_global)
    cudb_rows, count = process_wfdb_extracted(cudb_zip, "cudb", ref_records, total_target, count, t_start_global)

    all_rows = c2015_rows + vfdb_rows + cudb_rows
    physio_df = pd.DataFrame(all_rows)

    t_total = time.perf_counter() - t_start_global
    print("\n" + "=" * 78)
    print(f"EXTRACTION COMPLETE: {len(physio_df)} windows extracted in {t_total/60.0:.2f} min ({1000*t_total/len(physio_df):.2f} ms/win)")
    print("=" * 78)
    print("Record counts per source:", physio_df.groupby("dataset_source")["record_id"].nunique().to_dict())
    print("Window counts per source and label:")
    print(physio_df.groupby(["dataset_source", "label"]).size().to_string())
    sys.stdout.flush()

    # Integrity Assertions
    assert len(physio_df) == total_target, f"Window count mismatch: got {len(physio_df)}, expected {total_target}"
    feature_cols = [c for c in physio_df.columns if c not in {"record_id", "dataset_source", "label", "target", "window_index", "window_duration_seconds", "sampling_rate"}]
    assert np.isfinite(physio_df[feature_cols].to_numpy(dtype=float)).all(), "Non-finite feature values detected!"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    physio_df.to_csv(output_csv, index=False)
    print(f"\nSaved new physiological feature table to: {output_csv}")
    sys.stdout.flush()
    return physio_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Phase 5B physiological features (Strategy B).")
    parser.add_argument("--ref-csv", type=Path, default=Path("training/model3_phase3/artifacts/calibrated_vfvt_feature_table.csv"))
    parser.add_argument("--c2015-zip", type=Path, default=Path("mitbih-vtvtf.zip"))
    parser.add_argument("--vfdb-zip", type=Path, default=Path("VFDB.zip"))
    parser.add_argument("--cudb-zip", type=Path, default=Path("cu-ventricular-tachyarrhythmia-database-1.0.0.zip"))
    parser.add_argument("--output-csv", type=Path, default=Path("training/model3_phase5/artifacts/physiological_feature_table.csv"))
    args = parser.parse_args()

    run_full_extraction(args.ref_csv, args.c2015_zip, args.vfdb_zip, args.cudb_zip, args.output_csv)


if __name__ == "__main__":
    main()
