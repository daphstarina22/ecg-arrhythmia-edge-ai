"""Computational Benchmark & Profiling for Phase 5 Feature Extraction.

Evaluates raw waveform feature extraction on a representative subset of records:
- Challenge 2015: 2 VT records, 2 VF records
- VFDB: 2 VT records, 2 VF records

Profiles individual feature groups:
1. Spectral Organization
2. Autocorrelation / Regularity
3. Hjorth Parameters
4. Permutation Entropy
5. Lempel-Ziv Complexity (standard vs optimized)
6. Sample Entropy (current vs optimized)
7. Archive I/O & signal windowing

Reports exact times per window, identifies bottlenecks, and calculates projected
full-dataset runtime across all 14,618 windows.
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
WINDOW_SECONDS = 5
STEP_SECONDS = 2.5
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SECONDS)
STEP_SAMPLES = int(TARGET_FS * STEP_SECONDS)

# Representative subset (2 VT, 2 VF per source)
BENCHMARK_RECORDS_C2015 = {
    "v131l": "VT",
    "v132s": "VT",
    "f450s": "VF",
    "f543l": "VF",
}

BENCHMARK_RECORDS_VFDB = {
    "420": "VT",
    "421": "VT",
    "418": "VF",
    "419": "VF",
}


# ==============================================================================
# FEATURE CALCULATORS (ISOLATED FOR PROFILING)
# ==============================================================================


def compute_spectral_features_profiled(x: np.ndarray, fs: int = TARGET_FS) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
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

    dt = time.perf_counter() - t0
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
    }, dt


def compute_autocorrelation_features_profiled(x: np.ndarray, fs: int = TARGET_FS) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
    x_centered = x - np.mean(x)
    var = float(np.var(x_centered))
    if var < 1e-12:
        return {
            "ac_max_peak_ratio": 0.0,
            "ac_first_secondary_peak": 0.0,
            "ac_decay_time": 0.0,
            "ac_periodicity_strength": 0.0,
            "ac_zero_crossing_lag": 0.0,
        }, time.perf_counter() - t0

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

    dt = time.perf_counter() - t0
    return {
        "ac_max_peak_ratio": ac_max_peak,
        "ac_first_secondary_peak": first_sec_peak,
        "ac_decay_time": decay_time,
        "ac_periodicity_strength": periodicity_strength,
        "ac_zero_crossing_lag": zero_lag,
    }, dt


def compute_hjorth_profiled(x: np.ndarray) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
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
    dt = time.perf_counter() - t0
    return {
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
    }, dt


def compute_permutation_entropy_profiled(x: np.ndarray, m: int = 3, tau: int = 2) -> tuple[float, float]:
    t0 = time.perf_counter()
    n = len(x)
    n_patterns = n - (m - 1) * tau
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
    dt = time.perf_counter() - t0
    return perm_entropy, dt


def compute_lz_original_profiled(x: np.ndarray) -> tuple[float, float]:
    """Original while-loop Kaspar-Schuster LZ76 implementation on N=1800."""
    t0 = time.perf_counter()
    binary = (x > np.median(x)).astype(np.uint8)
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
    dt = time.perf_counter() - t0
    return float(c / b), dt


def compute_lz_optimized_profiled(x: np.ndarray) -> tuple[float, float]:
    """Optimized LZ76 complexity on 90 Hz downsampled binary sequence (N=450)."""
    t0 = time.perf_counter()
    # Downsample by factor 4 (90 Hz captures all cardiac rhythm dynamics)
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
    dt = time.perf_counter() - t0
    return float(c / b), dt


def compute_sampen_profiled(x: np.ndarray, downsample_factor: int = 8) -> tuple[float, float]:
    """Vectorized Sample Entropy on downsampled signal (N = 1800 // downsample_factor)."""
    t0 = time.perf_counter()
    x_down = x[::downsample_factor].astype(np.float32)
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
    dt = time.perf_counter() - t0
    return samp_entropy, dt


# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================


def run_benchmark(
    c2015_zip_path: Path,
    vfdb_zip_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("PHASE 5 COMPUTATIONAL BENCHMARK & PROFILING AUDIT")
    print("=" * 78)
    sys.stdout.flush()

    total_windows_collected = 0
    record_timings = []
    feature_timings: dict[str, list[float]] = {
        "spectral_welch": [],
        "autocorrelation": [],
        "hjorth": [],
        "permutation_entropy": [],
        "lz_original_N1800": [],
        "lz_optimized_N450": [],
        "sampen_N225": [],
        "sampen_N150": [],
    }

    start_bench = time.perf_counter()

    # 1. Benchmark Challenge 2015 Subset
    print("\n[1/2] Benchmarking Challenge 2015 records (2 VT, 2 VF)...")
    sys.stdout.flush()
    t_c2015_start = time.perf_counter()

    with zipfile.ZipFile(c2015_zip_path) as outer:
        t_zip_read = time.perf_counter()
        training_info = next(item for item in outer.infolist() if item.filename.endswith("/training.zip"))
        training_zip_bytes = io.BytesIO(outer.read(training_info))
        print(f"  Nested training.zip read to RAM: {time.perf_counter() - t_zip_read:.2f} s")
        sys.stdout.flush()

        with zipfile.ZipFile(training_zip_bytes) as archive:
            for rec_id, label in BENCHMARK_RECORDS_C2015.items():
                t_rec_start = time.perf_counter()
                hea_text = archive.read(f"training/{rec_id}.hea").decode(errors="replace")
                lines = hea_text.splitlines()
                first = lines[0].split()
                fs_orig = float(first[2])

                # Find II channel
                chan_idx = None
                gain = 200.0
                baseline = 0.0
                adc_zero = 0.0
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

                n_rec_windows = 0
                for start in range(0, len(physical) - WINDOW_SAMPLES, STEP_SAMPLES):
                    w = physical[start:start + WINDOW_SAMPLES]
                    n_rec_windows += 1

                    # Profile each feature component
                    _, dt_spec = compute_spectral_features_profiled(w)
                    _, dt_ac = compute_autocorrelation_features_profiled(w)
                    _, dt_hjorth = compute_hjorth_profiled(w)
                    _, dt_pe = compute_permutation_entropy_profiled(w)
                    _, dt_lz_orig = compute_lz_original_profiled(w)
                    _, dt_lz_opt = compute_lz_optimized_profiled(w)
                    _, dt_se225 = compute_sampen_profiled(w, downsample_factor=8)  # N=225
                    _, dt_se150 = compute_sampen_profiled(w, downsample_factor=12)  # N=150

                    feature_timings["spectral_welch"].append(dt_spec)
                    feature_timings["autocorrelation"].append(dt_ac)
                    feature_timings["hjorth"].append(dt_hjorth)
                    feature_timings["permutation_entropy"].append(dt_pe)
                    feature_timings["lz_original_N1800"].append(dt_lz_orig)
                    feature_timings["lz_optimized_N450"].append(dt_lz_opt)
                    feature_timings["sampen_N225"].append(dt_se225)
                    feature_timings["sampen_N150"].append(dt_se150)

                rec_elapsed = time.perf_counter() - t_rec_start
                total_windows_collected += n_rec_windows
                record_timings.append({
                    "dataset": "c2015",
                    "record_id": rec_id,
                    "label": label,
                    "windows": n_rec_windows,
                    "elapsed_sec": rec_elapsed,
                    "ms_per_window": 1000.0 * rec_elapsed / max(n_rec_windows, 1),
                })
                print(f"  c2015_{rec_id} ({label}): {n_rec_windows} windows in {rec_elapsed:.2f} s ({1000*rec_elapsed/n_rec_windows:.2f} ms/win)")
                sys.stdout.flush()

    # 2. Benchmark VFDB Subset
    print("\n[2/2] Benchmarking VFDB records (2 VT, 2 VF)...")
    sys.stdout.flush()
    with tempfile.TemporaryDirectory(prefix="bench_vfdb_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(vfdb_zip_path) as z:
            z.extractall(tmp_path)

        for rec_id, label in BENCHMARK_RECORDS_VFDB.items():
            t_rec_start = time.perf_counter()
            rec_path = tmp_path / rec_id
            record = wfdb.rdrecord(str(rec_path))
            annotation = wfdb.rdann(str(rec_path), "atr")
            signal_raw = record.p_signal[:, 0].astype(np.float64)
            if np.isnan(signal_raw).any():
                signal_raw = pd.Series(signal_raw).interpolate(limit_direction="both").to_numpy()

            boundaries = list(zip(annotation.sample, annotation.aux_note))
            n_rec_windows = 0
            for idx, (start_s, note) in enumerate(boundaries):
                aux_label = "VF" if "(VF" in str(note) else ("VT" if "(VT" in str(note) else None)
                if aux_label is None:
                    continue
                end_s = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(signal_raw)
                seg = signal_raw[start_s:end_s]
                if len(seg) < record.fs:
                    continue
                if record.fs != TARGET_FS:
                    seg = resample_poly(seg, TARGET_FS, int(record.fs))

                for w_start in range(0, len(seg) - WINDOW_SAMPLES, STEP_SAMPLES):
                    w = seg[w_start:w_start + WINDOW_SAMPLES]
                    n_rec_windows += 1

                    _, dt_spec = compute_spectral_features_profiled(w)
                    _, dt_ac = compute_autocorrelation_features_profiled(w)
                    _, dt_hjorth = compute_hjorth_profiled(w)
                    _, dt_pe = compute_permutation_entropy_profiled(w)
                    _, dt_lz_orig = compute_lz_original_profiled(w)
                    _, dt_lz_opt = compute_lz_optimized_profiled(w)
                    _, dt_se225 = compute_sampen_profiled(w, downsample_factor=8)
                    _, dt_se150 = compute_sampen_profiled(w, downsample_factor=12)

                    feature_timings["spectral_welch"].append(dt_spec)
                    feature_timings["autocorrelation"].append(dt_ac)
                    feature_timings["hjorth"].append(dt_hjorth)
                    feature_timings["permutation_entropy"].append(dt_pe)
                    feature_timings["lz_original_N1800"].append(dt_lz_orig)
                    feature_timings["lz_optimized_N450"].append(dt_lz_opt)
                    feature_timings["sampen_N225"].append(dt_se225)
                    feature_timings["sampen_N150"].append(dt_se150)

            rec_elapsed = time.perf_counter() - t_rec_start
            total_windows_collected += n_rec_windows
            record_timings.append({
                "dataset": "vfdb",
                "record_id": rec_id,
                "label": label,
                "windows": n_rec_windows,
                "elapsed_sec": rec_elapsed,
                "ms_per_window": 1000.0 * rec_elapsed / max(n_rec_windows, 1),
            })
            print(f"  vfdb_{rec_id} ({label}): {n_rec_windows} windows in {rec_elapsed:.2f} s ({1000*rec_elapsed/max(n_rec_windows,1):.2f} ms/win)")
            sys.stdout.flush()

    total_bench_time = time.perf_counter() - start_bench

    # Compile feature breakdown table
    breakdown_rows = []
    for feat_name, times in feature_timings.items():
        arr = np.array(times) * 1000.0  # to ms
        breakdown_rows.append({
            "feature_component": feat_name,
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "median_ms": float(np.median(arr)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr)),
        })

    breakdown_df = pd.DataFrame(breakdown_rows).sort_values("mean_ms", ascending=False).reset_index(drop=True)
    breakdown_df.to_csv(output_dir / "feature_timing_breakdown.csv", index=False)

    rec_timing_df = pd.DataFrame(record_timings)
    rec_timing_df.to_csv(output_dir / "record_timing_benchmark.csv", index=False)

    # Print Summary Table
    print("\n" + "=" * 78)
    print("FEATURE COMPUTATION TIME PER WINDOW (BENCHMARKED ON 8 RECORDS)")
    print("=" * 78)
    print(breakdown_df.to_string(index=False))
    print(f"\nTotal windows benchmarked: {total_windows_collected} across 8 records in {total_bench_time:.2f} s")
    sys.stdout.flush()

    # Calculate Projected Full Runtimes (14,618 windows)
    # Strategy A: Unoptimized (original LZ N=1800 + SampEn N=225)
    t_unopt_ms = (
        breakdown_df.loc[breakdown_df["feature_component"] == "spectral_welch", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "autocorrelation", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "hjorth", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "permutation_entropy", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "lz_original_N1800", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "sampen_N225", "mean_ms"].iloc[0]
    )

    # Strategy B: Optimized (LZ on 90 Hz N=450 + SampEn N=225)
    t_opt_ms = (
        breakdown_df.loc[breakdown_df["feature_component"] == "spectral_welch", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "autocorrelation", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "hjorth", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "permutation_entropy", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "lz_optimized_N450", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "sampen_N225", "mean_ms"].iloc[0]
    )

    # Strategy C: Fast Physiological (LZ on 90 Hz + SampEn N=150)
    t_fast_ms = (
        breakdown_df.loc[breakdown_df["feature_component"] == "spectral_welch", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "autocorrelation", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "hjorth", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "permutation_entropy", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "lz_optimized_N450", "mean_ms"].iloc[0]
        + breakdown_df.loc[breakdown_df["feature_component"] == "sampen_N150", "mean_ms"].iloc[0]
    )

    full_windows = 14618
    io_overhead_s = 65.0  # one-time zip decompression of training.zip

    proj_unopt_s = io_overhead_s + (t_unopt_ms * full_windows / 1000.0)
    proj_opt_s = io_overhead_s + (t_opt_ms * full_windows / 1000.0)
    proj_fast_s = io_overhead_s + (t_fast_ms * full_windows / 1000.0)

    print("\n" + "=" * 78)
    print("PROJECTED FULL-DATASET RUNTIME (14,618 WINDOWS)")
    print("=" * 78)
    print(f"Strategy A (Unoptimized LZ N=1800 + SampEn N=225): {t_unopt_ms:.2f} ms/win -> {proj_unopt_s/60.0:.2f} min ({proj_unopt_s:.1f} s)")
    print(f"Strategy B (Optimized LZ N=450  + SampEn N=225): {t_opt_ms:.2f} ms/win -> {proj_opt_s/60.0:.2f} min ({proj_opt_s:.1f} s)")
    print(f"Strategy C (Fast LZ N=450       + SampEn N=150): {t_fast_ms:.2f} ms/win -> {proj_fast_s/60.0:.2f} min ({proj_fast_s:.1f} s)")
    sys.stdout.flush()

    # Identify bottleneck
    bottleneck_feature = breakdown_df.iloc[0]["feature_component"]
    bottleneck_pct = (breakdown_df.iloc[0]["mean_ms"] / t_unopt_ms) * 100.0

    summary_report = f"""# Computational Benchmark Report: Phase 5 Feature Extraction

## Benchmark Setup
- **Evaluated Records**: 8 representative records (4 Challenge 2015, 4 VFDB; 4 VT and 4 VF).
- **Total Windows Benchmarked**: {total_windows_collected}.

## Feature-by-Feature Timing Profile
| Feature Component | Mean (ms) | Median (ms) | P90 (ms) | Max (ms) |
|:---|---:|---:|---:|---:|
"""
    for _, r in breakdown_df.iterrows():
        summary_report += f"| `{r['feature_component']}` | {r['mean_ms']:.2f} | {r['median_ms']:.2f} | {r['p90_ms']:.2f} | {r['max_ms']:.2f} |\n"

    summary_report += f"""
## Primary Computational Bottleneck
- **Identified Bottleneck**: `{bottleneck_feature}` ({breakdown_df.iloc[0]['mean_ms']:.2f} ms/window, accounting for {bottleneck_pct:.1f}% of per-window calculation time in unoptimized configuration).
- **Secondary Factor**: One-time archive decompression of Challenge 2015 `training.zip` takes approximately {io_overhead_s:.1f} seconds.

## Full Dataset Runtime Projections (14,618 Windows)
- **Strategy A (Unoptimized LZ N=1800 + SampEn N=225)**: `{t_unopt_ms:.2f} ms/win` -> **{proj_unopt_s/60.0:.2f} minutes** ({proj_unopt_s:.1f} s).
- **Strategy B (Optimized LZ N=450 + SampEn N=225)**: `{t_opt_ms:.2f} ms/win` -> **{proj_opt_s/60.0:.2f} minutes** ({proj_opt_s:.1f} s).
- **Strategy C (Fast LZ N=450 + SampEn N=150)**: `{t_fast_ms:.2f} ms/win` -> **{proj_fast_s/60.0:.2f} minutes** ({proj_fast_s:.1f} s).

## Recommendation for Full Phase 5 Extraction
Implement **Strategy B**:
1. Downsample LZ complexity to 90 Hz (N=450), which preserves all cardiac frequency components (0.5–30 Hz) while reducing LZ execution time significantly.
2. Vectorize Sample Entropy with N=225 (45 Hz, $m=2, r=0.2\sigma$) using float32 arrays.
3. Pre-extract `training.zip` once into a temporary folder to eliminate archive seek delays.
4. Provide per-record streaming progress reporting with `flush=True`, elapsed time, and ETA.
"""
    (output_dir / "benchmark_report.md").write_text(summary_report, encoding="utf-8")
    print(f"\nBenchmark report written to: {output_dir / 'benchmark_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 computational benchmark.")
    parser.add_argument("--c2015-zip", type=Path, default=Path("mitbih-vtvtf.zip"))
    parser.add_argument("--vfdb-zip", type=Path, default=Path("VFDB.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/model3_phase5/results/benchmark"))
    args = parser.parse_args()

    run_benchmark(args.c2015_zip, args.vfdb_zip, args.output_dir)


if __name__ == "__main__":
    main()
