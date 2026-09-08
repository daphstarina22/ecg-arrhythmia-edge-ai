"""Phase 6C Smoke Test: Audit verification on cu04, cu05, cu06."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model3_phase5.extract_physiological_features import (
    compute_spectral_features,
    compute_autocorrelation_features,
    compute_complexity_features_strategy_b,
)

ZIP_PATH = PROJECT_ROOT / "cu-ventricular-tachyarrhythmia-database-1.0.0.zip"
PLOTS_DIR = PROJECT_ROOT / "training" / "model3_phase6" / "phase6c" / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS = 250
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)


def parse_cudb_episodes(ann: wfdb.Annotation, sig_len: int) -> list[dict]:
    episodes = []
    current_start = None
    ep_idx = 1
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        if sym == "[":
            if current_start is not None:
                episodes.append({
                    "episode_id": f"ep{ep_idx:02d}",
                    "start_sample": current_start,
                    "end_sample": s,
                    "status": "UNCLOSED_PREVIOUS",
                })
                ep_idx += 1
            current_start = s
        elif sym == "]":
            if current_start is not None:
                episodes.append({
                    "episode_id": f"ep{ep_idx:02d}",
                    "start_sample": current_start,
                    "end_sample": s,
                    "status": "NORMAL_CLOSED",
                })
                ep_idx += 1
                current_start = None
    if current_start is not None:
        episodes.append({
            "episode_id": f"ep{ep_idx:02d}",
            "start_sample": current_start,
            "end_sample": sig_len,
            "status": "END_OF_RECORD",
        })
    return episodes


def compute_diagnostics(x_window: np.ndarray, fs: int = TARGET_FS) -> dict[str, float]:
    spec = compute_spectral_features(x_window, fs)
    ac = compute_autocorrelation_features(x_window, fs)
    comp = compute_complexity_features_strategy_b(x_window)
    amp_stats = {
        "amp_mean": float(np.mean(x_window)),
        "amp_std": float(np.std(x_window)),
        "amp_min": float(np.min(x_window)),
        "amp_max": float(np.max(x_window)),
        "amp_peak_to_peak": float(np.max(x_window) - np.min(x_window)),
        "amp_rms": float(np.sqrt(np.mean(x_window ** 2))),
    }
    return {
        "dominant_frequency": spec["dominant_freq"],
        "spectral_entropy": spec["spectral_entropy"],
        "spectral_concentration": spec["spectral_concentration"],
        "spectral_flatness": spec["spectral_flatness"],
        "ac_periodicity_strength": ac["ac_periodicity_strength"],
        "ac_decay_time": ac["ac_decay_time"],
        "ac_max_peak_ratio": ac["ac_max_peak_ratio"],
        "sample_entropy": comp["sample_entropy"],
        "lz_complexity": comp["lz_complexity"],
        "hjorth_mobility": comp["hjorth_mobility"],
        "hjorth_complexity": comp["hjorth_complexity"],
        **amp_stats,
    }


def generate_episode_plot(
    rec_id: str,
    ep_id: str,
    raw_signal: np.ndarray,
    start_s: int,
    end_s: int,
    w_rep: np.ndarray,
    diagnostics: dict[str, float],
    output_path: Path,
    fs: int = TARGET_FS,
) -> None:
    t_full = np.arange(start_s, end_s) / fs
    ep_wave = raw_signal[start_s:end_s]
    t_win = np.arange(len(w_rep)) / fs

    # Welch PSD for window
    freqs, psd = signal.welch(w_rep, fs=fs, nperseg=min(256, len(w_rep)))
    mask = (freqs >= 0.5) & (freqs <= 30.0)
    f_band = freqs[mask]
    p_band = psd[mask]

    # ACF for window
    x_c = w_rep - np.mean(w_rep)
    r = signal.fftconvolve(x_c, x_c[::-1], mode="full")[len(x_c) - 1 :]
    r_norm = r / (r[0] + 1e-12)
    lags = np.arange(len(r_norm)) / fs

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    fig.suptitle(
        f"CUDB Diagnostic Audit: {rec_id} — {ep_id} (Duration: {(end_s - start_s)/fs:.1f}s)\n"
        f"DomFreq={diagnostics['dominant_frequency']:.2f}Hz | SpecEntropy={diagnostics['spectral_entropy']:.2f} | "
        f"Periodicity={diagnostics['ac_periodicity_strength']:.2f} | LZ={diagnostics['lz_complexity']:.2f} | "
        f"SampEn={diagnostics['sample_entropy']:.2f}",
        fontsize=12,
        fontweight="bold",
    )

    # 1. Full episode waveform
    axes[0, 0].plot(t_full, ep_wave, color="#1f77b4", lw=0.8)
    axes[0, 0].set_title("Full Episode Waveform")
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Amplitude (mV)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Representative 5-second window
    axes[0, 1].plot(t_win, w_rep, color="#d62728", lw=1.2)
    axes[0, 1].set_title("Representative 5.0s Window")
    axes[0, 1].set_xlabel("Time (seconds)")
    axes[0, 1].set_ylabel("Amplitude (mV)")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. PSD
    axes[1, 0].plot(f_band, p_band, color="#2ca02c", lw=1.5)
    dom_f = diagnostics["dominant_frequency"]
    axes[1, 0].axvline(dom_f, color="orange", linestyle="--", label=f"Dom Peak: {dom_f:.2f} Hz")
    axes[1, 0].set_title("Power Spectral Density (0.5 - 30 Hz)")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Power / PSD")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. ACF
    max_lag_show = min(int(1.5 * fs), len(lags))
    axes[1, 1].plot(lags[:max_lag_show], r_norm[:max_lag_show], color="#9467bd", lw=1.5)
    axes[1, 1].axhline(0, color="gray", lw=0.8, linestyle=":")
    axes[1, 1].axvline(diagnostics["ac_decay_time"], color="red", linestyle="--", label=f"Decay: {diagnostics['ac_decay_time']:.2f}s")
    axes[1, 1].set_title("Autocorrelation Function (Lag 0 to 1.5s)")
    axes[1, 1].set_xlabel("Lag (seconds)")
    axes[1, 1].set_ylabel("Normalized ACF")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_path)
    plt.close(fig)


def run_smoke_test():
    print("Running Smoke Test on cu04, cu05, cu06...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        with tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)
            root = Path(tmp) / "cu-ventricular-tachyarrhythmia-database-1.0.0"
            for rec in ["cu04", "cu05", "cu06"]:
                p = root / rec
                record = wfdb.rdrecord(str(p))
                ann = wfdb.rdann(str(p), "atr")
                raw = record.p_signal[:, 0].astype(np.float64)
                episodes = parse_cudb_episodes(ann, len(raw))
                print(f"Record {rec}: found {len(episodes)} episodes")
                for ep in episodes:
                    s, e = ep["start_sample"], ep["end_sample"]
                    dur = (e - s) / TARGET_FS
                    ep_wave = raw[s:e]
                    if len(ep_wave) < WINDOW_SAMPLES:
                        continue
                    # take middle 5s window
                    mid_idx = (len(ep_wave) - WINDOW_SAMPLES) // 2
                    w_rep = ep_wave[mid_idx : mid_idx + WINDOW_SAMPLES]
                    diag = compute_diagnostics(w_rep, TARGET_FS)
                    out_png = PLOTS_DIR / f"{rec}_{ep['episode_id']}.png"
                    generate_episode_plot(rec, ep["episode_id"], raw, s, e, w_rep, diag, out_png)
                    print(f"  --> {rec}_{ep['episode_id']} ({dur:.1f}s): DomFreq={diag['dominant_frequency']:.2f}Hz, PSD/ACF computed, plot saved: {out_png.name}")
    print("Smoke Test PASSED successfully!")


if __name__ == "__main__":
    run_smoke_test()
