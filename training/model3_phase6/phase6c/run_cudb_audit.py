"""Phase 6C: Read-Only CUDB Episode Audit & Re-Adjudication.

Performs a rigorous, read-only diagnostic audit of all quarantined CUDB records
(cu04 through cu35) from cu-ventricular-tachyarrhythmia-database-1.0.0.zip.
Calculates waveform diagnostics, generates 4-panel diagnostic visualizations,
applies strict rhythm classification governance, and exports:
  training/model3_phase6/phase6c/results/cudb_episode_audit.csv
  training/model3_phase6/phase6c/results/plots/{record_id}_{episode_id}.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
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

TARGET_FS = 250
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)

PACED_RECORDS = {"cu12", "cu15", "cu24", "cu25", "cu32"}


def parse_cudb_annotations(ann: wfdb.Annotation, sig_len: int) -> tuple[list[dict[str, Any]], str]:
    """Parse episode intervals and surrounding annotation context."""
    episodes = []
    current_start = None
    ep_idx = 1

    all_notes = [str(n).strip().rstrip("\x00") for n in ann.aux_note if str(n).strip()]
    unique_notes = sorted(set(all_notes))
    context_str = f"Notes: {unique_notes}" if unique_notes else "No aux notes (brackets only)"

    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        aux_clean = str(aux).strip().rstrip("\x00") if aux else ""
        if sym == "[":
            if current_start is not None:
                episodes.append({
                    "episode_id": f"ep{ep_idx:02d}",
                    "start_sample": current_start,
                    "end_sample": s,
                    "status": "UNCLOSED_PREVIOUS",
                    "orig_annot": f"[{aux_clean}",
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
                    "orig_annot": f"[{aux_clean}]",
                })
                ep_idx += 1
                current_start = None

    if current_start is not None:
        episodes.append({
            "episode_id": f"ep{ep_idx:02d}",
            "start_sample": current_start,
            "end_sample": sig_len,
            "status": "END_OF_RECORD",
            "orig_annot": "[unclosed",
        })

    return episodes, context_str


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


def classify_episode(
    rec_id: str,
    duration_s: float,
    diag: dict[str, float],
    context: str,
) -> tuple[str, str, str, str]:
    """Governance rule: classify candidate episode based on waveform & annotations."""
    if duration_s < 5.0:
        return (
            "INSUFFICIENT_EVIDENCE",
            "LOW",
            "EXCLUDE_TOO_SHORT",
            f"Duration {duration_s:.1f}s < 5.0s minimum window threshold.",
        )

    dom_f = diag["dominant_frequency"]
    spec_ent = diag["spectral_entropy"]
    spec_conc = diag["spectral_concentration"]
    ac_period = diag["ac_periodicity_strength"]
    ac_decay = diag["ac_decay_time"]
    samp_en = diag["sample_entropy"]
    lz = diag["lz_complexity"]
    rms = diag["amp_rms"]
    p2p = diag["amp_peak_to_peak"]

    is_paced = rec_id in PACED_RECORDS
    has_af = "(AF" in context

    # 1. Non-target / Artifact / Paced baseline
    if is_paced and ac_period < 0.15 and rms < 0.15:
        return (
            "NON_TARGET_RHYTHM",
            "MEDIUM",
            "EXCLUDE",
            f"Known paced patient ({rec_id}). Low RMS amplitude ({rms:.3f}mV) and low periodicity ({ac_period:.2f}) indicate paced artifact or baseline drift.",
        )

    # 2. Very low amplitude disorganized -> Fine VF
    if p2p < 0.30 and rms < 0.08 and spec_ent > 0.65:
        return (
            "FINE_VF",
            "MEDIUM",
            "REQUIRES_MANUAL_REVIEW",
            f"Low amplitude (p2p={p2p:.2f}mV, RMS={rms:.3f}mV) with disorganized high entropy (spec_ent={spec_ent:.2f}, SampEn={samp_en:.2f}). Subtype ambiguous without clinical review.",
        )

    # 3. Highly periodic sinusoidal oscillations at 3.5 - 6.0 Hz -> Ventricular Flutter
    if (3.2 <= dom_f <= 6.5) and (ac_period >= 0.35 or spec_conc >= 0.30) and spec_ent < 0.65:
        conf = "MEDIUM"  # Source lacks explicit (VFL note
        rec = "REQUIRES_MANUAL_REVIEW"
        return (
            "VENTRICULAR_FLUTTER",
            conf,
            rec,
            f"Sinusoidal flutter morphology (dom_f={dom_f:.2f}Hz, periodicity={ac_period:.2f}, spec_conc={spec_conc:.2f}). Strong VFL candidate, but source lacks explicit subtype note.",
        )

    # 4. Discrete periodic wide-complex morphology at 2.0 - 4.0 Hz -> Monomorphic VT
    if (1.8 <= dom_f <= 3.5) and ac_period >= 0.28 and spec_ent < 0.70 and lz < 8.5:
        conf = "MEDIUM"
        rec = "REQUIRES_MANUAL_REVIEW"
        return (
            "MONOMORPHIC_VT",
            conf,
            rec,
            f"Regular wide-complex tachycardia morphology (dom_f={dom_f:.2f}Hz, periodicity={ac_period:.2f}, LZ={lz:.2f}). Strong VT candidate, but source lacks explicit subtype note.",
        )

    # 5. Chaotic, disorganized broadband oscillations -> Coarse VF
    if spec_ent >= 0.70 and ac_period < 0.25 and lz >= 7.5 and samp_en >= 0.40:
        conf = "MEDIUM"
        rec = "REQUIRES_MANUAL_REVIEW"
        return (
            "COARSE_VF",
            conf,
            rec,
            f"Chaotic fibrillatory waveform (dom_f={dom_f:.2f}Hz, spec_ent={spec_ent:.2f}, ac_period={ac_period:.2f}, LZ={lz:.2f}). Strong VF candidate, but source lacks explicit subtype note.",
        )

    # 6. Default: Ambiguous / Polymorphic / Transitional
    return (
        "AMBIGUOUS",
        "LOW",
        "REQUIRES_MANUAL_REVIEW",
        f"Transitional or mixed morphology (dom_f={dom_f:.2f}Hz, spec_ent={spec_ent:.2f}, periodicity={ac_period:.2f}). Bracket marker only; subtype cannot be determined without expert panel review.",
    )


def generate_episode_plot(
    rec_id: str,
    ep_id: str,
    raw_signal: np.ndarray,
    start_s: int,
    end_s: int,
    w_rep: np.ndarray,
    diagnostics: dict[str, float],
    category: str,
    recommendation: str,
    output_path: Path,
    fs: int = TARGET_FS,
) -> None:
    t_full = np.arange(start_s, end_s) / fs
    ep_wave = raw_signal[start_s:end_s]
    t_win = np.arange(len(w_rep)) / fs

    freqs, psd = signal.welch(w_rep, fs=fs, nperseg=min(256, len(w_rep)))
    mask = (freqs >= 0.5) & (freqs <= 30.0)
    f_band = freqs[mask]
    p_band = psd[mask]

    x_c = w_rep - np.mean(w_rep)
    r = signal.fftconvolve(x_c, x_c[::-1], mode="full")[len(x_c) - 1 :]
    r_norm = r / (r[0] + 1e-12)
    lags = np.arange(len(r_norm)) / fs

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    dur_s = (end_s - start_s) / fs
    fig.suptitle(
        f"CUDB Audit: {rec_id} — {ep_id} (Dur: {dur_s:.1f}s) | Category: {category} | Rec: {recommendation}\n"
        f"DomFreq={diagnostics['dominant_frequency']:.2f}Hz | SpecEntropy={diagnostics['spectral_entropy']:.2f} | "
        f"Periodicity={diagnostics['ac_periodicity_strength']:.2f} | LZ={diagnostics['lz_complexity']:.2f} | "
        f"SampEn={diagnostics['sample_entropy']:.2f} | RMS={diagnostics['amp_rms']:.3f}mV",
        fontsize=11,
        fontweight="bold",
    )

    # 1. Full waveform
    axes[0, 0].plot(t_full, ep_wave, color="#1f77b4", lw=0.7)
    axes[0, 0].set_title(f"Full Episode Waveform ({dur_s:.1f}s)")
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
    axes[1, 0].axvline(dom_f, color="orange", linestyle="--", label=f"Peak: {dom_f:.2f} Hz")
    axes[1, 0].set_title("Power Spectral Density (0.5 - 30 Hz)")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("PSD")
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. ACF
    max_lag = min(int(1.5 * fs), len(lags))
    axes[1, 1].plot(lags[:max_lag], r_norm[:max_lag], color="#9467bd", lw=1.5)
    axes[1, 1].axhline(0, color="gray", lw=0.8, linestyle=":")
    axes[1, 1].axvline(diagnostics["ac_decay_time"], color="red", linestyle="--", label=f"Decay: {diagnostics['ac_decay_time']:.2f}s")
    axes[1, 1].set_title("Autocorrelation Function (Lag 0 to 1.5s)")
    axes[1, 1].set_xlabel("Lag (seconds)")
    axes[1, 1].set_ylabel("Normalized ACF")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_path)
    plt.close(fig)


def run_full_audit(
    zip_path: Path,
    output_csv: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    print("=" * 78)
    print("PHASE 6C: FULL AUDIT OF QUARANTINED CUDB RECORDS (cu04 - cu35)")
    print("=" * 78)
    t0 = time.perf_counter()

    audit_rows = []
    plots_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        with tempfile.TemporaryDirectory(prefix="phase6c_cudb_") as tmp:
            z.extractall(tmp)
            root = Path(tmp) / "cu-ventricular-tachyarrhythmia-database-1.0.0"
            rec_ids = [f"cu{i:02d}" for i in range(4, 36)]

            total_episodes = 0
            for idx, rec_id in enumerate(rec_ids):
                rec_path = root / rec_id
                if not rec_path.with_suffix(".hea").exists():
                    print(f"  [{idx+1}/{len(rec_ids)}] {rec_id}: Header not found, skipping.")
                    continue

                record = wfdb.rdrecord(str(rec_path))
                ann = wfdb.rdann(str(rec_path), "atr")
                raw_signal = record.p_signal[:, 0].astype(np.float64)
                if np.isnan(raw_signal).any():
                    raw_signal = pd.Series(raw_signal).interpolate(limit_direction="both").to_numpy()
                sig_len = len(raw_signal)

                episodes, context_str = parse_cudb_annotations(ann, sig_len)
                if not episodes:
                    print(f"  [{idx+1}/{len(rec_ids)}] {rec_id}: 0 candidate episodes found (Normal/Ectopy only).")
                    continue

                for ep in episodes:
                    total_episodes += 1
                    ep_id = ep["episode_id"]
                    start_s = ep["start_sample"]
                    end_s = ep["end_sample"]
                    dur_s = (end_s - start_s) / TARGET_FS

                    ep_wave = raw_signal[start_s:end_s]
                    if len(ep_wave) < WINDOW_SAMPLES:
                        # Too short for 5s window
                        cat, conf, rec, ev = classify_episode(rec_id, dur_s, {}, context_str)
                        audit_rows.append({
                            "record_id": f"cudb_{rec_id}",
                            "episode_id": ep_id,
                            "start_sample": start_s,
                            "end_sample": end_s,
                            "duration_seconds": round(dur_s, 2),
                            "sampling_rate": TARGET_FS,
                            "original_annotation": ep["orig_annot"],
                            "annotation_context": context_str,
                            "proposed_category": cat,
                            "confidence": conf,
                            "dominant_frequency": np.nan,
                            "spectral_entropy": np.nan,
                            "spectral_concentration": np.nan,
                            "ac_periodicity_strength": np.nan,
                            "ac_decay_time": np.nan,
                            "sample_entropy": np.nan,
                            "lz_complexity": np.nan,
                            "hjorth_mobility": np.nan,
                            "hjorth_complexity": np.nan,
                            "inclusion_recommendation": rec,
                            "evidence_summary": ev,
                        })
                        continue

                    # Representative window: central 5.0 s
                    mid_idx = (len(ep_wave) - WINDOW_SAMPLES) // 2
                    w_rep = ep_wave[mid_idx : mid_idx + WINDOW_SAMPLES]

                    diag = compute_diagnostics(w_rep, TARGET_FS)
                    cat, conf, rec, ev = classify_episode(rec_id, dur_s, diag, context_str)

                    # Plot
                    out_png = plots_dir / f"{rec_id}_{ep_id}.png"
                    generate_episode_plot(
                        rec_id,
                        ep_id,
                        raw_signal,
                        start_s,
                        end_s,
                        w_rep,
                        diag,
                        cat,
                        rec,
                        out_png,
                        TARGET_FS,
                    )

                    audit_rows.append({
                        "record_id": f"cudb_{rec_id}",
                        "episode_id": ep_id,
                        "start_sample": start_s,
                        "end_sample": end_s,
                        "duration_seconds": round(dur_s, 2),
                        "sampling_rate": TARGET_FS,
                        "original_annotation": ep["orig_annot"],
                        "annotation_context": context_str,
                        "proposed_category": cat,
                        "confidence": conf,
                        "dominant_frequency": round(diag["dominant_frequency"], 3),
                        "spectral_entropy": round(diag["spectral_entropy"], 3),
                        "spectral_concentration": round(diag["spectral_concentration"], 3),
                        "ac_periodicity_strength": round(diag["ac_periodicity_strength"], 3),
                        "ac_decay_time": round(diag["ac_decay_time"], 3),
                        "sample_entropy": round(diag["sample_entropy"], 3),
                        "lz_complexity": round(diag["lz_complexity"], 3),
                        "hjorth_mobility": round(diag["hjorth_mobility"], 3),
                        "hjorth_complexity": round(diag["hjorth_complexity"], 3),
                        "inclusion_recommendation": rec,
                        "evidence_summary": ev,
                    })

                print(
                    f"  [{idx+1}/{len(rec_ids)}] {rec_id}: Processed {len(episodes)} episodes "
                    f"(Total so far: {total_episodes} eps) | Time: {time.perf_counter()-t0:.1f}s"
                )
                sys.stdout.flush()

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(output_csv, index=False)
    print(f"\n[Audit Completed] Exported {len(df_audit)} episode audit rows to: {output_csv}")
    print(f"  Plots saved to: {plots_dir}")
    return df_audit


def main():
    parser = argparse.ArgumentParser(description="Phase 6C CUDB Audit Runner")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=PROJECT_ROOT / "cu-ventricular-tachyarrhythmia-database-1.0.0.zip",
        help="Path to CUDB zip file",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "training" / "model3_phase6" / "phase6c" / "results" / "cudb_episode_audit.csv",
        help="Output path for audit CSV",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PROJECT_ROOT / "training" / "model3_phase6" / "phase6c" / "results" / "plots",
        help="Directory to save diagnostic plots",
    )
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_audit = run_full_audit(args.zip_path, args.output_csv, args.plots_dir)


if __name__ == "__main__":
    main()
