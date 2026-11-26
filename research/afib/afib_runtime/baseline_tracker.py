"""
Causal Patient-Adaptive Baseline Tracker Runtime Module
======================================================
Hardware Target: Embedded Wearable / Edge ECG Monitor (Raspberry Pi ARM64)

Design Principles:
1. STRICT CAUSALITY: Window k at time t depends exclusively on windows strictly
   preceding time t (t_i < t). Zero access to current or future windows.
2. ZERO CROSS-RECORD LEAKAGE: Complete memory purge and state reset at patient / record boundaries.
3. ANTI-CONTAMINATION: Utilizes robust lower percentiles (e.g. 20th percentile) and update gating
   to prevent prolonged AFib or extreme rate bursts from hijacking the resting baseline.
4. GRACEFUL INITIALIZATION: Population priors anchored from training set prevent NaNs or erratic
   spikes during initial cold start / warm-up.
5. ROBUSTNESS TO MISSING/INVALID DATA: Missing or rejected windows do not corrupt history;
   excessive time gaps (> 300s) trigger automatic re-warmup.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Default population priors fitted exclusively on the 15 AFDB training patients
DEFAULT_POPULATION_PRIORS = {
    "hr_mean_bpm": 83.5591,
    "rr_cv": 0.1016,
    "f_band_power_ratio": 0.3952,
}


class CausalBaselineTracker:
    """
    Causal, streaming patient-adaptive baseline tracker for sequential 5-second ECG windows.
    Maintains a sliding history buffer over a configurable time horizon T_history (seconds).
    """

    def __init__(
        self,
        history_window_sec: float = 300.0,
        min_warmup_windows: int = 12,
        max_gap_sec: float = 300.0,
        population_priors: Optional[Dict[str, float]] = None,
        anti_contamination: bool = True,
        rate_percentile: float = 25.0,
        cv_percentile: float = 20.0,
        gating_cv_threshold: float = 0.18,
    ):
        """
        Args:
            history_window_sec: Length of history memory buffer in seconds (default 300s = 5 min).
            min_warmup_windows: Number of valid windows required for full warm-up confidence.
            max_gap_sec: Max allowed elapsed time between valid windows before buffer reset.
            population_priors: Prior means for HR, RR CV, f-band power (derived strictly from training set).
            anti_contamination: If True, uses lower percentiles and gates updates during chaotic bursts.
            rate_percentile: Percentile for resting heart rate (default 25.0 for resting baseline).
            cv_percentile: Percentile for resting RR variability (default 20.0 to reject paroxysmal AFib).
            gating_cv_threshold: If current window RR CV exceeds this, quarantine window from updating baseline.
        """
        self.history_window_sec = float(history_window_sec)
        self.min_warmup_windows = int(min_warmup_windows)
        self.max_gap_sec = float(max_gap_sec)
        self.priors = dict(population_priors or DEFAULT_POPULATION_PRIORS)
        self.anti_contamination = bool(anti_contamination)
        self.rate_percentile = float(rate_percentile)
        self.cv_percentile = float(cv_percentile)
        self.gating_cv_threshold = float(gating_cv_threshold)

        # State storage (circular FIFO buffers of tuples: (timestamp_sec, hr, rr_cv, f_band))
        self.history_buffer = deque()
        self.current_record_id: Optional[str] = None
        self.last_valid_timestamp: Optional[float] = None
        self.total_processed_windows = 0
        self.total_valid_updates = 0

    def reset(self, new_record_id: Optional[str] = None):
        """Strictly resets all internal history to prevent inter-patient or inter-record leakage."""
        self.history_buffer.clear()
        self.current_record_id = new_record_id
        self.last_valid_timestamp = None
        self.total_processed_windows = 0
        self.total_valid_updates = 0

    def _purge_stale_history(self, current_time_sec: float):
        """Removes windows older than (current_time_sec - history_window_sec)."""
        cutoff_time = current_time_sec - self.history_window_sec
        while self.history_buffer and self.history_buffer[0][0] < cutoff_time:
            self.history_buffer.popleft()

    def get_baseline_estimates(self, current_time_sec: float) -> Dict[str, float]:
        """
        Computes causal baseline estimates strictly using windows preceding current_time_sec.
        Returns:
            Dict containing hr_baseline, rr_cv_baseline, f_band_baseline, hr_mad, warmup_fraction.
        """
        self._purge_stale_history(current_time_sec)
        n_buffered = len(self.history_buffer)
        warmup_fraction = min(1.0, float(n_buffered) / float(self.min_warmup_windows)) if self.min_warmup_windows > 0 else 1.0

        if n_buffered == 0:
            # Fall back completely to population prior
            return {
                "hr_baseline": float(self.priors["hr_mean_bpm"]),
                "rr_cv_baseline": float(self.priors["rr_cv"]),
                "f_band_baseline": float(self.priors["f_band_power_ratio"]),
                "hr_mad": 2.0,
                "warmup_fraction": 0.0,
                "buffer_count": 0,
            }

        hrs = np.array([item[1] for item in self.history_buffer], dtype=np.float64)
        cvs = np.array([item[2] for item in self.history_buffer], dtype=np.float64)
        f_bands = np.array([item[3] for item in self.history_buffer], dtype=np.float64)

        if self.anti_contamination:
            sample_hr = float(np.percentile(hrs, self.rate_percentile))
            sample_cv = float(np.percentile(cvs, self.cv_percentile))
            sample_f = float(np.percentile(f_bands, 25.0))
        else:
            sample_hr = float(np.median(hrs))
            sample_cv = float(np.median(cvs))
            sample_f = float(np.median(f_bands))

        med_hr = float(np.median(hrs))
        sample_hr_mad = float(np.median(np.abs(hrs - med_hr))) if len(hrs) > 1 else 2.0
        sample_hr_mad = max(1.0, sample_hr_mad)

        # Smooth blend with population priors during initial warm-up phase
        hr_baseline = warmup_fraction * sample_hr + (1.0 - warmup_fraction) * self.priors["hr_mean_bpm"]
        rr_cv_baseline = warmup_fraction * sample_cv + (1.0 - warmup_fraction) * self.priors["rr_cv"]
        f_band_baseline = warmup_fraction * sample_f + (1.0 - warmup_fraction) * self.priors["f_band_power_ratio"]

        # Ensure physiological clipping bounds
        hr_baseline = float(np.clip(hr_baseline, 40.0, 160.0))
        rr_cv_baseline = float(np.clip(rr_cv_baseline, 0.015, 0.25))
        f_band_baseline = float(np.clip(f_band_baseline, 0.05, 0.90))

        return {
            "hr_baseline": round(hr_baseline, 2),
            "rr_cv_baseline": round(rr_cv_baseline, 4),
            "f_band_baseline": round(f_band_baseline, 4),
            "hr_mad": round(sample_hr_mad, 2),
            "warmup_fraction": round(warmup_fraction, 3),
            "buffer_count": n_buffered,
        }

    def process_window(
        self,
        current_time_sec: float,
        hr_mean_bpm: float,
        rr_cv: float,
        f_band_power_ratio: float,
        n_beats: int = 5,
        is_valid_window: bool = True,
        record_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Causally evaluates window at time current_time_sec, then updates history buffer.

        Execution Order:
        1. Boundary Check: If record_id changes, reset all history.
        2. Time Gap Check: If elapsed time > max_gap_sec, reset history buffer.
        3. Causal Feature Computation: Compute baseline features using strictly t_i < current_time_sec.
        4. State Update: If current window is valid and passes anti-contamination check, append to buffer.
        """
        # 1. Boundary Check
        if record_id is not None and record_id != self.current_record_id:
            self.reset(new_record_id=record_id)

        # 2. Time Gap Check
        if self.last_valid_timestamp is not None:
            time_gap = current_time_sec - self.last_valid_timestamp
            if time_gap > self.max_gap_sec:
                # Gap is too long (lead lost/disconnected); purge history to prevent stale reference
                self.history_buffer.clear()

        # 3. Compute Causal Baseline from PREVIOUS windows
        base = self.get_baseline_estimates(current_time_sec)

        hr = float(hr_mean_bpm)
        cv = float(rr_cv)
        f_band = float(f_band_power_ratio)

        # Candidate 1: Delta Heart Rate (bpm)
        delta_hr = hr - base["hr_baseline"]

        # Candidate 2: Relative RR Interval Variability Ratio
        eps_cv = 0.01
        rr_cv_rel = cv / (base["rr_cv_baseline"] + eps_cv)

        # Candidate 3: Robust Heart Rate Z-Score
        eps_mad = 2.0
        hr_z_robust = (hr - base["hr_baseline"]) / (1.4826 * base["hr_mad"] + eps_mad)

        # Candidate 4: Baseline-Adjusted f-Band / Rate Product
        eps_f = 0.05
        f_band_rel = f_band / (base["f_band_baseline"] + eps_f)
        afib_index_rel = rr_cv_rel * f_band_rel

        features = {
            "delta_hr_bpm": round(float(delta_hr), 2),
            "rr_cv_relative": round(float(rr_cv_rel), 4),
            "hr_z_robust": round(float(hr_z_robust), 4),
            "afib_index_relative": round(float(afib_index_rel), 4),
            "baseline_warmup_fraction": float(base["warmup_fraction"]),
        }

        # 4. State Update (Causal registration for FUTURE windows)
        self.total_processed_windows += 1

        is_plausible = (
            is_valid_window
            and (n_beats >= 3)
            and (30.0 <= hr <= 250.0)
            and (cv >= 0.0)
            and not np.isnan(hr)
            and not np.isnan(cv)
            and not np.isnan(f_band)
        )

        if is_plausible:
            should_update = True
            if self.anti_contamination:
                # Gating rule: If window is strongly chaotic (e.g. CV > 0.18), do not let it drag up resting baseline
                if cv > self.gating_cv_threshold:
                    should_update = False

            if should_update:
                self.history_buffer.append((current_time_sec, hr, cv, f_band))
                self.total_valid_updates += 1
                self.last_valid_timestamp = current_time_sec

        return features
