"""
Synthetic Unit Tests for Causal Baseline Tracker
=================================================
Verifies:
1. Strict Anti-Leakage / Causality (changing future window has ZERO effect on past features).
2. Pure Prior-Only Computation (step k baseline depends strictly on t_i < t_k).
3. Boundary Isolation (switching records completely resets state with zero cross-patient memory).
4. Cold Start & Warm-Up Behavior (finite values, smooth transition from priors, no NaNs).
5. Missing / Corrupted Window Handling (NaNs, extreme values, long signal dropout).
6. Anti-Contamination Resistance (chaotic AFib bursts do not hijack resting baseline).
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from baseline_tracker import CausalBaselineTracker, DEFAULT_POPULATION_PRIORS


class TestCausalBaselineTracker(unittest.TestCase):

    def setUp(self):
        self.priors = {
            "hr_mean_bpm": 80.0,
            "rr_cv": 0.05,
            "f_band_power_ratio": 0.30
        }

    def test_future_leakage_invariance(self):
        """
        CRITICAL CAUSALITY TEST:
        Changing ANY future window (t > t_k) must have ZERO effect on features at step k.
        """
        # Baseline sequence: 50 normal windows
        np.random.seed(42)
        n_windows = 50
        times = [i * 2.5 for i in range(n_windows)]
        hrs = [75.0 + float(np.sin(i / 5.0) * 5.0) for i in range(n_windows)]
        cvs = [0.04 + float(np.random.uniform(0, 0.02)) for i in range(n_windows)]
        fbands = [0.25 for _ in range(n_windows)]

        # Run Run 1: original sequence
        tracker1 = CausalBaselineTracker(population_priors=self.priors, min_warmup_windows=10)
        feats_run1 = []
        for i in range(n_windows):
            f = tracker1.process_window(
                current_time_sec=times[i],
                hr_mean_bpm=hrs[i],
                rr_cv=cvs[i],
                f_band_power_ratio=fbands[i],
                record_id="REC_TEST"
            )
            feats_run1.append(f)

        # Alter future windows starting from index 25 (extreme values: HR=220, CV=0.35)
        hrs_altered = list(hrs)
        cvs_altered = list(cvs)
        for i in range(25, n_windows):
            hrs_altered[i] = 220.0
            cvs_altered[i] = 0.40

        # Run Run 2: altered future
        tracker2 = CausalBaselineTracker(population_priors=self.priors, min_warmup_windows=10)
        feats_run2 = []
        for i in range(n_windows):
            f = tracker2.process_window(
                current_time_sec=times[i],
                hr_mean_bpm=hrs_altered[i],
                rr_cv=cvs_altered[i],
                f_band_power_ratio=fbands[i],
                record_id="REC_TEST"
            )
            feats_run2.append(f)

        # Verification: Steps 0 to 24 must be EXACTLY identical between Run 1 and Run 2
        for k in range(25):
            for feat_name in ["delta_hr_bpm", "rr_cv_relative", "hr_z_robust", "afib_index_relative", "baseline_warmup_fraction"]:
                val1 = feats_run1[k][feat_name]
                val2 = feats_run2[k][feat_name]
                self.assertEqual(
                    val1, val2,
                    f"LEAKAGE VIOLATION at step {k} on feature {feat_name}: {val1} != {val2}"
                )

    def test_strict_prior_window_dependency(self):
        """
        Verifies that window k's baseline is computed strictly using windows t_i < t_k.
        At window 0, the baseline must equal the prior, regardless of what window 0's HR is.
        """
        tracker = CausalBaselineTracker(population_priors=self.priors, min_warmup_windows=10)
        # Even if window 0 has an extreme HR of 150 bpm, the baseline at window 0 must be the prior (80 bpm)
        f0 = tracker.process_window(
            current_time_sec=0.0,
            hr_mean_bpm=150.0,
            rr_cv=0.15,
            f_band_power_ratio=0.40,
            record_id="REC_TEST"
        )
        # delta_hr should be 150 - 80 = 70 bpm
        self.assertAlmostEqual(f0["delta_hr_bpm"], 70.0, delta=0.5)
        self.assertEqual(f0["baseline_warmup_fraction"], 0.0)

    def test_boundary_reset_cross_patient_isolation(self):
        """
        Verifies that switching record_id completely purges the buffer.
        Record B must start in an identical state to a freshly instantiated tracker.
        """
        tracker = CausalBaselineTracker(population_priors=self.priors, min_warmup_windows=10)

        # Feed Patient A 100 tachycardic windows (HR = 140)
        for i in range(100):
            tracker.process_window(
                current_time_sec=i * 2.5,
                hr_mean_bpm=140.0,
                rr_cv=0.10,
                f_band_power_ratio=0.30,
                record_id="PATIENT_A"
            )
        self.assertEqual(tracker.total_valid_updates, 100)

        # Now switch to Patient B (HR = 70)
        f_b0 = tracker.process_window(
            current_time_sec=0.0,
            hr_mean_bpm=70.0,
            rr_cv=0.04,
            f_band_power_ratio=0.25,
            record_id="PATIENT_B"
        )

        # Baseline at window 0 of Patient B must be prior (80 bpm), NOT Patient A's 140 bpm
        self.assertEqual(tracker.total_processed_windows, 1)
        self.assertEqual(tracker.total_valid_updates, 1)
        self.assertEqual(f_b0["baseline_warmup_fraction"], 0.0)
        # delta_hr should be 70 - 80 = -10 bpm, NOT 70 - 140 = -70 bpm
        self.assertAlmostEqual(f_b0["delta_hr_bpm"], -10.0, delta=0.5)

    def test_warmup_monotonicity_and_finite_outputs(self):
        """
        Verifies that warm-up fraction increases monotonically and all feature outputs are finite (no NaN/Inf).
        """
        tracker = CausalBaselineTracker(population_priors=self.priors, min_warmup_windows=12)
        prev_warmup = -1.0
        for i in range(20):
            f = tracker.process_window(
                current_time_sec=i * 2.5,
                hr_mean_bpm=75.0,
                rr_cv=0.04,
                f_band_power_ratio=0.25,
                record_id="REC_1"
            )
            # Monotonicity check
            self.assertGreaterEqual(f["baseline_warmup_fraction"], prev_warmup)
            prev_warmup = f["baseline_warmup_fraction"]

            # Finite check
            for k, v in f.items():
                self.assertFalse(np.isnan(v), f"NaN found in {k} at step {i}")
                self.assertFalse(np.isinf(v), f"Inf found in {k} at step {i}")

        self.assertEqual(prev_warmup, 1.0)

    def test_missing_invalid_window_handling(self):
        """
        Verifies that invalid windows (NaNs, 0 beats, flatline) do not crash the tracker
        and do not corrupt the baseline buffer.
        """
        tracker = CausalBaselineTracker(population_priors=self.priors, min_warmup_windows=10)

        # 5 normal windows
        for i in range(5):
            tracker.process_window(
                current_time_sec=i * 2.5,
                hr_mean_bpm=72.0,
                rr_cv=0.03,
                f_band_power_ratio=0.20,
                record_id="REC_1"
            )
        self.assertEqual(tracker.total_valid_updates, 5)

        # Invalid window with NaN HR
        f_nan = tracker.process_window(
            current_time_sec=12.5,
            hr_mean_bpm=float("nan"),
            rr_cv=0.03,
            f_band_power_ratio=0.20,
            record_id="REC_1"
        )
        # Should not update buffer
        self.assertEqual(tracker.total_valid_updates, 5)
        self.assertEqual(tracker.total_processed_windows, 6)

        # Invalid window with 0 beats
        f_0 = tracker.process_window(
            current_time_sec=15.0,
            hr_mean_bpm=0.0,
            rr_cv=0.0,
            f_band_power_ratio=0.0,
            n_beats=0,
            record_id="REC_1"
        )
        self.assertEqual(tracker.total_valid_updates, 5)

        # Huge time gap (600 seconds = 10 minutes)
        f_after_gap = tracker.process_window(
            current_time_sec=615.0,
            hr_mean_bpm=75.0,
            rr_cv=0.04,
            f_band_power_ratio=0.25,
            record_id="REC_1"
        )
        # Stale buffer should have been cleared because gap > max_gap_sec (300s)
        self.assertEqual(f_after_gap["baseline_warmup_fraction"], 0.0)

    def test_anti_contamination_resistance(self):
        """
        Verifies that when anti-contamination is active, high-CV AFib bursts (CV > gating threshold)
        are quarantined from corrupting the resting baseline.
        """
        tracker = CausalBaselineTracker(
            population_priors=self.priors,
            min_warmup_windows=10,
            anti_contamination=True,
            gating_cv_threshold=0.18
        )

        # 20 normal sinus windows (HR = 70, CV = 0.04)
        for i in range(20):
            tracker.process_window(
                current_time_sec=i * 2.5,
                hr_mean_bpm=70.0,
                rr_cv=0.04,
                f_band_power_ratio=0.20,
                record_id="REC_AFIB_TEST"
            )
        self.assertEqual(tracker.total_valid_updates, 20)
        base_before = tracker.get_baseline_estimates(current_time_sec=50.0)
        self.assertAlmostEqual(base_before["rr_cv_baseline"], 0.04, delta=0.015)

        # Now simulate 30 consecutive AFib windows (HR = 120, CV = 0.28)
        for i in range(20, 50):
            tracker.process_window(
                current_time_sec=i * 2.5,
                hr_mean_bpm=120.0,
                rr_cv=0.28,
                f_band_power_ratio=0.60,
                record_id="REC_AFIB_TEST"
            )

        # Due to gating (CV > 0.18), the AFib windows should have been quarantined from updating the buffer
        self.assertEqual(tracker.total_valid_updates, 20)
        base_after = tracker.get_baseline_estimates(current_time_sec=125.0)
        # Resting baseline must remain locked around 0.04, NOT inflated to 0.28!
        self.assertLess(base_after["rr_cv_baseline"], 0.10)
        self.assertAlmostEqual(base_after["hr_baseline"], 70.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
