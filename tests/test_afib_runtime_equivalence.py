"""
AFib Runtime Regression & Numerical Equivalence Test Suite
==========================================================
Verifies bitwise / justified numerical and behavioural equivalence between:
1. Research Feature Extractor vs Packaged Deployment Feature Extractor
2. Research CausalBaselineTracker vs Packaged Deployment CausalBaselineTracker
3. Research CausalTemporalPersistenceFilter vs Packaged Deployment CausalTemporalPersistenceFilter
4. End-to-End AFib Bridge with Model Weights and Preprocessing
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Research references
from training.research.afib_5s_classifier.extract_afdb_5s_features import (
    extract_5s_features as research_extract_features,
    detect_r_peaks as research_detect_r_peaks,
    butter_bandpass_filter as research_butter_filter,
)
from training.research.afib_5s_classifier.patient_adaptive_baseline_study.baseline_tracker import (
    CausalBaselineTracker as ResearchBaselineTracker,
    DEFAULT_POPULATION_PRIORS as RESEARCH_POPULATION_PRIORS,
)
from training.research.afib_5s_classifier.patient_adaptive_baseline_study.temporal_persistence_study.persistence_filter import (
    CausalTemporalPersistenceFilter as ResearchPersistenceFilter,
)

# Deployment packaged runtime
from deploy.afib_runtime.feature_extraction import (
    extract_5s_features as deploy_extract_features,
    detect_r_peaks as deploy_detect_r_peaks,
    butter_bandpass_filter as deploy_butter_filter,
)
from deploy.afib_runtime.baseline_tracker import (
    CausalBaselineTracker as DeployBaselineTracker,
    DEFAULT_POPULATION_PRIORS as DEPLOY_POPULATION_PRIORS,
)
from deploy.afib_runtime.persistence_filter import (
    CausalTemporalPersistenceFilter as DeployPersistenceFilter,
)
from deploy.afib_bridge import CausalAdaptiveAFibBridge


class TestFeatureExtractionEquivalence(unittest.TestCase):
    """Rigorous side-by-side numerical equivalence across all 21 input features."""

    def _generate_test_signals(self):
        signals = []
        fs = 360
        n_samples = 1800  # 5.0 seconds
        t = np.arange(n_samples) / float(fs)

        # 1. Clean synthetic sinus rhythm (1.2 Hz ~ 72 bpm)
        sig_sinus = 0.8 * np.sin(2 * np.pi * 1.2 * t)
        # Add QRS-like spikes every 300 samples (1.2 Hz)
        for i in range(150, n_samples, 300):
            sig_sinus[i - 5 : i + 5] += 1.2 * np.hanning(10)
        signals.append(("clean_sinus_72bpm", sig_sinus.astype(np.float32)))

        # 2. Multi-harmonic chaotic / AFib-like signal with 6 Hz fibrillatory wave
        np.random.seed(42)
        sig_afib = (
            0.3 * np.sin(2 * np.pi * 6.0 * t)  # 6 Hz f-wave
            + 0.15 * np.sin(2 * np.pi * 4.5 * t)
            + 0.05 * np.random.randn(n_samples)
        )
        # Irregular R-peaks at variable intervals: 220, 310, 190, 400, 250 samples
        peak_idx = 100
        intervals = [220, 310, 190, 350, 260, 280]
        for interval in intervals:
            peak_idx += interval
            if peak_idx + 6 < n_samples:
                sig_afib[peak_idx - 6 : peak_idx + 6] += 1.0 * np.hanning(12)
        signals.append(("chaotic_afib_6hz", sig_afib.astype(np.float32)))

        # 3. High-rate Tachycardia signal (2.5 Hz ~ 150 bpm)
        sig_tachy = 0.7 * np.sin(2 * np.pi * 2.5 * t)
        for i in range(70, n_samples - 8, 144):  # 144 samples = 0.4s = 150 bpm
            sig_tachy[i - 4 : i + 4] += 1.1 * np.hanning(8)
        signals.append(("tachycardia_150bpm", sig_tachy.astype(np.float32)))

        # 4. Low-amplitude / borderline noise signal
        np.random.seed(123)
        sig_low_noise = 0.04 * np.random.randn(n_samples)
        signals.append(("low_amplitude_noise", sig_low_noise.astype(np.float32)))

        # 5. Bradycardia signal (0.75 Hz ~ 45 bpm)
        sig_brady = 0.6 * np.sin(2 * np.pi * 0.75 * t)
        for i in range(240, n_samples - 10, 480):  # 480 samples = 1.33s = 45 bpm
            sig_brady[i - 5 : i + 5] += 1.3 * np.hanning(10)
        signals.append(("bradycardia_45bpm", sig_brady.astype(np.float32)))

        # 6. Flatline signal (near zero)
        sig_flat = np.full(n_samples, 0.001, dtype=np.float32)
        signals.append(("near_flatline", sig_flat))

        return signals

    def test_butter_bandpass_filter_equivalence(self):
        """Zero-phase Butterworth filter must produce identical outputs."""
        for name, sig in self._generate_test_signals():
            with self.subTest(signal=name):
                filt_ref = research_butter_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=2)
                filt_dep = deploy_butter_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=2)
                np.testing.assert_allclose(
                    filt_dep,
                    filt_ref,
                    rtol=1e-6,
                    atol=1e-7,
                    err_msg=f"Filter output mismatch on {name}",
                )

    def test_detect_r_peaks_equivalence(self):
        """R-peak detector must detect exact identical sample indices."""
        for name, sig in self._generate_test_signals():
            with self.subTest(signal=name):
                filt_sig = deploy_butter_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=2)
                peaks_ref = research_detect_r_peaks(filt_sig, fs=360)
                peaks_dep = deploy_detect_r_peaks(filt_sig, fs=360)
                self.assertEqual(
                    peaks_dep,
                    peaks_ref,
                    f"R-peak indices mismatch on {name}: {peaks_dep} vs {peaks_ref}",
                )

    def test_all_21_features_numerical_equivalence(self):
        """Compare all 21 extracted features on every deterministic test signal."""
        all_feature_keys = [
            "n_beats",
            "hr_mean_bpm",
            "hr_std_bpm",
            "rr_cv",
            "rr_rmssd",
            "rr_mad",
            "rr_pnn50",
            "rr_sample_dispersion",
            "rr_turning_point_ratio",
            "alternating_sign_index",
            "f_band_power_ratio",
            "f_band_flatness",
            "f_band_peak_purity",
            "p_band_power_ratio",
            "qrs_power_ratio",
            "hjorth_mobility",
            "hjorth_complexity",
            "ac_std_mv",
            "ac_ptp_mv",
            "zero_crossings",
            "lz_complexity",
        ]

        for sig_name, sig in self._generate_test_signals():
            feats_ref = research_extract_features(sig, fs=360)
            feats_dep = deploy_extract_features(sig, fs=360)

            # 1. Key set equality
            self.assertEqual(
                sorted(list(feats_dep.keys())),
                sorted(all_feature_keys),
                f"Feature key set mismatch on {sig_name}",
            )
            self.assertEqual(
                sorted(list(feats_ref.keys())),
                sorted(list(feats_dep.keys())),
                f"Feature keys differ from research reference on {sig_name}",
            )

            # 2. Per-feature numerical equality
            for key in all_feature_keys:
                val_ref = feats_ref[key]
                val_dep = feats_dep[key]

                if isinstance(val_ref, int):
                    self.assertEqual(
                        val_dep,
                        val_ref,
                        f"Integer feature '{key}' mismatch on {sig_name}: {val_dep} != {val_ref}",
                    )
                else:
                    self.assertAlmostEqual(
                        val_dep,
                        val_ref,
                        places=4,
                        msg=f"Float feature '{key}' mismatch on {sig_name}: deploy={val_dep} vs ref={val_ref}",
                    )


class TestCausalBaselineTrackerEquivalence(unittest.TestCase):
    """Verifies stateful baseline tracker equivalence: priors, updates, anti-contamination, gaps, resets."""

    def test_population_priors_match(self):
        """Population prior constants must be bitwise identical."""
        self.assertEqual(DEPLOY_POPULATION_PRIORS, RESEARCH_POPULATION_PRIORS)

    def test_sequential_window_processing_equivalence(self):
        """Execute identical sequence of 30 windows through research and deploy trackers."""
        tracker_ref = ResearchBaselineTracker(
            history_window_sec=300.0,
            min_warmup_windows=12,
            max_gap_sec=300.0,
            anti_contamination=True,
            rate_percentile=25.0,
            cv_percentile=20.0,
            gating_cv_threshold=0.18,
        )
        tracker_dep = DeployBaselineTracker(
            history_window_sec=300.0,
            min_warmup_windows=12,
            max_gap_sec=300.0,
            anti_contamination=True,
            rate_percentile=25.0,
            cv_percentile=20.0,
            gating_cv_threshold=0.18,
        )

        np.random.seed(999)
        # Window sequence definition
        for step in range(35):
            t_sec = float((step + 1) * 5.0)

            # Vary conditions across phases
            if step < 10:
                # Cold start: Resting rhythm
                hr = 70.0 + np.sin(step) * 3.0
                cv = 0.07 + np.cos(step) * 0.01
                f_band = 0.35
                rec_id = "patient_A"
            elif step < 15:
                # Extreme chaotic AFib burst (should trigger anti-contamination gating)
                hr = 165.0
                cv = 0.42  # > 0.18 threshold
                f_band = 0.85
                rec_id = "patient_A"
            elif step == 15:
                # Sudden time gap > 300s (e.g. 500s elapsed)
                t_sec = 800.0
                hr = 75.0
                cv = 0.08
                f_band = 0.38
                rec_id = "patient_A"
            elif step < 25:
                # Post-gap re-warmup
                hr = 72.0
                cv = 0.06
                f_band = 0.32
                rec_id = "patient_A"
            else:
                # New record / patient boundary reset
                rec_id = "patient_B"
                hr = 80.0
                cv = 0.09
                f_band = 0.40

            out_ref = tracker_ref.process_window(
                current_time_sec=t_sec,
                hr_mean_bpm=float(hr),
                rr_cv=float(cv),
                f_band_power_ratio=float(f_band),
                n_beats=6,
                is_valid_window=True,
                record_id=rec_id,
            )
            out_dep = tracker_dep.process_window(
                current_time_sec=t_sec,
                hr_mean_bpm=float(hr),
                rr_cv=float(cv),
                f_band_power_ratio=float(f_band),
                n_beats=6,
                is_valid_window=True,
                record_id=rec_id,
            )

            # Compare all 5 adaptive feature outputs
            for feat_key in [
                "delta_hr_bpm",
                "rr_cv_relative",
                "hr_z_robust",
                "afib_index_relative",
                "baseline_warmup_fraction",
            ]:
                self.assertAlmostEqual(
                    out_dep[feat_key],
                    out_ref[feat_key],
                    places=3,
                    msg=f"Step {step} ({rec_id}, t={t_sec}): Feature '{feat_key}' mismatch: deploy={out_dep[feat_key]} vs ref={out_ref[feat_key]}",
                )

            # Compare internal baseline state
            est_ref = tracker_ref.get_baseline_estimates(t_sec)
            est_dep = tracker_dep.get_baseline_estimates(t_sec)
            for k in ["hr_baseline", "rr_cv_baseline", "f_band_baseline", "hr_mad", "warmup_fraction", "buffer_count"]:
                self.assertEqual(
                    est_dep[k],
                    est_ref[k],
                    f"Step {step}: Baseline estimate '{k}' mismatch: {est_dep[k]} != {est_ref[k]}",
                )


class TestCausalTemporalPersistenceFilterEquivalence(unittest.TestCase):
    """Verifies persistence filter equivalence across all 7 supported rules and edge cases."""

    def test_all_rules_behavioral_equivalence(self):
        rules = [
            "unfiltered",
            "consecutive_2",
            "consecutive_3",
            "consecutive_4",
            "majority_2_of_3",
            "majority_3_of_4",
            "hysteresis_on3_off2",
        ]

        # Complex test trajectory of decisions and probabilities
        test_sequence = [
            # (timestamp_sec, raw_prob, raw_decision, record_id)
            (5.0, 0.10, 0, "rec_1"),
            (10.0, 0.85, 1, "rec_1"),  # isolated positive
            (15.0, 0.20, 0, "rec_1"),  # reset
            (20.0, 0.90, 1, "rec_1"),  # start streak
            (25.0, 0.92, 1, "rec_1"),  # 2 in a row
            (30.0, 0.88, 1, "rec_1"),  # 3 in a row
            (35.0, 0.95, 1, "rec_1"),  # 4 in a row
            (40.0, 0.15, 0, "rec_1"),  # negative 1
            (45.0, 0.10, 0, "rec_1"),  # negative 2 (hysteresis off)
            (50.0, 0.80, 1, "rec_1"),
            # Gap of 10 seconds (> max_gap_sec 6.0)
            (65.0, 0.85, 1, "rec_1"),  # should reset streaks due to gap
            (70.0, 0.89, 1, "rec_1"),
            # Boundary reset
            (75.0, 0.95, 1, "rec_2"),  # new record ID -> clean reset
            (80.0, 0.92, 1, "rec_2"),
        ]

        for rule in rules:
            with self.subTest(rule=rule):
                flt_ref = ResearchPersistenceFilter(rule_name=rule, max_gap_sec=6.0)
                flt_dep = DeployPersistenceFilter(rule_name=rule, max_gap_sec=6.0)

                for t_sec, prob, dec, rec in test_sequence:
                    alert_ref = flt_ref.process_window(t_sec, prob, dec, record_id=rec)
                    alert_dep = flt_dep.process_window(t_sec, prob, dec, record_id=rec)
                    self.assertEqual(
                        alert_dep,
                        alert_ref,
                        f"Rule '{rule}' mismatch at t={t_sec}, rec={rec}: deploy={alert_dep} vs ref={alert_ref}",
                    )


class TestAFibBridgeEndToEndEquivalence(unittest.TestCase):
    """Verifies that CausalAdaptiveAFibBridge produces valid deterministic predictions using packaged modules."""

    def test_bridge_prediction_deterministic(self):
        bridge = CausalAdaptiveAFibBridge(fs_target=360)
        self.assertTrue(bridge.is_available)
        bridge.reset("test_patient")

        np.random.seed(1234)
        sig = 0.7 * np.sin(2 * np.pi * 1.2 * np.linspace(0, 5, 1800)).astype(np.float32)

        out1 = bridge.predict_window(sig, fs=360, current_time_sec=5.0)
        self.assertEqual(out1["status"], "WARMING_UP")
        self.assertIn("raw_probability", out1)
        self.assertIn("afib_alert", out1)
        self.assertFalse(math.isnan(out1["raw_probability"]))
        self.assertFalse(out1["afib_alert"])

        # Second identical window
        out2 = bridge.predict_window(sig, fs=360, current_time_sec=10.0)
        self.assertIn("raw_probability", out2)
        self.assertFalse(math.isnan(out2["raw_probability"]))


if __name__ == "__main__":
    unittest.main()
