"""
Comprehensive Integration Test Suite for Unified ECG Arrhythmia Edge Pipeline
=============================================================================
Tests:
  1. Complete 9-Label Output Routing:
     - NORMAL_SINUS_RHYTHM
     - ATRIAL_FIBRILLATION
     - BRADYCARDIA
     - TACHYCARDIA
     - VENTRICULAR_ARRHYTHMIA
     - VENTRICULAR_TACHYCARDIA
     - VENTRICULAR_FIBRILLATION
     - POOR_SIGNAL_OR_LEAD_OFF
     - UNCERTAIN_OR_CONFLICTING_PREDICTIONS
  2. Safety Priority Invariant (AFib never downgrades shockable or ventricular rhythms)
  3. Deterministic Conflict Resolution (Tachy vs AFib, Brady vs AFib)
  4. Signal Quality Diagnostic Guard (Flatline, non-finite, rail saturation)
  5. AFib Bridge & Preprocessing Compatibility (360 Hz native vs 250 Hz polyphase resampling)
  6. Causal Baseline Tracker & Temporal Persistence State Machine
  7. Failure Isolation (Non-crashing graceful degradation)
  8. Latency Benchmark (< 50 ms / window on edge CPU)
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.unified_decision_layer import (
    UnifiedRhythmLabel,
    UnifiedDecision,
    arbitrate_decision,
    TRIAGE_CRITICAL_EMERGENCY,
    TRIAGE_URGENT_MONITORING,
    TRIAGE_ROUTINE_NORMAL,
    TRIAGE_SIGNAL_INVALID,
    TRIAGE_UNCERTAIN,
)
from deploy.afib_bridge import CausalAdaptiveAFibBridge
from deploy.edge_runtime import HierarchicalEdgeClassifier, evaluate_signal_quality


class TestUnifiedArbitrationLayer(unittest.TestCase):
    """Verifies deterministic arbitration across all 9 output routes and safety constraints."""

    def setUp(self):
        self.valid_quality = {"status": "SIGNAL_OK", "is_valid_for_ml": True, "detail": "OK"}
        self.invalid_quality_flat = {"status": "BAD_SIGNAL_FLAT", "is_valid_for_ml": False, "detail": "Flatline"}
        self.invalid_quality_nan = {"status": "BAD_SIGNAL_NONFINITE", "is_valid_for_ml": False, "detail": "NaN detected"}
        self.invalid_quality_sat = {"status": "BAD_SIGNAL_SATURATED", "is_valid_for_ml": False, "detail": "Rail saturated"}

        self.t1_nonshock = {"label": "NON_SHOCKABLE", "prob_shockable": 0.10, "is_shockable": False}
        self.t1_shockable = {"label": "SHOCKABLE", "prob_shockable": 0.88, "is_shockable": True}

        self.t2_nsr = {"predicted_class": "NSR", "confidence": 0.85, "class_probabilities": {"NSR": 0.85}}
        self.t2_tachy = {"predicted_class": "TACHY", "confidence": 0.90, "class_probabilities": {"TACHY": 0.90}}
        self.t2_brady = {"predicted_class": "BRADY_ASY", "confidence": 0.88, "class_probabilities": {"BRADY_ASY": 0.88}}
        self.t2_ventricular = {"predicted_class": "VENTRICULAR", "confidence": 0.92, "class_probabilities": {"VENTRICULAR": 0.92}}

        self.t3_vt = {"subtype": "VT", "prob_vt": 0.85, "prob_vf_vfl": 0.15}
        self.t3_vf = {"subtype": "VF", "prob_vt": 0.10, "prob_vf_vfl": 0.90}

        self.afib_negative = {"afib_alert": False, "raw_probability": 0.05, "status": "OK"}
        self.afib_positive = {"afib_alert": True, "raw_probability": 0.92, "status": "OK"}

    def test_route_1_poor_signal_flatline(self):
        """Route 1: Signal Quality Failure -> POOR_SIGNAL_OR_LEAD_OFF."""
        dec = arbitrate_decision(
            quality_report=self.invalid_quality_flat,
            tier1_result=self.t1_shockable,
            tier2_result=self.t2_ventricular,
            tier3_result=self.t3_vf,
            afib_result=self.afib_positive,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.POOR_SIGNAL_OR_LEAD_OFF.value)
        self.assertEqual(dec.triage_level, TRIAGE_SIGNAL_INVALID)
        self.assertFalse(dec.defibrillator_advised)

    def test_route_1_poor_signal_nonfinite(self):
        """Route 1b: Non-finite Signal Quality Failure -> POOR_SIGNAL_OR_LEAD_OFF."""
        dec = arbitrate_decision(
            quality_report=self.invalid_quality_nan,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_nsr,
            tier3_result=None,
            afib_result=self.afib_negative,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.POOR_SIGNAL_OR_LEAD_OFF.value)

    def test_route_2_ventricular_fibrillation(self):
        """Route 2: Shockable/Ventricular with VF Subtype -> VENTRICULAR_FIBRILLATION."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_shockable,
            tier2_result=self.t2_ventricular,
            tier3_result=self.t3_vf,
            afib_result=self.afib_negative,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.VENTRICULAR_FIBRILLATION.value)
        self.assertEqual(dec.triage_level, TRIAGE_CRITICAL_EMERGENCY)
        self.assertTrue(dec.defibrillator_advised)
        self.assertTrue(dec.is_shockable)

    def test_route_3_ventricular_tachycardia_shockable(self):
        """Route 3: Shockable VT -> VENTRICULAR_TACHYCARDIA with shock advice."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_shockable,
            tier2_result=self.t2_ventricular,
            tier3_result=self.t3_vt,
            afib_result=self.afib_negative,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.VENTRICULAR_TACHYCARDIA.value)
        self.assertEqual(dec.triage_level, TRIAGE_CRITICAL_EMERGENCY)
        self.assertTrue(dec.defibrillator_advised)

    def test_route_3b_ventricular_tachycardia_nonshockable(self):
        """Route 3b: Non-Shockable VT -> VENTRICULAR_TACHYCARDIA with urgent monitoring."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_ventricular,
            tier3_result=self.t3_vt,
            afib_result=self.afib_negative,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.VENTRICULAR_TACHYCARDIA.value)
        self.assertEqual(dec.triage_level, TRIAGE_URGENT_MONITORING)
        self.assertFalse(dec.defibrillator_advised)

    def test_route_4_ventricular_arrhythmia_indeterminate(self):
        """Route 4: Ventricular rhythm with bypassed/indeterminate Tier 3 -> VENTRICULAR_ARRHYTHMIA."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_ventricular,
            tier3_result=None,
            afib_result=self.afib_negative,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.VENTRICULAR_ARRHYTHMIA.value)
        self.assertEqual(dec.triage_level, TRIAGE_URGENT_MONITORING)

    def test_safety_invariant_afib_never_downgrades_shockable(self):
        """CRITICAL INVARIANT: AFib positive must NEVER override or downgrade VF/VT."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_shockable,
            tier2_result=self.t2_ventricular,
            tier3_result=self.t3_vf,
            afib_result=self.afib_positive,  # Strong AFib signal present
        )
        # Must strictly preserve VF
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.VENTRICULAR_FIBRILLATION.value)
        self.assertTrue(dec.defibrillator_advised)
        # Check that AFib confounder interception was logged in safety flags
        flag_found = any("VENTRICULAR_CONFOUNDER_INTERCEPTED" in f for f in dec.safety_flags)
        self.assertTrue(flag_found)

    def test_route_5_normal_sinus_rhythm(self):
        """Route 5: Tier 2 NSR and AFib negative -> NORMAL_SINUS_RHYTHM."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_nsr,
            tier3_result=None,
            afib_result=self.afib_negative,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.NORMAL_SINUS_RHYTHM.value)
        self.assertEqual(dec.triage_level, TRIAGE_ROUTINE_NORMAL)
        self.assertFalse(dec.defibrillator_advised)

    def test_route_6_atrial_fibrillation(self):
        """Route 6: Tier 2 NSR and AFib confirmed alert -> ATRIAL_FIBRILLATION."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_nsr,
            tier3_result=None,
            afib_result=self.afib_positive,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.ATRIAL_FIBRILLATION.value)
        self.assertEqual(dec.triage_level, TRIAGE_URGENT_MONITORING)
        self.assertFalse(dec.defibrillator_advised)

    def test_route_7_tachycardia(self):
        """Route 7: Tier 2 TACHY and AFib negative -> TACHYCARDIA."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_tachy,
            tier3_result=None,
            afib_result=self.afib_negative,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.TACHYCARDIA.value)
        self.assertEqual(dec.triage_level, TRIAGE_URGENT_MONITORING)
        self.assertFalse(dec.conflict_detected)

    def test_route_8_bradycardia(self):
        """Route 8: Tier 2 BRADY_ASY and AFib negative -> BRADYCARDIA."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_brady,
            tier3_result=None,
            afib_result=self.afib_negative,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.BRADYCARDIA.value)
        self.assertEqual(dec.triage_level, TRIAGE_URGENT_MONITORING)
        self.assertFalse(dec.conflict_detected)

    def test_route_9_conflict_tachy_vs_afib(self):
        """Route 9a: Conflict between Tachycardia and AFib -> UNCERTAIN_OR_CONFLICTING_PREDICTIONS."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_tachy,
            tier3_result=None,
            afib_result=self.afib_positive,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.UNCERTAIN_OR_CONFLICTING_PREDICTIONS.value)
        self.assertEqual(dec.triage_level, TRIAGE_UNCERTAIN)
        self.assertTrue(dec.conflict_detected)
        self.assertIn("TACHYCARDIA", dec.conflict_detail)
        self.assertIn("ATRIAL_FIBRILLATION", dec.conflict_detail)

    def test_route_9_conflict_brady_vs_afib(self):
        """Route 9b: Conflict between Bradycardia and AFib -> UNCERTAIN_OR_CONFLICTING_PREDICTIONS."""
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_brady,
            tier3_result=None,
            afib_result=self.afib_positive,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.UNCERTAIN_OR_CONFLICTING_PREDICTIONS.value)
        self.assertEqual(dec.triage_level, TRIAGE_UNCERTAIN)
        self.assertTrue(dec.conflict_detected)
        self.assertIn("BRADYCARDIA", dec.conflict_detail)

    def test_afib_unavailable_never_silently_confirms_nsr(self):
        """Requirement 4: When AFib inference is unavailable, NSR must NOT be confirmed; routes to UNCERTAIN."""
        afib_unavailable = {"afib_alert": False, "raw_probability": 0.0, "status": "UNAVAILABLE", "is_available": False}
        dec = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_nonshock,
            tier2_result=self.t2_nsr,
            tier3_result=None,
            afib_result=afib_unavailable,
        )
        self.assertEqual(dec.final_label, UnifiedRhythmLabel.UNCERTAIN_OR_CONFLICTING_PREDICTIONS.value)
        self.assertEqual(dec.triage_level, TRIAGE_UNCERTAIN)
        flag_found = any("AFIB_UNAVAILABLE" in f for f in dec.safety_flags)
        self.assertTrue(flag_found)

    def test_afib_unavailable_preserves_shockable_and_ventricular_protection(self):
        """Requirement 4: Shockable/ventricular protection must remain intact even if AFib is unavailable."""
        afib_unavailable = {"afib_alert": False, "raw_probability": 0.0, "status": "UNAVAILABLE", "is_available": False}
        dec_vf = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_shockable,
            tier2_result=self.t2_ventricular,
            tier3_result=self.t3_vf,
            afib_result=afib_unavailable,
        )
        self.assertEqual(dec_vf.final_label, UnifiedRhythmLabel.VENTRICULAR_FIBRILLATION.value)
        self.assertTrue(dec_vf.is_shockable)
        self.assertTrue(dec_vf.defibrillator_advised)

        dec_vt = arbitrate_decision(
            quality_report=self.valid_quality,
            tier1_result=self.t1_shockable,
            tier2_result=self.t2_ventricular,
            tier3_result=self.t3_vt,
            afib_result=afib_unavailable,
        )
        self.assertEqual(dec_vt.final_label, UnifiedRhythmLabel.VENTRICULAR_TACHYCARDIA.value)
        self.assertTrue(dec_vt.is_shockable)
        self.assertTrue(dec_vt.defibrillator_advised)

    def test_clean_deployment_imports_without_research_dir(self):
        """Verify deployment runtime imports succeed independently without training/research on sys.path."""
        # Ensure no research directory is on sys.path
        research_paths = [p for p in sys.path if "training" in p or "research" in p]
        for p in research_paths:
            sys.path.remove(p)

        from deploy.afib_runtime.feature_extraction import extract_5s_features
        from deploy.afib_runtime.baseline_tracker import CausalBaselineTracker
        from deploy.afib_runtime.persistence_filter import CausalTemporalPersistenceFilter

        self.assertIsNotNone(extract_5s_features)
        self.assertIsNotNone(CausalBaselineTracker)
        self.assertIsNotNone(CausalTemporalPersistenceFilter)

    def test_all_nine_labels_represented(self):
        """Verify that every single label in UnifiedRhythmLabel is tested and supported."""
        all_labels = {label.value for label in UnifiedRhythmLabel}
        expected = {
            "NORMAL_SINUS_RHYTHM",
            "ATRIAL_FIBRILLATION",
            "BRADYCARDIA",
            "TACHYCARDIA",
            "VENTRICULAR_ARRHYTHMIA",
            "VENTRICULAR_TACHYCARDIA",
            "VENTRICULAR_FIBRILLATION",
            "POOR_SIGNAL_OR_LEAD_OFF",
            "UNCERTAIN_OR_CONFLICTING_PREDICTIONS",
        }
        self.assertEqual(all_labels, expected)


class TestAFibBridgeAndPreprocessing(unittest.TestCase):
    """Verifies AFib bridge, 250->360Hz resampling, feature extraction, and causal filters."""

    @classmethod
    def setUpClass(cls):
        cls.bridge = CausalAdaptiveAFibBridge(fs_target=360)

    def setUp(self):
        self.bridge.reset(new_record_id="test_record")

    def test_feature_cols_schema(self):
        """Verify 26 features (21 absolute + 5 adaptive)."""
        self.assertEqual(len(self.bridge.feature_cols), 26)
        self.assertIn("n_beats", self.bridge.feature_cols)
        self.assertIn("f_band_power_ratio", self.bridge.feature_cols)
        self.assertIn("delta_hr_bpm", self.bridge.feature_cols)
        self.assertIn("rr_cv_relative", self.bridge.feature_cols)
        self.assertIn("afib_index_relative", self.bridge.feature_cols)

    def test_polyphase_resampling_250_to_360(self):
        """Verify 250 Hz window (1,250 samples) is correctly resampled to 360 Hz (1,800 samples)."""
        sig_250 = np.random.randn(1250).astype(np.float32)
        res = self.bridge.resample_if_needed(sig_250, fs=250)
        self.assertEqual(len(res), 1800)
        out = self.bridge.predict_window(sig_250, fs=250, current_time_sec=5.0)
        self.assertIn("raw_probability", out)
        self.assertIn("afib_alert", out)
        self.assertFalse(np.isnan(out["raw_probability"]))

    def test_native_360_hz_window(self):
        """Verify native 360 Hz window (1,800 samples) processes directly without resampling."""
        sig_360 = np.random.randn(1800).astype(np.float32)
        res = self.bridge.resample_if_needed(sig_360, fs=360)
        self.assertEqual(len(res), 1800)
        out = self.bridge.predict_window(sig_360, fs=360, current_time_sec=5.0)
        self.assertIn("raw_probability", out)
        self.assertIn("afib_alert", out)

    def test_causal_temporal_persistence_streaks(self):
        """Verify that 2-consecutive persistence filter requires 2 consecutive positives to trigger alert."""
        # Simulated sequence of probabilities directly through persistence filter
        filt = self.bridge.persistence_filter
        filt.reset(new_record_id="streak_test")

        # Window 1: Negative
        a1 = filt.process_window(timestamp_sec=5.0, raw_prob=0.10, raw_decision=0)
        self.assertEqual(a1, 0)

        # Window 2: First Positive (isolated) -> Alert must remain 0
        a2 = filt.process_window(timestamp_sec=10.0, raw_prob=0.85, raw_decision=1)
        self.assertEqual(a2, 0)

        # Window 3: Second Positive (streak=2) -> Alert must fire 1
        a3 = filt.process_window(timestamp_sec=15.0, raw_prob=0.90, raw_decision=1)
        self.assertEqual(a3, 1)

        # Window 4: Negative -> Alert resets to 0
        a4 = filt.process_window(timestamp_sec=20.0, raw_prob=0.20, raw_decision=0)
        self.assertEqual(a4, 0)

    def test_causal_tracker_anti_contamination(self):
        """Verify baseline tracker rejects extreme CV bursts from hijacking baseline."""
        tracker = self.bridge.tracker
        tracker.reset("anti_contam_test")

        # Process a series of normal resting windows (HR=70, CV=0.08)
        for i in range(15):
            tracker.process_window(
                current_time_sec=float(i * 5.0),
                hr_mean_bpm=70.0,
                rr_cv=0.08,
                f_band_power_ratio=0.35,
                n_beats=6,
                is_valid_window=True,
            )

        base_before = tracker.get_baseline_estimates(current_time_sec=75.0)
        self.assertAlmostEqual(base_before["hr_baseline"], 70.0, delta=5.0)

        # Process a massive AFib burst with extreme CV=0.45 and HR=180
        tracker.process_window(
            current_time_sec=80.0,
            hr_mean_bpm=180.0,
            rr_cv=0.45,  # Exceeds gating threshold (0.18)
            f_band_power_ratio=0.80,
            n_beats=15,
            is_valid_window=True,
        )

        # Baseline should NOT be hijacked by the burst
        base_after = tracker.get_baseline_estimates(current_time_sec=85.0)
        self.assertLess(base_after["hr_baseline"], 85.0)

    def test_failure_isolation(self):
        """Verify that bad/corrupted data returns graceful fallback without crashing."""
        corrupted = np.full(1800, np.nan, dtype=np.float32)
        out = self.bridge.predict_window(corrupted, fs=360, is_valid_signal=False)
        self.assertEqual(out["status"], "SIGNAL_INVALID")
        self.assertFalse(out["afib_alert"])
        self.assertEqual(out["raw_probability"], 0.0)


class TestEdgeClassifierEndToEnd(unittest.TestCase):
    """Verifies complete HierarchicalEdgeClassifier execution on full 1,800-sample windows."""

    @classmethod
    def setUpClass(cls):
        cls.clf = HierarchicalEdgeClassifier()

    def test_full_pipeline_normal_sinus_rhythm(self):
        t = np.linspace(0, 5, 1800)
        # Synthetic clean sinus rhythm (1.2 Hz ~ 72 bpm)
        sig = 0.8 * np.sin(2 * np.pi * 1.2 * t) + 0.02 * np.random.randn(1800)
        dec = self.clf.predict_window(sig.astype(np.float32), fs=360)
        self.assertIn(dec.primary_rhythm, [UnifiedRhythmLabel.NORMAL_SINUS_RHYTHM.value, UnifiedRhythmLabel.TACHYCARDIA.value])
        self.assertIsNotNone(dec.tier1_result)
        self.assertIsNotNone(dec.tier2_result)
        self.assertIsNotNone(dec.afib_result)
        self.assertIn(dec.primary_rhythm, [l.value for l in UnifiedRhythmLabel])

    def test_full_pipeline_flatline_rejection(self):
        # Pure flatline
        sig = np.zeros(1800, dtype=np.float32)
        dec = self.clf.predict_window(sig, fs=360)
        self.assertEqual(dec.primary_rhythm, UnifiedRhythmLabel.POOR_SIGNAL_OR_LEAD_OFF.value)
        self.assertEqual(dec.triage_level, TRIAGE_SIGNAL_INVALID)

    def test_full_pipeline_latency(self):
        """Measure full cascade cycle time and assert it is well within edge budget (< 50 ms)."""
        dummy = np.random.randn(1800).astype(np.float32)
        latencies = []
        for i in range(25):
            dec = self.clf.predict_window(dummy, current_time_sec=float(i * 5.0))
            latencies.append(dec.latencies_ms["total_cycle"])

        mean_lat = np.mean(latencies[2:])  # skip initial cold-start
        print(f"\n[BENCHMARK] Full Unified Cascade Mean Latency: {mean_lat:.2f} ms (Target: < 50 ms)")
        self.assertLess(mean_lat, 50.0)


if __name__ == "__main__":
    unittest.main()
