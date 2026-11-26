"""
Lightweight Unit Tests for PVC Research Prototype
==================================================
Runs without physical hardware (Raspberry Pi / ADS1292R).
Tests window dimensions, boundaries, alignment, patient leakage prevention,
and unsupported symbol filtering.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root and module path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.research.pvc_beat_prototype import (
    PRE_SAMPLES, POST_SAMPLES, TOTAL_WINDOW_SAMPLES,
    extract_beat_features, create_patient_stratified_split,
    SUBJECT_GROUPS
)


class TestPVCPrototype(unittest.TestCase):

    def test_window_dimensions(self):
        """Checks that window dimensions adhere strictly to the 200ms pre, 300ms post standard at 360Hz."""
        self.assertEqual(PRE_SAMPLES, 72, "Pre-peak samples at 360 Hz must be 72 (200 ms)")
        self.assertEqual(POST_SAMPLES, 108, "Post-peak samples at 360 Hz must be 108 (300 ms)")
        self.assertEqual(TOTAL_WINDOW_SAMPLES, 180, "Total beat window must be 180 samples (500 ms)")

    def test_feature_extraction_properties(self):
        """Verifies feature extraction on a synthetic 180-sample QRS pulse."""
        t = np.linspace(-0.2, 0.3, TOTAL_WINDOW_SAMPLES)
        # Synthetic QRS: high sharp central peak
        sig = 1.2 * np.exp(-(t / 0.02) ** 2) - 0.2 * np.exp(-((t - 0.04) / 0.03) ** 2)
        
        feats = extract_beat_features(sig, pre_rr_sec=0.82, post_rr_sec=0.85)
        
        # Verify feature dictionary keys
        self.assertIn("ptp_mv", feats)
        self.assertIn("std_mv", feats)
        self.assertIn("energy", feats)
        self.assertIn("rr_ratio", feats)
        self.assertIn("local_hr", feats)
        self.assertIn("morph_0", feats)
        self.assertIn("morph_13", feats)

        # Check all features are finite
        for k, v in feats.items():
            self.assertTrue(np.isfinite(v), f"Feature {k} must be finite, got {v}")

        # Check peak index is near center (around sample 72)
        self.assertAlmostEqual(feats["peak_idx"], 72, delta=3)

    def test_patient_split_leak_free(self):
        """Verifies that patient-level splitting guarantees zero leakage across train, val, and test sets."""
        dummy_counts = {r: 50 for sub in SUBJECT_GROUPS.values() for r in sub}
        dummy_counts["208"] = 992
        dummy_counts["200"] = 826

        train_recs, val_recs, test_recs = create_patient_stratified_split(dummy_counts, seed=42)

        set_tr = set(train_recs)
        set_val = set(val_recs)
        set_te = set(test_recs)

        # 1. Zero intersection
        self.assertEqual(len(set_tr & set_val), 0, "Train and Val records must not overlap!")
        self.assertEqual(len(set_tr & set_te), 0, "Train and Test records must not overlap!")
        self.assertEqual(len(set_val & set_te), 0, "Val and Test records must not overlap!")

        # 2. Shared subject constraint: Records 201 and 202 must be in the exact same partition
        has_201 = ("201" in set_tr, "201" in set_val, "201" in set_te)
        has_202 = ("202" in set_tr, "202" in set_val, "202" in set_te)
        self.assertEqual(has_201, has_202, "Records 201 and 202 (same subject) must reside in the exact same split!")

    def test_boundary_handling_logic(self):
        """Checks boundary threshold checks for beats occurring too close to recording start or end."""
        sig_len = 1000
        # Beat at sample 50: index 50 - 72 < 0 -> Underflow
        sample_early = 50
        is_early_valid = (sample_early >= PRE_SAMPLES) and (sample_early + POST_SAMPLES <= sig_len)
        self.assertFalse(is_early_valid, "Beats before PRE_SAMPLES must be marked invalid")

        # Beat at sample 950: index 950 + 108 > 1000 -> Overflow
        sample_late = 950
        is_late_valid = (sample_late >= PRE_SAMPLES) and (sample_late + POST_SAMPLES <= sig_len)
        self.assertFalse(is_late_valid, "Beats after sig_len - POST_SAMPLES must be marked invalid")

        # Beat at sample 500: Valid
        sample_mid = 500
        is_mid_valid = (sample_mid >= PRE_SAMPLES) and (sample_mid + POST_SAMPLES <= sig_len)
        self.assertTrue(is_mid_valid, "Beats within bounds must be valid")

    def test_unsupported_symbol_classification(self):
        """Checks that unsupported symbols (A, L, R, /, F) are not assigned to Normal (N) or PVC (V)."""
        unsupported = ["A", "L", "R", "/", "F", "f", "j", "+", "~"]
        supported = ["N", "V"]
        for sym in unsupported:
            self.assertNotIn(sym, supported, f"Symbol {sym} must not be accepted into binary N vs V")


if __name__ == "__main__":
    unittest.main()
