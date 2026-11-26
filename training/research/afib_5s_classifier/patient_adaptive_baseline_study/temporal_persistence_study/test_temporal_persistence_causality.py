"""
Synthetic Unit Tests: Causality, Gap Handling, and Boundary Isolation for Persistence Filters
=============================================================================================
"""

import sys
import unittest
from pathlib import Path
import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from persistence_filter import CausalTemporalPersistenceFilter


class TestTemporalPersistenceCausality(unittest.TestCase):

    def test_future_invariance_leakage(self):
        """
        Changing future predictions at t >= t_k must have ZERO effect on alert at t_k or earlier.
        """
        n_windows = 40
        times = [i * 2.5 for i in range(n_windows)]
        raw_probs1 = [0.85 if 10 <= i <= 20 else 0.15 for i in range(n_windows)]
        raw_decs1 = [1 if p >= 0.56 else 0 for p in raw_probs1]

        for rule in CausalTemporalPersistenceFilter.SUPPORTED_RULES:
            flt1 = CausalTemporalPersistenceFilter(rule_name=rule)
            alerts1 = [
                flt1.process_window(times[i], raw_probs1[i], raw_decs1[i], record_id="REC_A")
                for i in range(n_windows)
            ]

            # Modify future starting at index 15 (e.g. change to zeros or ones)
            raw_probs2 = list(raw_probs1)
            raw_decs2 = list(raw_decs1)
            for i in range(15, n_windows):
                raw_probs2[i] = 0.99
                raw_decs2[i] = 1

            flt2 = CausalTemporalPersistenceFilter(rule_name=rule)
            alerts2 = [
                flt2.process_window(times[i], raw_probs2[i], raw_decs2[i], record_id="REC_A")
                for i in range(n_windows)
            ]

            # Steps 0 to 14 must be bit-for-bit identical
            for k in range(15):
                self.assertEqual(
                    alerts1[k], alerts2[k],
                    f"Causality violation in rule '{rule}' at step {k}: {alerts1[k]} != {alerts2[k]}"
                )

    def test_boundary_reset_cross_patient_isolation(self):
        """
        Record change must purge streak so that Patient B does NOT inherit Patient A's streak.
        """
        flt = CausalTemporalPersistenceFilter(rule_name="consecutive_3")

        # Patient A ends with 2 consecutive positive windows
        a1 = flt.process_window(0.0, 0.9, 1, record_id="PATIENT_A")
        a2 = flt.process_window(2.5, 0.9, 1, record_id="PATIENT_A")
        self.assertEqual(a1, 0)
        self.assertEqual(a2, 0)  # Needs 3

        # Switch to Patient B with 1 positive window
        # If leaked, Patient B would get 3rd positive and alert!
        b1 = flt.process_window(0.0, 0.9, 1, record_id="PATIENT_B")
        self.assertEqual(b1, 0, "LEAKAGE: Patient B inherited Patient A's consecutive streak!")

    def test_gap_streak_reset(self):
        """
        If elapsed time exceeds max_gap_sec (e.g. 6.0s), streak resets to 0.
        """
        flt = CausalTemporalPersistenceFilter(rule_name="consecutive_2", max_gap_sec=6.0)

        # Window 1 at t=0s
        w1 = flt.process_window(0.0, 0.8, 1, record_id="REC_1")
        self.assertEqual(w1, 0)

        # Window 2 after a gap of 20 seconds (t=20.0s)
        # Because gap > 6.0s, window 1 was forgotten, so this is streak=1, NOT 2!
        w2 = flt.process_window(20.0, 0.8, 1, record_id="REC_1")
        self.assertEqual(w2, 0, "GAP LEAKAGE: Consecutive streak survived large time gap!")

        # Window 3 at t=22.5s (contiguous, gap=2.5s <= 6.0s)
        w3 = flt.process_window(22.5, 0.8, 1, record_id="REC_1")
        self.assertEqual(w3, 1, "Consecutive 2 should confirm alert on contiguous windows")

    def test_hysteresis_behavior(self):
        """
        Hysteresis: Onset requires 3 positives; offset requires 2 negatives.
        """
        flt = CausalTemporalPersistenceFilter(rule_name="hysteresis_on3_off2")

        # 1 positive
        self.assertEqual(flt.process_window(0.0, 0.8, 1), 0)
        # 2 positives
        self.assertEqual(flt.process_window(2.5, 0.8, 1), 0)
        # 3 positives -> TRIGGERS ON
        self.assertEqual(flt.process_window(5.0, 0.8, 1), 1)

        # Single negative drop -> STAYS ON (requires 2 negatives to clear)
        self.assertEqual(flt.process_window(7.5, 0.2, 0), 1)

        # Second negative -> CLEARS OFF
        self.assertEqual(flt.process_window(10.0, 0.2, 0), 0)


if __name__ == "__main__":
    unittest.main()
