"""
Causal Temporal Persistence Filter Runtime Module
=================================================
Hardware Target: Embedded Wearable / Edge ECG Monitor (Raspberry Pi ARM64)

Design Constraints:
1. STRICT CAUSALITY: An alert decision at time t depends exclusively on windows
   at time <= t. Never uses future windows to retroactively confirm an alert.
2. ZERO CROSS-RECORD LEAKAGE: When switching patient or record IDs, all persistence
   queues, streak counters, and hysteresis states are purged.
3. GAP-AWARE: If the elapsed time between consecutive windows exceeds max_gap_sec
   (due to transition guard band rejection or signal dropout), streaks reset.
4. COLD-START SAFE: Operates gracefully during initial windows without crashes or NaNs.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple


class CausalTemporalPersistenceFilter:
    """
    Causal temporal persistence state machine.
    Processes sequential (timestamp_sec, raw_prob, raw_decision) tuples.
    """

    SUPPORTED_RULES = [
        "unfiltered",
        "consecutive_2",
        "consecutive_3",
        "consecutive_4",
        "majority_2_of_3",
        "majority_3_of_4",
        "hysteresis_on3_off2",
    ]

    def __init__(self, rule_name: str = "consecutive_3", max_gap_sec: float = 6.0):
        if rule_name not in self.SUPPORTED_RULES:
            raise ValueError(f"Unknown rule '{rule_name}'. Supported: {self.SUPPORTED_RULES}")
        self.rule_name = str(rule_name)
        self.max_gap_sec = float(max_gap_sec)

        # Internal state
        self.current_record_id: Optional[str] = None
        self.last_timestamp_sec: Optional[float] = None
        self.window_history = deque(maxlen=10)  # stores (timestamp_sec, raw_decision, raw_prob)
        self.consecutive_positives = 0
        self.consecutive_negatives = 0
        self.hysteresis_active = False

    def reset(self, new_record_id: Optional[str] = None):
        """Purges all temporal memory at record or patient boundary."""
        self.current_record_id = new_record_id
        self.last_timestamp_sec = None
        self.window_history.clear()
        self.consecutive_positives = 0
        self.consecutive_negatives = 0
        self.hysteresis_active = False

    def process_window(
        self,
        timestamp_sec: float,
        raw_prob: float,
        raw_decision: int,
        record_id: Optional[str] = None,
    ) -> int:
        """
        Causally evaluates window at timestamp_sec and returns confirmed alert (0 or 1).
        """
        # 1. Boundary Check
        if record_id is not None and record_id != self.current_record_id:
            self.reset(new_record_id=record_id)

        # 2. Time Gap Check
        is_contiguous = True
        if self.last_timestamp_sec is not None:
            gap = timestamp_sec - self.last_timestamp_sec
            if gap > self.max_gap_sec:
                # Signal dropout or guard band rejection broke continuity
                is_contiguous = False
                self.window_history.clear()
                self.consecutive_positives = 0
                self.consecutive_negatives = 0
                self.hysteresis_active = False

        self.last_timestamp_sec = timestamp_sec

        # 3. Update Streaks
        if raw_decision == 1:
            self.consecutive_positives = self.consecutive_positives + 1 if is_contiguous else 1
            self.consecutive_negatives = 0
        else:
            self.consecutive_negatives = self.consecutive_negatives + 1 if is_contiguous else 1
            self.consecutive_positives = 0

        self.window_history.append((timestamp_sec, int(raw_decision), float(raw_prob)))

        # 4. Evaluate Selected Persistence Rule
        confirmed_alert = 0

        if self.rule_name == "unfiltered":
            confirmed_alert = raw_decision

        elif self.rule_name == "consecutive_2":
            confirmed_alert = 1 if self.consecutive_positives >= 2 else 0

        elif self.rule_name == "consecutive_3":
            confirmed_alert = 1 if self.consecutive_positives >= 3 else 0

        elif self.rule_name == "consecutive_4":
            confirmed_alert = 1 if self.consecutive_positives >= 4 else 0

        elif self.rule_name == "majority_2_of_3":
            if len(self.window_history) >= 2:
                recent = list(self.window_history)[-3:]
                pos_count = sum(item[1] for item in recent)
                confirmed_alert = 1 if pos_count >= 2 else 0
            else:
                confirmed_alert = 0

        elif self.rule_name == "majority_3_of_4":
            if len(self.window_history) >= 3:
                recent = list(self.window_history)[-4:]
                pos_count = sum(item[1] for item in recent)
                confirmed_alert = 1 if pos_count >= 3 else 0
            else:
                confirmed_alert = 0

        elif self.rule_name == "hysteresis_on3_off2":
            # Onset requires 3 consecutive positives; offset requires 2 consecutive negatives
            if not self.hysteresis_active:
                if self.consecutive_positives >= 3:
                    self.hysteresis_active = True
                    confirmed_alert = 1
                else:
                    confirmed_alert = 0
            else:
                if self.consecutive_negatives >= 2:
                    self.hysteresis_active = False
                    confirmed_alert = 0
                else:
                    confirmed_alert = 1

        return int(confirmed_alert)
