"""Final HMI decision under Policy B. Table-driven, no hidden overrides.

Exhaustive M1 (shockability) x M2 (rhythm) truth table (8 combinations):
    SHOCKABLE     + NSR         -> "CONFLICT — NOT CLASSIFIED"
    SHOCKABLE     + TACHY       -> "VF/VT SHOCKABLE"
    SHOCKABLE     + BRADY_ASY   -> "CONFLICT — NOT CLASSIFIED"
    SHOCKABLE     + VENTRICULAR -> "VF/VT SHOCKABLE"
    NON_SHOCKABLE + NSR         -> "NORMAL SINUS RHYTHM"
    NON_SHOCKABLE + TACHY       -> "TACHYCARDIA"
    NON_SHOCKABLE + BRADY_ASY   -> "OTHER — NOT CLASSIFIED"
    NON_SHOCKABLE + VENTRICULAR -> "VENTRICULAR (NON-SHOCKABLE)"

Model 3 is removed from the runtime decision path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .models import ModelOutput

DASH = "—"
NON, SHK = "NON_SHOCKABLE", "SHOCKABLE"

TABLE = {
    (SHK, "NSR"): (
        "CONFLICT",
        f"CONFLICT {DASH} NOT CLASSIFIED",
        "WARNING",
        "Model disagreement. No classification. Re-check electrodes and repeat.",
    ),
    (SHK, "TACHY"): (
        "CLASSIFIED",
        "VF/VT SHOCKABLE",
        "CRITICAL",
        "Shockable rhythm detected. VF or VT.",
    ),
    (SHK, "BRADY_ASY"): (
        "CONFLICT",
        f"CONFLICT {DASH} NOT CLASSIFIED",
        "WARNING",
        "Model disagreement. No classification. Re-check electrodes and repeat.",
    ),
    (SHK, "VENTRICULAR"): (
        "CLASSIFIED",
        "VF/VT SHOCKABLE",
        "CRITICAL",
        "Shockable rhythm detected. VF or VT.",
    ),
    (NON, "NSR"): (
        "CLASSIFIED",
        "NORMAL SINUS RHYTHM",
        "NORMAL",
        "Normal sinus rhythm. Non shockable.",
    ),
    (NON, "TACHY"): (
        "CLASSIFIED",
        "TACHYCARDIA",
        "WARNING",
        "Tachycardia. Non shockable.",
    ),
    (NON, "BRADY_ASY"): (
        "CLASSIFIED",
        f"OTHER {DASH} NOT CLASSIFIED",
        "WARNING",
        "Other rhythm. Not classified.",
    ),
    (NON, "VENTRICULAR"): (
        "CLASSIFIED",
        "VENTRICULAR (NON-SHOCKABLE)",
        "WARNING",
        "Ventricular rhythm, non shockable.",
    ),
}


@dataclass
class Decision:
    status: str                    # CLASSIFIED | CONFLICT | POOR_SIGNAL | ERROR
    headline: str                  # exactly what the HMI shows in the big banner
    severity: str                  # NORMAL | WARNING | CRITICAL | INVALID
    speech: str
    details: str = ""
    notes: List[str] = field(default_factory=list)


def decide(m1: ModelOutput, m2: ModelOutput, m3: Optional[ModelOutput] = None) -> Decision:
    shock, rhythm = m1.label, m2.label
    row = TABLE.get((shock, rhythm))
    if row is None:
        why = f"Unrecognized model output: shock={shock}, rhythm={rhythm}"
        return Decision(
            "CONFLICT",
            f"CONFLICT {DASH} NOT CLASSIFIED",
            "WARNING",
            "Model disagreement. No classification. Re-check electrodes and repeat.",
            _details(m1, m2),
            [why],
        )

    status, headline, severity, speech = row
    notes = []
    if status == "CONFLICT":
        notes.append(f"Shockability model says {shock}, rhythm model says {rhythm}")

    return Decision(status, headline, severity, speech, _details(m1, m2), notes)


def _details(m1: ModelOutput, m2: ModelOutput, m3: Optional[ModelOutput] = None) -> str:
    return f"M1 {m1.label} ({m1.confidence:.0%})  |  M2 {m2.label} ({m2.confidence:.0%})"


def poor_signal(reason: str) -> Decision:
    return Decision(
        "POOR_SIGNAL",
        f"NO RESULT {DASH} POOR SIGNAL",
        "INVALID",
        "Poor signal. Check electrode contact and press Start.",
        reason,
    )


def error(msg: str) -> Decision:
    return Decision(
        "ERROR",
        f"NO RESULT {DASH} ACQUISITION ERROR",
        "INVALID",
        "Acquisition error. Check connections and press Start.",
        msg,
    )
