"""Final HMI decision. Table-driven, no hidden overrides.

Required outputs (M2 rhythm + M1 shockability [+ M3 subtype]):
    NSR         + NON_SHOCKABLE             -> "NSR — NON-SHOCKABLE"
    TACHY       + NON_SHOCKABLE             -> "TACHY — NON-SHOCKABLE"
    BRADY_ASY   + NON_SHOCKABLE             -> "BRADY_ASY — NON-SHOCKABLE"
    VENTRICULAR + SHOCKABLE + subtype VF    -> "VF — SHOCKABLE"
    VENTRICULAR + SHOCKABLE + subtype VT    -> "VT — SHOCKABLE"

Every other combination of the two models is a disagreement. It is NOT
resolved silently: the HMI shows CONFLICT, never a shock advisory, and lists
the raw model outputs. See docs/CLASSIFICATION_LOGIC.md (open question).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .models import ModelOutput

DASH = "—"
NON, SHK = "NON_SHOCKABLE", "SHOCKABLE"

# (rhythm, shockability) -> display for the unambiguous non-ventricular rows
_NON_VENT = {"NSR": "NSR", "TACHY": "TACHY", "BRADY_ASY": "BRADY_ASY"}


@dataclass
class Decision:
    status: str                    # CLASSIFIED | CONFLICT | POOR_SIGNAL | ERROR
    headline: str                  # exactly what the HMI shows in the big banner
    severity: str                  # NORMAL | WARNING | CRITICAL | INVALID
    speech: str
    details: str = ""
    notes: List[str] = field(default_factory=list)


def decide(m1: ModelOutput, m2: ModelOutput, m3: Optional[ModelOutput]) -> Decision:
    shock, rhythm = m1.label, m2.label
    sub = m3.label if m3 is not None else None

    if rhythm in _NON_VENT and shock == NON:
        name = _NON_VENT[rhythm]
        sev = "NORMAL" if rhythm == "NSR" else "WARNING"
        return Decision("CLASSIFIED", f"{name} {DASH} NON-SHOCKABLE", sev,
                        f"{name.replace('_', ' ')}. Non shockable.", _details(m1, m2, m3))

    if rhythm == "VENTRICULAR" and shock == SHK and sub in ("VF", "VT"):
        return Decision("CLASSIFIED", f"{sub} {DASH} SHOCKABLE", "CRITICAL",
                        f"{'Ventricular fibrillation' if sub == 'VF' else 'Ventricular tachycardia'}. Shockable.",
                        _details(m1, m2, m3))

    # --- anything else: models disagree (or M3 missing) -> do not assert shockability
    why = (f"Shockability model says {shock}, rhythm model says {rhythm}"
           + (f", subtype model says {sub}" if sub else ""))
    return Decision("CONFLICT", f"CONFLICT {DASH} NOT CLASSIFIED", "WARNING",
                    "Model disagreement. No classification. Re-check electrodes and repeat.",
                    _details(m1, m2, m3), [why])


def _details(m1, m2, m3) -> str:
    s = f"M1 {m1.label} ({m1.confidence:.0%})  |  M2 {m2.label} ({m2.confidence:.0%})"
    s += f"  |  M3 {m3.label} ({m3.confidence:.0%})" if m3 else "  |  M3 not run"
    return s


def poor_signal(reason: str) -> Decision:
    return Decision("POOR_SIGNAL", f"NO RESULT {DASH} POOR SIGNAL", "INVALID",
                    "Poor signal. Check electrode contact and press Start.", reason)


def error(msg: str) -> Decision:
    return Decision("ERROR", f"NO RESULT {DASH} ACQUISITION ERROR", "INVALID",
                    "Acquisition error. Check connections and press Start.", msg)
