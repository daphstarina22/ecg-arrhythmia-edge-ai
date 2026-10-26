"""
Unified ECG Arrhythmia Decision and Arbitration Layer
=====================================================
Target: Raspberry Pi 4B+ (Edge AI Runtime)

Defines the 9 unified clinical output labels, deterministic priority arbitration,
and safety-first conflict resolution rules combining:
  - Signal Quality Guard (Tier 0)
  - Shockable Screener (Tier 1 ONNX)
  - 4-Class Arrhythmia Classifier (Tier 2 ONNX: NSR, TACHY, BRADY_ASY, VENTRICULAR)
  - Ventricular Subtype Specialist (Tier 3 ONNX: VT, VF_VFL)
  - Patient-Adaptive AFib Branch (Research Config C: 26 features + Causal Persistence)

CRITICAL INVARIANTS:
1. AFib predictions NEVER suppress, override, or downgrade shockable or ventricular rhythms.
2. Signal quality failures immediately yield POOR_SIGNAL_OR_LEAD_OFF without calling ML.
3. Model 2's 4 classes are fully preserved (NSR, TACHY, BRADY_ASY, VENTRICULAR).
4. Bradycardia research is preserved and mapped to BRADYCARDIA.
5. All underlying model outputs and probabilities are retained in telemetry for auditability.
6. Conflicting predictions (e.g. TACHY vs AFib, BRADY vs AFib) are explicitly resolved to
   UNCERTAIN_OR_CONFLICTING_PREDICTIONS rather than comparing uncalibrated probabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class UnifiedRhythmLabel(str, Enum):
    """The 9 explicit output labels required for the unified edge system."""
    NORMAL_SINUS_RHYTHM = "NORMAL_SINUS_RHYTHM"
    ATRIAL_FIBRILLATION = "ATRIAL_FIBRILLATION"
    BRADYCARDIA = "BRADYCARDIA"
    TACHYCARDIA = "TACHYCARDIA"
    VENTRICULAR_ARRHYTHMIA = "VENTRICULAR_ARRHYTHMIA"
    VENTRICULAR_TACHYCARDIA = "VENTRICULAR_TACHYCARDIA"
    VENTRICULAR_FIBRILLATION = "VENTRICULAR_FIBRILLATION"
    POOR_SIGNAL_OR_LEAD_OFF = "POOR_SIGNAL_OR_LEAD_OFF"
    UNCERTAIN_OR_CONFLICTING_PREDICTIONS = "UNCERTAIN_OR_CONFLICTING_PREDICTIONS"


# Triage severity levels
TRIAGE_CRITICAL_EMERGENCY = "CRITICAL_EMERGENCY"
TRIAGE_URGENT_MONITORING = "URGENT_MONITORING"
TRIAGE_ROUTINE_NORMAL = "ROUTINE_NORMAL"
TRIAGE_SIGNAL_INVALID = "SIGNAL_INVALID"
TRIAGE_UNCERTAIN = "UNCERTAIN"


@dataclass
class UnifiedDecision:
    """Complete unified classification result with telemetry and audit trail."""
    final_label: str
    triage_level: str
    is_shockable: bool
    defibrillator_advised: bool
    confidence: float
    recommended_action: str
    conflict_detected: bool
    conflict_detail: Optional[str]
    tier0_quality: Dict[str, Any]
    tier1_shockable: Dict[str, Any]
    tier2_multiclass: Dict[str, Any]
    tier3_subtype: Optional[Dict[str, Any]]
    afib_branch: Dict[str, Any]
    latencies_ms: Dict[str, float]
    safety_flags: List[str] = field(default_factory=list)
    clinical_notice: str = (
        "RESEARCH PROTOTYPE ONLY: Not clinically validated for diagnostic use or automated therapy. "
        "Atrial Fibrillation detection is an experimental research module."
    )


def arbitrate_decision(
    quality_report: Dict[str, Any],
    tier1_result: Dict[str, Any],
    tier2_result: Dict[str, Any],
    tier3_result: Optional[Dict[str, Any]],
    afib_result: Dict[str, Any],
    latencies_ms: Optional[Dict[str, float]] = None,
) -> UnifiedDecision:
    """
    Deterministically arbitrates between Tier 0, Tier 1, Tier 2, Tier 3, and AFib outputs.

    Arbitration Hierarchy:
      Step 1: Signal Quality Guard -> POOR_SIGNAL_OR_LEAD_OFF
      Step 2: Shockable Screener & Ventricular Priority -> VF / VT / VENTRICULAR_ARRHYTHMIA
      Step 3: Non-Ventricular Rhythms -> NSR / AFib / TACHYCARDIA / BRADYCARDIA / CONFLICT
    """
    safety_flags: List[str] = []
    latencies = latencies_ms or {}

    # ------------------------------------------------------------------
    # Step 1: Signal Quality Diagnostic Guard (Tier 0)
    # ------------------------------------------------------------------
    is_valid_signal = quality_report.get("is_valid_for_ml", False)
    quality_status = quality_report.get("status", "BAD_SIGNAL")

    if not is_valid_signal:
        return UnifiedDecision(
            final_label=UnifiedRhythmLabel.POOR_SIGNAL_OR_LEAD_OFF.value,
            triage_level=TRIAGE_SIGNAL_INVALID,
            is_shockable=False,
            defibrillator_advised=False,
            confidence=0.0,
            recommended_action=f"INVALID SIGNAL: Check electrode contact and cable connection ({quality_status}).",
            conflict_detected=False,
            conflict_detail=None,
            tier0_quality=quality_report,
            tier1_shockable=tier1_result,
            tier2_multiclass=tier2_result,
            tier3_subtype=tier3_result,
            afib_branch=afib_result,
            latencies_ms=latencies,
            safety_flags=[f"SIGNAL_REJECTED: {quality_report.get('detail', quality_status)}"],
        )

    # ------------------------------------------------------------------
    # Step 2: Shockable Screener (Tier 1) & Ventricular Classification (Tier 2)
    # Priority: Ventricular and shockable rhythms must NEVER be suppressed by AFib.
    # ------------------------------------------------------------------
    is_shockable = bool(tier1_result.get("is_shockable", False))
    p_shockable = float(tier1_result.get("prob_shockable", tier1_result.get("p_shockable", 0.0)))
    tier2_pred = str(tier2_result.get("predicted_class", "NSR")).upper()
    is_ventricular = is_shockable or (tier2_pred == "VENTRICULAR")

    afib_res = afib_result or {}
    afib_active = bool(afib_res.get("afib_alert", False))
    afib_prob = float(afib_res.get("raw_probability", 0.0))
    afib_status = str(afib_res.get("status", "OK")).upper()
    afib_available = bool(afib_res.get("is_available", True)) and afib_status not in ("UNAVAILABLE", "ERROR")

    if not afib_available and is_valid_signal:
        safety_flags.append(
            f"AFIB_UNAVAILABLE: AFib inference branch is unavailable ({afib_status}); "
            "absence of atrial fibrillation cannot be verified."
        )

    if is_ventricular:
        # Note if AFib was triggered as a confounder during ventricular arrhythmia
        if afib_active or afib_prob >= 0.50:
            safety_flags.append(
                f"VENTRICULAR_CONFOUNDER_INTERCEPTED: AFib probability {afib_prob:.3f} was suppressed "
                "because high-priority ventricular rhythm was detected."
            )

        if tier3_result is not None:
            subtype = str(tier3_result.get("subtype", "VT")).upper()
            p_vf = float(tier3_result.get("prob_vf_vfl", tier3_result.get("p_vf", 0.0)))
            p_vt = float(tier3_result.get("prob_vt", tier3_result.get("p_vt", 0.0)))

            if 0.40 <= p_vf <= 0.60:
                safety_flags.append("EQUIVOCAL_VENTRICULAR_SUBTYPE: Borderline morphological complexity between VT and VF.")

            if subtype in ("VF", "VF_VFL") or (p_vf >= 0.50):
                return UnifiedDecision(
                    final_label=UnifiedRhythmLabel.VENTRICULAR_FIBRILLATION.value,
                    triage_level=TRIAGE_CRITICAL_EMERGENCY,
                    is_shockable=True,
                    defibrillator_advised=True,
                    confidence=p_vf if p_vf > 0 else (p_shockable if p_shockable > 0 else 0.90),
                    recommended_action="CRITICAL: Ventricular Fibrillation detected! Immediate defibrillation shock advised.",
                    conflict_detected=False,
                    conflict_detail=None,
                    tier0_quality=quality_report,
                    tier1_shockable=tier1_result,
                    tier2_multiclass=tier2_result,
                    tier3_subtype=tier3_result,
                    afib_branch=afib_result,
                    latencies_ms=latencies,
                    safety_flags=safety_flags,
                )
            else:
                defib_advised = is_shockable or (p_shockable >= 0.50)
                action = (
                    "CRITICAL: Sustained Ventricular Tachycardia (Shockable)! Prepare defibrillator / cardioversion."
                    if defib_advised
                    else "URGENT: Ventricular Tachycardia detected. Evaluate hemodynamics and notify cardiology immediately."
                )
                return UnifiedDecision(
                    final_label=UnifiedRhythmLabel.VENTRICULAR_TACHYCARDIA.value,
                    triage_level=TRIAGE_CRITICAL_EMERGENCY if defib_advised else TRIAGE_URGENT_MONITORING,
                    is_shockable=defib_advised,
                    defibrillator_advised=defib_advised,
                    confidence=p_vt if p_vt > 0 else 0.85,
                    recommended_action=action,
                    conflict_detected=False,
                    conflict_detail=None,
                    tier0_quality=quality_report,
                    tier1_shockable=tier1_result,
                    tier2_multiclass=tier2_result,
                    tier3_subtype=tier3_result,
                    afib_branch=afib_result,
                    latencies_ms=latencies,
                    safety_flags=safety_flags,
                )
        else:
            # Ventricular predicted by Tier 2 but Tier 3 did not run or is missing
            return UnifiedDecision(
                final_label=UnifiedRhythmLabel.VENTRICULAR_ARRHYTHMIA.value,
                triage_level=TRIAGE_CRITICAL_EMERGENCY if is_shockable else TRIAGE_URGENT_MONITORING,
                is_shockable=is_shockable,
                defibrillator_advised=is_shockable,
                confidence=tier2_result.get("confidence", 0.80),
                recommended_action="URGENT: Ventricular Arrhythmia detected. Assess patient rhythm and vitals.",
                conflict_detected=False,
                conflict_detail=None,
                tier0_quality=quality_report,
                tier1_shockable=tier1_result,
                tier2_multiclass=tier2_result,
                tier3_subtype=tier3_result,
                afib_branch=afib_result,
                latencies_ms=latencies,
                safety_flags=safety_flags,
            )

    # ------------------------------------------------------------------
    # Step 3: Non-Ventricular Rhythm Arbitration
    # Tier 2 options: NSR, TACHY, BRADY_ASY
    # ------------------------------------------------------------------
    tier2_conf = float(tier2_result.get("confidence", 0.50))

    if tier2_pred == "NSR":
        if not afib_available:
            return UnifiedDecision(
                final_label=UnifiedRhythmLabel.UNCERTAIN_OR_CONFLICTING_PREDICTIONS.value,
                triage_level=TRIAGE_UNCERTAIN,
                is_shockable=False,
                defibrillator_advised=False,
                confidence=tier2_conf,
                recommended_action="UNCERTAIN: Normal sinus rhythm unconfirmed because AFib detection is unavailable. Clinical rhythm review advised.",
                conflict_detected=True,
                conflict_detail=f"AFib branch is unavailable ({afib_status}); cannot confirm absence of AFib.",
                tier0_quality=quality_report,
                tier1_shockable=tier1_result,
                tier2_multiclass=tier2_result,
                tier3_subtype=tier3_result,
                afib_branch=afib_result,
                latencies_ms=latencies,
                safety_flags=safety_flags,
            )
        elif afib_active:
            # Tier 2 detects normal background rate/morphology, but AFib branch confirms persistent irregular rhythm
            return UnifiedDecision(
                final_label=UnifiedRhythmLabel.ATRIAL_FIBRILLATION.value,
                triage_level=TRIAGE_URGENT_MONITORING,
                is_shockable=False,
                defibrillator_advised=False,
                confidence=afib_prob,
                recommended_action="URGENT: Atrial Fibrillation detected by patient-adaptive rhythm analysis. Monitor pulse regularity.",
                conflict_detected=False,
                conflict_detail=None,
                tier0_quality=quality_report,
                tier1_shockable=tier1_result,
                tier2_multiclass=tier2_result,
                tier3_subtype=tier3_result,
                afib_branch=afib_result,
                latencies_ms=latencies,
                safety_flags=safety_flags,
            )
        else:
            return UnifiedDecision(
                final_label=UnifiedRhythmLabel.NORMAL_SINUS_RHYTHM.value,
                triage_level=TRIAGE_ROUTINE_NORMAL,
                is_shockable=False,
                defibrillator_advised=False,
                confidence=tier2_conf,
                recommended_action="STABLE: Normal Sinus Rhythm. Rhythm within expected baseline limits.",
                conflict_detected=False,
                conflict_detail=None,
                tier0_quality=quality_report,
                tier1_shockable=tier1_result,
                tier2_multiclass=tier2_result,
                tier3_subtype=tier3_result,
                afib_branch=afib_result,
                latencies_ms=latencies,
                safety_flags=safety_flags,
            )

    elif tier2_pred == "TACHY":
        if afib_active:
            # Conflict: Model 2 predicts regular narrow-complex SVT / Tachycardia while AFib branch predicts AFib
            conflict_msg = (
                f"Prediction conflict: Tier 2 classifies TACHYCARDIA (conf={tier2_conf:.2f}), "
                f"while AFib branch classifies ATRIAL_FIBRILLATION (p={afib_prob:.2f})."
            )
            safety_flags.append(conflict_msg)
            return UnifiedDecision(
                final_label=UnifiedRhythmLabel.UNCERTAIN_OR_CONFLICTING_PREDICTIONS.value,
                triage_level=TRIAGE_UNCERTAIN,
                is_shockable=False,
                defibrillator_advised=False,
                confidence=max(tier2_conf, afib_prob),
                recommended_action="UNCERTAIN: Conflicting supraventricular rhythm predictions (Rapid SVT vs AFib). Clinical verification required.",
                conflict_detected=True,
                conflict_detail=conflict_msg,
                tier0_quality=quality_report,
                tier1_shockable=tier1_result,
                tier2_multiclass=tier2_result,
                tier3_subtype=tier3_result,
                afib_branch=afib_result,
                latencies_ms=latencies,
                safety_flags=safety_flags,
            )
        else:
            return UnifiedDecision(
                final_label=UnifiedRhythmLabel.TACHYCARDIA.value,
                triage_level=TRIAGE_URGENT_MONITORING,
                is_shockable=False,
                defibrillator_advised=False,
                confidence=tier2_conf,
                recommended_action="URGENT: Tachycardia detected. Evaluate patient heart rate and clinical tolerance.",
                conflict_detected=False,
                conflict_detail=None,
                tier0_quality=quality_report,
                tier1_shockable=tier1_result,
                tier2_multiclass=tier2_result,
                tier3_subtype=tier3_result,
                afib_branch=afib_result,
                latencies_ms=latencies,
                safety_flags=safety_flags,
            )

    elif tier2_pred == "BRADY_ASY":
        if afib_active:
            # Conflict: Model 2 predicts Bradycardia / Asystolic pause while AFib branch predicts AFib
            conflict_msg = (
                f"Prediction conflict: Tier 2 classifies BRADYCARDIA (conf={tier2_conf:.2f}), "
                f"while AFib branch classifies ATRIAL_FIBRILLATION (p={afib_prob:.2f})."
            )
            safety_flags.append(conflict_msg)
            return UnifiedDecision(
                final_label=UnifiedRhythmLabel.UNCERTAIN_OR_CONFLICTING_PREDICTIONS.value,
                triage_level=TRIAGE_UNCERTAIN,
                is_shockable=False,
                defibrillator_advised=False,
                confidence=max(tier2_conf, afib_prob),
                recommended_action="UNCERTAIN: Conflicting rhythm predictions (Bradycardia/Pause vs Slow AFib). Check patient pulse.",
                conflict_detected=True,
                conflict_detail=conflict_msg,
                tier0_quality=quality_report,
                tier1_shockable=tier1_result,
                tier2_multiclass=tier2_result,
                tier3_subtype=tier3_result,
                afib_branch=afib_result,
                latencies_ms=latencies,
                safety_flags=safety_flags,
            )
        else:
            return UnifiedDecision(
                final_label=UnifiedRhythmLabel.BRADYCARDIA.value,
                triage_level=TRIAGE_URGENT_MONITORING,
                is_shockable=False,
                defibrillator_advised=False,
                confidence=tier2_conf,
                recommended_action="URGENT: Bradycardia / conduction pause detected. Monitor patient vitals.",
                conflict_detected=False,
                conflict_detail=None,
                tier0_quality=quality_report,
                tier1_shockable=tier1_result,
                tier2_multiclass=tier2_result,
                tier3_subtype=tier3_result,
                afib_branch=afib_result,
                latencies_ms=latencies,
                safety_flags=safety_flags,
            )

    # Fallback for unrecognized class
    return UnifiedDecision(
        final_label=UnifiedRhythmLabel.UNCERTAIN_OR_CONFLICTING_PREDICTIONS.value,
        triage_level=TRIAGE_UNCERTAIN,
        is_shockable=False,
        defibrillator_advised=False,
        confidence=0.0,
        recommended_action="UNCERTAIN: Unrecognized rhythm classification. Check system logs.",
        conflict_detected=True,
        conflict_detail=f"Unrecognized Tier 2 predicted class: {tier2_pred}",
        tier0_quality=quality_report,
        tier1_shockable=tier1_result,
        tier2_multiclass=tier2_result,
        tier3_subtype=tier3_result,
        afib_branch=afib_result,
        latencies_ms=latencies,
        safety_flags=safety_flags,
    )
