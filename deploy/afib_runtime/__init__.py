"""
Patient-Adaptive AFib Runtime Package
=====================================
Self-contained edge deployment package for 5-second window AFib inference.
Zero dependency on research training scripts or datasets.
"""

from .feature_extraction import extract_5s_features, detect_r_peaks, butter_bandpass_filter
from .baseline_tracker import CausalBaselineTracker, DEFAULT_POPULATION_PRIORS
from .persistence_filter import CausalTemporalPersistenceFilter

__all__ = [
    "extract_5s_features",
    "detect_r_peaks",
    "butter_bandpass_filter",
    "CausalBaselineTracker",
    "DEFAULT_POPULATION_PRIORS",
    "CausalTemporalPersistenceFilter",
]
