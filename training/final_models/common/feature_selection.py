"""Feature schema definitions and selection dictionaries for the three final models."""

# Model 1: Shockable vs. Non-Shockable (QRS-independent chaos & spectral features)
MODEL1_FEATURES = [
    "vf_band_power_ratio",
    "dominant_freq",
    "spectral_entropy",
    "spectral_peak_purity",
    "zero_crossings",
    "hjorth_mobility",
    "hjorth_complexity",
    "lz_complexity",
    "calibrated_ptp",
    "calibrated_std",
]

# Model 2: Multi-Arrhythmia (timing, rate, morphology, spectral, baseline)
MODEL2_FEATURES = [
    "mean_rr",
    "rr_cv",
    "qrs_width",
    "peak_count",
    "dominant_freq",
    "vf_band_power_ratio",
    "spectral_entropy",
    "zero_crossings",
    "hjorth_mobility",
    "hjorth_complexity",
    "calibrated_ptp",
    "calibrated_std",
]

# Model 3: VT vs. VF Subtype Classifier Feature Sets
# Rigorously constructed from Phase 5 feature audit (feature_audit_table.csv & report)

# Set A: Full Physiological Representation (21 candidate features)
MODEL3_SET_A_FULL = [
    "dominant_freq",
    "dominant_peak_power",
    "dominant_peak_prominence",
    "spectral_peak_power_ratio",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_concentration",
    "spectral_peak_purity",
    "ac_max_peak_ratio",
    "ac_first_secondary_peak",
    "ac_decay_time",
    "ac_periodicity_strength",
    "ac_zero_crossing_lag",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "lz_complexity",
    "permutation_entropy",
    "sample_entropy",
]

# Set B: Validated Domain-Stable Subset (6 features)
# Strictly filtered: direction_consistent == True across sources AND mean_source_eta2 <= 0.20.
# All 12 direction-inverted features (including dominant_freq, spectral_entropy,
# spectral_peak_purity, ac_periodicity_strength) and high-confounding features are excluded.
MODEL3_SET_B_DOMAIN_STABLE = [
    "lz_complexity",
    "ac_decay_time",
    "ac_zero_crossing_lag",
    "ac_max_peak_ratio",
    "sample_entropy",
    "hjorth_mobility",
]

# Set C: Minimal Edge Subset (4 top consistent features by Signal-to-Confounder Ratio)
# Optimal ultra-low-complexity subset for edge microcontrollers
MODEL3_SET_C_MINIMAL_EDGE = [
    "lz_complexity",
    "ac_decay_time",
    "ac_zero_crossing_lag",
    "sample_entropy",
]

MODEL3_FEATURE_SETS = {
    "set_a_full": MODEL3_SET_A_FULL,
    "set_b_domain_stable": MODEL3_SET_B_DOMAIN_STABLE,
    "set_c_minimal_edge": MODEL3_SET_C_MINIMAL_EDGE,
}
