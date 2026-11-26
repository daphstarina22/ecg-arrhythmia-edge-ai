"""
Patient-Adaptive AFib Edge Bridge Module
=========================================
Target: Raspberry Pi 4B+ (Edge AI Runtime)

Encapsulates:
  1. Causal Patient-Adaptive AFib Model (Config C: 21 absolute + 5 adaptive features)
  2. StandardScaler and Logistic Regression frozen from training set (afdb_5s_features_train.csv)
  3. CausalBaselineTracker (sliding 300s memory, anti-contamination, cold-start safe)
  4. CausalTemporalPersistenceFilter (2-consecutive window persistence filter)
  5. Polyphase Resampling (handles 250 Hz AFDB inputs or native 360 Hz ADS1292R windows)
  6. Robust Exception Isolation: inference failures return a safe fallback and do not crash the app.

CLINICAL NOTICE:
  This AFib classifier is an experimental research module and is NOT clinically validated.
  All outputs are intended solely for engineering prototyping and algorithmic research.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import joblib
except ImportError:
    joblib = None
import numpy as np
import scipy.signal as signal

# Ensure repository root and deploy directory are accessible
REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = Path(__file__).resolve().parent

for p in [REPO_ROOT, DEPLOY_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from deploy.afib_runtime.feature_extraction import extract_5s_features
    from deploy.afib_runtime.baseline_tracker import CausalBaselineTracker
    from deploy.afib_runtime.persistence_filter import CausalTemporalPersistenceFilter
except ImportError:
    try:
        from afib_runtime.feature_extraction import extract_5s_features
        from afib_runtime.baseline_tracker import CausalBaselineTracker
        from afib_runtime.persistence_filter import CausalTemporalPersistenceFilter
    except ImportError:
        extract_5s_features = None
        CausalBaselineTracker = None
        CausalTemporalPersistenceFilter = None


# Model artifact paths
NPZ_WEIGHTS_PATH = REPO_ROOT / "deploy" / "models" / "afib_lr_config_c_weights.npz"
DEPLOY_PACKAGE_PATH = REPO_ROOT / "deploy" / "models" / "afib_frozen_models_package.joblib"


class CausalAdaptiveAFibBridge:
    """
    Edge bridge for the causal patient-adaptive AFib classifier.
    Designed for real-time streaming execution on Raspberry Pi 4B+.
    Supports pure NumPy weights (no scikit-learn dependency) or joblib package.
    """

    def __init__(
        self,
        package_path: Optional[Path] = None,
        probability_threshold: float = 0.500,
        persistence_rule: str = "consecutive_2",
        fs_target: int = 360,
        strict: bool = False,
    ):
        self.fs_target = int(fs_target)
        self.threshold = float(probability_threshold)
        self.persistence_rule = str(persistence_rule)
        self.is_available = False
        self.unavailable_reason: Optional[str] = None
        self.model_path: Optional[Path] = None

        if extract_5s_features is None or CausalBaselineTracker is None or CausalTemporalPersistenceFilter is None:
            self.is_available = False
            self.unavailable_reason = "AFib runtime modules (feature_extraction, baseline_tracker, persistence_filter) unavailable."
            if strict:
                raise ImportError(self.unavailable_reason)
            return

        try:
            # 1. First priority: pure NumPy weights (zero sklearn requirement on Pi)
            if package_path is not None and str(package_path).endswith(".npz") and Path(package_path).exists():
                self.model_path = Path(package_path)
                self._load_from_npz(self.model_path)
            elif NPZ_WEIGHTS_PATH.exists():
                self.model_path = NPZ_WEIGHTS_PATH
                self._load_from_npz(self.model_path)
            # 2. Second priority: joblib package
            elif package_path is not None and Path(package_path).exists():
                self.model_path = Path(package_path)
                self._load_from_joblib(self.model_path)
            elif DEPLOY_PACKAGE_PATH.exists():
                self.model_path = DEPLOY_PACKAGE_PATH
                self._load_from_joblib(self.model_path)
            else:
                self.is_available = False
                self.unavailable_reason = (
                    f"Frozen AFib model weights not found. Checked:\n"
                    f"  - {NPZ_WEIGHTS_PATH}\n"
                    f"  - {package_path}\n"
                    f"  - {DEPLOY_PACKAGE_PATH}"
                )
                if strict:
                    raise FileNotFoundError(self.unavailable_reason)
                return
        except Exception as e:
            self.is_available = False
            self.unavailable_reason = f"Failed to load AFib model: {e}"
            if strict:
                raise
            return

        # Initialize causal trackers
        self.tracker = CausalBaselineTracker(
            history_window_sec=300.0,
            min_warmup_windows=12,
            max_gap_sec=300.0,
            anti_contamination=True,
            rate_percentile=25.0,
            cv_percentile=20.0,
            gating_cv_threshold=0.18,
        )
        self.persistence_filter = CausalTemporalPersistenceFilter(
            rule_name=self.persistence_rule,
            max_gap_sec=6.0,
        )

        self.last_prediction_time: Optional[float] = None
        self.window_counter = 0
        self.is_available = True

    def _load_from_npz(self, path: Path):
        """Loads weights into NumPy arrays for dependency-free edge execution."""
        data = np.load(path, allow_pickle=True)
        self.scaler_mean = data["scaler_mean"]
        self.scaler_scale = data["scaler_scale"]
        self.model_coef = data["model_coef"]
        self.model_intercept = data["model_intercept"]
        self.feature_cols = [str(c) for c in data["feature_cols"]]
        self.is_pure_numpy = True

    def _load_from_joblib(self, path: Path):
        """Loads from joblib scikit-learn package."""
        if joblib is None:
            raise ImportError(
                "joblib is required to load .joblib packages, but is not installed. "
                "Please install joblib or use .npz weights."
            )
        pkg = joblib.load(str(path))
        self.scaler = pkg["scaler_Config_C"]
        self.model = pkg["model_lr_Config_C"]
        self.feature_cols = pkg["feature_cols_C"]
        self.scaler_mean = self.scaler.mean_
        self.scaler_scale = self.scaler.scale_
        self.model_coef = self.model.coef_
        self.model_intercept = self.model.intercept_
        self.is_pure_numpy = False
        self.persistence_filter = CausalTemporalPersistenceFilter(
            rule_name=self.persistence_rule,
            max_gap_sec=6.0,
        )

        self.last_prediction_time: Optional[float] = None
        self.window_counter = 0

    def reset(self, new_record_id: Optional[str] = None):
        """Purges history at record/patient boundaries to prevent leakage."""
        if hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.reset(new_record_id=new_record_id)
        if hasattr(self, "persistence_filter") and self.persistence_filter is not None:
            self.persistence_filter.reset(new_record_id=new_record_id)
        self.last_prediction_time = None
        self.window_counter = 0

    def resample_if_needed(self, sig: np.ndarray, fs: int) -> np.ndarray:
        """
        Ensures input window is calibrated to target sampling rate (360 Hz).
        If 250 Hz (AFDB native), applies rational polyphase resampling (up=36, down=25).
        """
        if fs == self.fs_target and len(sig) == int(self.fs_target * 5.0):
            return sig.astype(np.float32)

        if fs == 250 or len(sig) == 1250:
            # Polyphase anti-aliasing resampling: 250 Hz -> 360 Hz (up=36, down=25)
            sig_360 = signal.resample_poly(sig, up=36, down=25).astype(np.float32)
            expected_samples = int(self.fs_target * 5.0)  # 1800
            if len(sig_360) != expected_samples:
                sig_360 = sig_360[:expected_samples] if len(sig_360) > expected_samples else np.pad(sig_360, (0, expected_samples - len(sig_360)))
            return sig_360

        # General resampling if sampling rate differs from target
        if fs != self.fs_target:
            target_samples = int(self.fs_target * 5.0)
            return signal.resample(sig, target_samples).astype(np.float32)

        return sig.astype(np.float32)

    def predict_window(
        self,
        sig_window: np.ndarray,
        fs: int = 360,
        current_time_sec: Optional[float] = None,
        is_valid_signal: bool = True,
        record_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes causal 5-second AFib inference.

        Returns dictionary with:
          - raw_probability: Instantaneous logistic regression probability
          - instantaneous_pred: Raw threshold decision (raw_prob >= 0.500)
          - afib_alert: Persisted decision via 2-consecutive consensus
          - baseline_features: 5 causal adaptive baseline features
          - tracker_warmup_fraction: Fraction of warm-up completed
          - latency_ms: Execution time in milliseconds
          - status: "OK", "WARMING_UP", "SIGNAL_INVALID", "UNAVAILABLE", or "ERROR"
          - is_available: bool
        """
        t0 = time.perf_counter()

        if not getattr(self, "is_available", False):
            return {
                "raw_probability": 0.0,
                "instantaneous_pred": False,
                "afib_alert": False,
                "threshold": self.threshold,
                "persistence_rule": self.persistence_rule,
                "baseline_features": {},
                "tracker_warmup_fraction": 0.0,
                "is_available": False,
                "status": "UNAVAILABLE",
                "error_detail": getattr(self, "unavailable_reason", "AFib model unavailable"),
                "latency_ms": 0.0,
                "notice": "AFib branch is unavailable; AFib inference bypassed.",
            }

        self.window_counter += 1
        if current_time_sec is None:
            current_time_sec = float(self.window_counter * 5.0)

        # Isolated execution guard to prevent crashing edge runtime
        try:
            if not is_valid_signal:
                # Corrupted/invalid signal: do not corrupt baseline tracker
                return {
                    "raw_probability": 0.0,
                    "instantaneous_pred": False,
                    "afib_alert": False,
                    "threshold": self.threshold,
                    "persistence_rule": self.persistence_rule,
                    "baseline_features": {},
                    "tracker_warmup_fraction": 0.0,
                    "is_available": True,
                    "status": "SIGNAL_INVALID",
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "notice": "Signal quality check failed; AFib inference bypassed.",
                }

            # 1. Resample to 360 Hz if needed
            w_sig_360 = self.resample_if_needed(sig_window, fs=fs)

            # 2. Extract 21 absolute features
            abs_feats = extract_5s_features(w_sig_360, fs=self.fs_target)

            # 3. Update causal baseline tracker (strictly using previous history)
            adapt_feats = self.tracker.process_window(
                current_time_sec=current_time_sec,
                hr_mean_bpm=float(abs_feats["hr_mean_bpm"]),
                rr_cv=float(abs_feats["rr_cv"]),
                f_band_power_ratio=float(abs_feats["f_band_power_ratio"]),
                n_beats=int(abs_feats["n_beats"]),
                is_valid_window=True,
                record_id=record_id,
            )

            # 4. Construct 26-dimensional combined feature vector
            all_feats = {**abs_feats, **adapt_feats}
            x_vec = np.array([[all_feats[c] for c in self.feature_cols]], dtype=np.float32)

            # 5. Model inference (StandardScaler + Logistic Regression)
            if hasattr(self, "model_coef") and self.model_coef is not None:
                # Pure NumPy evaluation (zero scikit-learn runtime overhead)
                z = np.dot((x_vec - self.scaler_mean) / self.scaler_scale, self.model_coef.T) + self.model_intercept
                p_afib = float(1.0 / (1.0 + np.exp(-z[0, 0])))
            else:
                x_scaled = self.scaler.transform(x_vec)
                p_afib = float(self.model.predict_proba(x_scaled)[0, 1])
            instant_pred = int(p_afib >= self.threshold)

            # 6. Apply causal temporal persistence filter
            persisted_alert = bool(
                self.persistence_filter.process_window(
                    timestamp_sec=current_time_sec,
                    raw_prob=p_afib,
                    raw_decision=instant_pred,
                    record_id=record_id,
                )
            )

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.last_prediction_time = current_time_sec

            warmup_frac = float(adapt_feats.get("baseline_warmup_fraction", 0.0))
            status = "WARMING_UP" if warmup_frac < 1.0 else "OK"

            return {
                "raw_probability": round(p_afib, 4),
                "instantaneous_pred": bool(instant_pred),
                "afib_alert": persisted_alert,
                "threshold": self.threshold,
                "persistence_rule": self.persistence_rule,
                "baseline_features": adapt_feats,
                "tracker_warmup_fraction": warmup_frac,
                "is_available": True,
                "status": status,
                "latency_ms": round(dt_ms, 2),
                "features_summary": {
                    "hr_bpm": abs_feats["hr_mean_bpm"],
                    "rr_cv": abs_feats["rr_cv"],
                    "f_band": abs_feats["f_band_power_ratio"],
                    "delta_hr": adapt_feats["delta_hr_bpm"],
                    "afib_index_rel": adapt_feats["afib_index_relative"],
                },
            }

        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "raw_probability": 0.0,
                "instantaneous_pred": False,
                "afib_alert": False,
                "threshold": self.threshold,
                "persistence_rule": self.persistence_rule,
                "baseline_features": {},
                "tracker_warmup_fraction": 0.0,
                "is_available": False,
                "status": "ERROR",
                "error_detail": str(e),
                "latency_ms": round(dt_ms, 2),
                "notice": f"AFib processing error; AFib branch unavailable ({e}).",
            }
