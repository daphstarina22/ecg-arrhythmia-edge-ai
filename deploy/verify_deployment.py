"""Lightweight deployment verification and self-test script for Raspberry Pi.

Verifies:
  1. Python runtime and dependencies (numpy, scipy, onnxruntime, joblib)
  2. Existence of all three ONNX candidate models
  3. Model loading and graph inspection (input/output metadata)
  4. Safe synthetic inference through all three ONNX models
  5. Patient-Adaptive AFib Bridge (Config C 26-features, tracker, persistence, 250->360Hz resampling)
  6. Unified 9-label Decision Layer execution on 1,800-sample window
"""

import sys
import time
from pathlib import Path

# Resolve base directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = SCRIPT_DIR / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = PROJECT_ROOT / "training" / "final_models" / "export_candidates"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_MODELS = {
    "Model 1 (Shockable Screener)": {
        "filename": "shockable_classifier_candidate.onnx",
        "expected_n_features": 10,
        "expected_classes": 2,
    },
    "Model 2 (Multi-Arrhythmia 4-Class)": {
        "filename": "arrhythmia_multiclass_4class_candidate.onnx",
        "expected_n_features": 12,
        "expected_classes": 4,
    },
    "Model 3 (VT vs VF Subtype Specialist)": {
        "filename": "vf_vt_subtype_classifier_candidate.onnx",
        "expected_n_features": 6,
        "expected_classes": 2,
    },
}


def run_checks() -> bool:
    all_passed = True
    print("=" * 78)
    print("  RASPBERRY PI DEPLOYMENT VERIFICATION & UNIFIED SELF-TEST")
    print("=" * 78)
    print(f"Python Version : {sys.version.split()[0]} ({sys.executable})")
    print(f"Models Dir     : {MODELS_DIR}")
    print("-" * 78)

    # 1. Dependency checks
    print("[CHECK 1/6] Verifying core edge dependencies...")
    for pkg in ["numpy", "scipy", "onnxruntime", "joblib"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [PASS] {pkg:<14} : {ver}")
        except ImportError as e:
            print(f"  [FAIL] {pkg:<14} : NOT FOUND ({e})")
            all_passed = False

    if not all_passed:
        print("\nERROR: Required dependencies are missing. Run: pip install -r deploy/requirements-pi.txt")
        return False

    import numpy as np
    import onnxruntime as ort

    # 2. File existence checks
    print("\n[CHECK 2/6] Verifying ONNX model candidate files...")
    sessions = {}
    for name, spec in EXPECTED_MODELS.items():
        path = MODELS_DIR / spec["filename"]
        if path.exists():
            size_kb = path.stat().st_size / 1024.0
            print(f"  [PASS] {spec['filename']:<44} ({size_kb:,.1f} KB)")
        else:
            print(f"  [FAIL] {spec['filename']:<44} NOT FOUND in {MODELS_DIR}")
            all_passed = False

    if not all_passed:
        return False

    # 3. Model loading and metadata inspection
    print("\n[CHECK 3/6] Loading ONNX runtime sessions & inspecting graphs...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    for name, spec in EXPECTED_MODELS.items():
        path = MODELS_DIR / spec["filename"]
        try:
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            out = sess.get_outputs()
            sessions[name] = sess
            print(f"  [PASS] {name}:")
            print(f"         Input  : name='{inp.name}', shape={inp.shape}, type={inp.type}")
            print(f"         Outputs: {[o.name + ':' + str(o.shape) for o in out]}")
        except Exception as e:
            print(f"  [FAIL] {name}: Failed to initialize session ({e})")
            all_passed = False

    if not all_passed:
        return False

    # 4. Synthetic inference check (3 ONNX models)
    print("\n[CHECK 4/6] Executing safe synthetic inference tests on ONNX models...")
    for name, spec in EXPECTED_MODELS.items():
        sess = sessions[name]
        n_feat = spec["expected_n_features"]
        dummy = np.zeros((1, n_feat), dtype=np.float32)
        try:
            t0 = time.perf_counter()
            outputs = sess.run(None, {sess.get_inputs()[0].name: dummy})
            dt_ms = (time.perf_counter() - t0) * 1000.0
            pred_label = outputs[0]
            probs = outputs[1]
            print(f"  [PASS] {name} ({n_feat} features):")
            print(f"         Latency: {dt_ms:.2f} ms | Output Label: {pred_label} | Probs: {probs}")
        except Exception as e:
            print(f"  [FAIL] {name}: Inference failed ({e})")
            all_passed = False

    # 5. AFib bridge verification
    print("\n[CHECK 5/6] Verifying Patient-Adaptive AFib Bridge & Resampling...")
    try:
        from deploy.afib_bridge import CausalAdaptiveAFibBridge
        af_bridge = CausalAdaptiveAFibBridge(fs_target=360)
        print(f"  [PASS] Loaded AFib model package: {af_bridge.model_path.name}")
        print(f"         Feature schema: {len(af_bridge.feature_cols)} features ({af_bridge.feature_cols[:4]} ... {af_bridge.feature_cols[-2:]})")

        # Test native 360 Hz window (1800 samples)
        dummy_360 = np.random.randn(1800).astype(np.float32)
        t0 = time.perf_counter()
        out_360 = af_bridge.predict_window(dummy_360, fs=360, current_time_sec=5.0)
        dt_360 = (time.perf_counter() - t0) * 1000.0
        print(f"  [PASS] 360 Hz Window: Latency={dt_360:.2f} ms | P(AFib)={out_360['raw_probability']:.4f} | Alert={out_360['afib_alert']}")

        # Test 250 Hz window (1250 samples) -> polyphase resampled to 360 Hz
        dummy_250 = np.random.randn(1250).astype(np.float32)
        t0 = time.perf_counter()
        out_250 = af_bridge.predict_window(dummy_250, fs=250, current_time_sec=10.0)
        dt_250 = (time.perf_counter() - t0) * 1000.0
        print(f"  [PASS] 250 Hz Resampling: Latency={dt_250:.2f} ms | P(AFib)={out_250['raw_probability']:.4f} | Alert={out_250['afib_alert']}")

    except Exception as e:
        print(f"  [FAIL] AFib Bridge verification failed: {e}")
        all_passed = False

    # 6. Unified decision layer verification
    print("\n[CHECK 6/6] Verifying Unified Decision Layer and Complete Runtime Cascade...")
    try:
        from deploy.edge_runtime import HierarchicalEdgeClassifier
        from deploy.unified_decision_layer import UnifiedRhythmLabel

        clf = HierarchicalEdgeClassifier()
        dummy_win = np.random.randn(1800).astype(np.float32)
        t0 = time.perf_counter()
        decision = clf.predict_window(dummy_win)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        valid_labels = [label.value for label in UnifiedRhythmLabel]
        if decision.primary_rhythm in valid_labels:
            print(f"  [PASS] Full Unified Cascade executed successfully:")
            print(f"         Total Latency : {dt_ms:.2f} ms (Cycle: {decision.latencies_ms.get('total_cycle', 0.0):.2f} ms)")
            print(f"         Output Label  : {decision.primary_rhythm}")
            print(f"         Triage Level  : {decision.triage_level}")
            print(f"         AED Shock     : {'YES' if decision.defibrillator_advised else 'NO'}")
        else:
            print(f"  [FAIL] Output label '{decision.primary_rhythm}' is not in the 9 unified labels: {valid_labels}")
            all_passed = False

    except Exception as e:
        print(f"  [FAIL] Unified cascade verification failed: {e}")
        all_passed = False

    print("=" * 78)
    if all_passed:
        print("  STATUS: ALL DEPLOYMENT CHECKS PASSED [OK]")
        print("  The Raspberry Pi edge runtime environment is verified and ready.")
        print("=" * 78)
        return True
    else:
        print("  STATUS: DEPLOYMENT CHECKS FAILED [ERROR]")
        print("=" * 78)
        return False


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
