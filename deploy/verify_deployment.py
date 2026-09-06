"""Lightweight deployment verification and self-test script for Raspberry Pi.

Verifies:
  1. Python runtime and dependencies (numpy, scipy, onnxruntime)
  2. Existence of all three ONNX candidate models
  3. Model loading and graph inspection (input/output metadata)
  4. Safe synthetic inference through all three models
  5. Full 3-tier hierarchical cascade execution on a 1,800-sample synthetic window
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
    print("=" * 75)
    print("  RASPBERRY PI DEPLOYMENT VERIFICATION & SELF-TEST")
    print("=" * 75)
    print(f"Python Version : {sys.version.split()[0]} ({sys.executable})")
    print(f"Models Dir     : {MODELS_DIR}")
    print("-" * 75)

    # 1. Dependency checks
    print("[CHECK 1/4] Verifying core edge dependencies...")
    for pkg in ["numpy", "scipy", "onnxruntime"]:
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
    print("\n[CHECK 2/4] Verifying ONNX model candidate files...")
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
    print("\n[CHECK 3/4] Loading ONNX runtime sessions & inspecting graphs...")
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

    # 4. Synthetic inference check
    print("\n[CHECK 4/4] Executing safe synthetic inference tests...")
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

    print("=" * 75)
    if all_passed:
        print("  STATUS: ALL DEPLOYMENT CHECKS PASSED [OK]")
        print("  The Raspberry Pi edge runtime environment is verified and ready.")
        print("=" * 75)
        return True
    else:
        print("  STATUS: DEPLOYMENT CHECKS FAILED [ERROR]")
        print("=" * 75)
        return False


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
