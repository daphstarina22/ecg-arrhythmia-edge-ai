"""
PVC Beat Research Prototype (Isolated Module)
==============================================
Evaluates Normal (N) vs. Premature Ventricular Contraction (V) beat classification
using verified local MIT-BIH Arrhythmia Database annotations.

SAFETY & ARCHITECTURAL ISOLATION:
- Standalone research prototype; NOT imported by deploy/ or production pipelines.
- Zero modifications to ADS1292R drivers, ONNX models, or edge runtime.
- Strictly leak-free: patient-level record grouping ensures beats from the same
  subject never co-exist in training, validation, or test sets.
"""

import os
import zipfile
import collections
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)

# Physical constants
TARGET_FS = 360  # Native MIT-BIH sampling rate (Hz)
PRE_PEAK_MS = 200.0   # Window duration before annotated peak
POST_PEAK_MS = 300.0  # Window duration after annotated peak
PRE_SAMPLES = int(round(PRE_PEAK_MS / 1000.0 * TARGET_FS))   # 72 samples
POST_SAMPLES = int(round(POST_PEAK_MS / 1000.0 * TARGET_FS)) # 108 samples
TOTAL_WINDOW_SAMPLES = PRE_SAMPLES + POST_SAMPLES             # 180 samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MITBIH_ZIP = PROJECT_ROOT / "MITBIH.zip"


# ======================================================================
# 1. PATIENT-LEVEL RECORD GROUPING & SPLITTING
# ======================================================================
# Records 201 and 202 originate from the same patient (male 68).
# Records 102 and 104 lack Lead MLII (have only V5/V2) and are excluded from MLII evaluation.
SUBJECT_GROUPS = {
    # Key: Subject ID, Value: List of Record IDs
    "sub_100": ["100"], "sub_101": ["101"], "sub_103": ["103"], "sub_105": ["105"],
    "sub_106": ["106"], "sub_107": ["107"], "sub_108": ["108"], "sub_109": ["109"],
    "sub_111": ["111"], "sub_112": ["112"], "sub_113": ["113"], "sub_114": ["114"],
    "sub_115": ["115"], "sub_116": ["116"], "sub_117": ["117"], "sub_118": ["118"],
    "sub_119": ["119"], "sub_121": ["121"], "sub_122": ["122"], "sub_123": ["123"],
    "sub_124": ["124"], "sub_200": ["200"],
    "sub_201_202": ["201", "202"], # Shared patient constraint
    "sub_203": ["203"], "sub_205": ["205"], "sub_207": ["207"], "sub_208": ["208"],
    "sub_209": ["209"], "sub_210": ["210"], "sub_212": ["212"], "sub_213": ["213"],
    "sub_214": ["214"], "sub_215": ["215"], "sub_217": ["217"], "sub_219": ["219"],
    "sub_220": ["220"], "sub_221": ["221"], "sub_222": ["222"], "sub_223": ["223"],
    "sub_228": ["228"], "sub_230": ["230"], "sub_231": ["231"], "sub_232": ["232"],
    "sub_233": ["233"], "sub_234": ["234"],
}


def create_patient_stratified_split(
    record_v_counts: Dict[str, int],
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Partitions subjects into Train (approx 60%), Val (approx 20%), Test (approx 20%)
    ensuring records from the same patient never cross splits and V-beat distribution is balanced.
    """
    rng = np.random.RandomState(seed)

    # Classify subjects into: High V (>= 100), Medium V (1-99), Zero V (0)
    high_v_subs, med_v_subs, zero_v_subs = [], [], []
    for sub, recs in SUBJECT_GROUPS.items():
        v_sum = sum(record_v_counts.get(r, 0) for r in recs)
        if v_sum >= 100:
            high_v_subs.append(sub)
        elif v_sum > 0:
            med_v_subs.append(sub)
        else:
            zero_v_subs.append(sub)

    rng.shuffle(high_v_subs)
    rng.shuffle(med_v_subs)
    rng.shuffle(zero_v_subs)

    def split_list(lst: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(lst)
        n_train = int(round(n * 0.60))
        n_val = int(round(n * 0.20))
        train = lst[:n_train]
        val = lst[n_train : n_train + n_val]
        test = lst[n_train + n_val :]
        return train, val, test

    tr_h, val_h, te_h = split_list(high_v_subs)
    tr_m, val_m, te_m = split_list(med_v_subs)
    tr_z, val_z, te_z = split_list(zero_v_subs)

    train_subs = tr_h + tr_m + tr_z
    val_subs = val_h + val_m + val_z
    test_subs = te_h + te_m + te_z

    # Expand subjects to records
    train_recs = sorted([r for s in train_subs for r in SUBJECT_GROUPS[s]])
    val_recs = sorted([r for s in val_subs for r in SUBJECT_GROUPS[s]])
    test_recs = sorted([r for s in test_subs for r in SUBJECT_GROUPS[s]])

    return train_recs, val_recs, test_recs


# ======================================================================
# 2. FEATURE EXTRACTION FOR BEAT WINDOWS
# ======================================================================

def extract_beat_features(
    window_mv: np.ndarray,
    pre_rr_sec: float,
    post_rr_sec: float
) -> Dict[str, float]:
    """
    Extracts 30 physiological and morphological features from a single 180-sample beat window.
    """
    # 1. Baseline leveling: subtract mean of the first 15 samples (pre-P wave baseline)
    base = float(np.mean(window_mv[:15]))
    sig = window_mv - base

    ptp_val = float(np.ptp(sig))
    std_val = float(np.std(sig))
    max_val = float(np.max(sig))
    min_val = float(np.min(sig))
    energy = float(np.sum(sig ** 2))
    zero_cross = int(np.sum(np.diff(sig > 0) != 0))

    # QRS timing & symmetry
    peak_idx = int(np.argmax(sig))
    trough_idx = int(np.argmin(sig))
    skew_val = float(skew(sig))
    kurt_val = float(kurtosis(sig))

    # Slope features
    diff1 = np.diff(sig)
    max_pos_slope = float(np.max(diff1)) if len(diff1) > 0 else 0.0
    max_neg_slope = float(np.min(diff1)) if len(diff1) > 0 else 0.0

    # RR timing metrics
    rr_ratio = float(pre_rr_sec / (post_rr_sec + 1e-4))
    local_hr = float(60.0 / (pre_rr_sec + 1e-4))

    feats = {
        "ptp_mv": ptp_val,
        "std_mv": std_val,
        "max_mv": max_val,
        "min_mv": min_val,
        "energy": energy,
        "zero_cross": zero_cross,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "max_pos_slope": max_pos_slope,
        "max_neg_slope": max_neg_slope,
        "pre_rr_sec": pre_rr_sec,
        "post_rr_sec": post_rr_sec,
        "rr_ratio": rr_ratio,
        "local_hr": local_hr,
    }

    # Decimated waveform morphology (14 points: 180 samples downsampled by ~13)
    decimated = sig[::13][:14]
    for idx, val in enumerate(decimated):
        feats[f"morph_{idx}"] = float(val)

    return feats


# ======================================================================
# 3. DATASET LOADER (STREAMING DIRECTLY FROM ZIP)
# ======================================================================

def load_record_beats(
    zip_ref: zipfile.ZipFile,
    record_id: str
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extracts beat windows and annotations for a single MIT-BIH record.
    Returns (valid_beats, excluded_symbol_counts).
    """
    csv_name = f"{record_id}.csv"
    ann_name = f"{record_id}annotations.txt"

    with zip_ref.open(csv_name) as f:
        df_csv = pd.read_csv(f)

    # Lead selection: require MLII
    mlii_cols = [c for c in df_csv.columns if "MLII" in c]
    if not mlii_cols:
        return [], {"NO_MLII": len(df_csv)}

    raw_sig = df_csv[mlii_cols[0]].values.astype(np.float64)
    # Physical mV calibration: gain = 200, baseline = 1024
    calib_sig = (raw_sig - 1024.0) / 200.0
    sig_len = len(calib_sig)

    with zip_ref.open(ann_name) as f:
        ann_lines = f.read().decode("utf-8", errors="replace").splitlines()

    # Parse all annotated beats
    annotations = []
    excluded_counts = collections.Counter()

    for line in ann_lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                sample_idx = int(parts[1])
                symbol = parts[2]
                annotations.append((sample_idx, symbol))
            except ValueError:
                pass

    valid_samples = [s for s, sym in annotations if sym in ("N", "V")]
    if len(valid_samples) < 2:
        return [], excluded_counts

    # Calculate median RR for edge boundaries
    diffs = np.diff(valid_samples) / TARGET_FS
    median_rr = float(np.median(diffs)) if len(diffs) > 0 else 0.833

    beats = []
    for i, (sample_idx, sym) in enumerate(annotations):
        if sym not in ("N", "V"):
            excluded_counts[sym] += 1
            continue

        # Boundary safety check
        if sample_idx < PRE_SAMPLES or sample_idx + POST_SAMPLES > sig_len:
            excluded_counts["BOUNDARY_OVERFLOW"] += 1
            continue

        # RR interval calculation
        pre_rr = (sample_idx - annotations[i - 1][0]) / TARGET_FS if i > 0 else median_rr
        post_rr = (annotations[i + 1][0] - sample_idx) / TARGET_FS if i < len(annotations) - 1 else median_rr

        # Sanity bound RR intervals (0.15s to 4.0s)
        pre_rr = float(np.clip(pre_rr, 0.15, 4.0))
        post_rr = float(np.clip(post_rr, 0.15, 4.0))

        # Extract 180-sample window
        win_mv = calib_sig[sample_idx - PRE_SAMPLES : sample_idx + POST_SAMPLES]
        feat_dict = extract_beat_features(win_mv, pre_rr, post_rr)
        feat_dict["record_id"] = record_id
        feat_dict["sample_idx"] = sample_idx
        feat_dict["symbol"] = sym
        feat_dict["label"] = 1 if sym == "V" else 0  # Binary target: 1=V (PVC), 0=N (Normal)

        beats.append(feat_dict)

    return beats, excluded_counts


# ======================================================================
# 4. TRAINING & EVALUATION PIPELINE
# ======================================================================

def run_pvc_prototype_experiment() -> Dict[str, Any]:
    """
    Executes a complete, leak-free, patient-stratified Normal vs. PVC research benchmark.
    """
    print("===========================================================================")
    print("  MIT-BIH PVC BEAT CLASSIFICATION RESEARCH PROTOTYPE")
    print("===========================================================================\n")

    if not MITBIH_ZIP.exists():
        raise FileNotFoundError(f"MIT-BIH dataset archive not found at: {MITBIH_ZIP}")

    # First pass: collect PVC counts per record to stratify subjects
    print("[1/5] Auditing records and building patient-level split...")
    rec_v_counts = {}
    with zipfile.ZipFile(MITBIH_ZIP) as z:
        for fname in z.namelist():
            if fname.endswith("annotations.txt") and "/" not in fname:
                rec = fname.replace("annotations.txt", "")
                content = z.read(fname).decode("utf-8", errors="replace")
                v_count = sum(1 for line in content.splitlines()[1:] if len(line.split()) >= 3 and line.split()[2] == "V")
                rec_v_counts[rec] = v_count

    train_recs, val_recs, test_recs = create_patient_stratified_split(rec_v_counts, seed=42)
    print(f"  Train Records ({len(train_recs)}): {train_recs}")
    print(f"  Val Records   ({len(val_recs)}): {val_recs}")
    print(f"  Test Records  ({len(test_recs)}): {test_recs}")

    # Verify zero leakage across splits
    assert not (set(train_recs) & set(val_recs)), "Train and Val records overlap!"
    assert not (set(train_recs) & set(test_recs)), "Train and Test records overlap!"
    assert not (set(val_recs) & set(test_recs)), "Val and Test records overlap!"
    print("  Patient-level leakage verification: PASSED (0 record overlap).")

    # Second pass: Extract beats per split
    print("\n[2/5] Extracting beat windows and morphological features...")
    all_data = {"train": [], "val": [], "test": []}
    total_excluded = collections.Counter()

    with zipfile.ZipFile(MITBIH_ZIP) as z:
        for split_name, rec_list in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
            for rec in rec_list:
                beats, excl = load_record_beats(z, rec)
                all_data[split_name].extend(beats)
                for k, v in excl.items():
                    total_excluded[k] += v

    print(f"\n[3/5] Annotation Symbol Filtering Summary (Unsupported symbols excluded):")
    for sym, count in total_excluded.most_common():
        print(f"  Symbol '{sym}': {count:>6} annotations excluded")

    # Build DataFrames
    df_train = pd.DataFrame(all_data["train"])
    df_val = pd.DataFrame(all_data["val"])
    df_test = pd.DataFrame(all_data["test"])

    meta_cols = ["record_id", "sample_idx", "symbol", "label"]
    feature_cols = [c for c in df_train.columns if c not in meta_cols]

    print(f"\n[4/5] Class Distribution Across Patient-Level Splits:")
    for split_label, df_s in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        n_normal = int(np.sum(df_s["label"] == 0))
        n_pvc = int(np.sum(df_s["label"] == 1))
        v_pct = (n_pvc / len(df_s)) * 100.0 if len(df_s) > 0 else 0.0
        print(f"  {split_label:<10}: Total {len(df_s):>5} beats | Normal: {n_normal:>5} | PVC (V): {n_pvc:>5} ({v_pct:>4.1f}%)")

    # Scale features strictly on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    y_train = df_train["label"].values

    X_val = scaler.transform(df_val[feature_cols].values)
    y_val = df_val["label"].values

    X_test = scaler.transform(df_test[feature_cols].values)
    y_test = df_test["label"].values

    print(f"\n[5/5] Training Classifiers (Evaluated on Held-Out Test Patients)...")

    # Baseline 1: Majority Class (Always predict Normal = 0)
    maj_pred = np.zeros_like(y_test)
    maj_acc = accuracy_score(y_test, maj_pred)
    maj_bal_acc = balanced_accuracy_score(y_test, maj_pred)
    print(f"\n  A. Majority Class Baseline (Predicts 100% Normal):")
    print(f"     Accuracy: {maj_acc * 100:.2f}% | Balanced Accuracy: {maj_bal_acc * 100:.2f}% | PVC Recall: 0.0%")

    # Model 1: Balanced Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_bal_acc = balanced_accuracy_score(y_test, lr_pred)
    lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average=None, labels=[0, 1])

    print(f"\n  B. Balanced Logistic Regression:")
    print(f"     Accuracy: {lr_acc * 100:.2f}% | Balanced Accuracy: {lr_bal_acc * 100:.2f}%")
    print(f"     Normal (N) - Prec: {lr_p[0]:.4f}, Rec: {lr_r[0]:.4f}, F1: {lr_f1[0]:.4f}")
    print(f"     PVC    (V) - Prec: {lr_p[1]:.4f}, Rec: {lr_r[1]:.4f}, F1: {lr_f1[1]:.4f}")
    print(f"     Confusion Matrix:\n{confusion_matrix(y_test, lr_pred)}")

    # Model 2: Balanced Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_bal_acc = balanced_accuracy_score(y_test, rf_pred)
    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average=None, labels=[0, 1])

    print(f"\n  C. Balanced Random Forest (100 Trees, Depth 8):")
    print(f"     Accuracy: {rf_acc * 100:.2f}% | Balanced Accuracy: {rf_bal_acc * 100:.2f}%")
    print(f"     Normal (N) - Prec: {rf_p[0]:.4f}, Rec: {rf_r[0]:.4f}, F1: {rf_f1[0]:.4f}")
    print(f"     PVC    (V) - Prec: {rf_p[1]:.4f}, Rec: {rf_r[1]:.4f}, F1: {rf_f1[1]:.4f}")
    cm_rf = confusion_matrix(y_test, rf_pred)
    print(f"     Confusion Matrix:\n{cm_rf}")

    # Per-Record Breakdown on Held-Out Test Set
    print(f"\n=== PER-RECORD PERFORMANCE BREAKDOWN ON HELD-OUT TEST SET (Random Forest) ===")
    df_test_res = df_test[["record_id", "symbol", "label"]].copy()
    df_test_res["pred"] = rf_pred

    record_summary = []
    for rec, group in df_test_res.groupby("record_id"):
        n_tot = len(group)
        n_v = int(np.sum(group["label"] == 1))
        n_n = int(np.sum(group["label"] == 0))
        acc = accuracy_score(group["label"], group["pred"])
        bal_acc = balanced_accuracy_score(group["label"], group["pred"]) if n_v > 0 and n_n > 0 else acc
        v_rec = float(np.sum((group["label"] == 1) & (group["pred"] == 1)) / n_v) if n_v > 0 else 1.0
        v_prec = float(np.sum((group["label"] == 1) & (group["pred"] == 1)) / np.sum(group["pred"] == 1)) if np.sum(group["pred"] == 1) > 0 else 0.0
        record_summary.append({
            "record": rec,
            "total_beats": n_tot,
            "n_normal": n_n,
            "n_pvc": n_v,
            "accuracy": round(acc * 100, 1),
            "balanced_acc": round(bal_acc * 100, 1),
            "pvc_recall": round(v_rec * 100, 1) if n_v > 0 else "N/A",
            "pvc_precision": round(v_prec * 100, 1) if n_v > 0 else "N/A",
        })
        print(f"  Record {rec:>3}: Total {n_tot:>4} | V: {n_v:>4} | Acc: {acc*100:>5.1f}% | V-Recall: {v_rec*100 if n_v > 0 else 'N/A'}% | V-Prec: {v_prec*100 if n_v > 0 else 'N/A'}%")

    results = {
        "train_records": train_recs,
        "val_records": val_recs,
        "test_records": test_recs,
        "class_counts": {
            "train": {"normal": int(np.sum(y_train == 0)), "pvc": int(np.sum(y_train == 1))},
            "val": {"normal": int(np.sum(y_val == 0)), "pvc": int(np.sum(y_val == 1))},
            "test": {"normal": int(np.sum(y_test == 0)), "pvc": int(np.sum(y_test == 1))},
        },
        "majority_acc": maj_acc,
        "rf_metrics": {
            "accuracy": rf_acc,
            "balanced_acc": rf_bal_acc,
            "normal_p": rf_p[0], "normal_r": rf_r[0], "normal_f1": rf_f1[0],
            "pvc_p": rf_p[1], "pvc_r": rf_r[1], "pvc_f1": rf_f1[1],
            "confusion_matrix": cm_rf.tolist(),
        },
        "per_record_summary": record_summary,
    }

    return results


if __name__ == "__main__":
    run_pvc_prototype_experiment()
