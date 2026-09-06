"""Data loading, caching, and dataset assembly for the three final models."""

from pathlib import Path
import pandas as pd
import numpy as np

from .feature_selection import (
    MODEL1_FEATURES,
    MODEL2_FEATURES,
    MODEL3_SET_A_FULL,
    MODEL3_SET_B_DOMAIN_STABLE,
    MODEL3_SET_C_MINIMAL_EDGE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_model3_dataset(
    csv_path: Path | str | None = None,
    feature_set: str = "set_a_full",
) -> tuple[pd.DataFrame, list[str]]:
    """Loads the validated Phase 6B/7 expanded physiological feature dataset for Model 3.

    Target mapping:
      0 = Ventricular Tachycardia (VT)
      1 = Ventricular Fibrillation / Flutter (VF/VFL)
    """
    if csv_path is None:
        csv_path = (
            PROJECT_ROOT
            / "training"
            / "model3_phase6"
            / "artifacts"
            / "expanded_physiological_feature_table.csv"
        )
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Model 3 dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate target column
    if "target" not in df.columns:
        df["target"] = (df["label"] == "VF").astype(int)

    feature_map = {
        "set_a_full": MODEL3_SET_A_FULL,
        "set_b_domain_stable": MODEL3_SET_B_DOMAIN_STABLE,
        "set_c_minimal_edge": MODEL3_SET_C_MINIMAL_EDGE,
    }
    selected_features = feature_map.get(feature_set, MODEL3_SET_A_FULL)

    # Ensure all selected features exist in dataframe
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing required features in Model 3 table: {missing}")

    return df, selected_features


def load_model1_dataset(
    artifact_path: Path | str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Loads or prepares the balanced multi-source dataset for Model 1 (Shockable vs Non-Shockable).

    Target mapping:
      0 = Non-Shockable (Normal, Tachycardia, Bradycardia, Asystole, SVTA, AFIB)
      1 = Shockable (VT, VF, VFL)
    """
    if artifact_path is None:
        artifact_path = (
            PROJECT_ROOT
            / "training"
            / "final_models"
            / "artifacts"
            / "model1_shockable_dataset.csv"
        )
    artifact_path = Path(artifact_path)
    if artifact_path.exists():
        df = pd.read_csv(artifact_path)
        return df, MODEL1_FEATURES

    # If artifact does not yet exist, return empty or raise notice for preparation step
    raise FileNotFoundError(
        f"Model 1 dataset artifact not found at {artifact_path}. Run dataset preparation first."
    )


def load_model2_dataset(
    artifact_path: Path | str | None = None,
    taxonomy: str = "4class",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Loads the balanced multi-source dataset for Model 2 (Multi-Arrhythmia).

    Taxonomies supported:
      '4class': [NSR, TACHY, BRADY_ASY, VENTRICULAR]
      '5class': [NSR, TACHY, BRADY_ASY, VT, VF_VFL]
    """
    if artifact_path is None:
        filename = (
            "model2_multiclass_4class_dataset.csv"
            if taxonomy == "4class"
            else "model2_multiclass_5class_dataset.csv"
        )
        artifact_path = (
            PROJECT_ROOT
            / "training"
            / "final_models"
            / "artifacts"
            / filename
        )
    artifact_path = Path(artifact_path)
    if artifact_path.exists():
        df = pd.read_csv(artifact_path)
        class_names = (
            ["NSR", "TACHY", "BRADY_ASY", "VENTRICULAR"]
            if taxonomy == "4class"
            else ["NSR", "TACHY", "BRADY_ASY", "VT", "VF_VFL"]
        )
        return df, MODEL2_FEATURES, class_names

    raise FileNotFoundError(
        f"Model 2 dataset artifact not found at {artifact_path}. Run dataset preparation first."
    )
