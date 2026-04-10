from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ShapArtifacts:
    shap_values: np.ndarray
    expected_value: float
    instance_index: int


def _extract_positive_class_shap(
    shap_values_raw,
    expected_value_raw,
    positive_class_idx: int = 1,
) -> Tuple[np.ndarray, float]:
    if isinstance(shap_values_raw, list):
        shap_values = np.asarray(shap_values_raw[positive_class_idx])
    else:
        shap_arr = np.asarray(shap_values_raw)
        if shap_arr.ndim == 3:
            shap_values = shap_arr[:, :, positive_class_idx]
        else:
            shap_values = shap_arr

    if isinstance(expected_value_raw, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value_raw)[positive_class_idx])
    else:
        expected_value = float(expected_value_raw)

    return shap_values, expected_value


def shap_explain(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    plots_dir: Path,
    instance_index: int,
) -> ShapArtifacts:
    """Generate SHAP summary, force, and waterfall plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(
        model,
        data=X_train,
        model_output="probability",
        feature_perturbation="interventional",
    )
    shap_values_raw = explainer.shap_values(X_test)
    shap_values, expected_value = _extract_positive_class_shap(
        shap_values_raw,
        explainer.expected_value,
        positive_class_idx=1,
    )

    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary_plot.png", dpi=220, bbox_inches="tight")
    plt.close()

    try:
        plt.figure(figsize=(14, 3.5))
        shap.force_plot(
            expected_value,
            shap_values[instance_index],
            X_test.iloc[instance_index],
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(plots_dir / "shap_force_plot.png", dpi=220, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"Warning: SHAP force plot export failed: {exc}")

    explanation = shap.Explanation(
        values=shap_values[instance_index],
        base_values=expected_value,
        data=X_test.iloc[instance_index].values,
        feature_names=X_test.columns.tolist(),
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_waterfall_plot.png", dpi=220, bbox_inches="tight")
    plt.close()

    return ShapArtifacts(
        shap_values=shap_values,
        expected_value=expected_value,
        instance_index=instance_index,
    )


def validate_shap(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    shap_artifacts: ShapArtifacts,
) -> Dict[str, float]:
    """Validate local accuracy: prediction ~= base value + sum(feature SHAP values)."""
    idx = shap_artifacts.instance_index
    model_pred = float(model.predict_proba(X_test.iloc[[idx]])[0, 1])
    shap_reconstruction = float(
        shap_artifacts.expected_value + shap_artifacts.shap_values[idx].sum()
    )
    abs_diff = abs(model_pred - shap_reconstruction)

    return {
        "model_prediction": model_pred,
        "shap_reconstruction": shap_reconstruction,
        "absolute_difference": abs_diff,
    }
