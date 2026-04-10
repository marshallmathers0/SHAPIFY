from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _map_lime_expression_to_feature(expression: str, known_features: List[str]) -> str:
    for feature in known_features:
        if feature in expression:
            return feature

    tokens = (
        expression.replace("<=", " ")
        .replace(">=", " ")
        .replace("<", " ")
        .replace(">", " ")
        .replace("=", " ")
        .split()
    )
    return tokens[0] if tokens else expression


def compare_methods(
    shap_values: np.ndarray,
    feature_names: List[str],
    lime_df: pd.DataFrame,
    model: RandomForestClassifier,
    plots_dir: Path,
) -> pd.DataFrame:
    """Compare SHAP and LIME rankings and model-native feature importance."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    shap_importance = np.mean(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_mean_abs": shap_importance,
            "model_importance": model.feature_importances_,
        }
    )

    lime_tmp = lime_df.copy()
    lime_tmp["feature_mapped"] = lime_tmp["feature"].apply(
        lambda expr: _map_lime_expression_to_feature(expr, feature_names)
    )
    lime_agg = (
        lime_tmp.groupby("feature_mapped", as_index=False)["abs_weight"]
        .sum()
        .sort_values("abs_weight", ascending=False)
    )

    comparison = shap_df.merge(
        lime_agg,
        left_on="feature",
        right_on="feature_mapped",
        how="left",
    ).drop(columns=["feature_mapped"])

    comparison = comparison.sort_values("shap_mean_abs", ascending=False)

    plot_df = comparison.head(20).sort_values("shap_mean_abs", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)

    axes[0].barh(plot_df["feature"], plot_df["shap_mean_abs"], color="#1f77b4")
    axes[0].set_title("SHAP Global Importance (mean |value|)")
    axes[0].set_xlabel("mean |SHAP value|")

    axes[1].barh(plot_df["feature"], plot_df["model_importance"], color="#ff7f0e")
    axes[1].set_title("RandomForest feature_importances_")
    axes[1].set_xlabel("Model importance")

    fig.suptitle("SHAP vs Model Feature Importance", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "shap_vs_model_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return comparison
