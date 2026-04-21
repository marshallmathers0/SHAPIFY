from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


def _detect_numeric_columns(frame: pd.DataFrame) -> List[str]:
    return frame.select_dtypes(include=[np.number]).columns.tolist()


def _compute_mixed_distance(
    source_row: pd.Series,
    candidate_row: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_scale: Dict[str, float],
) -> float:
    distance = 0.0

    for col in numeric_cols:
        scale = numeric_scale.get(col, 1.0) or 1.0
        distance += abs(float(candidate_row[col]) - float(source_row[col])) / scale

    for col in categorical_cols:
        distance += 0.0 if str(candidate_row[col]) == str(source_row[col]) else 1.0

    return distance


def _build_change_table(original_row: pd.Series, cf_row: pd.Series, feature_names: List[str]) -> pd.DataFrame:
    records = []
    for col in feature_names:
        original_value = original_row[col]
        counterfactual_value = cf_row[col]

        if isinstance(original_value, (int, float, np.number)) and isinstance(counterfactual_value, (int, float, np.number)):
            change_value = float(counterfactual_value) - float(original_value)
            change_type = "numeric"
        else:
            change_value = 0.0 if str(original_value) == str(counterfactual_value) else 1.0
            change_type = "categorical"

        records.append(
            {
                "feature": col,
                "original_value": original_value,
                "counterfactual_value": counterfactual_value,
                "change": change_value,
                "change_type": change_type,
            }
        )

    return pd.DataFrame(records)


def plot_counterfactual_change_scores(
    change_table: pd.DataFrame,
    plots_dir: Path,
    file_prefix: str,
    max_features_to_show: int = 10,
) -> None:
    """Plot the largest feature changes between the original row and counterfactual row."""
    if change_table.empty:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_df = change_table.copy()
    plot_df = plot_df.sort_values("change", key=lambda s: s.abs(), ascending=False).head(max_features_to_show)
    plot_df = plot_df[plot_df["change"].abs() > 0]

    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(plot_df["feature"], plot_df["change"].abs(), color="#2ca02c")
    ax.set_xlabel("Absolute Change")
    ax.set_ylabel("Feature")
    ax.set_title("Counterfactual Feature Changes")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{file_prefix}_counterfactual_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_counterfactuals(
    model: RandomForestClassifier,
    preprocessor: ColumnTransformer,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    instance_index: int,
    plots_dir: Path,
    feature_names: Optional[List[str]] = None,
    file_prefix: str = "counterfactual",
    max_features_to_show: int = 10,
) -> pd.DataFrame:
    """
    Generate a practical counterfactual by finding the nearest training instance
    predicted as the opposite class by the trained model.

    This approach is stable for the assignment and easy to explain in a demo:
    show one rejected case and one accepted case, then explain the minimum feature
    changes needed to flip the prediction.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    original_row = X_test_raw.iloc[instance_index].copy()
    transformed_columns = preprocessor.get_feature_names_out().tolist()

    original_transformed_arr = preprocessor.transform(X_test_raw.iloc[[instance_index]])
    original_transformed = pd.DataFrame(original_transformed_arr, columns=transformed_columns)
    current_prediction = int(model.predict(original_transformed)[0])
    current_probability = float(model.predict_proba(original_transformed)[0, 1])
    target_class = 0 if current_prediction == 1 else 1

    transformed_train_arr = preprocessor.transform(X_train_raw)
    transformed_train = pd.DataFrame(transformed_train_arr, columns=transformed_columns, index=X_train_raw.index)
    train_predictions = model.predict(transformed_train)
    candidate_pool = X_train_raw.loc[train_predictions == target_class].copy()

    if candidate_pool.empty:
        return pd.DataFrame()

    numeric_cols = _detect_numeric_columns(X_train_raw)
    categorical_cols = [c for c in X_train_raw.columns if c not in numeric_cols]
    numeric_scale = {col: float(X_train_raw[col].std()) if float(X_train_raw[col].std()) > 0 else 1.0 for col in numeric_cols}

    best_candidate = None
    best_distance = float("inf")

    for _, candidate in candidate_pool.iterrows():
        distance = _compute_mixed_distance(
            source_row=original_row,
            candidate_row=candidate,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            numeric_scale=numeric_scale,
        )
        if distance < best_distance:
            best_distance = distance
            best_candidate = candidate.copy()

    if best_candidate is None:
        return pd.DataFrame()

    cf_transformed_arr = preprocessor.transform(pd.DataFrame([best_candidate], columns=X_train_raw.columns))
    cf_transformed = pd.DataFrame(cf_transformed_arr, columns=transformed_columns)
    cf_prediction = int(model.predict(cf_transformed)[0])
    cf_probability = float(model.predict_proba(cf_transformed)[0, 1])

    raw_feature_names = feature_names or list(X_train_raw.columns)
    change_table = _build_change_table(original_row, best_candidate, raw_feature_names)
    change_table.insert(0, "counterfactual_prediction", cf_prediction)
    change_table.insert(0, "counterfactual_probability", cf_probability)
    change_table.insert(0, "original_probability", current_probability)
    change_table.insert(0, "original_prediction", current_prediction)
    change_table.insert(0, "target_class", target_class)
    change_table.insert(0, "distance", best_distance)
    change_table.insert(0, "counterfactual_id", 1)

    plot_counterfactual_change_scores(
        change_table=change_table,
        plots_dir=plots_dir,
        file_prefix=file_prefix,
        max_features_to_show=max_features_to_show,
    )

    return change_table


def summarize_counterfactuals(counterfactual_df: pd.DataFrame) -> Dict[str, str]:
    """Create a short human-readable summary for the chosen counterfactual."""
    if counterfactual_df.empty:
        return {"summary": "No counterfactual could be generated for this instance."}

    original_prediction = int(counterfactual_df["original_prediction"].iloc[0])
    target_class = int(counterfactual_df["target_class"].iloc[0])
    original_probability = float(counterfactual_df["original_probability"].iloc[0])
    counterfactual_probability = float(counterfactual_df["counterfactual_probability"].iloc[0])
    distance = float(counterfactual_df["distance"].iloc[0])

    start_label = "Rejected" if original_prediction == 1 else "Accepted"
    target_label = "Accepted" if target_class == 0 else "Rejected"

    change_rows = counterfactual_df[counterfactual_df["change"] != 0].copy()
    if change_rows.empty:
        return {
            "summary": (
                f"Current prediction: {start_label} (probability: {original_probability:.4f})\n"
                f"Target prediction: {target_label} (probability: {counterfactual_probability:.4f})\n"
                f"Nearest counterfactual distance: {distance:.4f}\n"
                "No visible feature changes in the selected counterfactual."
            )
        }

    summary_lines = [
        f"Current prediction: {start_label} (probability: {original_probability:.4f})",
        f"Target prediction: {target_label} (probability: {counterfactual_probability:.4f})",
        f"Nearest counterfactual distance: {distance:.4f}",
        "Minimum changes suggested:",
    ]

    for _, row in change_rows.head(5).iterrows():
        if row["change_type"] == "categorical":
            summary_lines.append(
                f"- {row['feature']}: change category from {row['original_value']} to {row['counterfactual_value']}"
            )
        else:
            direction = "increase" if float(row["change"]) > 0 else "decrease"
            summary_lines.append(f"- {row['feature']}: {direction} by {abs(float(row['change'])):.4f}")

    return {
        "summary": "\n".join(summary_lines),
        "num_counterfactuals": 1,
        "target_label": target_label,
    }
