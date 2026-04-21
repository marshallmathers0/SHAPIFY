from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_data
from src.evaluation import compare_methods
from src.lime_explainer import lime_explain
from src.counterfactual_explainer import (
    generate_counterfactuals,
    summarize_counterfactuals,
)
from src.model import evaluate_model, train_model
from src.preprocessing import preprocess_data
from src.shap_explainer import shap_explain, validate_shap

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
OUTPUT_TXT = RESULTS_DIR / "outputs.txt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAPIFY pipeline with row-based counterfactual explanation.")
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Row index from data/dataset.csv for Assignment 2 counterfactual explanation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(DATA_PATH)
    if args.row_index < 0 or args.row_index >= len(X):
        raise ValueError(f"row-index must be between 0 and {len(X) - 1}. Got: {args.row_index}")

    prep = preprocess_data(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = train_model(prep.X_train_transformed, prep.y_train, random_state=RANDOM_STATE)
    metrics = evaluate_model(model, prep.X_test_transformed, prep.y_test)

    target_instance = 0
    shap_artifacts = shap_explain(
        model=model,
        X_train=prep.X_train_transformed,
        X_test=prep.X_test_transformed,
        plots_dir=PLOTS_DIR,
        instance_index=target_instance,
    )

    shap_validation = validate_shap(model, prep.X_test_transformed, shap_artifacts)

    lime_df = lime_explain(
        model=model,
        X_train=prep.X_train_transformed,
        X_test=prep.X_test_transformed,
        plots_dir=PLOTS_DIR,
        instance_index=target_instance,
        random_state=RANDOM_STATE,
    )

    comparison_df = compare_methods(
        shap_values=shap_artifacts.shap_values,
        feature_names=prep.feature_names,
        lime_df=lime_df,
        model=model,
        plots_dir=PLOTS_DIR,
    )
    comparison_df.to_csv(RESULTS_DIR / "method_comparison.csv", index=False)

    # Assignment 2: single user-selected row counterfactual explanation.
    selected_row = X.iloc[[args.row_index]].copy()
    counterfactual_df = generate_counterfactuals(
        model=model,
        preprocessor=prep.preprocessor,
        X_train_raw=prep.X_train_raw,
        X_test_raw=selected_row,
        instance_index=0,
        plots_dir=PLOTS_DIR,
        feature_names=prep.X_train_raw.columns.tolist(),
        file_prefix=f"row_{args.row_index}",
        max_features_to_show=10,
    )

    if not counterfactual_df.empty:
        counterfactual_df.insert(0, "dataset_row_index", args.row_index)
    counterfactual_df.to_csv(RESULTS_DIR / "counterfactual_examples.csv", index=False)
    cf_summary = summarize_counterfactuals(counterfactual_df)

    top_shap = comparison_df.head(10)["feature"].tolist()
    top_lime = [
        val for val in comparison_df.dropna(subset=["abs_weight"]).head(10)["feature"].tolist()
    ]
    overlap = sorted(set(top_shap).intersection(top_lime))

    report_lines = [
        "XAI Evaluation Output",
        "=====================",
        f"Dataset path: {DATA_PATH}",
        f"Samples: {len(X)} | Features: {X.shape[1]}",
        f"Counterfactual query row index: {args.row_index}",
        "",
        "Model Metrics",
        "-------------",
        f"Accuracy: {metrics['accuracy']}",
        f"ROC-AUC: {metrics['roc_auc']}",
        f"Precision: {metrics['precision']}",
        f"Recall: {metrics['recall']}",
        "Confusion Matrix:",
        metrics["confusion_matrix"],
        "Feature Importances:",
        metrics["feature_importances"],
        "Classification Report:",
        metrics["classification_report"],
        "",
        "SHAP Local Accuracy Validation",
        "------------------------------",
        f"Model prediction: {shap_validation['model_prediction']:.8f}",
        f"SHAP reconstruction: {shap_validation['shap_reconstruction']:.8f}",
        f"Absolute difference: {shap_validation['absolute_difference']:.8e}",
        "",
        "SHAP vs LIME (Top-10 overlap)",
        "---------------------------",
        f"Top SHAP features: {top_shap}",
        f"Top LIME features (mapped): {top_lime}",
        f"Overlap count: {len(overlap)}",
        f"Overlapping features: {overlap}",
        "",
        "Saved artifacts",
        "-------------",
        f"- {PLOTS_DIR / 'shap_summary_plot.png'}",
        f"- {PLOTS_DIR / 'shap_force_plot.png'}",
        f"- {PLOTS_DIR / 'shap_waterfall_plot.png'}",
        f"- {PLOTS_DIR / 'lime_local_explanation.png'}",
        f"- {PLOTS_DIR / 'shap_vs_model_importance.png'}",
        f"- {RESULTS_DIR / 'method_comparison.csv'}",
        f"- {RESULTS_DIR / 'counterfactual_examples.csv'}",
        f"- {PLOTS_DIR / f'row_{args.row_index}_counterfactual_comparison.png'}",
        "",
        "Counterfactual Explanation",
        "--------------------------",
        cf_summary.get("summary", "No counterfactuals generated."),
    ]

    OUTPUT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
