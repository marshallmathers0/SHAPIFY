from __future__ import annotations

from pathlib import Path

from src.data_loader import load_data
from src.evaluation import compare_methods
from src.lime_explainer import lime_explain
from src.model import evaluate_model, train_model
from src.preprocessing import preprocess_data
from src.shap_explainer import shap_explain, validate_shap

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
OUTPUT_TXT = RESULTS_DIR / "outputs.txt"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(DATA_PATH)
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
    ]

    OUTPUT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
