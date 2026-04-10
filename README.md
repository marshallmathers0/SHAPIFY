# Explainable AI Project: SHAP Reproduction and LIME Comparison

This project reproduces core SHAP behavior from the paper:
**A Unified Approach to Interpreting Model Predictions**.

It trains a black-box classifier on a real-world credit dataset and compares:
- SHAP (global + local additive explanations)
- LIME (local surrogate explanation)
- RandomForest native feature importance

## What this project implements

1. Real-world tabular dataset (German Credit from OpenML)
2. Data preprocessing:
   - Missing value imputation
   - Categorical one-hot encoding
   - Numeric scaling
   - Train-test split
3. Black-box model training: RandomForestClassifier
4. SHAP explainability using TreeExplainer:
   - Summary plot
   - Force plot
   - Waterfall plot
5. SHAP local accuracy validation:
   - prediction ~= expected_value + sum(SHAP values)
6. LIME explanation for the exact same instance
7. SHAP vs LIME ranking comparison
8. SHAP importance vs model.feature_importances_ comparison plot
9. Modular function structure:
   - load_data()
   - preprocess_data()
   - train_model()
   - shap_explain()
   - lime_explain()
   - validate_shap()
   - compare_methods()

## Project Structure

XAI_PROJECT/
|
+-- data/
|   +-- dataset.csv
|
+-- src/
|   +-- data_loader.py
|   +-- preprocessing.py
|   +-- model.py
|   +-- shap_explainer.py
|   +-- lime_explainer.py
|   +-- evaluation.py
|
+-- notebooks/
|   +-- shap_analysis.ipynb
|
+-- results/
|   +-- plots/
|   +-- outputs.txt
|
+-- paper/
|   +-- shap_paper.pdf
|
+-- main.py
+-- requirements.txt
+-- README.md

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Expected outputs

Inside `results/plots/`:
- `shap_summary_plot.png`
- `shap_force_plot.png`
- `shap_waterfall_plot.png`
- `lime_local_explanation.png`
- `shap_vs_model_importance.png`

Inside `results/`:
- `method_comparison.csv`
- `outputs.txt`

## Notes

- The dataset is fetched from OpenML at runtime, so internet access is required.
- SHAP output format can differ by library version; the script handles common variants.
