# SHAPIFY

SHAPIFY is an end-to-end Explainable AI project that trains a black-box classifier and explains predictions using SHAP and LIME. It validates SHAP local accuracy numerically and compares SHAP attributions with LIME and model-native feature importance.

## Objective

Reproduce SHAP explanations from the paper *A Unified Approach to Interpreting Model Predictions*, validate SHAP's theoretical guarantee (local accuracy), and compare interpretability behavior against LIME.

<<<<<<< HEAD
=======
## Assignments

### Assignment 1: Core SHAP and LIME Framework
- Implement SHAP TreeExplainer with global and local visualizations
- Validate SHAP local accuracy theorem numerically
- Compare SHAP with LIME explanations
- Compare SHAP importance with model native feature importance

### Assignment 2: Counterfactual Explanations (Extension)
- Generate counterfactual examples using a stable nearest-opposite-instance method
- Show minimum feature changes to flip the prediction for a user-selected dataset row
- Quantify feature changes (deltas) required for prediction flip
- Visualize original vs counterfactual feature changes
- Practical demo: user gives one row index from `dataset.csv`

>>>>>>> 35c1c0d (added counter factual explainations for jackfruit problem)
## Dataset

- Source: OpenML `credit-g` (German Credit Risk)
- Type: Real-world tabular financial dataset
- Samples: 1,000
- Original features: 20
- Target: `class` (`good` / `bad`), mapped to binary `is_bad_credit` (`bad` = 1, `good` = 0)
- Local cache: `data/dataset.csv`

## Requirements Coverage

This project includes all required steps:

1. Real-world tabular dataset with multiple features
2. Preprocessing:
   - Missing value handling (median / most-frequent imputation)
   - Categorical encoding (one-hot encoding)
   - Normalization/scaling (StandardScaler for numeric features)
   - Train-test split (stratified)
3. Black-box model training (RandomForestClassifier)
4. SHAP implementation with TreeExplainer
5. SHAP visualizations:
   - Summary plot (global)
   - Force plot (local)
   - Waterfall plot (local)
6. SHAP local-accuracy validation:
   - Verify prediction is approximately equal to expected_value + sum(SHAP values)
7. LIME explanation for the same instance
8. SHAP vs LIME comparison (feature ranking overlap and differences)
9. SHAP vs model feature importance comparison
10. Modular implementation of required functions:
   - `load_data()`
   - `preprocess_data()`
   - `train_model()`
   - `shap_explain()`
   - `lime_explain()`
   - `validate_shap()`
   - `compare_methods()`
11. Saved outputs (images + CSV + clear textual report)
12. Clean, readable, production-style structure

## Project Structure

```text
Xai_Project/
├── data/
│   └── dataset.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── shap_explainer.py
│   ├── lime_explainer.py
<<<<<<< HEAD
=======
│   ├── counterfactual_explainer.py
>>>>>>> 35c1c0d (added counter factual explainations for jackfruit problem)
│   └── evaluation.py
├── notebooks/
│   └── shap_analysis.ipynb
├── results/
│   ├── plots/
│   ├── method_comparison.csv
<<<<<<< HEAD
=======
│   ├── counterfactual_examples.csv
>>>>>>> 35c1c0d (added counter factual explainations for jackfruit problem)
│   └── outputs.txt
├── paper/
│   ├── shap_paper.pdf
│   └── SHAPIFY_Report.md
├── .vscode/
│   └── tasks.json
├── main.py
├── requirements.txt
└── README.md
```

## Setup

### 1) Create virtual environment (recommended)

```bash
python -m venv .venv
```

### 2) Activate environment

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## How To Run

### Option A: Run from terminal

```bash
python main.py
```

<<<<<<< HEAD
=======
### Assignment 2 (row-based counterfactual)

```bash
python main.py --row-index 25
```

This returns:
- original prediction for row 25 (Accepted/Rejected)
- minimum feature changes to flip that prediction
- saved counterfactual table and plot for that row

>>>>>>> 35c1c0d (added counter factual explainations for jackfruit problem)
### Option B: One-click from VS Code Task

1. Open Command Palette (`Ctrl+Shift+P`)
2. Run `Tasks: Run Task`
3. Select `Run XAI Pipeline`

## Expected Outputs

Generated files:

- `results/plots/shap_summary_plot.png`
- `results/plots/shap_force_plot.png`
- `results/plots/shap_waterfall_plot.png`
- `results/plots/lime_local_explanation.png`
- `results/plots/shap_vs_model_importance.png`
<<<<<<< HEAD
- `results/method_comparison.csv`
=======
- `results/plots/row_<index>_counterfactual_comparison.png`
- `results/method_comparison.csv`
- `results/counterfactual_examples.csv`
>>>>>>> 35c1c0d (added counter factual explainations for jackfruit problem)
- `results/outputs.txt`

## Key Evaluation Metrics Printed

- Accuracy
- ROC-AUC
- Precision
- Recall
- Confusion Matrix
- Classification report
- Model `feature_importances_`
- SHAP local-accuracy numeric check
- SHAP vs LIME top-feature overlap
<<<<<<< HEAD
=======
- Counterfactual explanations (minimum feature changes to flip prediction)
- Counterfactual feature deltas (quantified changes)
- Counterfactual vs Original comparison visualizations
>>>>>>> 35c1c0d (added counter factual explainations for jackfruit problem)

## Architecture Notes

- `main.py` orchestrates full pipeline execution.
- `src/shap_explainer.py` handles SHAP values, plots, and local-accuracy validation.
- `src/lime_explainer.py` generates local LIME explanation for the same SHAP instance.
- `src/evaluation.py` compares SHAP/LIME rankings and SHAP/model feature importance.
<<<<<<< HEAD
=======
- `src/counterfactual_explainer.py` generates a practical counterfactual for representative cases.
>>>>>>> 35c1c0d (added counter factual explainations for jackfruit problem)
- Plotting uses a non-interactive backend for stable script execution.

## Troubleshooting

1. If packages are missing:

```bash
pip install -r requirements.txt
```

2. If OpenML download fails (network issue), retry once connection is available.

3. If PowerShell blocks venv activation, run:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

then activate the environment again.
