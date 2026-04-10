# SHAPIFY: Explainable AI Framework for SHAP-Based Model Interpretation and LIME Comparison

## Project Description

SHAPIFY is a production-style Explainable AI project that builds a black-box classification model on tabular data and explains its predictions using SHAP and LIME. The framework combines global and local interpretability, validates SHAP's local-accuracy theorem numerically, and compares attribution consistency across SHAP, LIME, and model-native feature importance.

## Abstract

This report presents SHAPIFY, an end-to-end Explainable AI (XAI) pipeline for credit risk prediction on tabular financial data. The project reproduces key ideas from the SHAP framework introduced in the paper *A Unified Approach to Interpreting Model Predictions* and evaluates interpretability through side-by-side comparison with LIME and model-native feature importance.

The implementation uses the German Credit dataset (OpenML: `credit-g`) and a RandomForestClassifier as the black-box model. The workflow includes robust preprocessing (imputation, one-hot encoding, scaling, train-test split), model training and evaluation, SHAP explanation generation (summary, force, and waterfall plots), numerical validation of SHAP local accuracy, LIME local explanation on the same test instance, and comparative analysis of feature attribution consistency.

Experimental results show that SHAP and LIME overlap on several dominant local drivers (especially checking account status and savings-related variables), while they differ in lower-ranked features due to methodological differences. SHAP satisfies its additive reconstruction property with very small numerical error in our run, confirming local accuracy in practice. The report also discusses methodological tradeoffs, stability considerations, and production-readiness aspects of the modular codebase.

---

## 1. Introduction

### 1.1 Motivation

In high-stakes domains such as lending, model performance alone is insufficient. Stakeholders need to know *why* a model predicts high default risk for one applicant and low risk for another. Without transparent explanations, deployment can introduce operational, legal, and ethical risks.

Tree-based ensemble models such as Random Forest are often effective for tabular data, but they are not naturally interpretable in the same way as linear models. Explainability frameworks bridge this gap by producing feature-level attributions for individual predictions and global patterns.

This project addresses the following practical question:

How can we build a production-style credit-risk pipeline that keeps strong predictive behavior while also providing trustworthy local and global explanations?

### 1.2 Objective

The objective is to reproduce and evaluate SHAP-based interpretability, validate a core theoretical guarantee (local accuracy), and compare SHAP against LIME and model-native importance. The pipeline is implemented in modular Python files for clarity, maintainability, and reproducibility.

### 1.3 Contributions

This project delivers:

1. A complete modular XAI workflow over a real-world financial dataset.
2. SHAP global and local visual explanations.
3. Numeric verification of SHAP local accuracy.
4. LIME explanation for the same sample used in SHAP.
5. Ranking-level SHAP vs LIME comparison and overlap analysis.
6. SHAP vs RandomForest feature importance comparison.
7. Save-to-disk artifacts for reporting (`plots`, `CSV`, and textual run summary).

---

## 2. Dataset and Problem Setup

### 2.1 Dataset Source

The dataset is loaded through OpenML (`credit-g`) and cached to local CSV for repeatability. It contains 1,000 credit applicants and 20 original features plus one target column.

### 2.2 Prediction Task

The original target is `class` with values `good` and `bad`. In the implementation, the target is converted to a binary risk label:

- `bad` -> 1 (high risk / default-like outcome)
- `good` -> 0 (lower risk)

This turns the task into binary classification of bad credit risk probability.

### 2.3 Feature Space

The 20 original predictors include account status, loan duration, credit history, loan purpose, credit amount, savings, employment, installment burden, demographic status indicators, housing, job type, and communication/foreign-worker fields.

These are appropriate risk determinants in retail credit settings because they encode borrower liquidity, repayment history, and exposure profile.

---

## 3. Methodology

### 3.1 Preprocessing Pipeline

A structured preprocessing pipeline is applied:

1. Train-test split with stratification to preserve class ratio.
2. Numeric branch:
   - missing value imputation with median
   - normalization via StandardScaler
3. Categorical branch:
   - missing value imputation with most-frequent category
   - one-hot encoding with unknown-category handling
4. ColumnTransformer integration to produce final model matrix.

Although scaling is not strictly required for tree models, it helps maintain consistent transformed input spaces and can improve neighborhood behavior for LIME.

### 3.2 Black-Box Model

A RandomForestClassifier is used with class balancing and multiple trees to stabilize predictions. This model class is a strong baseline for tabular credit data and supports SHAP TreeExplainer efficiently.

The model evaluation includes:

- Accuracy
- ROC-AUC
- Precision
- Recall
- Confusion matrix
- Full classification report
- Native feature importances (`feature_importances_`)

### 3.3 SHAP Implementation

The project uses SHAP TreeExplainer with probability output.

For a prediction function $f(x)$, SHAP decomposes the output as:

$$
f(x) = \phi_0 + \sum_{i=1}^{M}\phi_i
$$

Where:

- $\phi_0$ is the expected model output (baseline)
- $\phi_i$ is the contribution of feature $i$
- $M$ is the number of input features in transformed space

The implementation exports:

1. Summary plot (global)
2. Force plot (single instance, local)
3. Waterfall plot (ordered local additive contributions)

### 3.4 SHAP Local Accuracy Validation

The report verifies the theoretical local-accuracy property numerically:

$$
\hat{y}(x) \approx \phi_0 + \sum_{i=1}^{M}\phi_i
$$

For one selected test instance, the model probability and SHAP reconstruction are printed together with absolute error.

### 3.5 LIME Implementation

LIME is applied to the exact same test instance used by SHAP for fair local comparison. A local surrogate explanation is generated with top weighted features. LIME output rules are mapped back (where possible) to transformed feature names to enable overlap and ranking comparison.

### 3.6 SHAP vs LIME and SHAP vs Model Importance

Two complementary comparisons are performed:

1. SHAP vs LIME:
   - compare top-ranked local/global-attribution features
   - compute overlap in top-$k$
   - analyze consistency and divergence drivers

2. SHAP vs model-native importance:
   - compare mean absolute SHAP values with RandomForest split importances
   - visualize both in side-by-side bar charts

---

## 4. Implementation Architecture

The project is organized for production-style maintainability:

- `src/data_loader.py`: OpenML fetch + local caching + target conversion.
- `src/preprocessing.py`: split + imputation + encoding + scaling.
- `src/model.py`: model training + metric reporting.
- `src/shap_explainer.py`: SHAP computation + plots + local-accuracy validation.
- `src/lime_explainer.py`: LIME explanation for matched instance.
- `src/evaluation.py`: cross-method comparison and importance plots.
- `main.py`: orchestrates full pipeline and writes artifacts.

Artifacts are saved under `results/`:

- `results/plots/*.png`
- `results/method_comparison.csv`
- `results/outputs.txt`

A notebook entry point (`notebooks/shap_analysis.ipynb`) mirrors the same modular workflow for interactive exploration.

---

## 5. Experimental Results

### 5.1 Predictive Performance

From a successful pipeline run:

- Accuracy: 0.7700
- ROC-AUC: 0.8046
- Precision: 0.6522
- Recall: 0.5000
- Confusion Matrix:
  - TN = 124
  - FP = 16
  - FN = 30
  - TP = 30

Interpretation:

The model separates classes reasonably well (ROC-AUC > 0.80), but recall for the bad-credit class indicates false negatives are still meaningful. This is expected in moderate-size credit datasets and highlights why explanation is essential for error analysis and policy calibration.

### 5.2 SHAP Global Insight

The SHAP summary plot reveals high-impact features tied to checking account status, duration, savings category, and credit history. These align with financial intuition:

- poor/no checking and low savings often raise risk
- longer duration and larger obligations can contribute to adverse predictions
- stronger historical repayment tends to reduce risk

### 5.3 SHAP Local Explanation

For the selected instance, force and waterfall plots provide directional contributions. The waterfall plot is particularly useful for auditing additive pushes toward higher or lower bad-credit probability.

### 5.4 Local Accuracy Check

Observed values from run output:

- Model prediction: 0.32488596
- SHAP reconstruction: 0.32479911
- Absolute difference: $8.68502989 \times 10^{-5}$

Interpretation:

The difference is extremely small, empirically validating SHAP local accuracy under numerical precision and implementation details.

### 5.5 LIME Local Explanation

LIME identifies a top set of local rule-style drivers around checking status, savings, credit history, and related binary indicators. The explanation supports qualitative consistency with SHAP while remaining methodologically distinct (local surrogate fit rather than game-theoretic additive decomposition).

### 5.6 SHAP-LIME Comparison

In the top-10 comparison for the selected run, overlap was 7 features. This indicates strong agreement on dominant signals and expected disagreement on lower-ranked signals.

Why differences occur:

1. SHAP is additive and tied to model output decomposition.
2. LIME approximates local behavior with perturbed samples and weighted regression.
3. SHAP summary importance is global aggregation; LIME here is local to one neighborhood.

### 5.7 SHAP vs Model-Native Importance

The side-by-side plot contrasts two notions:

- RandomForest `feature_importances_`: split-criterion-based, global, unsigned
- SHAP mean absolute values: contribution-based, globally aggregated from local attributions

Differences are normal and informative. SHAP often better reflects direct prediction impact, while model-native importance reflects training-time split utility.

---

## 6. Discussion

### 6.1 Practical Interpretability Value

The combined SHAP+LIME strategy provides layered interpretability:

- SHAP for principled additive attribution and globally consistent ranking.
- LIME for local surrogate perspective and rule-like local explanations.

For risk teams, this dual lens can improve trust, investigation speed, and stakeholder communication.

### 6.2 Stability and Consistency

Consistency is strong among top drivers but naturally weaker in tail features. This is expected and should be communicated clearly to non-technical users: agreement at the top matters most for policy-level insights.

### 6.3 Limitations

1. Single dataset scale (1,000 rows) limits generalization.
2. LIME sensitivity to perturbation settings and discretization can affect rankings.
3. Feature engineering remains basic; richer domain transforms may improve both performance and interpretability.
4. No fairness audit included yet.

### 6.4 Reproducibility and Engineering Strength

The project is reproducible and deployment-friendly due to:

- fixed random state
- modular function decomposition
- deterministic data cache path
- saved artifacts for audit
- one-click VS Code task (`Run XAI Pipeline`)

---

## 7. Conclusion

SHAPIFY successfully reproduces a practical SHAP interpretation workflow and validates its local-accuracy property on a real credit dataset. The project also demonstrates that SHAP and LIME frequently agree on high-impact explanatory factors while diverging in less dominant terms due to differing assumptions.

From an engineering perspective, the pipeline meets production-quality expectations for clarity, modularity, and artifact traceability. From a methodological perspective, it demonstrates how to combine performance and interpretability in financial risk modeling.

Key takeaways:

1. SHAP provides robust additive explanations with strong theoretical grounding.
2. LIME offers useful local surrogate explanations for complementary analysis.
3. Comparison across SHAP, LIME, and model-native importance reveals richer insight than any single method.
4. The modular codebase can be extended to XGBoost, fairness diagnostics, drift monitoring, and policy threshold optimization.

---

## 8. Future Work

1. Add XGBoost model branch and compare explanation behavior with RandomForest.
2. Add bootstrap stability analysis for SHAP/LIME rankings.
3. Include calibration and threshold optimization for business utility.
4. Add fairness and subgroup explainability checks.
5. Export final report to PDF and integrate experiment tracking.

---

## 9. References

1. Lundberg, S. M., and Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
2. Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). *Why Should I Trust You? Explaining the Predictions of Any Classifier*. KDD.
3. OpenML Dataset: `credit-g` (German Credit Data).
