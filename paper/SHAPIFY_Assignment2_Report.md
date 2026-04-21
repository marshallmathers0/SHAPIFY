# SHAPIFY - Assignment 2 Report
## Counterfactual Explanations as an Extension of Assignment 1

## Abstract

This report documents Assignment 2 of the SHAPIFY project, which extends the Assignment 1 explainability pipeline (SHAP + LIME + model importance analysis) with row-level counterfactual explanations. Assignment 1 established transparent model interpretation for a credit-risk classifier, while Assignment 2 adds actionable "what-if" guidance by answering: what minimum changes in input fields can flip a prediction from accepted to rejected or from rejected to accepted?

The extension introduces a user-driven interface where a specific row index from `data/dataset.csv` is supplied at runtime. The system returns the original prediction, target flipped prediction, and a compact list of minimal field-level changes. This aligns the project with real decision-support use cases in financial risk workflows where users need not only explanation of the current outcome, but also a path to an alternative outcome.

---

## 1. Assignment 1 Recap (One-Page Overview)

Assignment 1 built the core SHAPIFY framework for interpretable binary classification on a real-world financial dataset.

### 1.1 Problem Setup

- Dataset: OpenML `credit-g` (German Credit Risk), cached to `data/dataset.csv`
- Task: Predict bad credit risk
- Target mapping:
  - `bad` -> 1
  - `good` -> 0
- Feature space: 20 original mixed-type (numeric + categorical) fields

### 1.2 Data Engineering Pipeline

A robust preprocessing stack was implemented:

1. Train-test split with stratification
2. Missing value imputation
   - Numeric: median
   - Categorical: most frequent
3. Categorical one-hot encoding
4. Numeric scaling with StandardScaler
5. Unified transformed matrix for model and explanation modules

### 1.3 Model and Evaluation

A RandomForestClassifier was used as the black-box model. Assignment 1 evaluation included:

- Accuracy
- ROC-AUC
- Precision / Recall
- Confusion matrix
- Classification report
- Native `feature_importances_`

In the latest run used for this report context, the model produced:

- Accuracy: 0.7650
- ROC-AUC: 0.7994
- Precision: 0.6383
- Recall: 0.5000

### 1.4 Explainability Delivered in Assignment 1

Assignment 1 provided three interpretation lenses:

1. SHAP (TreeExplainer)
   - Global summary plot
   - Local force plot
   - Local waterfall plot

2. SHAP local accuracy validation

The additive SHAP property was verified numerically:

$$
\hat{y}(x) \approx \phi_0 + \sum_{i=1}^{M}\phi_i
$$

Latest observed validation:

- Model probability: 0.32077004
- SHAP reconstruction: 0.32062109
- Absolute error: $1.48947781 \times 10^{-4}$

3. LIME local explanation and SHAP-LIME comparison

- LIME generated local surrogate feature weights
- Overlap between top SHAP and top LIME features was measured
- Latest run overlap: 6 out of top 10 features

### 1.5 Assignment 1 Deliverable Quality

Assignment 1 produced a modular, production-style codebase with reproducible outputs in `results/` and `results/plots/`, plus a consolidated run summary in `results/outputs.txt`.

---

## 2. Assignment 2 Objective and Motivation

Assignment 2 extends interpretability from descriptive to actionable analysis.

Assignment 1 answers:
- Why did the model output this prediction?

Assignment 2 answers:
- What minimum changes can flip this prediction?

In lending and risk domains, this distinction is critical. Stakeholders often need guidance such as:
- "If this applicant is currently rejected, what concrete factors would likely move the decision to accepted?"

Hence, Assignment 2 focuses on row-level counterfactual explanations tied to the original dataset fields.

---

## 3. Assignment 2 Design Decision

A key design choice was made for usability and demonstration quality:

- Not all rows (too noisy and operationally heavy)
- Not random rows (low interpretive clarity)
- Use user-provided row input (`--row-index`) and generate one high-quality counterfactual explanation for that row

This approach is best for assignment demos because it is:

1. Deterministic and easy to present
2. Fast enough for live walkthroughs
3. Aligned with realistic analyst workflow (case-by-case explanation)
4. Directly actionable at the field level

---

## 4. Methodology for Counterfactual Generation

### 4.1 Core Approach

The extension uses a stable nearest-opposite-instance strategy in the raw feature space:

1. Take user-selected row from `dataset.csv`
2. Compute current model prediction and probability
3. Define target as opposite class
4. Search training pool for instances predicted as target class
5. Select nearest candidate under mixed distance:
   - Numeric features: absolute standardized difference
   - Categorical features: match/mismatch penalty
6. Report field-level deltas between original row and nearest opposite candidate

### 4.2 Why This Method

This method was selected because it is robust, transparent, and easy to explain in an assignment context. It avoids brittle dependencies and still provides clear minimum-change guidance.

### 4.3 Output Structure

For each queried row, Assignment 2 returns:

- Original prediction label and probability
- Counterfactual (flipped) prediction and probability
- Distance score to nearest opposite decision profile
- Top minimum feature changes (numeric deltas and category transitions)
- Plot of largest absolute changes

---

## 5. Implementation Details

### 5.1 Files Added/Updated

- `src/counterfactual_explainer.py`
  - Counterfactual generation logic
  - Human-readable summary generation
  - Feature-change plotting

- `main.py`
  - Added CLI argument `--row-index`
  - Added row validation
  - Integrated Assignment 2 flow after Assignment 1 pipeline

- `README.md`
  - Updated Assignment 2 run instructions and output artifacts

### 5.2 Runtime Interface

Command:

```bash
python main.py --row-index 25
```

Behavior:

1. Runs full Assignment 1 workflow
2. Runs Assignment 2 counterfactual for row 25
3. Writes outputs to `results/`

---

## 6. Assignment 2 Experimental Example

Using a recent run with row index 998:

- Original prediction: Accepted
- Original probability (bad-credit class): 0.4380
- Counterfactual target: Rejected
- Counterfactual probability (bad-credit class): 0.8618
- Nearest counterfactual distance: 8.4002

Top suggested changes:

1. `checking_status`: `0<=X<200` -> `<0`
2. `duration`: decrease by 27
3. `purpose`: `radio/tv` -> `new car`
4. `credit_amount`: increase by 342
5. `employment`: `4<=X<7` -> `<1`

These are the model-consistent changes that move the selected case across the learned decision boundary.

---

## 7. Artifacts Generated by Assignment 2

- `results/counterfactual_examples.csv`
  - Complete field-by-field original vs counterfactual values and deltas

- `results/plots/row_<index>_counterfactual_comparison.png`
  - Bar chart of top absolute feature changes

- `results/outputs.txt`
  - Consolidated text summary including Assignment 2 explanation block

---

## 8. How to Demo Assignment 2 (Recommended)

### 8.1 Suggested Demo Flow

1. Run baseline:

```bash
python main.py --row-index 25
```

2. Show three outputs in order:
- Console/text summary (original prediction + minimum changes)
- `counterfactual_examples.csv` table
- row-specific counterfactual plot

3. Repeat with a second row (e.g., high-risk/low-risk case) to show consistency

### 8.2 What to Say During Demo

- "This is the original decision and probability"
- "This is the target flipped class"
- "These are the minimum model-consistent feature changes"
- "Numeric fields show amount shifts; categorical fields show class transitions"

### 8.3 Why Not Demo All Rows

- Too much noise
- Hard to interpret in presentation
- Slower and less stakeholder-friendly
- Does not reflect real analyst use (case-by-case queries are standard)

---

## 9. Strengths, Limitations, and Future Work

### 9.1 Strengths

1. Clear extension path from Assignment 1 to Assignment 2
2. Actionable field-level recommendations
3. Row-level user control for targeted analysis
4. Reproducible and script-friendly outputs

### 9.2 Limitations

1. Counterfactuals are nearest-opposite approximations, not globally optimal under strict causal constraints
2. Some suggested changes may be practically difficult to realize in real-world operations
3. Domain feasibility constraints are not yet encoded (e.g., immutable attributes)

### 9.3 Future Improvements

1. Add immutable-feature constraints (e.g., prevent age edits)
2. Add plausibility constraints and business rules
3. Add top-N candidate counterfactual alternatives per row
4. Add API endpoint for interactive row-based counterfactual serving

---

## 10. Conclusion

Assignment 2 successfully extends SHAPIFY from interpretation to intervention-style guidance. Assignment 1 explained why predictions occur; Assignment 2 adds how to change them. The row-index query interface makes the extension practical for demonstration and decision support.

Together, Assignments 1 and 2 now provide a complete explainability stack:

- global understanding (SHAP summary)
- local attribution (SHAP/LIME)
- theoretical validation (SHAP local accuracy)
- actionable recourse guidance (counterfactual changes)

This combined pipeline is a strong foundation for real-world, trustworthy credit-risk model analysis.
