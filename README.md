# Adolescent Screen Time and Health Risk: A Statistical & Machine Learning Analysis

A research-oriented analysis examining the relationship between daily screen time and self-reported health risk in adolescents, combining machine learning classification, model interpretability (SHAP), and formal statistical hypothesis testing.

## Research Question

Does average daily screen time predict self-reported health risk in adolescents, and does this relationship differ between younger (≤14) and older (15–18) age groups?

## Dataset

- Source: `Screen_time.csv` — adolescent screen time survey data
- Target variable: `Reported_Health_Risk` (binary — derived from whether `Health_Impacts` was reported)
- Features used: average daily screen time (hours), gender, primary device, urban/rural residence
- Class distribution: ~67% reported health risk vs. ~33% none — a moderately imbalanced target
- Data was split into two age cohorts and analyzed **separately** to detect whether the relationship differs by developmental stage:
  - **Age ≤ 14**
  - **Age 15–18**

## Methodology

For each age group, a rigorous ML pipeline was built:

1. **80/20 stratified train/test split**, with the test set held out strictly for final reporting
2. **5 candidate models** compared: KNN, Logistic Regression, Random Forest, Naive Bayes, SVM
3. **Hyperparameter tuning** via `GridSearchCV` with 5-fold stratified cross-validation, optimizing for ROC-AUC
4. **Pipeline-based scaling** (StandardScaler) to prevent data leakage — scaling fit only on training folds
5. **Best model selection** based on mean cross-validated AUC-ROC
6. **SHAP interpretation** to explain which features drove predictions for the winning model (TreeExplainer for Random Forest, KernelExplainer otherwise)
7. **Statistical validation**: an optimal screening threshold for screen time was derived using Youden's J statistic from the ROC curve, then tested for association with health risk using a **chi-square test of independence** and **Cramér's V** effect size

## Results

### Age ≤ 14

| Model | Mean Accuracy | Mean AUC-ROC | Mean F1 | Mean Recall |
|---|---|---|---|---|
| **Random Forest** | 0.8604 | 0.8085 | 0.9008 | 1.0000 |
| SVM | 0.8416 | 0.8069 | 0.8860 | 0.9708 |
| KNN | 0.8312 | 0.8106 | 0.8767 | 0.9470 |
| Logistic Regression | 0.8227 | 0.8065 | 0.8690 | 0.9279 |
| Naive Bayes | 0.8054 | 0.7853 | 0.8564 | 0.9158 |

- **Final model**: KNN
- **Optimized screening threshold**: 4.78 hrs/day
- **Statistical significance**: p = 1.98e-116 (highly significant)
- **Effect size (Cramér's V)**: 0.291 (moderate association)

### Age 15–18

| Model | Mean Accuracy | Mean AUC-ROC | Mean F1 | Mean Recall |
|---|---|---|---|---|
| **Random Forest** | 0.8303 | 0.6951 | 0.8957 | 0.9976 |
| KNN | 0.8139 | 0.7021 | 0.8853 | 0.9834 |
| Naive Bayes | 0.8121 | 0.6820 | 0.8840 | 0.9804 |
| Logistic Regression | 0.8110 | 0.6890 | 0.8851 | 0.9966 |
| SVM | 0.7521 | 0.6933 | 0.8549 | 1.0000 |

- **Final model**: KNN
- **Optimized screening threshold**: 4.70 hrs/day
- **Statistical significance**: p = 2.96e-22 (highly significant)
- **Effect size (Cramér's V)**: 0.164 (weak-to-moderate association)

### Key Finding

Both age groups showed a **statistically significant association** between exceeding ~4.7–4.8 hours/day of screen time and reported health risk, with a **stronger effect size in the younger (≤14) cohort** — suggesting younger adolescents may be more sensitive to the health impacts of high screen time exposure than older teens.

## Tech Stack

- Python
- scikit-learn (model pipelines, GridSearchCV, cross-validation, metrics)
- SHAP (model interpretability)
- SciPy (chi-square test, Cramér's V)
- Pandas, NumPy
- Matplotlib, Seaborn

## How to Run

1. Place `Screen_time.csv` in the working directory
2. Install dependencies: `pip install scikit-learn shap scipy pandas numpy matplotlib seaborn`
3. Run all cells — the notebook automatically loops through both age groups and outputs results, plots, and statistical findings for each

## Possible Improvements

- Include additional covariates (e.g., sleep duration, academic performance) if available, to control for confounding
- Extend the age segmentation to finer-grained bins to detect a more precise inflection point
- Validate findings on an external/held-out dataset to test generalizability
- Report confidence intervals around the optimized screening threshold, not just a point estimate
