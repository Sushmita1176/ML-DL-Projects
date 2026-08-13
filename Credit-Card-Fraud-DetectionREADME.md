# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using multiple classification algorithms, with a focus on handling severe class imbalance.

## Overview

Credit card fraud detection is a classic **imbalanced classification** problem — fraudulent transactions make up a tiny fraction of all transactions. This project compares five different classifiers on a real-world-style transaction dataset, using **SMOTE (Synthetic Minority Oversampling Technique)** to address the imbalance, and evaluates models using metrics suited to rare-event detection (F1-score, ROC-AUC) rather than raw accuracy, which can be misleading on imbalanced data.

## Dataset

- 27,819 transactions with 30 anonymized numerical features (`V1`–`V28`, `Time`, `Amount`) plus a binary `Class` label
- **Class distribution**: 99.67% genuine transactions (Class 0) vs. only 0.33% fraudulent (Class 1) — a highly imbalanced dataset
- Missing values (1 row) were dropped
- Data split into training (80%), validation (10%), and test (10%) sets using stratified sampling to preserve class ratios

## Handling Class Imbalance

Since fraud cases are extremely rare, training a model directly on the raw data would bias it toward always predicting "genuine." To address this:
- **SMOTE** was applied to the training set only, generating synthetic examples of the minority (fraud) class
- This balanced the training set to an equal number of genuine and fraudulent samples, while validation/test sets were left untouched to reflect real-world class distribution

## Models Compared

Five classifiers were trained and evaluated on the same SMOTE-balanced training data:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

## Evaluation Metrics

Given the class imbalance, models were compared using:
- Accuracy
- Precision
- Recall
- **F1-score** (primary metric for selecting the best model)
- ROC-AUC

## Results

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.9993 | 0.818 | 1.000 | **0.900** | 0.9999 |
| Decision Tree | 0.9975 | 0.571 | 0.889 | 0.696 | 0.9434 |
| Logistic Regression | 0.9925 | 0.300 | 1.000 | 0.462 | 0.9978 |
| Naive Bayes | 0.9788 | 0.132 | 1.000 | 0.234 | 0.9951 |
| KNN | 0.9659 | 0.070 | 0.778 | 0.128 | 0.8770 |

**Random Forest** was selected as the best-performing model, achieving the highest F1-score (0.900) with perfect recall (catching 100% of fraud cases in the test set) and strong precision.

### Confusion Matrix (Random Forest, Test Set)

|  | Predicted Genuine | Predicted Fraud |
|---|---|---|
| **Actual Genuine** | 2714 | 59 |
| **Actual Fraud** | 0 | 9 |

The model correctly identified all 9 fraudulent transactions in the test set with zero false negatives, at the cost of 59 false positives — a reasonable trade-off for fraud detection, where missing actual fraud is far costlier than flagging a genuine transaction for review.

## Tech Stack

- Python
- scikit-learn (classifiers, metrics, train/test split)
- imbalanced-learn (SMOTE)
- Pandas, NumPy
- Matplotlib

## How to Run

1. Ensure `creditcard.csv` is available in the working directory
2. Install dependencies: `pip install imbalanced-learn scikit-learn pandas numpy matplotlib`
3. Run the notebook cells sequentially

## Possible Improvements

- Perform hyperparameter tuning (e.g., `GridSearchCV`) on Random Forest to further improve F1-score
- Try ensemble/boosting methods like XGBoost or LightGBM, which often perform well on imbalanced tabular data
- Experiment with alternative imbalance-handling techniques (e.g., class weighting, ADASYN) for comparison against SMOTE
- Use cost-sensitive learning to explicitly penalize false negatives more heavily, given the real-world cost asymmetry in fraud detection
