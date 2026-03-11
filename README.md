# Machine_Learning_Foundations-Data_Preparation

**Author:** Rodrigo Blanco Maldonado  
**Model:** Logistic Regression  
**Dataset:** `bank-additional.csv`

This project explores through **data preparation + modeling pipeline** to train and evaluate a **Logistic Regression**
model that predicts whether a bank client will subscribe to a term deposit.

At a high level, the objective is to transform raw data into a structured,
model format and develop a predictive model capable of estimating the likelihood
that a client will subscribe to a term deposit.

The work emphasizes in machine learning foundation methodology,
ensuring that model development follows statistical practices and avoids common mistakes
such as information leakage or biased evaluation.

---

## Project Goals

- Understand the dataset structure, feature types, and target distribution.
- Build a model without data leakage, and correct workflow.
- Handle:
  - missing/unknown values,
  - categorical encoding,
  - feature selection,
  - feature scaling,
  - class imbalance.
- Train a Logistic Regression model and evaluate it with appropriate metrics.

---

## Prediction Objective

### Target: `y`
- `y = 1` → client subscribed to a term deposit  
- `y = 0` → client did not subscribe

The dataset is moderately imbalanced (≈ **10.95%** positive class).  
Because of this, **accuracy alone is misleading** and a zero-rule classifier predicting “no” achieves ~89% accuracy.

Recommended evaluation focus:
- Precision, F1-score, ROC-AUC
- Confusion matrix + ROC curve
- Mainly **Recall**

---

## Why Task Ordering Matters (Avoiding Data Leakage)

Logistic Regression assumptions and practical requirements:
- Predictors must be available at prediction time
- Avoid multicollinearity
- Standardization improves optimization and coefficient comparability
- **No leakage** between train/validation/test

Key leakage examples avoided by the workflow:
- **Scaling before splitting** leaks global distribution into training.
- **Target encoding before splitting** leaks target information from validation/test into training.

---

## Task Workflow

1. **Data Loading**
2. **Target & Feature Exploration**
3. **Feature Visualization vs Target**
4. **Data Splitting** (critical: do before transformations)
5. **Missing/Unknown Handling**
6. **Categorical Encoding** (e.g., TargetEncoder / OneHotEncoder as needed)
7. **Feature Selection** (variance, collinearity, suitability)
8. **Scaling** (StandardScaler on numeric features)
9. **Class Imbalance Handling** (RandomOverSampler on training only)
10. **Train Logistic Regression**
11. **Evaluation** (compare to baseline)

---

## Dataset Notes

### Features
Includes client attributes (`age`, `job`, `education`, etc.), campaign details (`campaign`, `previous`, etc.), and macro indicators (`euribor3m`, `nr.employed`, etc.).

### Unknown values
Several categorical columns contain `"unknown"` values.

### Special case
`pdays = 999` represents *“client was never contacted”*.  
Treating it as a normal numeric value is misleading; it should be transformed into a meaningful category/value (e.g., map `999 → -1` or create a flag).

### Potential leakage feature
Call `duration` is **not available at prediction time** (only known after the call ends), so it should not be used in a real-world predictive setting.

---

## Dependencies

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (`imblearn`)
- category-encoders (`TargetEncoder`)

---

`README structure by AI`