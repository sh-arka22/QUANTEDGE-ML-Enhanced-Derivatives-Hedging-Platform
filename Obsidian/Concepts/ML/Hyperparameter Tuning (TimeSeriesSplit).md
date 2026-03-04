---
aliases: [TimeSeriesSplit, Cross-Validation, Hyperparameter Tuning]
tags: [machine-learning, validation, first-principles]
---

# Hyperparameter Tuning (TimeSeriesSplit)

## First Principles: Why Not Regular Cross-Validation?

Standard k-fold CV randomly splits data into folds. For time series, this creates **look-ahead bias** — the model trains on future data and tests on past data, which inflates performance estimates.

**TimeSeriesSplit** preserves temporal order: each fold uses earlier data for training and later data for testing, expanding the training window with each fold.

```
Fold 1: [Train: ——] [Test: —]
Fold 2: [Train: ————] [Test: —]
Fold 3: [Train: ——————] [Test: —]
Fold 4: [Train: ————————] [Test: —]
Fold 5: [Train: ——————————] [Test: —]
```

## How This Project Implements It

In `src/analytics/classification.py`, `tune_models()`:

1. Uses `TimeSeriesSplit(n_splits=5)` from sklearn
2. Wraps each model in `RandomizedSearchCV` with `scoring="f1"`
3. Tests hyperparameter combinations:
   - **Decision Tree**: max_depth [3,5,7,10], min_samples_leaf [10,20,50]
   - **Random Forest**: n_estimators [50,100,200], max_depth [5,10,15], max_features [sqrt, log2]
   - **KNN**: n_neighbors [3,5,7,11], weights [uniform, distance], metric [euclidean, manhattan]
   - **SVM**: C [0.1,1,10], gamma [scale, auto]
4. Caps n_iter at the smaller of `TUNING_N_ITER=20` and the total grid size

## Connections

- [[ML Classification Pipeline]] — Tuning is an optional step in the pipeline
- [[Feature Selection (Mutual Information)]] — Features are selected before tuning
- [[Asof Join and Look-Ahead Bias]] — Both address information leakage from different angles
