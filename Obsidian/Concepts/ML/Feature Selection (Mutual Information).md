---
aliases: [MI Feature Selection, Mutual Information]
tags: [machine-learning, feature-engineering, first-principles]
---

# Feature Selection (Mutual Information)

## First Principles

Not all features are useful. Including irrelevant features adds noise, increases computation, and can hurt model performance (curse of dimensionality). **Mutual information** measures how much knowing a feature reduces uncertainty about the target.

$$MI(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x) p(y)}$$

MI = 0 means the feature and target are completely independent. MI > 0 means they share information. Unlike correlation, MI captures **nonlinear** relationships.

## How This Project Uses It

In `src/analytics/classification.py`:

### `compute_mutual_info()`
Uses `sklearn.feature_selection.mutual_info_classif()` to score each feature against the binary target (BAC up/down).

### `select_features()`
Drops features with MI below `MI_THRESHOLD = 0.001`. If no features pass the threshold, all features are kept (fallback to prevent empty feature sets).

This step runs **after** the chronological train/test split but **before** model training, ensuring selection is based only on training data.

## Connections

- [[ML Classification Pipeline]] — MI selection is step 2 of the pipeline
- [[Technical Indicators as Features]] — The candidate features being filtered
- [[Hyperparameter Tuning (TimeSeriesSplit)]] — Feature selection complements hyperparameter tuning
