---
aliases: [BAC Classification, Direction Prediction, ML Models]
tags: [machine-learning, classification, first-principles]
---

# ML Classification Pipeline

## First Principles: Direction Prediction

Instead of predicting the exact return (regression), this module predicts the **direction** of BAC's next-day move: up (1) or down (0). This is a binary classification problem.

Why BAC (Bank of America)? It's in the banking portfolio, and direction prediction is more forgiving than exact return prediction — you don't need to know the magnitude, just the sign.

## The Full Pipeline

The orchestrator `run_classification()` in `src/analytics/classification.py` runs:

### 1. Feature Engineering
Uses 18+ [[Technical Indicators as Features]]: lagged returns, RSI, MACD, Bollinger Bands, Stochastic Oscillator, Rate of Change, realized volatility, SMA ratio, Z-score, and optionally FRED macro data. See [[Technical Indicators as Features]] for details.

### 2. Feature Selection via [[Feature Selection (Mutual Information)]]
Computes mutual information between each feature and the target, drops features below the MI threshold (0.001 default).

### 3. Scaling
**RobustScaler** (fitted on train only) — uses median and IQR instead of mean/std, making it resistant to outliers. This is important for financial data which has fat tails.

### 4. Model Training
Four base models, all with `class_weight="balanced"` to handle potential class imbalance:

| Model | Key Parameters | Strengths |
|-------|---------------|-----------|
| **Decision Tree** | max_depth=5, min_samples_leaf=20 | Interpretable, captures nonlinear splits |
| **Random Forest** | 100 trees, max_depth=10, OOB score | Ensemble reduces variance, gives feature importance |
| **KNN** | k=5 neighbors | Non-parametric, no assumptions about data distribution |
| **SVM** | RBF kernel, C=1.0, probability=True | Good for high-dimensional data, captures complex boundaries |

### 5. Ensemble: Soft Voting
A **VotingClassifier** combines RF + KNN + SVM (DT excluded for lower accuracy) using soft voting — averaging predicted probabilities rather than hard predictions. This typically improves robustness.

### 6. Optional: [[Hyperparameter Tuning (TimeSeriesSplit)]]
`RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)` — the correct cross-validation strategy for time series (no future data leaks into training folds).

### 7. Evaluation
Computes accuracy, precision, recall, F1, ROC-AUC, and confusion matrix per model.

## Project Results

All models achieve AUC ≈ 0.50, confirming that **daily BAC direction is essentially unpredictable** — consistent with the [[Efficient Market Hypothesis]]. The classification results mirror the [[OLS Regression with Diagnostics]] findings.

## Connections

- [[Technical Indicators as Features]] — The feature engineering layer
- [[Feature Selection (Mutual Information)]] — Pre-training feature filtering
- [[Hyperparameter Tuning (TimeSeriesSplit)]] — Avoiding temporal leakage in CV
- [[OLS Regression with Diagnostics]] — Linear complement to nonlinear classification
- [[Tab 2 - Price Prediction]] — ROC curves, confusion matrices, feature importance charts
- [[Efficient Market Hypothesis]] — The economic interpretation of near-random results
