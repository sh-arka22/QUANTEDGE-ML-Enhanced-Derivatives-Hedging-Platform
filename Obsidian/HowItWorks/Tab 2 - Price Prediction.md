---
aliases: [Prediction Tab]
tags: [ui, machine-learning, tab]
---

# Tab 2 — Price Prediction

## What It Shows

This tab runs two complementary prediction pipelines and displays their results side-by-side, collectively testing the [[Efficient Market Hypothesis]].

## Section 1: AAPL Regression

Uses [[OLS Regression with Diagnostics]] to predict next-day AAPL returns.

### Metrics Cards
- RMSE, R-squared (negative!), MAE
- Durbin-Watson statistic, Breusch-Pagan p-value, Jarque-Bera p-value, Max VIF

### Charts
- **Actual vs Predicted** scatter with perfect-prediction line — shows nearly random scatter
- **Residual plot** — should be random around zero (and is)
- **QQ Plot** — compares residual distribution to normal; fat tails visible
- **VIF Bar Chart** — shows multicollinearity levels per feature with threshold line

## Section 2: BAC Classification

Uses [[ML Classification Pipeline]] to predict next-day BAC direction (up/down).

### Model Comparison Table
Accuracy, Precision, Recall, F1, AUC-ROC for all five models (DT, RF, KNN, SVM, Voting Ensemble).

### Charts
- **ROC Curves** — All models hugging the diagonal (AUC ≈ 0.50)
- **Feature Importance** — Random Forest's feature importances ranked horizontally
- **Confusion Matrices** — 2×2 heatmaps for each base model
- **Class Distribution Pie** — Training set up/down split (~50/50)

## Key Insight

Both sections demonstrate that daily equity returns are essentially unpredictable with publicly available data — a strong empirical validation of market efficiency.

## Connections

- [[OLS Regression with Diagnostics]] — Regression pipeline details
- [[ML Classification Pipeline]] — Classification pipeline details
- [[Efficient Market Hypothesis]] — The economic interpretation
- [[Streamlit Dashboard]] — Parent UI structure
