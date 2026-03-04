---
aliases: [AAPL Regression, OLS, Linear Regression]
tags: [machine-learning, regression, diagnostics, first-principles]
---

# OLS Regression with Diagnostics

## First Principles: What Is OLS?

Ordinary Least Squares finds the linear relationship $y = X\beta + \epsilon$ that minimizes the sum of squared residuals. In this project, it's used to predict **AAPL's next-day return** from lagged features.

The key question isn't just "can we fit a line?" but "is this linear model statistically valid?" — hence the extensive diagnostics.

## Feature Engineering

The `prepare_features()` function in `src/analytics/regression.py` builds five features:

1. **Lag_1_Return**: Yesterday's AAPL log return
2. **Lag_2_Return**: Two days ago
3. **Volatility_5d**: 5-day rolling standard deviation of returns
4. **MomentumRatio_5_20**: $\frac{SMA_5}{SMA_{20}} - 1$ — short-term trend strength
5. **Market_Lag1**: Yesterday's S&P 500 return (single market factor)

All features are in return/ratio space to avoid multicollinearity issues (raw prices would be highly correlated).

### Anti-Look-Ahead Bias Design

- **Chronological 80/20 split** (no shuffle) — past predicts future, never the reverse
- **Winsorization** uses only **training** percentiles — test data is clipped using train boundaries
- Target is `aapl_ret.shift(-1)` — next-day return, so all features are lagged by at least 1 day

## Diagnostic Tests

### Variance Inflation Factor (VIF)
Tests for **multicollinearity** — if VIF > 10, features are too correlated with each other, inflating standard errors and making coefficients unreliable. The code computes VIF for each feature by regressing it on all others.

### Breusch-Pagan Test
Tests for **heteroscedasticity** — whether residual variance is constant across predictions. Financial returns often violate this (see [[GARCH(1,1) Model]]). If p < 0.05, residuals are heteroscedastic.

### Durbin-Watson Statistic
Tests for **autocorrelation** in residuals. DW ≈ 2 means no autocorrelation; DW < 1.5 suggests positive autocorrelation; DW > 2.5 suggests negative.

### Jarque-Bera / Normal Test
Tests whether residuals are **normally distributed** — an assumption of OLS inference. Financial residuals typically fail this due to fat tails.

## Robust Standard Errors

Since financial data violates homoscedasticity and may have autocorrelation, the code automatically refits with **Newey-West HAC standard errors** (`_refit_robust()`). HAC (Heteroscedasticity and Autocorrelation Consistent) standard errors correct the coefficient p-values without changing the point estimates.

## Project Results

The regression yields **R² < 0** on the test set, meaning the model is worse than predicting the mean. This is not a failure — it's a validation of the [[Efficient Market Hypothesis]]: daily equity returns contain very little exploitable linear signal.

## Connections

- [[CAPM (Capital Asset Pricing Model)]] — Also an OLS regression, but with a single market factor
- [[ML Classification Pipeline]] — Complementary approach using nonlinear models
- [[Technical Indicators as Features]] — The regression uses simpler features; classification uses richer ones
- [[Tab 2 - Price Prediction]] — Visualizes actual vs predicted, residuals, QQ plot, VIF chart
- [[Log Returns]] — The target and features are all in log return space
