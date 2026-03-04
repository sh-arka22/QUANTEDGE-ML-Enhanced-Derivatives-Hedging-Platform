---
aliases: [EMH]
tags: [finance-theory, markets, first-principles]
---

# Efficient Market Hypothesis

## First Principles

The EMH states that asset prices fully reflect all available information. In its **semi-strong form**, this means:

- Historical prices and publicly available information are already incorporated into the current price
- You cannot systematically profit by trading on past price patterns or public news
- Any apparent pattern would be quickly exploited away by rational traders

## Three Forms

1. **Weak**: Prices reflect all past trading information (prices, volume). Technical analysis shouldn't work.
2. **Semi-strong**: Prices also reflect all publicly available information. Fundamental analysis shouldn't work.
3. **Strong**: Prices reflect all information, including insider information.

## How This Project Tests It

The QUANTEDGE platform provides empirical evidence supporting the weak and semi-strong forms:

### [[OLS Regression with Diagnostics]]
AAPL regression using lagged returns, volatility, and market factors yields **R² < 0** on the test set — the model cannot outperform a simple mean prediction. This supports weak-form efficiency.

### [[ML Classification Pipeline]]
Six models (DT, RF, KNN, SVM, Ensemble) trained on 18+ [[Technical Indicators as Features]] achieve **AUC ≈ 0.50** for BAC direction prediction — essentially random. Even nonlinear models with sophisticated features cannot extract a tradable signal from daily price data.

### Why the ML Results Validate EMH

If markets were inefficient, we'd expect at least some models to achieve AUC > 0.55. The fact that multiple diverse models all converge to ~0.50 strongly suggests the signal simply isn't there in the data — not that the models are misconfigured.

## The Nuance

EMH doesn't mean all prediction is impossible — it means the **risk-adjusted** expected return from prediction is zero. The [[GARCH(1,1) Model]] successfully forecasts volatility (not returns), which is consistent with EMH: volatility is forecastable, returns are not.

## Connections

- [[OLS Regression with Diagnostics]] — Linear evidence for EMH
- [[ML Classification Pipeline]] — Nonlinear evidence for EMH
- [[GARCH(1,1) Model]] — Volatility is forecastable even if returns aren't
- [[Delta-Neutral Hedging]] — Hedging is about managing risk, not predicting direction
