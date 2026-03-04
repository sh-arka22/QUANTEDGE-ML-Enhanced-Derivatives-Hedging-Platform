---
aliases: [QUANTEDGE MOC, Project Overview]
tags: [MOC, quantedge, project]
---

# QUANTEDGE — ML-Enhanced Derivatives Hedging Platform

A production-grade quantitative finance platform built in Python 3.11, covering portfolio risk analytics, OLS regression with diagnostics, ML classification, GARCH volatility modeling, Black-Scholes and binomial tree derivatives pricing, and delta-neutral hedging simulation — all accessible through an interactive 4-tab Streamlit dashboard.

**Repository**: [GitHub](https://github.com/sh-arka22/QUANTEDGE-ML-Enhanced-Derivatives-Hedging-Platform)

---

## Core Concepts (First Principles)

These notes explain each concept from the ground up — what it is, why it matters, and then how this project uses it.

### Derivatives Pricing
- [[Black-Scholes Model]] — Closed-form option pricing via risk-neutral valuation
- [[Option Greeks]] — Sensitivity measures (Delta, Gamma, Vega, Theta, Rho)
- [[Binomial Tree (CRR)]] — Lattice-based pricing for European and American options
- [[Implied Volatility]] — Backing out market's volatility expectation from observed prices
- [[Volatility Surface]] — The full (Strike × Expiry) map of implied volatilities
- [[Put-Call Parity]] — The no-arbitrage relationship between calls and puts

### Portfolio Risk Analytics
- [[Portfolio Returns and Weighting]] — How portfolio returns are computed from constituent assets
- [[Sharpe and Sortino Ratios]] — Risk-adjusted return measures
- [[CAPM (Capital Asset Pricing Model)]] — Decomposing returns into alpha and beta
- [[Value at Risk (VaR)]] — Parametric and historical loss estimation
- [[Conditional VaR (CVaR)]] — Expected Shortfall beyond the VaR threshold
- [[Maximum Drawdown]] — Worst peak-to-trough decline

### Volatility Modeling
- [[GARCH(1,1) Model]] — Conditional volatility with volatility clustering
- [[EWMA Volatility]] — Exponentially weighted moving average as a simpler alternative
- [[ARCH LM Test]] — Testing whether ARCH effects exist in return series
- [[Volatility Cones]] — Realized vol distribution across rolling windows
- [[Volatility Forecasting]] — Predicting future vol from fitted GARCH

### Machine Learning
- [[OLS Regression with Diagnostics]] — Linear regression for AAPL return prediction
- [[ML Classification Pipeline]] — BAC direction prediction using DT, RF, KNN, SVM, Ensemble
- [[Technical Indicators as Features]] — RSI, MACD, Bollinger Bands, Stochastic Oscillator
- [[Feature Selection (Mutual Information)]] — Filtering features by information content
- [[Hyperparameter Tuning (TimeSeriesSplit)]] — Avoiding look-ahead bias in cross-validation

### Hedging
- [[Delta-Neutral Hedging]] — The core hedging simulation strategy
- [[Rebalance Bands and Transaction Costs]] — Balancing hedge accuracy vs. trading friction
- [[Gamma Scalping]] — Profiting from gamma when holding delta-neutral positions

### Data Pipeline
- [[Data Ingestion Layer]] — yfinance equity + FRED macro with retry logic
- [[Log Returns]] — Why log returns are used instead of simple returns
- [[Asof Join and Look-Ahead Bias]] — How macro data is aligned without information leakage

---

## How the Project Fits Together

- [[Project Architecture]] — Module map and data flow
- [[Streamlit Dashboard]] — The 4-tab interactive UI
- [[Tab 1 - Portfolio Analytics]] — Risk metrics, cumulative returns, correlation
- [[Tab 2 - Price Prediction]] — Regression + Classification
- [[Tab 3 - Derivatives Pricing]] — BS, CRR, Greeks, 3D surface
- [[Tab 4 - Hedging Simulator]] — Delta-neutral simulation with band optimization

---

## Key Insights from the Project

The project validates several important financial principles empirically:

1. **EMH Validation**: The regression R² < 0 and classification AUC ≈ 0.50 on BAC confirm that daily equity returns are largely unpredictable — consistent with the [[Efficient Market Hypothesis]].

2. **Volatility Clustering**: The [[ARCH LM Test]] confirms ARCH effects in JPM returns, justifying the use of [[GARCH(1,1) Model]] over simple historical vol.

3. **Hedging Costs**: The [[Rebalance Bands and Transaction Costs]] analysis shows that tighter bands improve hedge quality but increase costs — a real-world tradeoff every derivatives desk faces.

4. **Binomial Convergence**: The CRR tree with N ≥ 200 steps converges to within 1 cent of the [[Black-Scholes Model]] price, demonstrating the theoretical link between discrete and continuous models.
