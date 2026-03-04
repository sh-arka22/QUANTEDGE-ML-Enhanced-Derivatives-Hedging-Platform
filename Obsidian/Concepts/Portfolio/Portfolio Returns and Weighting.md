---
aliases: [Portfolio Construction, Weighted Returns]
tags: [portfolio, risk, first-principles]
---

# Portfolio Returns and Weighting

## First Principles

A portfolio is a collection of assets held in certain proportions. The portfolio's return at any point in time is the **weighted sum** of its constituent assets' returns:

$$R_p = \sum_{i=1}^{n} w_i \cdot R_i$$

where $w_i$ is the weight of asset $i$ and $R_i$ is its return. The weights must sum to 1 (fully invested) for a standard long-only portfolio.

## How This Project Uses It

The platform tracks a **banking sector portfolio** of three stocks:
- Morgan Stanley (MS): 33%
- JPMorgan (JPM): 34%
- Bank of America (BAC): 33%

Defined in `src/data/config.py` as `PORTFOLIO_WEIGHTS = {"MS": 0.33, "JPM": 0.34, "BAC": 0.33}`.

The `weighted_returns()` function in `src/analytics/portfolio.py`:
1. Validates that weights are non-zero and match available columns
2. Auto-normalizes weights to sum to 1.0 (with a warning if they don't already)
3. Computes the weighted sum via matrix multiplication: `returns_df[common].values @ w`

These portfolio returns then feed into all the risk metrics: [[Sharpe and Sortino Ratios]], [[CAPM (Capital Asset Pricing Model)]], [[Value at Risk (VaR)]], [[Conditional VaR (CVaR)]], and [[Maximum Drawdown]].

## Why Log Returns?

The project uses [[Log Returns]] throughout. Log returns have the property that multi-period returns are additive: $r_{t_1 \to t_3} = r_{t_1 \to t_2} + r_{t_2 \to t_3}$. However, log returns of a portfolio are **not** exactly the weighted sum of individual log returns — this is an approximation that works well for daily returns (which are small).

## Connections

- [[Log Returns]] — The return computation method used
- [[Sharpe and Sortino Ratios]] — Risk-adjusted performance of the weighted portfolio
- [[CAPM (Capital Asset Pricing Model)]] — Decomposes portfolio return into alpha and beta
- [[Tab 1 - Portfolio Analytics]] — Visualizes cumulative growth and individual stock performance
