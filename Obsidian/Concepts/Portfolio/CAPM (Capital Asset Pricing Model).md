---
aliases: [CAPM, Alpha, Beta]
tags: [portfolio, risk, regression, first-principles]
---

# CAPM (Capital Asset Pricing Model)

## First Principles

CAPM says that any asset's expected excess return is proportional to the market's excess return:

$$R_p - R_f = \alpha + \beta \cdot (R_m - R_f) + \epsilon$$

Where:
- **Beta (β)**: The asset's sensitivity to market movements. β = 1 means the asset moves with the market; β > 1 means it amplifies market moves; β < 1 means it dampens them.
- **Alpha (α)**: The return unexplained by market exposure — "skill" or "anomaly." Positive alpha means outperformance after adjusting for market risk.
- **R²**: How much of the portfolio's variance is explained by the market.

## Intuition

Think of beta as a **leverage multiplier on market risk**. A portfolio with β = 1.22 acts like holding 1.22x the market — it goes up more in bull markets but also falls harder in bear markets. Alpha captures whether the portfolio adds value beyond this passive market exposure.

## How This Project Implements It

In `src/analytics/portfolio.py`, the `capm()` function:

1. Runs OLS regression: $R_p = \alpha + \beta \cdot R_m$ using `statsmodels`
2. Returns alpha, beta, R², and p-values for both coefficients
3. Handles length mismatches by trimming to the shorter series
4. Returns NaN values on any fitting failure

## Project Results

The banking portfolio (MS + JPM + BAC) shows:
- **β ≈ 1.22**: The portfolio amplifies market movements by ~22%, expected for financials which are pro-cyclical
- **α**: Close to zero, suggesting no significant outperformance beyond market exposure
- **R² ≈ 0.68**: About 68% of portfolio variance is explained by the S&P 500

These results are displayed in [[Tab 1 - Portfolio Analytics]] alongside [[Sharpe and Sortino Ratios]].

## Connections

- [[Portfolio Returns and Weighting]] — The portfolio returns being regressed
- [[OLS Regression with Diagnostics]] — CAPM is fundamentally a linear regression
- [[Sharpe and Sortino Ratios]] — Different approaches to risk-adjusted return
- [[Value at Risk (VaR)]] — Beta amplification means VaR is larger than market VaR × weight
