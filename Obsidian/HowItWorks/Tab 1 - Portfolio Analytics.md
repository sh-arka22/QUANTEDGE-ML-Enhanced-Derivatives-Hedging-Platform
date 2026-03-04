---
aliases: [Portfolio Tab]
tags: [ui, portfolio, tab]
---

# Tab 1 — Portfolio Analytics

## What It Shows

This tab provides a comprehensive risk profile of the equal-weighted banking portfolio (MS 33%, JPM 34%, BAC 33%) over 2015-2024.

## Metrics Displayed

### Header Cards
- [[Sharpe and Sortino Ratios|Sharpe Ratio]] and [[Sharpe and Sortino Ratios|Sortino Ratio]]
- [[CAPM (Capital Asset Pricing Model)|Beta]] and annualized [[CAPM (Capital Asset Pricing Model)|Alpha]]
- [[Maximum Drawdown]]

### Value at Risk Section
- [[Value at Risk (VaR)|Parametric VaR]] (95%)
- [[Value at Risk (VaR)|Historical VaR]] (95%)
- [[Conditional VaR (CVaR)|CVaR / Expected Shortfall]]
- Fat-tail warning when parametric and historical VaR diverge

## Charts

1. **Cumulative Returns**: Individual stocks (MS, JPM, BAC) plus the weighted portfolio — shows diversification benefit and relative performance
2. **Return Distribution Histogram**: Portfolio daily returns with VaR thresholds overlaid as vertical lines
3. **Rolling 30-Day Volatility**: Time series of annualized portfolio volatility — reveals [[GARCH(1,1) Model|volatility clustering]]
4. **Correlation Heatmap**: Pairwise correlation between banking stocks — shows high correlation (0.7-0.8), limiting diversification benefit

## Implementation

`src/ui/tab_portfolio.py` calls `risk_summary()` from `src/analytics/portfolio.py`, which orchestrates all [[Portfolio Returns and Weighting|weighted return]] calculations and risk metrics.

## Connections

- [[Portfolio Returns and Weighting]] — How the portfolio is constructed
- [[Streamlit Dashboard]] — Parent UI structure
- [[Project Architecture]] — Tab 1 in the overall system
