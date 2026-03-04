---
aliases: [Sharpe Ratio, Sortino Ratio, Risk-Adjusted Returns]
tags: [portfolio, risk, performance, first-principles]
---

# Sharpe and Sortino Ratios

## First Principles: Why Not Just Use Returns?

Raw return alone doesn't tell you if a strategy is good — a 20% return with 50% volatility is much worse than 15% return with 10% volatility. We need to **normalize returns by the risk taken** to compare strategies fairly.

## Sharpe Ratio

$$\text{Sharpe} = \frac{E[R_p - R_f]}{\sigma(R_p - R_f)} \times \sqrt{252}$$

**Intuition**: Excess return per unit of total volatility, annualized. A Sharpe of 1.0 means you earn one standard deviation of excess return per year. The $\sqrt{252}$ annualization comes from the fact that standard deviation scales with $\sqrt{T}$.

**Limitation**: Sharpe penalizes **upside** volatility equally with downside volatility. If a strategy has big upswings, Sharpe makes it look worse than it really is.

## Sortino Ratio

$$\text{Sortino} = \frac{E[R_p - R_f]}{\sigma_{downside}} \times \sqrt{252}$$

**Intuition**: Same idea as Sharpe, but only penalizes **downside** deviation (returns below a minimum acceptable return, usually 0). This better captures the asymmetric risk profile that investors actually care about.

**Downside deviation** = $\sqrt{E[\min(R - MAR, 0)^2]}$ where MAR is the minimum acceptable return.

## How This Project Implements Them

In `src/analytics/portfolio.py`:

### `sharpe_ratio()`
- Subtracts the daily risk-free rate ($r_{annual}/252$) from each daily return
- Computes the standard deviation with `ddof=1` (sample correction)
- Returns `NaN` if std < 1e-8 (flat returns → undefined ratio)

### `sortino_ratio()`
- Filters to returns below the MAR (default 0)
- Requires at least 5 downside observations (otherwise returns `NaN`)
- Uses RMS of downside returns (not standard deviation) as the downside deviation

Both are displayed in [[Tab 1 - Portfolio Analytics]] as headline metrics.

## Project Results

The banking portfolio (MS + JPM + BAC) over 2015-2024 showed moderate risk-adjusted returns, with the Sharpe and Sortino ratios reflecting the high-beta nature of bank stocks relative to the S&P 500.

## Connections

- [[Portfolio Returns and Weighting]] — The returns being measured
- [[CAPM (Capital Asset Pricing Model)]] — Another way to decompose risk-adjusted return
- [[Value at Risk (VaR)]] — Complements Sharpe by quantifying tail risk
- [[Maximum Drawdown]] — Captures peak-to-trough risk that Sharpe misses
