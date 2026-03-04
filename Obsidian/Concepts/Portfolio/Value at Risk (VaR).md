---
aliases: [VaR, Value-at-Risk]
tags: [portfolio, risk, first-principles]
---

# Value at Risk (VaR)

## First Principles

VaR answers a simple question: **"What is the maximum loss I can expect over a given time horizon, at a given confidence level?"**

For example, a 1-day 95% VaR of -2.3% means: "On 95% of days, the portfolio won't lose more than 2.3%. On the worst 5% of days, losses can exceed this."

VaR is expressed as a **negative number** (a loss threshold).

## Two Methods

### Parametric VaR (Variance-Covariance)
Assumes returns follow a **normal distribution**:
$$\text{VaR}_\alpha = \mu + z_\alpha \cdot \sigma$$

where $z_\alpha$ is the z-score at confidence level $\alpha$ (e.g., $z_{0.95} \approx -1.645$).

**Advantage**: Fast, closed-form. **Weakness**: Underestimates risk if returns have fat tails (which financial returns almost always do).

### Historical VaR
Simply takes the empirical percentile of historical returns:
$$\text{VaR}_\alpha = \text{Percentile}(R, (1-\alpha) \times 100)$$

**Advantage**: No distributional assumption — captures fat tails, skew, and any empirical pattern. **Weakness**: Only as good as the historical sample.

## Fat-Tail Detection

This project compares both methods. When they diverge significantly (|parametric VaR - historical VaR| / |historical VaR| > 20%), it flags **kurtosis risk** — the returns distribution has fatter tails than the normal assumption, meaning parametric VaR underestimates true risk.

## How This Project Implements It

In `src/analytics/portfolio.py`:

- `var_parametric()`: Uses `scipy.stats.norm.ppf(1 - confidence)` to get the z-score, then computes $\mu + z \cdot \sigma$
- `var_historical()`: Uses `np.percentile(returns, (1-confidence) * 100)`. Warns if fewer than 100 observations.

Both are shown in [[Tab 1 - Portfolio Analytics]] with the [[Conditional VaR (CVaR)]] alongside them, plus a histogram with VaR lines overlaid.

## Connections

- [[Conditional VaR (CVaR)]] — Extends VaR by asking "given we're beyond VaR, how bad can it get?"
- [[Portfolio Returns and Weighting]] — The return distribution being measured
- [[Maximum Drawdown]] — A different risk measure: worst cumulative loss
- [[CAPM (Capital Asset Pricing Model)]] — High beta amplifies VaR
