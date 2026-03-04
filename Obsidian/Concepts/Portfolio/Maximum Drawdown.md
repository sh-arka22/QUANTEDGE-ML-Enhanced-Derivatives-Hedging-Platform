---
aliases: [MDD, Max Drawdown]
tags: [portfolio, risk, first-principles]
---

# Maximum Drawdown

## First Principles

Maximum drawdown measures the **worst peak-to-trough decline** in portfolio value over a period. If your portfolio grew from $100 to $150, then fell to $80, the drawdown from peak is $(150 - 80) / 150 = 46.7\%$.

MDD answers the question: "If I invested at the worst possible time, how much would I have lost before recovering?"

## How It's Computed

$$\text{MDD} = \max_t \frac{\text{Peak}(t) - \text{Value}(t)}{\text{Peak}(t)}$$

where $\text{Peak}(t) = \max_{\tau \leq t} \text{Value}(\tau)$ is the running maximum.

## Why It Matters

- MDD captures **sequence risk** — two portfolios can have the same Sharpe ratio but very different drawdowns
- It's psychologically meaningful: investors feel losses more than gains, and a 50% drawdown requires a 100% gain to recover
- Hedge funds often gate redemptions during large drawdowns

## How This Project Implements It

In `src/analytics/portfolio.py`, `max_drawdown()`:

1. Takes cumulative returns as input (product of (1 + daily returns))
2. Computes the running maximum using `np.maximum.accumulate()`
3. Calculates drawdowns as `(peak - current) / peak`
4. Returns the maximum drawdown

The banking portfolio (MS + JPM + BAC) showed an MDD of approximately 50.8% over 2015-2024, reflecting the severe drawdowns during COVID-19 and other stress periods.

## Connections

- [[Sharpe and Sortino Ratios]] — Complementary risk measures
- [[Value at Risk (VaR)]] — VaR is a single-day measure; MDD captures cumulative path risk
- [[Portfolio Returns and Weighting]] — The return series being analyzed
- [[Tab 1 - Portfolio Analytics]] — MDD displayed as a headline metric
