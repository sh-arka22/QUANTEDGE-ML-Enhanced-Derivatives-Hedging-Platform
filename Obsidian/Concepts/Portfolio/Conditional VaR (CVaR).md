---
aliases: [CVaR, Expected Shortfall, ES]
tags: [portfolio, risk, tail-risk, first-principles]
---

# Conditional VaR (CVaR)

## First Principles

VaR tells you the **threshold** of loss at a given confidence level, but it doesn't tell you what happens when you breach that threshold. [[Value at Risk (VaR)]] says "you won't lose more than X on 95% of days" — but on the worst 5% of days, how bad is it?

**CVaR (Expected Shortfall)** answers this: it's the **average loss in the tail** beyond the VaR threshold.

$$\text{CVaR}_\alpha = E[R \mid R \leq \text{VaR}_\alpha]$$

CVaR is always worse (more negative) than VaR because it averages over the entire tail, not just the edge.

## Why CVaR Is Better Than VaR

1. **Coherent risk measure**: CVaR satisfies subadditivity (diversification reduces risk), while VaR does not. This means VaR can perversely suggest that combining two portfolios increases risk.
2. **Tail sensitivity**: CVaR captures the severity of extreme losses, not just their probability.
3. **Regulatory preference**: Basel III moved toward Expected Shortfall for bank capital requirements.

## How This Project Implements It

In `src/analytics/portfolio.py`, `cvar()`:

1. Computes the historical VaR threshold
2. Filters returns to those at or below the VaR
3. Returns the mean of those tail returns
4. Returns `NaN` if no returns fall below VaR (very short series)

## Connections

- [[Value at Risk (VaR)]] — CVaR extends VaR into the tail
- [[Portfolio Returns and Weighting]] — The return distribution being analyzed
- [[Tab 1 - Portfolio Analytics]] — CVaR displayed alongside parametric and historical VaR
