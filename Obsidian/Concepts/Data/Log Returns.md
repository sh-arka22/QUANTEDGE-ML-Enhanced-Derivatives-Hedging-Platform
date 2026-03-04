---
aliases: [Logarithmic Returns, Continuously Compounded Returns]
tags: [data, finance, first-principles]
---

# Log Returns

## First Principles

There are two ways to compute returns:

**Simple return**: $r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$

**Log return**: $r_t = \ln\frac{P_t}{P_{t-1}}$

This project uses log returns throughout. Here's why:

## Advantages of Log Returns

1. **Time-additivity**: Multi-period log returns are the sum of single-period returns: $r_{1 \to 3} = r_{1 \to 2} + r_{2 \to 3}$. This makes rolling calculations and cumulative computations cleaner.

2. **Symmetry**: A +10% move followed by -10% move doesn't return to the starting price with simple returns, but does approximately with log returns. This is more intuitive for statistical modeling.

3. **Approximate normality**: Log returns are more normally distributed than simple returns, which matters for [[GARCH(1,1) Model]], [[OLS Regression with Diagnostics]], and [[Value at Risk (VaR)]].

4. **Continuous compounding**: Log returns correspond to continuous compounding, which aligns with the [[Black-Scholes Model]]'s assumption of geometric Brownian motion.

## How This Project Computes Them

In `src/data/loaders.py`, `compute_log_returns()`:
```python
returns = np.log(prices / prices.shift(1)).dropna()
```

Validates that all prices are strictly positive (required for the logarithm).

## The Approximation

For small returns (< ~5%), log and simple returns are nearly identical: $\ln(1+r) \approx r$. For daily equity returns (typically < 2%), the difference is negligible. The approximation breaks down for large moves (crashes, IPO pops).

## Connections

- [[Portfolio Returns and Weighting]] — Portfolio log returns are approximately the weighted sum of individual log returns
- [[OLS Regression with Diagnostics]] — Target and features are in log return space
- [[GARCH(1,1) Model]] — Models the conditional distribution of log returns
- [[Data Ingestion Layer]] — `compute_log_returns()` is part of the data pipeline
