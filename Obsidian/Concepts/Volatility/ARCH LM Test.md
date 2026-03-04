---
aliases: [Engle's ARCH Test, ARCH Effects Test]
tags: [volatility, diagnostics, econometrics, first-principles]
---

# ARCH LM Test

## First Principles

Before fitting a [[GARCH(1,1) Model]], you need to check whether ARCH effects (volatility clustering) actually exist in the data. The ARCH Lagrange Multiplier test does this by regressing squared residuals on their own lags and testing if the coefficients are jointly significant.

**Null hypothesis**: No ARCH effects (homoscedastic returns — constant variance).
**Alternative**: ARCH effects present (variance changes over time).

If p-value < 0.05, we reject the null → ARCH effects exist → GARCH modeling is justified.

## How This Project Implements It

In `src/analytics/volatility.py`, `arch_lm_test()`:

1. Uses `statsmodels.stats.diagnostic.het_arch()` with configurable lags (default 5)
2. Automatically reduces lags if the series is too short (needs at least `2 × lags` observations)
3. Returns the test statistic, p-value, and a boolean `has_arch_effects`

This is called as the first step in `run_volatility()` before any GARCH fitting.

## Connections

- [[GARCH(1,1) Model]] — ARCH test justifies whether GARCH fitting is appropriate
- [[EWMA Volatility]] — If no ARCH effects, even EWMA may be overkill vs simple rolling vol
