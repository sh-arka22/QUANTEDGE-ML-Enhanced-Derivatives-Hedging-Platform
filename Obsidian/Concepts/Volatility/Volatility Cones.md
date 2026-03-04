---
aliases: [Vol Cones, Realized Vol Distribution]
tags: [volatility, analytics, first-principles]
---

# Volatility Cones

## First Principles

A volatility cone shows the **distribution of realized volatility** across different rolling windows. For each window size (e.g., 10, 30, 60, 90 days), you compute the rolling standard deviation of returns over the entire history, then report percentiles: min, 25th, median, 75th, max.

This creates a "cone" shape: shorter windows show wider dispersion (more noise), longer windows converge toward the mean.

## Why It Matters

Volatility cones help you understand **where current vol sits relative to history**. If current 30-day realized vol is at the 90th percentile, vol is historically elevated — which might inform option pricing decisions (is [[Implied Volatility]] too low?).

## How This Project Implements It

In `src/analytics/volatility.py`, `volatility_cones()`:

1. Computes rolling standard deviation at windows [10, 30, 60, 90] days
2. Annualizes by multiplying by $\sqrt{252}$
3. Returns a DataFrame with min, q25, median, q75, max for each window
4. Skips windows that exceed the data length

## Connections

- [[GARCH(1,1) Model]] — Conditional vol vs realized vol comparison
- [[EWMA Volatility]] — Another way to estimate current vol
- [[Volatility Forecasting]] — Cones provide context for whether forecasted vol is extreme
