---
aliases: [EWMA, Exponentially Weighted Moving Average]
tags: [volatility, time-series, first-principles]
---

# EWMA Volatility

## First Principles

EWMA is a **simple, parameterless** (almost) volatility estimator that gives more weight to recent observations. It's essentially a special case of [[GARCH(1,1) Model]] where $\alpha + \beta = 1$ (integrated GARCH), meaning there's no mean-reversion — the latest observations dominate.

$$\sigma_t^2 = \lambda \cdot \sigma_{t-1}^2 + (1 - \lambda) \cdot r_{t-1}^2$$

Where $\lambda$ is the decay factor. The industry standard (set by RiskMetrics/JP Morgan) is $\lambda = 0.94$.

## Intuition

EWMA is like a moving average of squared returns, but with exponentially decaying weights. Yesterday's squared return gets weight $(1-\lambda) = 0.06$, the day before gets $0.06 \times 0.94 = 0.056$, and so on. The effective "memory" is about $1/(1-\lambda) = 16.7$ days.

## How This Project Uses It

In `src/analytics/volatility.py`, `fit_ewma()`:

1. Computes squared returns
2. Applies pandas `.ewm(alpha=1-lambda, adjust=False).mean()` — an exponentially weighted average
3. Takes the square root to get daily volatility
4. Annualizes by multiplying by $\sqrt{252}$
5. Forward-fills any NaN and fills remaining with 0

The decay factor defaults to `EWMA_LAMBDA = 0.94` from config, with clamping if the user provides an out-of-range value.

## EWMA vs GARCH

| Aspect | EWMA | GARCH(1,1) |
|--------|------|------------|
| Mean reversion | No ($\alpha + \beta = 1$) | Yes ($\alpha + \beta < 1$) |
| Parameters to fit | 0 (λ fixed) | 3 ($\omega, \alpha, \beta$) |
| Convergence issues | None | Can fail |
| Long-run vol | Undefined | $\sqrt{\omega/(1-\alpha-\beta)}$ |

## Role in the Project

EWMA serves as the **fallback** when GARCH convergence fails, and as a **complement** — the dashboard shows both GARCH conditional vol and EWMA vol for comparison. It's orchestrated in `run_volatility()`.

## Connections

- [[GARCH(1,1) Model]] — EWMA is the IGARCH special case
- [[Volatility Forecasting]] — EWMA provides a simpler forecast baseline
- [[Black-Scholes Model]] — EWMA vol can substitute as the $\sigma$ input
