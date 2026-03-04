---
aliases: [GARCH, Generalized ARCH, Conditional Volatility]
tags: [volatility, time-series, econometrics, first-principles]
---

# GARCH(1,1) Model

## First Principles: Why Not Just Use Standard Deviation?

Financial returns exhibit **volatility clustering**: large moves (positive or negative) tend to be followed by large moves, and calm periods tend to follow calm periods. A simple rolling standard deviation misses this temporal structure.

The GARCH (Generalized Autoregressive Conditional Heteroscedasticity) model captures this by making tomorrow's variance a function of today's return and today's variance.

## The Model

$$\sigma_t^2 = \omega + \alpha \cdot \epsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$$

Where:
- $\sigma_t^2$: Today's conditional variance (what we're predicting)
- $\epsilon_{t-1}^2$: Yesterday's squared return (the "news" or "shock")
- $\sigma_{t-1}^2$: Yesterday's conditional variance (persistence)
- $\omega$: The baseline variance (long-run component)
- $\alpha$: How much yesterday's shock matters (reactivity)
- $\beta$: How much yesterday's variance persists (memory)

**Key constraint**: $\alpha + \beta < 1$ for the process to be stationary (convergent to a long-run variance).

### Intuition

Think of it as a **weighted average** that blends three things: a long-run baseline ($\omega$), what just happened ($\alpha \cdot \epsilon^2$), and what was expected ($\beta \cdot \sigma^2$). The persistence $\alpha + \beta$ controls how long a volatility shock lingers — values near 1.0 (like 0.97) mean shocks take hundreds of days to decay.

### Long-Run Variance

$$\sigma_{LR}^2 = \frac{\omega}{1 - \alpha - \beta}$$

This is the unconditional (average) variance the process reverts to over time.

## How This Project Implements It

In `src/analytics/volatility.py`:

### `fit_garch()`
1. **Scales returns to percentages** — the `arch` library expects percentage returns, not decimal
2. Fits GARCH(1,1) with `dist="normal"` and `mean="Constant"`
3. **Tries two starting configurations** — if default fails, retries with `[0.01, 0.05, 0.90]` to help convergence
4. Extracts $\omega, \alpha, \beta$, persistence, and long-run vol
5. Converts conditional volatility from percentage-scaled back to annualized decimal: `cond_vol / 100 * sqrt(252)`
6. Flags IGARCH ($\alpha + \beta \geq 0.999$) — integrated GARCH, meaning volatility shocks are permanent

### Pre-Test: [[ARCH LM Test]]
Before fitting GARCH, the code tests whether ARCH effects exist at all using Engle's LM test. If the test fails (no ARCH effects), GARCH fitting would be unjustified.

### Fallback: [[EWMA Volatility]]
If GARCH convergence fails, the platform falls back to EWMA (a special case where $\alpha + \beta = 1$).

## Connections

- [[ARCH LM Test]] — Pre-test for GARCH effects
- [[EWMA Volatility]] — Simpler alternative (IGARCH special case)
- [[Volatility Forecasting]] — GARCH enables forward-looking vol estimates
- [[Volatility Cones]] — Compares realized vol distributions across windows
- [[Black-Scholes Model]] — GARCH output can serve as the $\sigma$ input
- [[Delta-Neutral Hedging]] — Better vol estimates → better hedge ratios
