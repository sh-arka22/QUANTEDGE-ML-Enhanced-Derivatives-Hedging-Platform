---
aliases: [Vol Forecast, GARCH Forecast]
tags: [volatility, forecasting, first-principles]
---

# Volatility Forecasting

## First Principles

Unlike returns (which are nearly unpredictable), volatility is **highly persistent and forecastable**. Today's volatility tells you a lot about tomorrow's. This is the economic foundation of the [[GARCH(1,1) Model]].

The GARCH forecast works by iterating the variance equation forward:

$$\sigma_{t+h}^2 = \omega \cdot \sum_{i=0}^{h-1}(\alpha+\beta)^i + (\alpha+\beta)^h \cdot \sigma_t^2$$

As $h \to \infty$, the forecast converges to the long-run variance $\omega/(1-\alpha-\beta)$.

## How This Project Implements It

In `src/analytics/volatility.py`, `forecast_volatility()`:

1. Takes a fitted GARCH model result
2. Uses the `arch` library's `.forecast(horizon=h)` method
3. Extracts the variance forecast for the last observation
4. Converts from percentage-squared to annualized decimal volatility
5. Caps horizon at $n/3$ where $n$ is the sample size (prevents unreliable long-horizon forecasts)

The function also includes `realized_vs_predicted()` which aligns the GARCH conditional vol with rolling realized vol for backtesting comparison.

## Connections

- [[GARCH(1,1) Model]] — Provides the fitted model for forecasting
- [[Volatility Cones]] — Context for whether the forecast is extreme
- [[Black-Scholes Model]] — Forecast vol can serve as the $\sigma$ input for forward-looking pricing
- [[Delta-Neutral Hedging]] — Better vol forecasts → better hedge ratios
