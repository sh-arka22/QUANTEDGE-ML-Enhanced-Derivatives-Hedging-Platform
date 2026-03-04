---
aliases: [Derivatives Tab, Pricing Tab]
tags: [ui, derivatives, tab]
---

# Tab 3 — Derivatives Pricing

## What It Shows

An interactive option pricer supporting both [[Black-Scholes Model]] and [[Binomial Tree (CRR)]] methods, with full [[Option Greeks]] display and sensitivity analysis.

## Input Controls

Five parameter sliders: Spot (S), Strike (K), Expiry (T), Risk-Free Rate (r), Volatility (σ). Plus option type (call/put), pricing method (BS/CRR), and for CRR: tree steps (N) and American exercise toggle.

## Pricing Display

- **BS method**: Single price metric
- **CRR method**: CRR price + BS reference price + difference (convergence gap)
- **American exercise**: Early exercise premium (American - European)

## Greeks Display

All five Greeks shown as metric cards: Delta, Gamma, Vega, Theta, Rho. For CRR, Delta/Gamma/Theta come from the tree; Vega/Rho from BS (since the tree doesn't directly compute them).

## Charts

1. **Payoff at Expiry**: Intrinsic value (hockey stick) + current option value curve — shows time value
2. **Binomial Convergence**: CRR price at N = [10, 20, 50, 100, 200, 500] vs BS reference line — demonstrates convergence
3. **Greeks Sensitivity (2×2)**: Delta, Gamma, Vega, Theta plotted against spot price — shows how each Greek varies with moneyness
4. **3D Price Surface**: Option price as a function of (Strike × Expiry) — rendered as a Plotly surface with Viridis colorscale

## Connections

- [[Black-Scholes Model]] — Analytical pricing engine
- [[Binomial Tree (CRR)]] — Lattice pricing engine
- [[Option Greeks]] — Sensitivity measures displayed
- [[Implied Volatility]] — Used internally for IV-related features
- [[Streamlit Dashboard]] — Parent UI structure
