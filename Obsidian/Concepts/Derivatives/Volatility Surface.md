---
aliases: [Vol Surface, IV Surface]
tags: [derivatives, volatility, first-principles]
---

# Volatility Surface

## First Principles

If the [[Black-Scholes Model]] assumptions held perfectly, [[Implied Volatility]] would be constant across all strikes and expiries. In reality, it's not — and the pattern of IV across (Strike, Expiry) pairs forms the **volatility surface**.

This surface reveals where the BS model breaks down and where the market prices in skew, smile, and term structure.

## Common Patterns

- **Volatility Smile**: OTM puts and OTM calls have higher IV than ATM options — common in FX and equity index options. This reflects fat tails (extreme moves are more likely than the normal distribution assumes).
- **Volatility Skew**: IV increases as strike decreases — OTM puts are "expensive" relative to BS. This reflects crash risk (markets fall faster than they rise).
- **Term Structure**: Short-dated IV can differ from long-dated IV. In calm markets, short-dated IV is often lower; during crises, it spikes.

## How This Project Implements It

In `src/pricing/black_scholes.py`, the `vol_surface()` function:

1. Takes a grid of market option prices across strikes and expiries
2. Calls `implied_volatility()` for each (K, T) pair
3. Returns a DataFrame with strikes as rows, expiries as columns, and IVs as values

In [[Tab 3 - Derivatives Pricing]], the dashboard shows a **3D price surface** (not strictly an IV surface, but a BS price surface across Strike × Expiry), rendered with Plotly's `Surface` trace. This visualizes how the option price depends on both moneyness and time.

## Connections

- [[Implied Volatility]] — Each point on the surface is one IV solve
- [[Black-Scholes Model]] — The surface reveals deviations from BS assumptions
- [[GARCH(1,1) Model]] — Realized vol captures actual clustering; IV surface captures market's forward-looking view
