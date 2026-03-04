---
aliases: [IV, Implied Vol]
tags: [derivatives, volatility, first-principles]
---

# Implied Volatility

## First Principles

The [[Black-Scholes Model]] takes five inputs: S, K, T, r, and σ. Of these, four are observable in the market. The fifth — volatility σ — is the **market's expectation of future price uncertainty**.

**Implied volatility** is the value of σ that, when plugged into the BS formula, produces the option's observed market price. It's found by **inverting** the BS pricing function:

$$C_{market} = BS(S, K, T, r, \sigma_{IV}) \quad \Rightarrow \quad \sigma_{IV} = BS^{-1}(C_{market})$$

Since the BS formula has no closed-form inverse, this requires numerical root-finding.

## Why It Matters

IV is arguably the most important number in options markets because it tells you **how expensive an option is relative to its fundamental inputs**. Two options with different strikes and expiries can be compared by their IVs — higher IV means higher relative cost.

IV also reveals market sentiment: a spike in IV (like VIX for S&P 500 options) signals that the market expects larger price swings ahead.

## How This Project Implements It

In `src/pricing/black_scholes.py`, the `implied_volatility()` function uses a **two-phase approach**:

### Phase 1: Newton-Raphson
Starting from $\sigma_0 = 0.20$ (a typical guess), iteratively update:
$$\sigma_{n+1} = \sigma_n - \frac{BS(\sigma_n) - C_{market}}{\text{vega}(\sigma_n)}$$

This converges quadratically when the initial guess is close. The [[Option Greeks]] vega serves as the derivative — it measures how sensitively the BS price responds to volatility.

**Failure conditions**: If vega ≈ 0 (deep ITM/OTM options) or σ goes negative, Newton-Raphson can fail.

### Phase 2: Bisection Fallback
If Newton fails, the algorithm switches to bisection search over $\sigma \in [0.001, 5.0]$ — guaranteed to converge (up to 200 iterations) as long as the market price is within the BS range.

### Edge Cases
- Returns `NaN` if $T \leq 0$ (expired option)
- Returns `NaN` if $C_{market} \leq 0$ (no meaningful IV for zero-price options)

## Connection to Volatility Surface

A single IV is just one point. The full [[Volatility Surface]] maps IV across all strikes and expiries — revealing patterns like the **volatility smile** (OTM options having higher IV) and the **term structure** (IV varying with maturity).

## Connections

- [[Black-Scholes Model]] — IV inverts the BS formula
- [[Option Greeks]] — Vega is the derivative used in Newton-Raphson
- [[Volatility Surface]] — Grid of IVs across (K, T)
- [[GARCH(1,1) Model]] — Statistical (realized) vol vs market-implied vol comparison
