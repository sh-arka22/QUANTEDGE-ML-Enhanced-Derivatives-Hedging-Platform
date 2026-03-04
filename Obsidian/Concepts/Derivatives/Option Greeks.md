---
aliases: [Greeks, Sensitivities]
tags: [derivatives, greeks, risk-management, first-principles]
---

# Option Greeks

## First Principles: Why Do We Need Greeks?

An option's price depends on multiple variables: spot price $S$, strike $K$, time to expiry $T$, risk-free rate $r$, and volatility $\sigma$. The Greeks measure **how sensitively the option price changes** when each variable moves, holding the others constant. They are the partial derivatives of the option pricing function.

Traders use Greeks to understand and manage the risk of their positions — especially for [[Delta-Neutral Hedging]].

## The Five Greeks

### Delta ($\Delta$) — Sensitivity to Spot Price
$$\Delta_{call} = N(d_1), \quad \Delta_{put} = N(d_1) - 1$$

**Intuition**: Delta tells you how many shares of the underlying you need to hold to replicate the option's price movement. A delta of 0.6 means the option moves \$0.60 for every \$1 move in the stock.

**Range**: Call delta ∈ [0, 1], Put delta ∈ [-1, 0].

### Gamma ($\Gamma$) — Rate of Change of Delta
$$\Gamma = \frac{n(d_1)}{S \sigma \sqrt{T}}$$

**Intuition**: Gamma measures how quickly delta changes as the stock moves. High gamma means the hedge (delta) becomes stale quickly, requiring more frequent rebalancing. This is central to [[Rebalance Bands and Transaction Costs]].

Gamma is highest when the option is at-the-money and near expiry.

### Vega ($\mathcal{V}$) — Sensitivity to Volatility
$$\mathcal{V} = S \cdot n(d_1) \cdot \sqrt{T}$$

**Intuition**: Vega measures the option price change per 1% change in implied volatility. Longer-dated options have higher vega because uncertainty has more time to manifest. Used critically in [[Implied Volatility]] solving (Newton-Raphson step size).

### Theta ($\Theta$) — Time Decay
$$\Theta_{call} = -\frac{S \cdot n(d_1) \cdot \sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)$$

**Intuition**: Theta is the "price of waiting." Options lose value as expiry approaches because the optionality shrinks. Theta is always negative for long options (time works against you) and is the counterpart to gamma — this is the [[Gamma Scalping]] tradeoff.

### Rho ($\rho$) — Sensitivity to Interest Rate
$$\rho_{call} = KTe^{-rT}N(d_2)$$

**Intuition**: Higher rates increase the present-value advantage of deferring the strike payment, making calls more valuable and puts less valuable. Rho is generally the least important Greek for short-dated equity options.

## How This Project Implements Them

In `src/pricing/black_scholes.py`, the `greeks()` function computes **all Greeks in a single vectorized pass**:

1. **Near-expiry handling**: When $T < 1/365$ (less than 1 day), the function returns limiting values (delta → 0 or 1, gamma/vega/theta/rho → 0) to avoid numerical instability.

2. **Vega scaling**: Reported per 1% vol move (divided by 100), matching market convention.

3. **Theta scaling**: Reported per calendar day (divided by 365), not per trading day.

4. **Rho scaling**: Per 1% rate move (divided by 100).

The [[Binomial Tree (CRR)]] also computes delta, gamma, and theta numerically from tree node values, providing an independent verification.

## Connections

- [[Delta-Neutral Hedging]] — Delta determines the hedge ratio; gamma determines rebalancing frequency
- [[Black-Scholes Model]] — Greeks are partial derivatives of the BS price formula
- [[Implied Volatility]] — Newton-Raphson uses vega as the derivative for root-finding
- [[Tab 3 - Derivatives Pricing]] — Interactive sensitivity charts show each Greek vs spot price
- [[Gamma Scalping]] — The theta/gamma tradeoff is the economic basis of hedging P&L
