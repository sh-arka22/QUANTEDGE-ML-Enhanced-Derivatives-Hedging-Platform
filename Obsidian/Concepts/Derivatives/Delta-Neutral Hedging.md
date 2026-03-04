---
aliases: [Delta Hedging, Hedging Simulation]
tags: [derivatives, hedging, risk-management, first-principles]
---

# Delta-Neutral Hedging

## First Principles: What Is a Hedge?

If you've sold (written) a call option, you have exposure to the underlying stock's price — if the stock rises, you owe more. **Delta-neutral hedging** eliminates this directional risk by holding shares that offset the option's delta.

The core idea: the option has a [[Option Greeks|delta]] telling you how much the option price moves per \$1 stock move. If delta = 0.6 and you've sold 1 contract (100 shares), you need to hold $0.6 \times 100 = 60$ shares to neutralize the exposure.

## The Problem: Delta Changes

Delta isn't constant — it changes as the stock moves ([[Option Greeks|gamma]]) and as time passes. So you must **rebalance** the hedge periodically. But each rebalance incurs transaction costs. This creates a fundamental tradeoff:

- **Rebalance too often** → Perfect hedge but crushed by transaction costs
- **Rebalance too rarely** → Cheap but poor hedge quality (large residual risk)

## How This Project Simulates It

In `src/pricing/hedging.py`, `simulate_delta_hedge()` runs a **daily simulation** over real historical spot prices:

### Day 0: Initialize
1. Compute the option's delta at current spot using [[Black-Scholes Model]]
2. Buy shares to neutralize: `target_shares = -position × delta × 100`
3. Record initial transaction cost

### Day 1 to N: Daily Loop
For each trading day:
1. **Compute new delta** at current spot and remaining time
2. **Calculate net delta exposure**: option delta × position × 100 + shares held
3. **Check [[Rebalance Bands and Transaction Costs|rebalance band]]**: if |net delta| > band × 100, trade to neutralize
4. **Track P&L**: share P&L (incremental), option P&L, cumulative transaction costs
5. **Net P&L** = share P&L + option P&L - transaction costs

### Incremental Share P&L
The code uses the correct incremental formula:
$$\text{Share P\&L}_t = \sum_{i=1}^{t} \text{shares}_{i-1} \times (S_i - S_{i-1})$$
This avoids the error of computing share P&L as `shares × (S_t - S_0)`, which ignores the changing position.

### Time Decay
$T_{remaining} = T - \text{day}/252$, clamped at $10^{-6}$ to prevent division by zero in [[Black-Scholes Model]].

## Summary Statistics

`hedging_summary()` computes:
- Net P&L and profitability
- Number and percentage of rebalancing events
- Average absolute delta (hedge quality measure)
- Daily P&L Sharpe ratio
- Transaction cost ratio (costs / |gross P&L|) — flags if > 50%

## Optimal Band Search

`optimal_band_search()` runs the simulation at multiple band widths [0.01, 0.02, 0.05, 0.1, 0.2] and compares net P&L, rebalance count, costs, and Sharpe — helping identify the [[Rebalance Bands and Transaction Costs|sweet spot]].

## Connections

- [[Option Greeks]] — Delta determines the hedge ratio; gamma determines rebalancing frequency
- [[Rebalance Bands and Transaction Costs]] — The core tradeoff in hedging
- [[Gamma Scalping]] — The economic mechanism of hedging P&L
- [[Black-Scholes Model]] — Provides delta and option values for the simulation
- [[Tab 4 - Hedging Simulator]] — Interactive UI for running simulations
