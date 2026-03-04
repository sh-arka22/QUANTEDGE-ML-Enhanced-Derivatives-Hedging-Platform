---
aliases: [Rebalance Band, Transaction Costs, Hedging Friction]
tags: [derivatives, hedging, practical, first-principles]
---

# Rebalance Bands and Transaction Costs

## First Principles

In theory, [[Delta-Neutral Hedging]] requires continuous rebalancing. In practice, every trade has a cost — commissions, bid-ask spread, market impact. The **rebalance band** is the threshold that triggers a hedge adjustment.

**How it works**: If the net delta exposure (option delta × position + shares held) exceeds `band × shares_per_contract`, the simulator trades to neutralize. Otherwise, it does nothing.

## The Tradeoff

| Band Width | Rebalances | Hedge Quality | Transaction Costs |
|-----------|------------|---------------|-------------------|
| Tight (0.01) | Very frequent | Excellent | Very high |
| Medium (0.05) | Moderate | Good | Moderate |
| Wide (0.20) | Rare | Poor | Low |

The optimal band depends on:
- **Volatility**: Higher vol → delta changes faster → needs tighter bands
- **Gamma**: ATM options near expiry have high gamma → need tighter bands
- **Transaction costs**: Higher costs → prefer wider bands
- **Time to expiry**: Short-dated options → gamma spike → tighter bands needed

## Implementation Details

In `src/pricing/hedging.py`:
- Transaction cost = `|shares traded| × price × bps / 10,000`
- Default: 5 bps (basis points) per trade
- The code warns if rebalancing occurs every single day (band too tight) or never (band too wide)
- `optimal_band_search()` tests bands [0.01, 0.02, 0.05, 0.1, 0.2] and reports net P&L, rebalance count, costs, and Sharpe for each

## Connections

- [[Delta-Neutral Hedging]] — Bands control the hedging frequency
- [[Gamma Scalping]] — The gamma/theta tradeoff determines whether tight bands are profitable
- [[Option Greeks]] — Gamma drives how quickly delta becomes stale
- [[Tab 4 - Hedging Simulator]] — Optimal band search chart
