---
aliases: [Gamma-Theta Tradeoff]
tags: [derivatives, hedging, trading, first-principles]
---

# Gamma Scalping

## First Principles

When you're [[Delta-Neutral Hedging|delta-hedged]], you have zero directional exposure. But you still have **gamma** and **theta** exposure, which are opposite sides of the same coin:

- **Gamma (positive for long options)**: You profit when the stock moves a lot (in either direction), because your delta hedge underperforms — you bought shares that the stock has now moved away from, and the option captured the convexity.
- **Theta (negative for long options)**: You lose money every day from time decay — the option is worth less today than yesterday, all else equal.

**Gamma scalping** is the process of repeatedly delta-hedging a long option position to capture gamma profits that (hopefully) exceed theta costs.

## The Economics

For a **short option position** (which this project defaults to):
- You **receive** theta (time decay works in your favor)
- You **pay** gamma (large moves hurt you because your hedge was set at an old delta)

Net P&L = Theta collected - Gamma cost - Transaction costs

If realized volatility is **lower** than the implied volatility you sold the option at, you profit. If realized vol is higher, you lose. This is the fundamental mechanism behind the hedging simulation's P&L.

## How It Appears in the Project

The [[Delta-Neutral Hedging]] simulation in `src/pricing/hedging.py` tracks:
- `share_pnl_cumulative`: The gamma component (share hedge underperformance)
- `option_pnl_cumulative`: The theta component (option value change)
- `transaction_costs_cumulative`: Friction from rebalancing

The [[Tab 4 - Hedging Simulator]] shows these as stacked P&L curves.

## Connections

- [[Delta-Neutral Hedging]] — The simulation that implements gamma scalping economics
- [[Option Greeks]] — Gamma and theta are the two Greeks that drive hedging P&L
- [[Rebalance Bands and Transaction Costs]] — Transaction costs eat into gamma scalping profits
- [[Black-Scholes Model]] — BS assumption of constant vol means gamma scalping should break even in theory
