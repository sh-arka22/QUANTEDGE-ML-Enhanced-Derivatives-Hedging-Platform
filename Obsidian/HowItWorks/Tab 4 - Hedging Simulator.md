---
aliases: [Hedging Tab]
tags: [ui, hedging, tab]
---

# Tab 4 — Hedging Simulator

## What It Shows

A [[Delta-Neutral Hedging]] simulator that runs over real historical spot prices, showing the tradeoff between hedge quality and [[Rebalance Bands and Transaction Costs|transaction costs]].

## Input Controls

- **Underlying**: Any non-index ticker from the data (MS, JPM, BAC, AAPL)
- **Option type**: Call or put
- **Position**: Short (-1) or Long (+1)
- **Strike, Expiry, Rate, Volatility**: Standard option parameters
- **Transaction cost**: 1-20 basis points per trade
- **Rebalance band**: 0.01-0.30 delta threshold

The spot price window is automatically limited to the most recent `T × 252` trading days.

## Simulation Output

### Summary Metrics
- **Net P&L** with profit/loss indicator
- **Total transaction costs**
- **Rebalance count and percentage**
- **Average absolute delta** (hedge quality)
- **P&L Sharpe ratio**
- Warning if transaction costs exceed 50% of gross profit

### Charts

1. **Spot Price & Rebalances**: Price line with orange triangles marking rebalance events, plus strike level
2. **Portfolio Delta**: Delta over time — should oscillate around zero for a good hedge
3. **Cumulative P&L Breakdown**: Stacked area chart showing share P&L, option P&L, costs (negative), and net P&L (white line overlay) — this is the [[Gamma Scalping]] economics visualized

### Optimal Band Search (Expandable)
- Table comparing bands [0.01, 0.02, 0.05, 0.1, 0.2] with net P&L, rebalance count, costs, Sharpe
- Dual-axis chart: bars for net P&L + line for rebalance count vs band width

## Connections

- [[Delta-Neutral Hedging]] — The simulation engine
- [[Rebalance Bands and Transaction Costs]] — The core tradeoff explored
- [[Gamma Scalping]] — The economic mechanism behind the P&L
- [[Black-Scholes Model]] — Provides delta and option values
- [[Streamlit Dashboard]] — Parent UI structure
