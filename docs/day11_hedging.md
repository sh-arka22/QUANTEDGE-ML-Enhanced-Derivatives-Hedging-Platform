# Day 11 — Delta-Neutral Hedging Simulation

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/pricing/hedging.py` | 163 | Delta-neutral hedging sim, summary stats, optimal band search |

## Module: `src/pricing/hedging.py`

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `simulate_delta_hedge()` | `(spot_prices, K, T, r, sigma, ...) -> pd.DataFrame` | Day-by-day hedging simulation with rebalance bands and transaction costs. |
| `hedging_summary()` | `(sim_df) -> dict` | P&L, costs, rebalance count, Sharpe, cost ratio. |
| `optimal_band_search()` | `(spot_prices, K, T, r, sigma, bands) -> pd.DataFrame` | Runs simulations across multiple bands for optimization. |

### Simulation Logic

1. **Initialize**: Compute BS delta at t=0, buy/sell shares to neutralize option delta
2. **Daily loop**: For each day, compute T_remaining, new delta, net exposure
3. **Rebalance check**: If `|net_delta| > band * shares_per_contract`, trade to neutralize
4. **Track**: Share P&L (incremental), option P&L, transaction costs, net P&L
5. **Output**: DataFrame with all daily values for charting

### Edge Cases

| Edge Case | Behavior |
|-----------|----------|
| < 5 days of data | Raises `ValueError` |
| T_remaining = 0 | Clamped to 1e-6 (BS handles near-expiry) |
| All days rebalanced | Warns "band too tight" |
| Zero rebalances | Warns "band too wide" |
| Txn costs > profit | Flagged via `txn_cost_ratio` in summary |

### Test Results (AAPL, 63 trading days)

| Band | Net P&L | Rebalances | Txn Costs | Sharpe |
|------|---------|------------|-----------|--------|
| 0.01 | $465 | 37 | $23.73 | 20.6 |
| 0.02 | $460 | 28 | $22.82 | 19.5 |
| 0.05 | $468 | 17 | $19.69 | 18.0 |
| 0.10 | $477 | 5 | $13.56 | 9.9 |
| 0.20 | $451 | 2 | $11.44 | 6.6 |

## Integration

```python
from src.pricing.hedging import simulate_delta_hedge, hedging_summary, optimal_band_search
```

Used by:
- Day 13 (Streamlit Tab 4) — hedging simulator dashboard with interactive controls
- Day 15 (metrics) — hedging efficiency benchmarks
