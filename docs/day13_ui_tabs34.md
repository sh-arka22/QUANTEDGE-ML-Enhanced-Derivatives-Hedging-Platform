# Day 13 — Streamlit UI Tabs 3 & 4

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/ui/tab_derivatives.py` | ~140 | Tab 3: Derivatives Pricing (BS + CRR + Greeks + 3D surface) |
| `src/ui/tab_hedging.py` | ~140 | Tab 4: Hedging Simulator (delta-neutral sim + band optimization) |

## Tab 3: Derivatives Pricing (`tab_derivatives.py`)

### Input Controls
| Control | Type | Range | Default |
|---------|------|-------|---------|
| Spot (S) | number_input | > 0 | 100 |
| Strike (K) | number_input | > 0 | 100 |
| Expiry (T) | slider | 0.0 – 3.0 | 1.0 |
| Risk-Free Rate | slider | 0.0 – 0.15 | 0.05 |
| Volatility | slider | 0.01 – 1.0 | 0.20 |
| Option Type | radio | call / put | call |
| Pricing Method | radio | BS / CRR | BS |
| Steps (N) | slider | 10 – 1000 | 200 |
| American | checkbox | bool | False |

### Sections

1. **Price Display** — BS price or CRR + BS comparison with delta. American early exercise premium.
2. **Greeks Row** — Delta, Gamma, Vega, Theta, Rho in metric cards
3. **Payoff at Expiry** — Intrinsic payoff (dashed) + option value curve vs spot
4. **Binomial Convergence** — CRR price at N=10,20,50,100,200,500 vs BS horizontal line
5. **Greeks Sensitivity** — 2x2 subplot: Delta, Gamma, Vega, Theta vs spot range
6. **3D Price Surface** — Strike x Expiry grid (20x15), Viridis colorscale

### Edge Cases
- T=0: shows intrinsic value only, returns early (Greeks not applicable)
- American premium displayed when American exercise selected

## Tab 4: Hedging Simulator (`tab_hedging.py`)

### Input Controls
| Control | Type | Range | Default |
|---------|------|-------|---------|
| Underlying | selectbox | equity tickers | first |
| Option Type | radio | call / put | call |
| Position | radio | Short / Long | Short |
| Strike (K) | number_input | > 0 | last spot |
| Expiry (T) | slider | 0.05 – 1.0 | 0.25 |
| Risk-Free Rate | slider | 0.0 – 0.15 | 0.05 |
| Volatility | slider | 0.05 – 1.0 | 0.20 |
| Txn Cost (bps) | slider | 1 – 20 | 5 |
| Rebalance Band | slider | 0.01 – 0.30 | 0.05 |

### Sections (after "Run Simulation" button)

1. **Summary Metrics** — Net P&L, Transaction Costs, Rebalances, Avg |Delta|, PnL Sharpe
2. **Spot Price & Rebalances** — Price line + orange triangle markers at rebalance events + strike line
3. **Portfolio Delta** — Delta over time with zero reference
4. **Cumulative P&L Breakdown** — Stacked area: share P&L, option P&L, costs (negative), net P&L (white)
5. **Optimal Band Search** (expander) — Table + dual-axis bar/line chart (P&L vs rebalances by band)

### Edge Cases
- No equity tickers: shows error
- < 5 days: shows error
- Txn costs > 50% of profit: warning displayed
