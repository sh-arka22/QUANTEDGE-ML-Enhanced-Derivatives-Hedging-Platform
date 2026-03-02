# Day 12 — Streamlit UI Tabs 1 & 2

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/ui/app.py` | 78 | Main entry point, sidebar, tab layout, lazy loading |
| `src/ui/tab_portfolio.py` | 98 | Tab 1: Portfolio Analytics dashboard |
| `src/ui/tab_prediction.py` | 182 | Tab 2: Price Prediction (regression + classification) |
| `src/ui/tab_derivatives.py` | 9 | Tab 3: Placeholder for Day 13 |
| `src/ui/tab_hedging.py` | 9 | Tab 4: Placeholder for Day 13 |

## App Architecture (`app.py`)

### Sidebar Controls
- **FRED API Key**: Password input, reads from `FRED_API_KEY` env var as default
- **Risk-Free Rate**: Number input, range 0–20%, default 2%

### Data Loading
- `get_all_data()` called once with `@st.cache_data(ttl=3600)` caching
- Returns dict with `prices`, `returns`, `market_returns`, `macro_aligned`

### Tab Structure
| Tab | Module | Data Passed |
|-----|--------|-------------|
| Portfolio Analytics | `tab_portfolio.render()` | `returns`, `market_returns`, `rf_rate` |
| Price Prediction | `tab_prediction.render()` | `prices`, `market_returns`, `macro_aligned` |
| Derivatives Pricing | `tab_derivatives.render()` | (none — placeholder) |
| Hedging Simulator | `tab_hedging.render()` | `prices` |

### Error Handling
Each tab wrapped in `try/except` — tab-level errors display `st.error()` without crashing the app.

## Tab 1: Portfolio Analytics (`tab_portfolio.py`)

### Sections

1. **Key Metrics** — 5-column layout: Sharpe, Sortino, Beta, Alpha (annualized), Max Drawdown
2. **Value at Risk (95%)** — 3-column layout: Parametric VaR, Historical VaR, CVaR/ES
   - Fat-tail warning if kurtosis detected
3. **Cumulative Returns** — Line chart: all tickers + portfolio (white, width=3)
4. **Return Distribution** — Histogram with Parametric VaR (red dash) and Historical VaR (orange dot) lines
5. **Rolling 30-Day Volatility** — Annualized rolling vol (left column)
6. **Correlation Matrix** — Heatmap with annotated values (right column)
7. **Full Risk Summary** — Expandable table with all metrics

### Chart Patterns
- `plotly_dark` template on all figures
- `height=350–400` for consistent sizing
- `float32` casting for memory efficiency
- `use_container_width=True` for responsive layout

## Tab 2: Price Prediction (`tab_prediction.py`)

### Section A: AAPL Regression

**Metrics Row**: RMSE, R-squared, MAE (3 columns)

**Diagnostics Row**: Durbin-Watson, Breusch-Pagan p, Jarque-Bera p, Max VIF (4 columns with delta flags)

**Charts** (2x2 grid):
| Position | Chart | Key Feature |
|----------|-------|-------------|
| Top-left | Actual vs Predicted scatter | Red dashed perfect-prediction line |
| Top-right | Residuals scatter | Red dashed zero line |
| Bottom-left | QQ Plot | Normal reference line |
| Bottom-right | VIF Bar Chart | Red threshold line at VIF=10 |

### Section B: BAC Direction Classification

**Model Comparison Table**: Accuracy, Precision, Recall, F1, AUC-ROC per model

**Charts** (2 columns):
- Left: ROC Curves — per-model lines (excluding Voting Ensemble), gray random baseline
- Right: Feature Importance — horizontal bar chart (Random Forest)

**Confusion Matrices**: Subplot grid with heatmaps for each base model (Blues colorscale)

**Class Distribution**: Pie chart showing Up/Down split in training data

## Integration

```bash
# Run the app
streamlit run src/ui/app.py

# With FRED macro data
FRED_API_KEY=your_key streamlit run src/ui/app.py
```

## Design Decisions

- **Lazy imports**: Tab modules imported inside tab context to avoid loading all analytics on startup
- **No `@st.cache_data` in tabs**: Caching handled by `get_all_data()` in loaders
- **Graceful degradation**: App works without FRED key (macro features disabled)
- **Separate render functions**: Each tab is a standalone module with `render()` entry point
