# Quantitative Finance Analytics Platform

Production-grade quantitative finance platform covering portfolio risk analytics, regression/classification modeling, GARCH volatility, derivatives pricing, and delta-neutral hedging -- with an interactive 4-tab Streamlit dashboard.

**4,400+ lines** | **171 tests (100% pass)** | **10 years of market data** | **6 trained models**

---

## Key Results

| Metric | Value |
|--------|-------|
| Portfolio cumulative return | **169.1%** (vs S&P 500 144.3%) |
| Portfolio Sharpe | 0.42 |
| CAPM Beta | 1.22 (banking sector exposure) |
| OLS RMSE improvement | **26.4%** over naive baseline |
| ML classifiers trained | 5 (DT, RF, KNN, SVM, Voting Ensemble) |
| Best classifier AUC-ROC | 0.54 |
| Feature engineering | 19 technical indicators, MI-filtered to 8 |
| GARCH persistence | 0.92 (mean-reverting) |
| Automated tests | 171 (100% pass rate) |

---

## Architecture

Four-layer modular monolith:

```
src/
  data/           Data ingestion (yfinance equity, FRED macro, merge_asof alignment)
    config.py       Single source of truth: tickers, weights, constants
    loaders.py      fetch_equity(), fetch_fred(), align_data(), get_all_data()

  analytics/      Portfolio metrics, OLS regression, ML classification, GARCH volatility
    portfolio.py    Sharpe, Sortino, CAPM, VaR, CVaR, max drawdown
    regression.py   AAPL OLS with VIF, Breusch-Pagan, Durbin-Watson, Jarque-Bera
    classification.py  BAC direction: DT, RF, KNN, SVM, Voting Ensemble, MI selection
    volatility.py   GARCH(1,1), EWMA, ARCH LM test, forecast, volatility cones

  pricing/        Black-Scholes, CRR binomial tree, delta-neutral hedging
    black_scholes.py  Vectorized BS pricing, all Greeks, IV solver, vol surface
    binomial.py       CRR tree (European/American), Greeks, early exercise premium
    hedging.py        Delta-neutral simulation, summary stats, optimal band search

  ui/             Streamlit dashboard (4 tabs)
    app.py            Main entry point, sidebar, tab routing
    tab_portfolio.py  Risk metrics, VaR, cumulative returns, correlation heatmap
    tab_prediction.py AAPL regression + BAC classification visualizations
    tab_derivatives.py  Interactive BS/CRR pricer, Greeks, 3D vol surface
    tab_hedging.py    Hedging simulator with P&L charts, band optimization
```

```mermaid
graph LR
    Config[config.py] --> Data[loaders.py]
    Data --> Analytics[analytics/]
    Data --> Pricing[pricing/]
    Analytics --> UI[ui/ Streamlit]
    Pricing --> UI
```

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd quant-platform
conda create -p ./quant python=3.11 -y
conda activate ./quant
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run src/ui/app.py

# 3. Run tests
pytest tests/ -v

# 4. Generate resume metrics
python scripts/generate_metrics.py
```

Opens at **http://localhost:8501**.

### With FRED macro data (optional)

```bash
FRED_API_KEY=your_key streamlit run src/ui/app.py
```

The app works fully without a FRED key -- macro features are gracefully disabled.

### Docker

```bash
docker build -t quant-platform .
docker run -p 8501:8501 quant-platform
```

---

## Dashboard Tabs

### Tab 1: Portfolio Analytics

Banking portfolio (MS, JPM, BAC) with equal weights.

- **Key metrics**: Sharpe, Sortino, Beta, Alpha (annualized), Max Drawdown
- **Value at Risk**: Parametric, Historical, CVaR with fat-tail detection
- **Charts**: Cumulative returns (all tickers + portfolio), return distribution with VaR lines, rolling 30-day volatility, correlation heatmap
- Full risk summary in expandable table

### Tab 2: Price Prediction

**AAPL Regression**: OLS with SMA, lag, and market features. Diagnostic suite (VIF, Breusch-Pagan, Durbin-Watson, Jarque-Bera). Actual vs predicted scatter, residuals, QQ plot, VIF bars.

**BAC Classification**: 5-model comparison table (accuracy, precision, recall, F1, AUC-ROC). ROC curves, Random Forest feature importance, confusion matrices, class distribution.

### Tab 3: Derivatives Pricing

Interactive option pricer with adjustable S, K, T, r, sigma.

- **Pricing**: Black-Scholes or CRR Binomial (with American exercise toggle)
- **Greeks**: Delta, Gamma, Vega, Theta, Rho in metric cards
- **Charts**: Payoff diagram, binomial convergence (N=10 to 500 vs BS), Greeks sensitivity (2x2), 3D price surface (20x15 strike x expiry grid)

### Tab 4: Hedging Simulator

Delta-neutral hedging simulation on historical spot prices.

- Select underlying ticker, strike, expiry, vol, transaction costs, rebalance band
- **Summary**: Net P&L, costs, rebalance count, Sharpe
- **Charts**: Spot price with rebalance markers, portfolio delta over time, cumulative P&L breakdown (share/option/costs/net)
- **Optimal band search**: Compare P&L and rebalances across band widths

---

## Modules

### Data Layer (`src/data/`)

| Function | Description |
|----------|-------------|
| `fetch_equity()` | yfinance download with retry, staleness check, NaN handling |
| `fetch_fred()` | FRED macro series with graceful degradation (returns None without key) |
| `align_data()` | `merge_asof(direction='backward', tolerance='90d')` -- no look-ahead bias |
| `compute_log_returns()` | `np.log(P / P.shift(1))` -- log returns only |
| `get_all_data()` | Orchestrator with `@st.cache_data(ttl=3600)` |

### Analytics Layer (`src/analytics/`)

**Portfolio** (`portfolio.py`): `sharpe_ratio`, `sortino_ratio`, `capm`, `var_parametric`, `var_historical`, `cvar`, `max_drawdown`, `risk_summary`

**Regression** (`regression.py`): `prepare_features` (SMA, lag, market features), `fit_ols` (with Ridge fallback), `diagnostics` (VIF, BP, DW, JB), `evaluate`, `run_regression`

**Classification** (`classification.py`): 18 technical indicators (SMA, RSI, MACD, Bollinger %B, Stochastic, ROC, RVol, Z-Score). RobustScaler. Mutual information feature selection. 4 classifiers + soft-voting ensemble. Optional hyperparameter tuning via `RandomizedSearchCV` with `TimeSeriesSplit`.

**Volatility** (`volatility.py`): `arch_lm_test`, `fit_garch` (with EWMA fallback), `fit_ewma`, `forecast_volatility`, `volatility_cones`, `realized_vs_predicted`, `run_volatility`

### Pricing Layer (`src/pricing/`)

**Black-Scholes** (`black_scholes.py`): Fully vectorized. `price()`, `greeks()` (all 6), `implied_volatility()` (Newton-Raphson + bisection), `vol_surface()`. Hull textbook validated.

**Binomial Tree** (`binomial.py`): CRR parameterization with vectorized backward induction. `crr_price()` (European/American), `crr_greeks()`, `american_premium()`. N capped at 1000. Converges to BS within 0.5% at N=500.

**Hedging** (`hedging.py`): `simulate_delta_hedge()` (daily rebalancing with bands and transaction costs), `hedging_summary()`, `optimal_band_search()`.

---

## Testing

171 tests across 7 files, all passing:

| File | Tests | Coverage |
|------|-------|----------|
| `test_data.py` | 3 | Data loaders, log returns, alignment |
| `test_portfolio.py` | 13 | Sharpe, Sortino, VaR, CVaR, CAPM, drawdown |
| `test_regression.py` | 16 | Feature prep, OLS fitting, diagnostics, evaluation |
| `test_classification.py` | 38 | All indicators, training, tuning, ensemble, MI, evaluation |
| `test_pricing.py` | 47 | BS (Hull validated), put-call parity, Greeks, IV, CRR convergence, American pricing |
| `test_volatility.py` | 34 | ARCH LM, GARCH params, EWMA, forecast, cones, realized vs predicted |
| `test_hedging.py` | 20 | Simulation, delta bounds, costs, band behavior, summary, band search |

```bash
pytest tests/ -v    # ~90 seconds
```

---

## Technical Constraints

These are enforced throughout the codebase:

- **Vectorization**: All numeric operations use numpy/pandas. No Python for-loops in pricing or analytics.
- **Log returns only**: `np.log(prices / prices.shift(1))`, never simple returns.
- **Adjusted Close only**: Never raw Close prices.
- **No look-ahead bias**: Macro data merged with `merge_asof(direction='backward')`.
- **Chronological splits**: Train/test splits are time-ordered (80/20), never random shuffle.
- **GARCH fallback**: If GARCH fails to converge, falls back to EWMA (lambda=0.94).
- **Black-Scholes T=0**: Returns intrinsic value, not NaN.
- **Binomial tree**: N capped at `MAX_TREE_STEPS=1000`.
- **FRED graceful degradation**: App works fully without a FRED API key.

---

## Configuration

All constants live in `src/data/config.py`:

| Constant | Value | Used By |
|----------|-------|---------|
| `TICKERS` | MS, JPM, BAC, AAPL, ^GSPC, ^IXIC | loaders |
| `PORTFOLIO_WEIGHTS` | 0.33 / 0.34 / 0.33 | portfolio |
| `DATE_RANGE` | 2015-01-01 to 2024-12-31 | loaders |
| `TRADING_DAYS` | 252 | all modules |
| `VIF_THRESHOLD` | 10 | regression |
| `VAR_CONFIDENCE` | 0.95 | portfolio |
| `EWMA_LAMBDA` | 0.94 | volatility |
| `MAX_TREE_STEPS` | 1000 | binomial |
| `MI_THRESHOLD` | 0.001 | classification |

---

## CI/CD

- **GitHub Actions**: `pytest tests/ -v` on every push/PR to main
- **Streamlit Community Cloud**: Auto-deploys from main branch
- **Docker**: Multi-stage build for local development

---

## Project Stats

| Metric | Value |
|--------|-------|
| Source code | 2,465 lines / 14 modules |
| Tests | 1,504 lines / 171 tests / 7 files |
| Scripts | 474 lines |
| Total | ~4,400 lines |
| Dependencies | 12 packages |
| Data range | 2015-2024 (10 years, ~2,500 trading days) |
| Models trained | 6 (OLS + DT + RF + KNN + SVM + Voting Ensemble) |
