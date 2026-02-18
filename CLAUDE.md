# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-grade Quantitative Finance platform (Python 3.11 + Streamlit) covering portfolio risk analytics, regression/classification modeling, GARCH volatility, derivatives pricing, and delta-neutral hedging. Deployed to Streamlit Community Cloud (1GB RAM).

Full technical spec: `AI Coding Assistant Prompt Engineering.docx` (in parent directory). 14-day build plan: `.memory/plan.md`.

## Commands

```bash
# Activate conda environment
conda activate ./quant

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_data.py::test_log_returns_known_values -v

# Run Streamlit app
streamlit run src/ui/app.py

# Run app with FRED macro data enabled
FRED_API_KEY=your_key streamlit run src/ui/app.py
```

No linter or formatter is configured. Code follows PEP 8 with type hints.

## Architecture

Four-layer modular monolith:

```
src/data/        → Data ingestion (yfinance equity, FRED macro, merge_asof alignment)
src/analytics/   → Portfolio metrics, OLS regression, ML classification, GARCH volatility
src/pricing/     → Black-Scholes, CRR binomial tree, delta-neutral hedging
src/ui/          → Streamlit dashboard (4 tabs: portfolio, prediction, derivatives, hedging)
```

`src/data/config.py` is the single source of truth for tickers, FRED series IDs, portfolio weights, date ranges, and all numeric constants (TRADING_DAYS=252, VIF_THRESHOLD=10, etc.).

`src/data/loaders.py` orchestrates all data fetching via `get_all_data()`, which is cached with `@st.cache_data(ttl=3600)`.

## Critical Constraints

These are non-negotiable rules from the project spec:

- **Vectorization**: ALL numeric operations use numpy/pandas. No Python for-loops in pricing or analytics.
- **Log returns only**: Always `np.log(prices / prices.shift(1))`, never simple returns.
- **Adjusted Close only**: Never use raw Close prices.
- **No look-ahead bias**: Macro data merged with `merge_asof(direction='backward', tolerance='90d')`.
- **Chronological splits**: Train/test splits are time-ordered, never random shuffle.
- **StandardScaler**: Required before KNN and SVM classifiers.
- **GARCH fallback**: If GARCH fails to converge, fall back to EWMA (lambda=0.94).
- **Black-Scholes T=0**: Return intrinsic value, not NaN.
- **Binomial tree**: N capped at MAX_TREE_STEPS=1000.
- **FRED graceful degradation**: App must work fully without a FRED API key.
- **Streamlit caching**: `@st.cache_data` on expensive computations, `@st.cache_resource` for models.
- **One-line docstrings only**: Keep files minimal and clean.

## Testing

Tests use pytest with no additional plugins. Test files mirror source structure in `tests/`. Black-Scholes tests should validate against Hull textbook values. All edge cases (empty data, zero prices, division by zero in Sharpe/Sortino) must be covered.

## UI Patterns

- Dark theme configured in `.streamlit/config.toml`
- `plotly_dark` template for all Plotly charts
- `st.spinner` wrapping all long operations
- Lazy data loading (only fetch when tab is active)
- float32 for plotting data to save memory
- Plotly 3D surfaces capped at 20x15 grid
