---
aliases: [Dashboard, UI, Streamlit App]
tags: [architecture, ui, streamlit]
---

# Streamlit Dashboard

## Overview

The platform is served as a single-page Streamlit app (`src/ui/app.py`) with four tabs and a global sidebar. All visualizations use Plotly with the `plotly_dark` theme.

## Global Controls (Sidebar)

- **FRED API Key**: Optional, enables macro features across all tabs
- **Risk-Free Rate**: Default 2%, affects [[CAPM (Capital Asset Pricing Model)]], [[Sharpe and Sortino Ratios]], [[Black-Scholes Model]], and [[Delta-Neutral Hedging]]

## Tab Structure

| Tab | Module | Primary Analytics |
|-----|--------|-------------------|
| [[Tab 1 - Portfolio Analytics]] | `tab_portfolio.py` | Risk metrics, cumulative returns, VaR, correlation |
| [[Tab 2 - Price Prediction]] | `tab_prediction.py` | AAPL regression + BAC classification |
| [[Tab 3 - Derivatives Pricing]] | `tab_derivatives.py` | BS/CRR pricing, Greeks, 3D surface |
| [[Tab 4 - Hedging Simulator]] | `tab_hedging.py` | Delta-neutral simulation, band optimization |

## Key Design Choices

1. **Lazy imports**: Each tab imports its analytics module only when rendered, reducing initial load time.
2. **Error isolation**: Each tab is wrapped in `try/except` — a crash in one tab doesn't kill the others.
3. **Caching**: `get_all_data()` uses `@st.cache_data(ttl=3600)` to avoid refetching data on every interaction.
4. **Plotly throughout**: Interactive charts with hover, zoom, pan — much richer than matplotlib.

## Deployment

- **Streamlit Community Cloud** with 1GB RAM constraint
- **Docker** containerization for reproducibility
- **GitHub Actions** CI/CD pipeline for automated testing

## Connections

- [[Project Architecture]] — Overall system design
- [[Data Ingestion Layer]] — Data fetched once, shared across tabs
