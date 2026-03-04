---
aliases: [Data Pipeline, Data Loaders]
tags: [data, infrastructure, first-principles]
---

# Data Ingestion Layer

## Overview

The data layer (`src/data/loaders.py`) handles all data acquisition, validation, and preparation. It follows a **graceful degradation** pattern — the platform works with or without macro data.

## Data Sources

### Equity Data (yfinance)
- **Tickers**: MS, JPM, BAC (banking), AAPL (tech), ^GSPC, ^IXIC (market indices)
- **Date range**: 2015-01-01 to 2024-12-31 (10 years)
- **Field**: Adjusted Close (falls back to Close with warning)
- **Retry logic**: Up to 3 attempts with exponential backoff
- **Validation**: Checks for empty data, NaN (forward/backward fill), stale data (warns if >5 business days old), minimum 100 rows

### Macro Data (FRED API — Optional)
Seven macroeconomic series from the Federal Reserve:
- 10Y Treasury yield, Yield Curve (10Y-2Y), VIX
- Real GDP (quarterly), CPI (monthly), USD Index, WTI Oil

Requires a FRED API key (from environment variable, Streamlit secrets, or sidebar input). If absent, the platform runs in **technical-only mode**.

## Data Alignment

See [[Asof Join and Look-Ahead Bias]] for how macro data is merged with equity data.

## Return Computation

See [[Log Returns]] for why log returns are used and how they're computed.

## Orchestration

`get_all_data()` ties everything together:
1. Fetch all equity tickers
2. Compute banking returns and market returns
3. Optionally fetch and align FRED macro data
4. Cache results for 1 hour via `@st.cache_data(ttl=3600)`

Returns a dictionary consumed by all four dashboard tabs.

## Connections

- [[Log Returns]] — Return computation method
- [[Asof Join and Look-Ahead Bias]] — Macro data alignment
- [[Streamlit Dashboard]] — Data flows into all tabs
- [[Project Architecture]] — Data layer is the foundation of the platform
