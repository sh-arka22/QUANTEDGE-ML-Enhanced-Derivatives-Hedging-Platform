---
aliases: [Architecture, Module Map]
tags: [architecture, project-structure]
---

# Project Architecture

## Module Map

```
QUANTEDGE/
├── src/
│   ├── data/                    # Data Layer
│   │   ├── config.py            # All constants and thresholds
│   │   └── loaders.py           # yfinance + FRED ingestion
│   ├── analytics/               # Analytics Layer
│   │   ├── portfolio.py         # Portfolio risk metrics
│   │   ├── regression.py        # AAPL OLS regression
│   │   ├── classification.py    # BAC ML classification
│   │   └── volatility.py        # GARCH + EWMA modeling
│   ├── pricing/                 # Derivatives Layer
│   │   ├── black_scholes.py     # BS pricing, Greeks, IV
│   │   ├── binomial.py          # CRR tree pricing
│   │   └── hedging.py           # Delta-neutral simulation
│   └── ui/                      # Presentation Layer
│       ├── app.py               # Streamlit entry point
│       ├── tab_portfolio.py     # Tab 1
│       ├── tab_prediction.py    # Tab 2
│       ├── tab_derivatives.py   # Tab 3
│       └── tab_hedging.py       # Tab 4
├── tests/                       # Unit tests
├── docs/                        # Development docs
├── Dockerfile                   # Container config
└── .github/workflows/ci.yml     # CI/CD pipeline
```

## Data Flow

```
yfinance API ──→ fetch_equity() ──→ prices DataFrame
FRED API ──────→ fetch_fred() ───→ macro DataFrame
                                          │
                      align_data() ←──────┘
                           │
                    compute_log_returns()
                           │
                    get_all_data() [cached 1hr]
                           │
              ┌────────────┼────────────────┬──────────────┐
              ▼            ▼                ▼              ▼
        Tab 1:        Tab 2:          Tab 3:          Tab 4:
       Portfolio     Prediction     Derivatives      Hedging
       Analytics     (OLS + ML)     (BS + CRR)      Simulator
```

## Design Principles

1. **Separation of Concerns**: Data, analytics, pricing, and UI are fully independent modules. You can use `src.pricing.black_scholes` without any Streamlit dependency.

2. **Graceful Degradation**: The platform works without FRED macro data — classification and regression fall back to technical-only features.

3. **Vectorization**: Heavy computation (BS pricing, Greeks, terminal payoffs) uses NumPy broadcasting instead of Python loops for performance under Streamlit Cloud's 1GB RAM constraint.

4. **Config-Driven**: All magic numbers live in `src/data/config.py` — trading days (252), VIF threshold (10), VaR confidence (95%), EWMA lambda (0.94), etc.

5. **Defensive Programming**: Input validation, NaN handling, convergence fallbacks, and warnings throughout.

## Connections

- [[Streamlit Dashboard]] — The UI layer
- [[Data Ingestion Layer]] — The foundation
- [[Black-Scholes Model]], [[Binomial Tree (CRR)]], [[Delta-Neutral Hedging]] — Pricing layer
- [[OLS Regression with Diagnostics]], [[ML Classification Pipeline]], [[GARCH(1,1) Model]] — Analytics layer
