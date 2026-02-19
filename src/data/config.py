"""Configuration constants for the Quantitative Finance Platform."""

TICKERS = {
    "banking": ["MS", "JPM", "BAC"],
    "tech": ["AAPL"],
    "market": ["^GSPC", "^IXIC"],
}

FRED_SERIES = {
    "DGS10": "10Y Treasury",
    "T10Y2Y": "Yield Curve",
    "VIXCLS": "VIX",
    "GDPC1": "Real GDP",
    "CPIAUCSL": "CPI",
    "DTWEXBGS": "USD Index",
    "POILWTIUSDQ": "WTI Oil",
}

PORTFOLIO_WEIGHTS = {"MS": 0.33, "JPM": 0.34, "BAC": 0.33}

DATE_RANGE = ("2015-01-01", "2024-12-31")

# Numeric constants
TRADING_DAYS = 252
MIN_PERIODS = 100
VIF_THRESHOLD = 10
VAR_CONFIDENCE = 0.95
EWMA_LAMBDA = 0.94
MAX_TREE_STEPS = 1000
MAX_RETRIES = 3
STALE_DAYS_THRESHOLD = 5

# Classification tuning
CV_SPLITS = 5
TUNING_N_ITER = 20
TUNING_SCORING = "f1"
MI_THRESHOLD = 0.001
