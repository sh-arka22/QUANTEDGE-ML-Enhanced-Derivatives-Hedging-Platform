"""Data ingestion layer: yfinance equity, FRED macro, alignment, and returns."""

import logging
import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from src.data.config import (
    DATE_RANGE,
    FRED_SERIES,
    MAX_RETRIES,
    MIN_PERIODS,
    PORTFOLIO_WEIGHTS,
    STALE_DAYS_THRESHOLD,
    TICKERS,
    TRADING_DAYS,
)

logger = logging.getLogger(__name__)


def fetch_equity(
    tickers: list[str], start: str = DATE_RANGE[0], end: str = DATE_RANGE[1]
) -> pd.DataFrame:
    """Fetch adjusted close prices with retry and validation."""
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES}: {e}")
            else:
                raise ConnectionError(f"Failed to fetch equity data after {MAX_RETRIES} attempts: {e}")

    if df.empty or len(df) == 0:
        raise ValueError(f"No data returned for tickers {tickers}")

    # Extract Adj Close
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]
        elif "Close" in df.columns.get_level_values(0):
            df = df["Close"]
            warnings.warn("Adj Close unavailable, using Close")
        else:
            raise ValueError("Neither Adj Close nor Close found in data")
    elif "Adj Close" in df.columns:
        df = df[["Adj Close"]]
    else:
        df = df[["Close"]]
        warnings.warn("Adj Close unavailable, using Close")

    # Handle single ticker returning Series
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0] if len(tickers) == 1 else "price")

    # Fill NaN: forward then backward
    df = df.ffill().bfill()
    if df.isnull().any().any():
        raise ValueError("Unfillable NaN values remain after forward/backward fill")

    # Stale data check
    last_date = df.index[-1]
    bdays_since = len(pd.bdate_range(last_date, pd.Timestamp.now())) - 1
    if bdays_since > STALE_DAYS_THRESHOLD:
        warnings.warn(f"Stale data: last date {last_date.date()}, {bdays_since} business days ago")

    if len(df) < MIN_PERIODS:
        raise ValueError(f"Insufficient data: {len(df)} rows < minimum {MIN_PERIODS}")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def fetch_fred(
    series_ids: Optional[dict] = None,
    start: str = DATE_RANGE[0],
    end: str = DATE_RANGE[1],
    api_key: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch FRED macro data. Returns None if no API key (graceful degradation)."""
    if not api_key:
        logger.info("No FRED API key provided — macro features disabled")
        return None

    try:
        from fredapi import Fred
    except ImportError:
        warnings.warn("fredapi not installed — macro features disabled")
        return None

    if series_ids is None:
        series_ids = FRED_SERIES

    fred = Fred(api_key=api_key)
    frames = {}

    for sid, label in series_ids.items():
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            if s is not None and len(s) > 0:
                frames[sid] = s
            else:
                warnings.warn(f"Empty FRED series: {sid} ({label})")
        except Exception as e:
            warnings.warn(f"Failed to fetch FRED {sid} ({label}): {e}")
            continue

    if not frames:
        warnings.warn("No FRED series fetched — macro features disabled")
        return None

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    return df


def align_data(equity_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Merge equity and macro data using backward asof join (no look-ahead bias)."""
    equity = equity_df.copy()
    macro = macro_df.copy()

    # Ensure sorted
    equity = equity.sort_index()
    macro = macro.sort_index()

    # Remove duplicate timestamps
    equity = equity[~equity.index.duplicated(keep="last")]
    macro = macro[~macro.index.duplicated(keep="last")]

    # merge_asof requires both to have the same index name
    equity.index.name = "date"
    macro.index.name = "date"

    merged = pd.merge_asof(
        equity.reset_index(),
        macro.reset_index(),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("90d"),
    ).set_index("date")

    # Drop rows where ALL macro columns are NaN
    macro_cols = [c for c in macro.columns if c in merged.columns]
    if macro_cols:
        merged = merged.dropna(subset=macro_cols, how="all")

    if len(merged) < MIN_PERIODS:
        raise ValueError(
            f"Insufficient data after alignment: {len(merged)} rows < {MIN_PERIODS}"
        )

    return merged


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price series."""
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive for log returns")
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def _get_fred_key() -> Optional[str]:
    """Retrieve FRED API key from environment or Streamlit secrets."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    try:
        key = st.secrets.get("FRED_API_KEY", None)
    except Exception:
        pass
    return key


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_data(
    start: str = DATE_RANGE[0],
    end: str = DATE_RANGE[1],
    fred_api_key: Optional[str] = None,
) -> dict:
    """Orchestrator: fetch all data, align, compute returns."""
    all_tickers = TICKERS["banking"] + TICKERS["tech"] + TICKERS["market"]
    prices = fetch_equity(all_tickers, start, end)

    # Compute returns for banking stocks
    banking_prices = prices[[t for t in TICKERS["banking"] if t in prices.columns]]
    banking_returns = compute_log_returns(banking_prices)

    # Market returns (S&P 500)
    market_col = "^GSPC" if "^GSPC" in prices.columns else prices.columns[0]
    market_returns = compute_log_returns(prices[[market_col]]).iloc[:, 0]

    # FRED macro data (graceful if no key)
    api_key = fred_api_key or _get_fred_key()
    macro_df = fetch_fred(api_key=api_key, start=start, end=end)

    macro_aligned = None
    has_macro = False
    if macro_df is not None:
        try:
            macro_aligned = align_data(banking_prices, macro_df)
            has_macro = True
        except ValueError as e:
            warnings.warn(f"Macro alignment failed: {e}")

    return {
        "prices": prices,
        "returns": banking_returns,
        "market_returns": market_returns,
        "macro_aligned": macro_aligned,
        "has_macro": has_macro,
    }
