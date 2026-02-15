"""Tests for data ingestion and alignment."""

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import align_data, compute_log_returns


def test_log_returns_known_values():
    """Log return of 100 -> 105 should be log(1.05)."""
    prices = pd.DataFrame({"A": [100.0, 105.0, 110.25]})
    returns = compute_log_returns(prices)
    expected_first = np.log(105.0 / 100.0)
    np.testing.assert_allclose(returns.iloc[0, 0], expected_first, atol=1e-10)


def test_log_returns_rejects_negative_prices():
    """Negative prices should raise ValueError."""
    prices = pd.DataFrame({"A": [100.0, -5.0, 110.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        compute_log_returns(prices)


def test_log_returns_rejects_zero_prices():
    """Zero prices should raise ValueError."""
    prices = pd.DataFrame({"A": [100.0, 0.0, 110.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        compute_log_returns(prices)


def test_align_backward_direction():
    """Ensure alignment uses backward direction (no look-ahead bias)."""
    equity = pd.DataFrame(
        {"price": [100, 101, 102, 103, 104]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    # Macro data available on Jan 1 and Jan 3
    macro = pd.DataFrame(
        {"rate": [5.0, 5.5]},
        index=pd.to_datetime(["2024-01-01", "2024-01-03"]),
    )
    # Pad equity to meet MIN_PERIODS for testing by lowering threshold
    import src.data.config as cfg
    orig = cfg.MIN_PERIODS
    cfg.MIN_PERIODS = 2
    try:
        merged = align_data(equity, macro)
        # Jan 2 should have rate=5.0 (backward from Jan 1), not 5.5 (Jan 3 = future)
        jan2_rate = merged.loc["2024-01-02", "rate"]
        assert jan2_rate == 5.0, f"Expected 5.0 but got {jan2_rate} — possible look-ahead bias"
    finally:
        cfg.MIN_PERIODS = orig


def test_align_drops_all_nan_macro():
    """Rows where ALL macro columns are NaN should be dropped."""
    equity = pd.DataFrame(
        {"price": range(200, 210)},
        index=pd.bdate_range("2024-01-01", periods=10),
    )
    macro = pd.DataFrame(
        {"rate": [5.0]},
        index=pd.to_datetime(["2024-01-15"]),  # After all equity dates
    )
    import src.data.config as cfg
    orig = cfg.MIN_PERIODS
    cfg.MIN_PERIODS = 1
    try:
        # With 90d tolerance and all equity before macro, should get empty after dropping NaN
        with pytest.raises(ValueError):
            align_data(equity, macro)
    finally:
        cfg.MIN_PERIODS = orig
