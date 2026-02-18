"""Tests for portfolio risk analytics."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.analytics.portfolio import (
    capm,
    cvar,
    max_drawdown,
    risk_summary,
    sharpe_ratio,
    sortino_ratio,
    var_historical,
    var_parametric,
    weighted_returns,
)


def test_sharpe_ratio_known():
    """Constant returns (std=0) should return NaN."""
    returns = np.full(252, 0.001)
    result = sharpe_ratio(returns, risk_free_rate=0.0)
    assert np.isnan(result)


def test_sharpe_ratio_positive():
    """Random returns with positive mean should yield positive finite Sharpe."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 252)
    result = sharpe_ratio(returns, risk_free_rate=0.0)
    assert np.isfinite(result)
    assert result > 0


def test_sortino_no_downside():
    """All positive returns should return NaN (fewer than 5 downside obs)."""
    returns = np.abs(np.random.default_rng(42).normal(0.01, 0.005, 252))
    result = sortino_ratio(returns, risk_free_rate=0.0, mar=0.0)
    assert np.isnan(result)


def test_var_parametric_vs_historical():
    """For normal returns, parametric and historical VaR should be within 10%."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.02, 10000)
    vp = var_parametric(returns, confidence=0.95)
    vh = var_historical(returns, confidence=0.95)
    assert abs(vp) > 0 and abs(vh) > 0
    relative_diff = abs(vp - vh) / abs(vh)
    assert relative_diff < 0.10, f"VaR divergence {relative_diff:.4f} exceeds 10%"


def test_cvar_worse_than_var():
    """CVaR (expected shortfall) should be <= VaR (more negative)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.02, 5000)
    vh = var_historical(returns, confidence=0.95)
    cv = cvar(returns, confidence=0.95)
    assert cv <= vh, f"CVaR {cv} should be <= VaR {vh}"


def test_max_drawdown():
    """Known series: peak 1.1 to trough 0.9 = 18.18% drawdown."""
    cum = np.array([1.0, 1.1, 1.05, 0.9, 1.0])
    result = max_drawdown(cum)
    expected = (1.1 - 0.9) / 1.1
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_max_drawdown_monotonic():
    """Monotonically increasing series should have zero drawdown."""
    cum = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    assert max_drawdown(cum) == 0.0


def test_weighted_returns_normalization():
    """Weights that don't sum to 1 should be normalized."""
    returns_df = pd.DataFrame({
        "A": [0.01, 0.02, 0.03],
        "B": [0.04, 0.05, 0.06],
        "C": [0.07, 0.08, 0.09],
    })
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = weighted_returns(returns_df, {"A": 0.5, "B": 0.5, "C": 0.5})
        assert len(w) == 1 and "normalizing" in str(w[0].message)
    # After normalization, weights are 1/3 each
    expected = returns_df.values @ np.array([1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_weighted_returns_empty_raises():
    """Empty weights dict should raise ValueError."""
    returns_df = pd.DataFrame({"A": [0.01]})
    with pytest.raises(ValueError, match="empty"):
        weighted_returns(returns_df, {})


def test_weighted_returns_no_match_raises():
    """No matching columns should raise ValueError."""
    returns_df = pd.DataFrame({"A": [0.01]})
    with pytest.raises(ValueError, match="No matching"):
        weighted_returns(returns_df, {"X": 1.0})


def test_capm_perfect_correlation():
    """Portfolio = 2 * market should yield beta ~ 2.0."""
    rng = np.random.default_rng(42)
    market = rng.normal(0.0005, 0.015, 500)
    portfolio = 2.0 * market
    result = capm(portfolio, market)
    np.testing.assert_allclose(result["beta"], 2.0, atol=1e-6)
    np.testing.assert_allclose(result["r_squared"], 1.0, atol=1e-6)


def test_capm_returns_dict_keys():
    """CAPM result should contain all expected keys."""
    rng = np.random.default_rng(42)
    market = rng.normal(0.0, 0.01, 100)
    port = rng.normal(0.0, 0.01, 100)
    result = capm(port, market)
    expected_keys = {"alpha", "beta", "r_squared", "alpha_pvalue", "beta_pvalue"}
    assert set(result.keys()) == expected_keys


def test_risk_summary_keys():
    """risk_summary should return all expected metrics."""
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame({
        "MS": rng.normal(0.0005, 0.02, 300),
        "JPM": rng.normal(0.0004, 0.018, 300),
        "BAC": rng.normal(0.0003, 0.022, 300),
    })
    market = rng.normal(0.0004, 0.015, 300)
    weights = {"MS": 0.33, "JPM": 0.34, "BAC": 0.33}
    result = risk_summary(returns_df, weights, market, risk_free_rate=0.02)
    expected_keys = {
        "annualized_return", "annualized_vol", "sharpe", "sortino",
        "beta", "alpha", "alpha_pvalue", "beta_pvalue", "r_squared",
        "var_parametric", "var_historical", "cvar", "max_drawdown",
        "skewness", "kurtosis", "kurtosis_risk",
        "portfolio_returns", "cumulative_returns",
    }
    assert set(result.keys()) == expected_keys
    assert isinstance(result["kurtosis_risk"], bool)
    assert np.isfinite(result["annualized_return"])
    assert np.isfinite(result["annualized_vol"])
    assert result["max_drawdown"] >= 0.0
