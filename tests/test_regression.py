"""Tests for AAPL OLS regression with diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.regression import (
    diagnostics,
    evaluate,
    fit_ols,
    prepare_features,
    run_regression,
)


# --- Fixtures ---

def _make_prices(n=500, seed=42):
    """Generate synthetic price data for AAPL and market indices."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Random walk prices
    aapl = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    gspc = 3000 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))
    ixic = 10000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    prices = pd.DataFrame({"AAPL": aapl, "^GSPC": gspc, "^IXIC": ixic}, index=dates)
    market = prices[["^GSPC", "^IXIC"]]
    return prices, market


# --- prepare_features tests ---

def test_prepare_features_shape():
    """Output shapes should reflect 80/20 split and feature count."""
    prices, market = _make_prices()
    X_train, X_test, y_train, y_test = prepare_features(prices, market)
    total = len(X_train) + len(X_test)
    assert abs(len(X_train) / total - 0.8) < 0.02  # ~80% train
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_prepare_features_expected_columns():
    """Feature set should include SMA, lag, and market lag columns."""
    prices, market = _make_prices()
    X_train, _, _, _ = prepare_features(prices, market)
    cols = X_train.columns.tolist()
    assert "SMA_5" in cols
    assert "SMA_20" in cols
    assert "Lag_1_Return" in cols
    assert "SP500_Lag1" in cols
    assert "NASDAQ_Lag1" in cols


def test_prepare_features_chronological_order():
    """Train dates must all precede test dates."""
    prices, market = _make_prices()
    X_train, X_test, _, _ = prepare_features(prices, market)
    assert X_train.index[-1] < X_test.index[0]


def test_prepare_features_no_nans():
    """No NaN values in features or target after preparation."""
    prices, market = _make_prices()
    X_train, X_test, y_train, y_test = prepare_features(prices, market)
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()
    assert not y_train.isnull().any()
    assert not y_test.isnull().any()


def test_prepare_features_missing_aapl_raises():
    """Should raise if AAPL column missing from prices."""
    prices = pd.DataFrame({"JPM": [100.0] * 200}, index=pd.bdate_range("2020-01-01", periods=200))
    market = pd.DataFrame({"^GSPC": [3000.0] * 200}, index=prices.index)
    with pytest.raises(ValueError, match="AAPL"):
        prepare_features(prices, market)


# --- fit_ols tests ---

def test_fit_ols_returns_model():
    """OLS fit should return a statsmodels results object."""
    prices, market = _make_prices()
    X_train, _, y_train, _ = prepare_features(prices, market)
    model = fit_ols(X_train, y_train)
    assert hasattr(model, "params")
    assert hasattr(model, "rsquared")


def test_fit_ols_param_count():
    """Model should have intercept + n_features parameters."""
    prices, market = _make_prices()
    X_train, _, y_train, _ = prepare_features(prices, market)
    model = fit_ols(X_train, y_train)
    assert len(model.params) == X_train.shape[1] + 1  # +1 for constant


# --- diagnostics tests ---

def test_diagnostics_keys():
    """Diagnostics dict should contain all expected keys."""
    prices, market = _make_prices()
    X_train, _, y_train, _ = prepare_features(prices, market)
    model = fit_ols(X_train, y_train)
    diag = diagnostics(model, X_train, y_train)
    expected_keys = {"vif", "vif_flag", "bp_stat", "bp_pvalue", "is_heteroscedastic",
                     "dw_stat", "dw_flag", "jb_stat", "jb_pvalue", "is_normal"}
    assert expected_keys.issubset(diag.keys())


def test_diagnostics_vif_values():
    """VIF values should be positive and finite for well-behaved data."""
    prices, market = _make_prices()
    X_train, _, y_train, _ = prepare_features(prices, market)
    model = fit_ols(X_train, y_train)
    diag = diagnostics(model, X_train, y_train)
    for name, val in diag["vif"].items():
        assert val > 0, f"VIF for {name} should be positive"
        assert np.isfinite(val), f"VIF for {name} should be finite"


def test_diagnostics_dw_range():
    """Durbin-Watson statistic should be in [0, 4]."""
    prices, market = _make_prices()
    X_train, _, y_train, _ = prepare_features(prices, market)
    model = fit_ols(X_train, y_train)
    diag = diagnostics(model, X_train, y_train)
    assert 0 <= diag["dw_stat"] <= 4


# --- evaluate tests ---

def test_evaluate_metrics_finite():
    """Evaluation metrics should be finite numbers."""
    prices, market = _make_prices()
    X_train, X_test, y_train, y_test = prepare_features(prices, market)
    model = fit_ols(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["r_squared"])


def test_evaluate_rmse_positive():
    """RMSE should be non-negative."""
    prices, market = _make_prices()
    X_train, X_test, y_train, y_test = prepare_features(prices, market)
    model = fit_ols(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert metrics["rmse"] >= 0


def test_evaluate_predictions_shape():
    """Predictions array should match test set size."""
    prices, market = _make_prices()
    X_train, X_test, y_train, y_test = prepare_features(prices, market)
    model = fit_ols(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert len(metrics["predictions"]) == len(y_test)


# --- run_regression integration test ---

def test_run_regression_full_pipeline():
    """Full pipeline should return all expected keys."""
    prices, market = _make_prices()
    result = run_regression(prices, market)
    expected_keys = {"model", "robust_model", "diagnostics", "metrics",
                     "feature_names", "X_train", "X_test", "y_train", "y_test"}
    assert expected_keys == set(result.keys())
    assert result["metrics"]["rmse"] >= 0
    assert len(result["feature_names"]) > 0
