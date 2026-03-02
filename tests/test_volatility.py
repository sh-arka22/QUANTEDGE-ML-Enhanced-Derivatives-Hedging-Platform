"""Tests for src.analytics.volatility — GARCH, EWMA, ARCH LM, cones, forecast."""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.volatility import (
    arch_lm_test,
    fit_ewma,
    fit_garch,
    forecast_volatility,
    realized_vs_predicted,
    run_volatility,
    volatility_cones,
)
from src.data import config as cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_returns():
    """Synthetic log returns with realistic GARCH-like properties."""
    rng = np.random.default_rng(42)
    n = 600
    # Simulate GARCH-like process: clustered volatility
    returns = np.zeros(n)
    sigma = np.zeros(n)
    sigma[0] = 0.01
    for t in range(1, n):
        sigma[t] = np.sqrt(1e-6 + 0.1 * returns[t - 1] ** 2 + 0.85 * sigma[t - 1] ** 2)
        returns[t] = sigma[t] * rng.standard_normal()
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(returns, index=idx, name="returns")


@pytest.fixture()
def short_returns():
    """Very short return series for edge-case testing."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-01", periods=20)
    return pd.Series(rng.standard_normal(20) * 0.01, index=idx)


# ---------------------------------------------------------------------------
# arch_lm_test
# ---------------------------------------------------------------------------

class TestArchLM:
    def test_returns_dict_keys(self, synthetic_returns):
        result = arch_lm_test(synthetic_returns)
        assert set(result.keys()) == {"statistic", "pvalue", "has_arch_effects", "lags"}

    def test_statistic_positive(self, synthetic_returns):
        result = arch_lm_test(synthetic_returns)
        assert result["statistic"] >= 0

    def test_pvalue_in_range(self, synthetic_returns):
        result = arch_lm_test(synthetic_returns)
        assert 0 <= result["pvalue"] <= 1

    def test_detects_arch_effects(self, synthetic_returns):
        result = arch_lm_test(synthetic_returns)
        assert result["has_arch_effects"]  # synthetic has vol clustering

    def test_short_data_reduces_lags(self, short_returns):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = arch_lm_test(short_returns, lags=50)
        assert result["lags"] < 50

    def test_custom_lags(self, synthetic_returns):
        result = arch_lm_test(synthetic_returns, lags=10)
        assert result["lags"] == 10


# ---------------------------------------------------------------------------
# fit_garch
# ---------------------------------------------------------------------------

class TestFitGarch:
    def test_returns_dict_with_expected_keys(self, synthetic_returns):
        result = fit_garch(synthetic_returns)
        assert result is not None
        assert "params" in result
        assert "conditional_vol" in result
        assert "aic" in result
        assert "bic" in result

    def test_params_in_expected_range(self, synthetic_returns):
        result = fit_garch(synthetic_returns)
        p = result["params"]
        assert 0 <= p["alpha"] <= 1
        assert 0 <= p["beta"] <= 1
        assert 0 < p["persistence"] < 1.05  # allow slight numerical excess

    def test_conditional_vol_nonneg(self, synthetic_returns):
        result = fit_garch(synthetic_returns)
        assert np.all(result["conditional_vol"] >= 0)

    def test_long_run_vol_finite(self, synthetic_returns):
        result = fit_garch(synthetic_returns)
        lrv = result["params"]["long_run_vol"]
        if result["params"]["persistence"] < 1.0:
            assert np.isfinite(lrv)
            assert lrv > 0

    def test_too_few_observations_raises(self):
        short = pd.Series(np.random.default_rng(1).standard_normal(10) * 0.01)
        with pytest.raises(ValueError, match="Insufficient data"):
            fit_garch(short)

    def test_zero_variance_raises(self):
        flat = pd.Series(np.zeros(200))
        with pytest.raises(ValueError, match="Zero variance"):
            fit_garch(flat)


# ---------------------------------------------------------------------------
# fit_ewma
# ---------------------------------------------------------------------------

class TestFitEWMA:
    def test_returns_series(self, synthetic_returns):
        result = fit_ewma(synthetic_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == len(synthetic_returns.dropna())

    def test_no_nans(self, synthetic_returns):
        result = fit_ewma(synthetic_returns)
        assert result.isna().sum() == 0

    def test_positive_values(self, synthetic_returns):
        result = fit_ewma(synthetic_returns)
        assert (result >= 0).all()

    def test_annualized_scale(self, synthetic_returns):
        result = fit_ewma(synthetic_returns)
        # Annualized vol should be in a reasonable range (not daily)
        median_vol = result.median()
        assert 0.01 < median_vol < 5.0  # very wide range, just sanity

    def test_bad_lambda_warns(self, synthetic_returns):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fit_ewma(synthetic_returns, lambda_=1.5)
            assert any("Lambda" in str(x.message) for x in w)
        assert len(result) > 0  # still produces output

    def test_custom_lambda(self, synthetic_returns):
        r1 = fit_ewma(synthetic_returns, lambda_=0.90)
        r2 = fit_ewma(synthetic_returns, lambda_=0.98)
        # Lower lambda = more weight on recent, higher reactivity
        # They should differ
        assert not np.allclose(r1.values, r2.values)


# ---------------------------------------------------------------------------
# forecast_volatility
# ---------------------------------------------------------------------------

class TestForecast:
    def test_returns_series(self, synthetic_returns):
        garch = fit_garch(synthetic_returns)
        fc = forecast_volatility(garch, horizon=10)
        assert isinstance(fc, pd.Series)
        assert len(fc) == 10

    def test_forecast_positive(self, synthetic_returns):
        garch = fit_garch(synthetic_returns)
        fc = forecast_volatility(garch, horizon=10)
        assert (fc > 0).all()

    def test_none_input_returns_empty(self):
        fc = forecast_volatility(None)
        assert len(fc) == 0

    def test_horizon_capping(self, synthetic_returns):
        garch = fit_garch(synthetic_returns)
        # Request absurdly long horizon
        fc = forecast_volatility(garch, horizon=10_000)
        assert len(fc) <= len(synthetic_returns) // 3


# ---------------------------------------------------------------------------
# volatility_cones
# ---------------------------------------------------------------------------

class TestVolatilityCones:
    def test_returns_dataframe(self, synthetic_returns):
        result = volatility_cones(synthetic_returns)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"min", "q25", "median", "q75", "max"}

    def test_default_windows(self, synthetic_returns):
        result = volatility_cones(synthetic_returns)
        assert list(result.index) == [10, 30, 60, 90]

    def test_custom_windows(self, synthetic_returns):
        result = volatility_cones(synthetic_returns, windows=[5, 20])
        assert list(result.index) == [5, 20]

    def test_ordering(self, synthetic_returns):
        result = volatility_cones(synthetic_returns)
        for _, row in result.iterrows():
            assert row["min"] <= row["q25"] <= row["median"] <= row["q75"] <= row["max"]

    def test_window_exceeds_data_skipped(self, short_returns):
        result = volatility_cones(short_returns, windows=[5, 10, 1000])
        assert 1000 not in result.index
        assert 5 in result.index

    def test_all_windows_exceed_warns(self, short_returns):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = volatility_cones(short_returns, windows=[1000, 2000])
            assert any("exceed" in str(x.message) for x in w)
        assert result.empty


# ---------------------------------------------------------------------------
# realized_vs_predicted
# ---------------------------------------------------------------------------

class TestRealizedVsPredicted:
    def test_two_columns(self, synthetic_returns):
        garch = fit_garch(synthetic_returns)
        result = realized_vs_predicted(synthetic_returns, garch)
        assert "realized" in result.columns
        assert "predicted" in result.columns

    def test_no_nans_after_dropna(self, synthetic_returns):
        garch = fit_garch(synthetic_returns)
        result = realized_vs_predicted(synthetic_returns, garch)
        assert result.isna().sum().sum() == 0

    def test_none_garch_fills_nan(self, synthetic_returns):
        result = realized_vs_predicted(synthetic_returns, None)
        # predicted should be NaN → entire df empty after dropna
        assert result.empty or "predicted" in result.columns


# ---------------------------------------------------------------------------
# run_volatility (orchestrator)
# ---------------------------------------------------------------------------

class TestRunVolatility:
    def test_returns_all_keys(self, synthetic_returns):
        result = run_volatility(synthetic_returns)
        expected = {
            "arch_lm_test", "garch", "ewma_vol", "forecast",
            "cones", "realized_vs_predicted", "used_ewma_fallback",
        }
        assert set(result.keys()) == expected

    def test_garch_present(self, synthetic_returns):
        result = run_volatility(synthetic_returns)
        assert result["garch"] is not None
        assert result["used_ewma_fallback"] is False

    def test_ewma_vol_series(self, synthetic_returns):
        result = run_volatility(synthetic_returns)
        assert isinstance(result["ewma_vol"], pd.Series)
        assert result["ewma_vol"].isna().sum() == 0
