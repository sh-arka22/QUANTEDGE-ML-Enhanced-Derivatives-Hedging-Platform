"""Tests for src.pricing.hedging — delta-neutral hedging simulation."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.pricing.hedging import (
    hedging_summary,
    optimal_band_search,
    simulate_delta_hedge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def spot_series():
    """Synthetic 63-day spot prices (random walk from S=100)."""
    rng = np.random.default_rng(42)
    n = 63
    steps = rng.standard_normal(n) * 0.015
    prices = 100 * np.exp(np.cumsum(steps))
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(prices, index=idx, name="spot")


@pytest.fixture()
def sim_df(spot_series):
    """Pre-computed simulation for summary tests."""
    return simulate_delta_hedge(
        spot_series, K=100, T=0.25, r=0.05, sigma=0.20, rebalance_band=0.05,
    )


# ---------------------------------------------------------------------------
# simulate_delta_hedge
# ---------------------------------------------------------------------------

class TestSimulateDeltaHedge:
    def test_returns_dataframe(self, spot_series):
        result = simulate_delta_hedge(spot_series, K=100, T=0.25, r=0.05, sigma=0.20)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, spot_series):
        result = simulate_delta_hedge(spot_series, K=100, T=0.25, r=0.05, sigma=0.20)
        expected = {
            "date", "spot", "delta", "shares_held", "option_value",
            "share_pnl_cumulative", "option_pnl_cumulative",
            "transaction_costs_cumulative", "net_pnl", "rebalance_triggered",
        }
        assert set(result.columns) == expected

    def test_length_matches_input(self, spot_series):
        result = simulate_delta_hedge(spot_series, K=100, T=0.25, r=0.05, sigma=0.20)
        assert len(result) == len(spot_series)

    def test_delta_in_range(self, spot_series):
        result = simulate_delta_hedge(
            spot_series, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call",
        )
        assert (result["delta"] >= 0).all()
        assert (result["delta"] <= 1).all()

    def test_put_delta_negative(self, spot_series):
        result = simulate_delta_hedge(
            spot_series, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put",
        )
        assert (result["delta"] <= 0).all()
        assert (result["delta"] >= -1).all()

    def test_transaction_costs_nondecreasing(self, spot_series):
        result = simulate_delta_hedge(spot_series, K=100, T=0.25, r=0.05, sigma=0.20)
        costs = result["transaction_costs_cumulative"].values
        assert np.all(np.diff(costs) >= -1e-10)

    def test_too_few_days_raises(self):
        short = pd.Series([100, 101, 102], index=pd.bdate_range("2024-01-01", periods=3))
        with pytest.raises(ValueError, match="Need >= 5 days"):
            simulate_delta_hedge(short, K=100, T=0.25, r=0.05, sigma=0.20)

    def test_tight_band_rebalances_often(self, spot_series):
        result = simulate_delta_hedge(
            spot_series, K=100, T=0.25, r=0.05, sigma=0.20, rebalance_band=0.001,
        )
        assert result["rebalance_triggered"].sum() > 20

    def test_wide_band_rebalances_rarely(self, spot_series):
        result = simulate_delta_hedge(
            spot_series, K=100, T=0.25, r=0.05, sigma=0.20, rebalance_band=0.50,
        )
        assert result["rebalance_triggered"].sum() < 10

    def test_no_nan_in_output(self, spot_series):
        result = simulate_delta_hedge(spot_series, K=100, T=0.25, r=0.05, sigma=0.20)
        assert result.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# hedging_summary
# ---------------------------------------------------------------------------

class TestHedgingSummary:
    def test_returns_dict_keys(self, sim_df):
        summary = hedging_summary(sim_df)
        expected = {
            "final_pnl", "total_transaction_costs", "n_rebalances",
            "rebalance_pct", "avg_abs_delta", "pnl_sharpe", "profitable",
            "txn_cost_ratio",
        }
        assert set(summary.keys()) == expected

    def test_pnl_is_float(self, sim_df):
        summary = hedging_summary(sim_df)
        assert isinstance(summary["final_pnl"], float)

    def test_n_rebalances_nonneg(self, sim_df):
        summary = hedging_summary(sim_df)
        assert summary["n_rebalances"] >= 0

    def test_rebalance_pct_range(self, sim_df):
        summary = hedging_summary(sim_df)
        assert 0 <= summary["rebalance_pct"] <= 100

    def test_costs_nonneg(self, sim_df):
        summary = hedging_summary(sim_df)
        assert summary["total_transaction_costs"] >= 0


# ---------------------------------------------------------------------------
# optimal_band_search
# ---------------------------------------------------------------------------

class TestOptimalBandSearch:
    def test_returns_dataframe(self, spot_series):
        result = optimal_band_search(spot_series, K=100, T=0.25, r=0.05, sigma=0.20)
        assert isinstance(result, pd.DataFrame)

    def test_default_bands(self, spot_series):
        result = optimal_band_search(spot_series, K=100, T=0.25, r=0.05, sigma=0.20)
        assert len(result) == 5  # default: [0.01, 0.02, 0.05, 0.1, 0.2]

    def test_custom_bands(self, spot_series):
        result = optimal_band_search(
            spot_series, K=100, T=0.25, r=0.05, sigma=0.20,
            bands=[0.03, 0.06],
        )
        assert len(result) == 2

    def test_columns(self, spot_series):
        result = optimal_band_search(spot_series, K=100, T=0.25, r=0.05, sigma=0.20)
        assert set(result.columns) == {"band", "net_pnl", "n_rebalances", "total_costs", "sharpe"}

    def test_costs_increase_with_tighter_bands(self, spot_series):
        result = optimal_band_search(
            spot_series, K=100, T=0.25, r=0.05, sigma=0.20,
            bands=[0.01, 0.20],
        )
        tight = result[result["band"] == 0.01]["total_costs"].values[0]
        wide = result[result["band"] == 0.20]["total_costs"].values[0]
        assert tight >= wide
