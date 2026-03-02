"""Tests for Black-Scholes pricing, Greeks, and implied volatility."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.pricing.black_scholes import price, greeks, implied_volatility, vol_surface


# ---------------------------------------------------------------------------
# Black-Scholes price
# ---------------------------------------------------------------------------

class TestBSPrice:
    def test_call_hull_textbook(self):
        """Hull Example: S=42, K=40, T=0.5, r=0.1, sigma=0.2 -> call ≈ 4.76."""
        c = price(42, 40, 0.5, 0.1, 0.2, "call")
        assert_allclose(c, 4.76, atol=0.02)

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        c = price(S, K, T, r, sigma, "call")
        p = price(S, K, T, r, sigma, "put")
        parity = S - K * np.exp(-r * T)
        assert_allclose(c - p, parity, atol=1e-6)

    def test_expiry_returns_intrinsic_call(self):
        """T=0 call returns max(S-K, 0)."""
        assert price(105, 100, 0, 0.05, 0.2, "call") == 5.0

    def test_expiry_returns_intrinsic_put(self):
        """T=0 put returns max(K-S, 0)."""
        assert price(95, 100, 0, 0.05, 0.2, "put") == 5.0

    def test_expiry_otm_call(self):
        """T=0, OTM call returns 0."""
        assert price(90, 100, 0, 0.05, 0.2, "call") == 0.0

    def test_vectorized_shapes(self):
        """Arrays of S and K produce matching shapes."""
        S = np.array([100, 105, 110])
        K = np.array([100, 100, 100])
        result = price(S, K, 1.0, 0.05, 0.2, "call")
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K*exp(-rT)."""
        S, K, T, r, sigma = 200, 100, 1.0, 0.05, 0.2
        c = price(S, K, T, r, sigma, "call")
        lower_bound = S - K * np.exp(-r * T)
        assert c >= lower_bound - 1e-6

    def test_put_positive(self):
        """Put price should be positive for OTM."""
        p = price(90, 100, 1.0, 0.05, 0.3, "put")
        assert p > 0

    def test_negative_S_raises(self):
        with pytest.raises(ValueError, match="Spot price"):
            price(-100, 100, 1.0, 0.05, 0.2)

    def test_negative_K_raises(self):
        with pytest.raises(ValueError, match="Strike price"):
            price(100, -100, 1.0, 0.05, 0.2)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="Volatility"):
            price(100, 100, 1.0, 0.05, -0.2)

    def test_call_geq_put(self):
        """For same params, call >= put when S >= K*exp(-rT)."""
        S, K, T, r, sigma = 100, 95, 0.5, 0.05, 0.2
        c = price(S, K, T, r, sigma, "call")
        p = price(S, K, T, r, sigma, "put")
        assert c >= p


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

class TestGreeks:
    def test_delta_call_bounds(self):
        """0 <= call_delta <= 1."""
        g = greeks(100, 100, 1.0, 0.05, 0.2)
        assert 0 <= g["delta_call"] <= 1

    def test_delta_put_bounds(self):
        """-1 <= put_delta <= 0."""
        g = greeks(100, 100, 1.0, 0.05, 0.2)
        assert -1 <= g["delta_put"] <= 0

    def test_atm_delta_near_half(self):
        """ATM call delta ≈ 0.5 for long T."""
        g = greeks(100, 100, 5.0, 0.0, 0.2)
        assert_allclose(g["delta_call"], 0.5, atol=0.1)

    def test_gamma_positive(self):
        """Gamma is always positive."""
        g = greeks(100, 100, 1.0, 0.05, 0.2)
        assert g["gamma"] > 0

    def test_vega_positive(self):
        """Vega is always positive."""
        g = greeks(100, 100, 1.0, 0.05, 0.2)
        assert g["vega"] > 0

    def test_theta_call_negative(self):
        """Call theta typically negative (time decay)."""
        g = greeks(100, 100, 1.0, 0.05, 0.2)
        assert g["theta_call"] < 0

    def test_near_expiry_delta(self):
        """Near-expiry ITM call delta = 1, OTM = 0."""
        g_itm = greeks(105, 100, 1e-5, 0.05, 0.2)
        g_otm = greeks(95, 100, 1e-5, 0.05, 0.2)
        assert g_itm["delta_call"] == 1.0
        assert g_otm["delta_call"] == 0.0

    def test_near_expiry_gamma_zero(self):
        """Near-expiry gamma = 0."""
        g = greeks(100, 100, 1e-5, 0.05, 0.2)
        assert g["gamma"] == 0.0

    def test_vectorized_greeks(self):
        """Arrays produce matching shapes."""
        S = np.array([90, 100, 110])
        g = greeks(S, 100, 1.0, 0.05, 0.2)
        assert g["delta_call"].shape == (3,)
        assert g["gamma"].shape == (3,)

    def test_put_call_delta_relation(self):
        """delta_put = delta_call - 1."""
        g = greeks(100, 100, 1.0, 0.05, 0.2)
        assert_allclose(g["delta_put"], g["delta_call"] - 1, atol=1e-10)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

class TestImpliedVol:
    def test_roundtrip(self):
        """IV of a BS-priced option recovers the original sigma."""
        sigma_true = 0.3
        c = price(100, 100, 1.0, 0.05, sigma_true, "call")
        iv = implied_volatility(float(c), 100, 100, 1.0, 0.05, "call")
        assert_allclose(iv, sigma_true, atol=1e-4)

    def test_roundtrip_put(self):
        """IV roundtrip for put."""
        sigma_true = 0.25
        p = price(100, 110, 0.5, 0.05, sigma_true, "put")
        iv = implied_volatility(float(p), 100, 110, 0.5, 0.05, "put")
        assert_allclose(iv, sigma_true, atol=1e-4)

    def test_high_vol_roundtrip(self):
        """IV works for high volatility."""
        sigma_true = 1.5
        c = price(100, 100, 1.0, 0.05, sigma_true, "call")
        iv = implied_volatility(float(c), 100, 100, 1.0, 0.05, "call")
        assert_allclose(iv, sigma_true, atol=1e-3)

    def test_expired_returns_nan(self):
        """T=0 -> nan."""
        assert np.isnan(implied_volatility(5.0, 100, 100, 0, 0.05, "call"))

    def test_zero_price_returns_nan(self):
        """Market price 0 -> nan."""
        assert np.isnan(implied_volatility(0, 100, 100, 1.0, 0.05, "call"))


# ---------------------------------------------------------------------------
# Vol surface
# ---------------------------------------------------------------------------

class TestVolSurface:
    def test_surface_shape(self):
        """Surface has correct dimensions."""
        strikes = np.array([90, 100, 110])
        expiries = np.array([0.25, 0.5, 1.0])
        # Generate prices from known sigma=0.2
        prices_grid = np.array([
            [price(100, k, t, 0.05, 0.2, "call") for t in expiries]
            for k in strikes
        ])
        surf = vol_surface(100, strikes, expiries, 0.05, prices_grid, "call")
        assert surf.shape == (3, 3)
        # All IVs should recover ~0.2
        assert_allclose(surf.values, 0.2, atol=1e-3)

    def test_surface_nan_on_bad_price(self):
        """Unreasonable prices -> nan in surface."""
        strikes = np.array([100])
        expiries = np.array([1.0])
        bad_prices = np.array([[0.0]])  # price=0 -> no IV
        surf = vol_surface(100, strikes, expiries, 0.05, bad_prices, "call")
        assert np.isnan(surf.values[0, 0])
