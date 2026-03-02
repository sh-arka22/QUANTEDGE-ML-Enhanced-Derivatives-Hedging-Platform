"""Black-Scholes-Merton pricing, Greeks, implied volatility, and vol surface."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def _validate_inputs(S: np.ndarray, K: np.ndarray, sigma: np.ndarray) -> None:
    """Validate that S, K > 0 and sigma > 0."""
    if np.any(np.asarray(S) <= 0):
        raise ValueError("Spot price S must be positive")
    if np.any(np.asarray(K) <= 0):
        raise ValueError("Strike price K must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("Volatility sigma must be positive")


def _d1d2(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d1 and d2, clamped to [-50, 50] to prevent overflow."""
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    d1 = np.clip(d1, -50, 50)
    d2 = np.clip(d2, -50, 50)
    return d1, d2


def price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str = "call",
) -> np.ndarray:
    """Black-Scholes option price, fully vectorized."""
    S, K, T, r, sigma = (np.asarray(x, dtype=np.float64) for x in (S, K, T, r, sigma))
    _validate_inputs(S, K, sigma)

    # Intrinsic value for T <= 0
    intrinsic = np.maximum(S - K, 0) if option_type == "call" else np.maximum(K - S, 0)
    expired = T <= 0

    # BS formula where T > 0
    d1, d2 = _d1d2(S, K, np.where(expired, 1.0, T), r, sigma)
    discount = K * np.exp(-r * T)

    if option_type == "call":
        bs = S * norm.cdf(d1) - discount * norm.cdf(d2)
    else:
        bs = discount * norm.cdf(-d2) - S * norm.cdf(-d1)

    result = np.where(expired, intrinsic, bs)
    return float(result) if result.ndim == 0 else result


def greeks(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
) -> dict:
    """All Black-Scholes Greeks, fully vectorized."""
    S, K, T, r, sigma = (np.asarray(x, dtype=np.float64) for x in (S, K, T, r, sigma))
    _validate_inputs(S, K, sigma)

    near_expiry = T < 1 / 365
    T_safe = np.where(near_expiry, 1.0, T)

    d1, d2 = _d1d2(S, K, T_safe, r, sigma)
    sqrt_T = np.sqrt(T_safe)
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    discount = K * np.exp(-r * T_safe)

    # Delta
    delta_call = np.where(near_expiry, np.where(S > K, 1.0, 0.0), N_d1)
    delta_put = np.where(near_expiry, np.where(S < K, -1.0, 0.0), N_d1 - 1)

    # Gamma
    gamma = np.where(near_expiry, 0.0, n_d1 / (S * sigma * sqrt_T))

    # Vega (per 1% vol move)
    vega = np.where(near_expiry, 0.0, S * n_d1 * sqrt_T / 100)

    # Theta (per calendar day)
    theta_common = -(S * n_d1 * sigma) / (2 * sqrt_T)
    theta_call = np.where(
        near_expiry, 0.0, (theta_common - r * discount * N_d2) / 365
    )
    theta_put = np.where(
        near_expiry, 0.0, (theta_common + r * discount * norm.cdf(-d2)) / 365
    )

    # Rho (per 1% rate move)
    rho_call = np.where(near_expiry, 0.0, discount * T_safe * N_d2 / 100)
    rho_put = np.where(near_expiry, 0.0, -discount * T_safe * norm.cdf(-d2) / 100)

    return {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "vega": vega,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "rho_call": rho_call,
        "rho_put": rho_put,
    }


def implied_volatility(
    price_market: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson IV solve with bisection fallback."""
    if T <= 0:
        return np.nan
    if price_market <= 0:
        return np.nan

    S, K, T, r = float(S), float(K), float(T), float(r)

    # Newton-Raphson
    sigma = 0.2
    for _ in range(max_iter):
        bs_price = float(price(S, K, T, r, sigma, option_type))
        diff = bs_price - price_market
        if abs(diff) < tol:
            return sigma
        vega_val = float(greeks(S, K, T, r, sigma)["vega"]) * 100  # undo /100 scaling
        if abs(vega_val) < 1e-12:
            break  # fall through to bisection
        sigma -= diff / vega_val
        if sigma <= 0:
            break

    # Bisection fallback
    lo, hi = 0.001, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        bs_price = float(price(S, K, T, r, mid, option_type))
        if abs(bs_price - price_market) < tol:
            return mid
        if bs_price > price_market:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return mid

    return np.nan


def vol_surface(
    S: float,
    strikes: np.ndarray,
    expiries: np.ndarray,
    r: float,
    option_prices: np.ndarray,
    option_type: str = "call",
) -> pd.DataFrame:
    """Implied vol surface from a grid of market prices."""
    strikes = np.asarray(strikes, dtype=np.float64)
    expiries = np.asarray(expiries, dtype=np.float64)
    option_prices = np.asarray(option_prices, dtype=np.float64)

    surface = np.full((len(strikes), len(expiries)), np.nan)
    for i, k in enumerate(strikes):
        for j, t in enumerate(expiries):
            surface[i, j] = implied_volatility(
                float(option_prices[i, j]), S, k, t, r, option_type
            )

    return pd.DataFrame(surface, index=strikes, columns=expiries)
