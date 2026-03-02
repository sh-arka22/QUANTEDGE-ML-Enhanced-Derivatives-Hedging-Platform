"""Vectorized CRR binomial tree: European/American pricing and Greeks."""

import warnings

import numpy as np

from src.data import config as cfg


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 200,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """CRR binomial tree price with vectorized backward induction."""
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return float(intrinsic)
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if N < 1:
        raise ValueError("N must be >= 1")
    if N > cfg.MAX_TREE_STEPS:
        warnings.warn(f"N={N} exceeds max {cfg.MAX_TREE_STEPS}, capping")
        N = cfg.MAX_TREE_STEPS

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    if not 0 < p < 1:
        raise ValueError(f"Risk-neutral probability p={p:.4f} outside (0,1)")

    disc = np.exp(-r * dt)

    # Terminal asset prices: S * u^(N-2j) for j = 0..N
    j = np.arange(N + 1)
    S_T = S * u ** (N - 2 * j)

    # Terminal option values
    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # Backward induction (single loop, vectorized within each step)
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            j_step = np.arange(i + 1)
            S_step = S * u ** (i - 2 * j_step)
            if option_type == "call":
                exercise = np.maximum(S_step - K, 0.0)
            else:
                exercise = np.maximum(K - S_step, 0.0)
            V = np.maximum(V, exercise)

    return float(V[0])


def crr_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 200,
    option_type: str = "call",
    american: bool = False,
) -> dict:
    """Binomial tree Greeks: delta, gamma, theta."""
    if T <= 0:
        return {"delta": np.nan, "gamma": np.nan, "theta": np.nan}
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive")
    if N < 3:
        N = max(N, 3)
    if N > cfg.MAX_TREE_STEPS:
        N = cfg.MAX_TREE_STEPS

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Build full tree prices at steps 0, 1, 2
    # Step 0: V[0] = f_0
    # Step 1: f_u (j=0), f_d (j=1)
    # Step 2: f_uu (j=0), f_ud (j=1), f_dd (j=2)

    # Terminal values
    j = np.arange(N + 1)
    S_T = S * u ** (N - 2 * j)
    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # Store values at steps N, N-1, N-2 as we roll back
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            j_step = np.arange(i + 1)
            S_step = S * u ** (i - 2 * j_step)
            if option_type == "call":
                exercise = np.maximum(S_step - K, 0.0)
            else:
                exercise = np.maximum(K - S_step, 0.0)
            V = np.maximum(V, exercise)
        if i == 2:
            V2 = V.copy()  # f_uu, f_ud, f_dd
        elif i == 1:
            V1 = V.copy()  # f_u, f_d

    f_0 = float(V[0])
    f_u, f_d = float(V1[0]), float(V1[1])
    f_uu, f_ud, f_dd = float(V2[0]), float(V2[1]), float(V2[2])

    S_u = S * u
    S_d = S * d
    S_uu = S * u * u
    S_dd = S * d * d

    # Delta
    delta = (f_u - f_d) / (S_u - S_d)

    # Gamma
    gamma_num = (f_uu - f_ud) / (S_uu - S) - (f_ud - f_dd) / (S - S_dd)
    gamma = gamma_num / (0.5 * (S_uu - S_dd))

    # Theta (per calendar day)
    theta = (f_ud - f_0) / (2 * dt) / 365

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
    }


def american_premium(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 200,
    option_type: str = "put",
) -> float:
    """Early exercise premium: American price minus European price."""
    am = crr_price(S, K, T, r, sigma, N, option_type, american=True)
    eu = crr_price(S, K, T, r, sigma, N, option_type, american=False)
    return float(am - eu)
