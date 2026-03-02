"""Delta-neutral hedging simulation with gamma scalping."""

import warnings

import numpy as np
import pandas as pd

from src.data import config as cfg
from src.pricing.black_scholes import price as bs_price, greeks as bs_greeks


def simulate_delta_hedge(
    spot_prices: pd.Series,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    option_position: int = -1,
    shares_per_contract: int = 100,
    transaction_cost_bps: float = 5.0,
    rebalance_band: float = 0.05,
) -> pd.DataFrame:
    """Simulate daily delta-neutral hedging over historical spot prices."""
    spots = spot_prices.dropna()
    if len(spots) < 5:
        raise ValueError(f"Need >= 5 days of spot prices, got {len(spots)}")

    n_days = len(spots)
    spot_arr = spots.values.astype(np.float64)
    dates = spots.index

    # Pre-allocate output arrays
    delta_arr = np.zeros(n_days)
    shares_held = np.zeros(n_days)
    option_val = np.zeros(n_days)
    share_pnl_cum = np.zeros(n_days)
    option_pnl_cum = np.zeros(n_days)
    txn_cost_cum = np.zeros(n_days)
    rebalanced = np.zeros(n_days, dtype=bool)

    # Day 0: initialize
    T_rem = max(T, 1e-6)
    g = bs_greeks(spot_arr[0], K, T_rem, r, sigma)
    delta_key = "delta_call" if option_type == "call" else "delta_put"
    d0 = float(g[delta_key])
    delta_arr[0] = d0

    # Initial hedge: neutralize the option delta
    target_shares = -option_position * d0 * shares_per_contract
    shares_held[0] = target_shares
    option_val[0] = float(bs_price(spot_arr[0], K, T_rem, r, sigma, option_type))

    # Initial transaction cost
    txn = abs(target_shares) * spot_arr[0] * transaction_cost_bps / 10_000
    txn_cost_cum[0] = txn

    # Daily simulation
    for day in range(1, n_days):
        T_rem = max(T - day / cfg.TRADING_DAYS, 1e-6)
        S = spot_arr[day]

        # Option value
        opt_val = float(bs_price(S, K, T_rem, r, sigma, option_type))
        option_val[day] = opt_val

        # New delta
        g = bs_greeks(S, K, T_rem, r, sigma)
        new_delta = float(g[delta_key])
        delta_arr[day] = new_delta

        # Net delta exposure
        option_delta_shares = option_position * new_delta * shares_per_contract
        net_delta = option_delta_shares + shares_held[day - 1]

        # Check rebalance band
        if abs(net_delta) > rebalance_band * shares_per_contract:
            # Trade to neutralize
            trade = -net_delta
            shares_held[day] = shares_held[day - 1] + trade
            txn = abs(trade) * S * transaction_cost_bps / 10_000
            txn_cost_cum[day] = txn_cost_cum[day - 1] + txn
            rebalanced[day] = True
        else:
            shares_held[day] = shares_held[day - 1]
            txn_cost_cum[day] = txn_cost_cum[day - 1]

        # Cumulative P&L
        share_pnl_cum[day] = shares_held[day] * (S - spot_arr[0])
        # More accurate: track incremental share PnL
        # share_pnl = sum of shares_held[i] * (S[i+1] - S[i])
        option_pnl_cum[day] = option_position * shares_per_contract * (opt_val - option_val[0])

    # Recompute share PnL more accurately: incremental
    share_pnl_incremental = np.zeros(n_days)
    for day in range(1, n_days):
        share_pnl_incremental[day] = (
            share_pnl_incremental[day - 1]
            + shares_held[day - 1] * (spot_arr[day] - spot_arr[day - 1])
        )
    share_pnl_cum = share_pnl_incremental

    net_pnl = share_pnl_cum + option_pnl_cum - txn_cost_cum

    return pd.DataFrame({
        "date": dates,
        "spot": spot_arr,
        "delta": delta_arr,
        "shares_held": shares_held,
        "option_value": option_val,
        "share_pnl_cumulative": share_pnl_cum,
        "option_pnl_cumulative": option_pnl_cum,
        "transaction_costs_cumulative": txn_cost_cum,
        "net_pnl": net_pnl,
        "rebalance_triggered": rebalanced,
    })


def hedging_summary(sim_df: pd.DataFrame) -> dict:
    """Summary statistics from a hedging simulation."""
    n_rebalances = int(sim_df["rebalance_triggered"].sum())
    total_days = len(sim_df)
    total_txn = float(sim_df["transaction_costs_cumulative"].iloc[-1])
    final_pnl = float(sim_df["net_pnl"].iloc[-1])

    # Delta stats
    net_delta_arr = (
        sim_df["delta"] * sim_df["shares_held"].iloc[0]  # rough proxy
    )
    # Better: compute net delta at each step
    avg_abs_delta = float(np.abs(sim_df["delta"]).mean())

    # Daily P&L changes
    daily_pnl = sim_df["net_pnl"].diff().dropna()
    pnl_std = daily_pnl.std()
    pnl_sharpe = (
        float(daily_pnl.mean() / pnl_std * np.sqrt(cfg.TRADING_DAYS))
        if pnl_std > 1e-10 else np.nan
    )

    if n_rebalances == total_days - 1 and total_days > 10:
        warnings.warn("Band too tight: rebalanced every day")
    if n_rebalances == 0 and total_days > 10:
        warnings.warn("Band too wide: zero rebalances")

    return {
        "final_pnl": final_pnl,
        "total_transaction_costs": total_txn,
        "n_rebalances": n_rebalances,
        "rebalance_pct": n_rebalances / max(total_days - 1, 1) * 100,
        "avg_abs_delta": avg_abs_delta,
        "pnl_sharpe": pnl_sharpe,
        "profitable": final_pnl > 0,
        "txn_cost_ratio": total_txn / abs(final_pnl) if abs(final_pnl) > 1e-10 else np.nan,
    }


def optimal_band_search(
    spot_prices: pd.Series,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    option_position: int = -1,
    bands: list[float] | None = None,
) -> pd.DataFrame:
    """Search for optimal rebalance band by running multiple simulations."""
    if bands is None:
        bands = [0.01, 0.02, 0.05, 0.1, 0.2]

    records = []
    for band in bands:
        sim = simulate_delta_hedge(
            spot_prices, K, T, r, sigma,
            option_type=option_type,
            option_position=option_position,
            rebalance_band=band,
        )
        summary = hedging_summary(sim)
        records.append({
            "band": band,
            "net_pnl": summary["final_pnl"],
            "n_rebalances": summary["n_rebalances"],
            "total_costs": summary["total_transaction_costs"],
            "sharpe": summary["pnl_sharpe"],
        })

    return pd.DataFrame(records)
