"""Tab 4: Hedging Simulator — delta-neutral hedging with rebalance bands."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.pricing.hedging import simulate_delta_hedge, hedging_summary, optimal_band_search


def render(prices_df: pd.DataFrame) -> None:
    """Render hedging simulator tab with interactive controls and charts."""

    # --- Input Controls ---
    st.subheader("Hedging Parameters")

    tickers = [c for c in prices_df.columns if not c.startswith("^")]
    if not tickers:
        st.error("No equity tickers available for hedging.")
        return

    c1, c2, c3 = st.columns(3)
    ticker = c1.selectbox("Underlying", tickers, index=0)
    option_type = c2.radio("Option Type", ["call", "put"], horizontal=True, key="hedge_option_type")
    position = c3.radio("Position", ["Short (-1)", "Long (+1)"], horizontal=True, key="hedge_position")
    option_position = -1 if "Short" in position else 1

    c4, c5, c6, c7 = st.columns(4)
    spot = prices_df[ticker].dropna()
    last_spot = float(spot.iloc[-1])
    K = c4.number_input("Strike (K)", value=round(last_spot, 2), min_value=0.01, step=1.0, format="%.2f", key="hedge_K")
    T = c5.slider("Expiry (T years)", min_value=0.05, max_value=1.0, value=0.25, step=0.05, key="hedge_T")
    r = c6.slider("Risk-Free Rate", min_value=0.0, max_value=0.15, value=0.05, step=0.005, format="%.3f", key="hedge_r")
    sigma = c7.slider("Volatility", min_value=0.05, max_value=1.0, value=0.20, step=0.01, format="%.2f", key="hedge_sigma")

    c8, c9 = st.columns(2)
    txn_bps = c8.slider("Transaction Cost (bps)", min_value=1, max_value=20, value=5, key="hedge_txn")
    band = c9.slider("Rebalance Band", min_value=0.01, max_value=0.30, value=0.05, step=0.01, format="%.2f", key="hedge_band")

    # Limit spot to ~T years of data
    n_days = max(int(T * 252), 10)
    spot_window = spot.iloc[-n_days:] if len(spot) > n_days else spot

    st.divider()

    # --- Run Simulation ---
    if st.button("Run Simulation", type="primary"):
        if len(spot_window) < 5:
            st.error("Not enough price data for simulation.")
            return

        with st.spinner("Running delta-neutral hedging simulation..."):
            sim_df = simulate_delta_hedge(
                spot_window, K=K, T=T, r=r, sigma=sigma,
                option_type=option_type, option_position=option_position,
                transaction_cost_bps=float(txn_bps), rebalance_band=band,
            )
            summary = hedging_summary(sim_df)

        # --- Summary Metrics ---
        st.subheader("Simulation Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Net P&L", f"${summary['final_pnl']:.2f}",
                  delta="Profit" if summary["profitable"] else "Loss",
                  delta_color="normal" if summary["profitable"] else "inverse")
        m2.metric("Transaction Costs", f"${summary['total_transaction_costs']:.2f}")
        m3.metric("Rebalances", f"{summary['n_rebalances']} ({summary['rebalance_pct']:.0f}%)")
        m4.metric("Avg |Delta|", f"{summary['avg_abs_delta']:.3f}")
        sharpe_str = f"{summary['pnl_sharpe']:.1f}" if np.isfinite(summary["pnl_sharpe"]) else "N/A"
        m5.metric("PnL Sharpe", sharpe_str)

        if np.isfinite(summary["txn_cost_ratio"]) and summary["txn_cost_ratio"] > 0.5:
            st.warning(f"Transaction costs are {summary['txn_cost_ratio']:.0%} of gross profit — consider wider bands.")

        st.divider()

        # --- Charts ---
        col_a, col_b = st.columns(2)

        with col_a:
            # Spot Price + Rebalance Events
            st.subheader("Spot Price & Rebalances")
            rebal_mask = sim_df["rebalance_triggered"]
            fig_spot = go.Figure()
            fig_spot.add_trace(go.Scatter(
                x=sim_df["date"], y=sim_df["spot"].astype(np.float32),
                mode="lines", name="Spot",
            ))
            if rebal_mask.any():
                fig_spot.add_trace(go.Scatter(
                    x=sim_df.loc[rebal_mask, "date"],
                    y=sim_df.loc[rebal_mask, "spot"].astype(np.float32),
                    mode="markers", name="Rebalance",
                    marker=dict(size=6, color="orange", symbol="triangle-up"),
                ))
            fig_spot.add_hline(y=K, line_dash="dot", line_color="red", annotation_text=f"K={K:.0f}")
            fig_spot.update_layout(template="plotly_dark", height=350, yaxis_title="Price")
            st.plotly_chart(fig_spot, use_container_width=True)

        with col_b:
            # Delta Over Time
            st.subheader("Portfolio Delta")
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=sim_df["date"], y=sim_df["delta"].astype(np.float32),
                mode="lines", name="Delta",
            ))
            fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_delta.update_layout(template="plotly_dark", height=350, yaxis_title="Delta")
            st.plotly_chart(fig_delta, use_container_width=True)

        # Cumulative P&L Breakdown
        st.subheader("Cumulative P&L Breakdown")
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=sim_df["date"], y=sim_df["share_pnl_cumulative"].astype(np.float32),
            mode="lines", name="Share P&L", stackgroup="pnl",
        ))
        fig_pnl.add_trace(go.Scatter(
            x=sim_df["date"], y=sim_df["option_pnl_cumulative"].astype(np.float32),
            mode="lines", name="Option P&L", stackgroup="pnl",
        ))
        fig_pnl.add_trace(go.Scatter(
            x=sim_df["date"], y=(-sim_df["transaction_costs_cumulative"]).astype(np.float32),
            mode="lines", name="Costs (neg)", stackgroup="pnl",
        ))
        fig_pnl.add_trace(go.Scatter(
            x=sim_df["date"], y=sim_df["net_pnl"].astype(np.float32),
            mode="lines", name="Net P&L", line=dict(width=3, color="white"),
        ))
        fig_pnl.update_layout(template="plotly_dark", height=400, yaxis_title="Cumulative P&L ($)")
        st.plotly_chart(fig_pnl, use_container_width=True)

        # --- Optimal Band Search ---
        with st.expander("Optimal Band Search"):
            with st.spinner("Searching optimal rebalance band..."):
                band_df = optimal_band_search(
                    spot_window, K=K, T=T, r=r, sigma=sigma,
                    option_type=option_type, option_position=option_position,
                )

            st.dataframe(band_df.style.format({
                "band": "{:.2f}", "net_pnl": "${:.2f}",
                "total_costs": "${:.2f}", "sharpe": "{:.1f}",
            }), use_container_width=True)

            fig_band = make_subplots(specs=[[{"secondary_y": True}]])
            fig_band.add_trace(go.Bar(
                x=band_df["band"].astype(str), y=band_df["net_pnl"],
                name="Net P&L", marker_color="steelblue",
            ), secondary_y=False)
            fig_band.add_trace(go.Scatter(
                x=band_df["band"].astype(str), y=band_df["n_rebalances"],
                mode="lines+markers", name="Rebalances", marker_color="orange",
            ), secondary_y=True)
            fig_band.update_layout(template="plotly_dark", height=350,
                                   xaxis_title="Rebalance Band")
            fig_band.update_yaxes(title_text="Net P&L ($)", secondary_y=False)
            fig_band.update_yaxes(title_text="Rebalances", secondary_y=True)
            st.plotly_chart(fig_band, use_container_width=True)
