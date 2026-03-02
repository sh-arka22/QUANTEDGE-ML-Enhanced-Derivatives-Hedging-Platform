"""Tab 1: Portfolio Analytics — risk metrics, cumulative returns, VaR, correlation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.portfolio import risk_summary, weighted_returns
from src.data import config as cfg


def render(returns_df: pd.DataFrame, market_returns, risk_free_rate: float) -> None:
    """Render portfolio analytics tab."""
    mkt = np.asarray(market_returns, dtype=np.float64)

    with st.spinner("Computing portfolio risk metrics..."):
        summary = risk_summary(returns_df, cfg.PORTFOLIO_WEIGHTS, mkt, risk_free_rate)

    # --- Key Metrics ---
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe", f"{summary['sharpe']:.2f}")
    c2.metric("Sortino", f"{summary['sortino']:.2f}")
    c3.metric("Beta", f"{summary['beta']:.2f}")
    c4.metric("Alpha (ann.)", f"{summary['alpha'] * cfg.TRADING_DAYS:.2%}")
    c5.metric("Max Drawdown", f"{summary['max_drawdown']:.2%}")

    # --- VaR ---
    st.subheader("Value at Risk (95%)")
    v1, v2, v3 = st.columns(3)
    v1.metric("Parametric VaR", f"{summary['var_parametric']:.4f}")
    v2.metric("Historical VaR", f"{summary['var_historical']:.4f}")
    v3.metric("CVaR (ES)", f"{summary['cvar']:.4f}")

    if summary["kurtosis_risk"]:
        st.warning("Fat tail risk detected: parametric VaR may underestimate true risk.")

    # --- Cumulative Returns Chart ---
    st.subheader("Cumulative Returns")
    port_ret = weighted_returns(returns_df, cfg.PORTFOLIO_WEIGHTS)
    cum_port = np.cumprod(1 + port_ret)

    fig_cum = go.Figure()
    for col in returns_df.columns:
        cum_col = np.cumprod(1 + returns_df[col].values)
        fig_cum.add_trace(go.Scatter(
            x=returns_df.index, y=cum_col.astype(np.float32),
            name=col, mode="lines",
        ))
    fig_cum.add_trace(go.Scatter(
        x=returns_df.index, y=cum_port.astype(np.float32),
        name="Portfolio", mode="lines", line=dict(width=3, color="white"),
    ))
    fig_cum.update_layout(template="plotly_dark", height=400, yaxis_title="Growth of $1")
    st.plotly_chart(fig_cum, use_container_width=True)

    # --- Return Distribution ---
    st.subheader("Return Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=port_ret, nbinsx=80, name="Portfolio Returns"))
    fig_hist.add_vline(x=summary["var_parametric"], line_dash="dash", line_color="red",
                       annotation_text="Parametric VaR")
    fig_hist.add_vline(x=summary["var_historical"], line_dash="dot", line_color="orange",
                       annotation_text="Historical VaR")
    fig_hist.update_layout(template="plotly_dark", height=350, xaxis_title="Daily Return")
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Rolling Volatility ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Rolling 30-Day Volatility")
        port_series = pd.Series(port_ret, index=returns_df.index)
        rolling_vol = port_series.rolling(30).std() * np.sqrt(cfg.TRADING_DAYS)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol.values.astype(np.float32),
            mode="lines", name="Portfolio Vol",
        ))
        fig_vol.update_layout(template="plotly_dark", height=350, yaxis_title="Annualized Vol")
        st.plotly_chart(fig_vol, use_container_width=True)

    # --- Correlation Heatmap ---
    with col_b:
        st.subheader("Correlation Matrix")
        corr = returns_df.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
        ))
        fig_corr.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- Full Summary Table ---
    with st.expander("Full Risk Summary"):
        display = {k: v for k, v in summary.items()
                   if k not in ("portfolio_returns", "cumulative_returns", "kurtosis_risk")}
        st.dataframe(pd.DataFrame(display, index=["Value"]).T.rename(columns={"Value": "Metric"}))
