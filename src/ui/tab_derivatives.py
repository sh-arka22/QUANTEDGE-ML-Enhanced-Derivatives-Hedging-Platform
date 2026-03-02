"""Tab 3: Derivatives Pricing — Black-Scholes, binomial tree, Greeks, vol surface."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.pricing.black_scholes import price as bs_price, greeks as bs_greeks
from src.pricing.binomial import crr_price, crr_greeks, american_premium


def render() -> None:
    """Render derivatives pricing tab with interactive controls and charts."""

    # --- Input Controls ---
    st.subheader("Option Parameters")
    c1, c2, c3, c4, c5 = st.columns(5)
    S = c1.number_input("Spot (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K = c2.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T = c3.slider("Expiry (T years)", min_value=0.0, max_value=3.0, value=1.0, step=0.01)
    r = c4.slider("Risk-Free Rate", min_value=0.0, max_value=0.15, value=0.05, step=0.005, format="%.3f")
    sigma = c5.slider("Volatility", min_value=0.01, max_value=1.0, value=0.20, step=0.01, format="%.2f")

    p1, p2, p3 = st.columns(3)
    option_type = p1.radio("Option Type", ["call", "put"], horizontal=True)
    method = p2.radio("Pricing Method", ["Black-Scholes", "Binomial CRR"], horizontal=True)
    american = False
    N = 200
    if method == "Binomial CRR":
        N = p3.slider("Tree Steps (N)", min_value=10, max_value=1000, value=200, step=10)
        american = p3.checkbox("American Exercise")

    st.divider()

    # --- Pricing ---
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        st.metric("Price (Intrinsic)", f"${intrinsic:.4f}")
        st.info("At expiry — Greeks not applicable.")
        return

    with st.spinner("Pricing..."):
        bs_p = float(bs_price(S, K, T, r, sigma, option_type))
        g = bs_greeks(S, K, T, r, sigma)
        delta_key = "delta_call" if option_type == "call" else "delta_put"
        theta_key = "theta_call" if option_type == "call" else "theta_put"
        rho_key = "rho_call" if option_type == "call" else "rho_put"

        if method == "Binomial CRR":
            crr_p = crr_price(S, K, T, r, sigma, N, option_type, american)
            crr_g = crr_greeks(S, K, T, r, sigma, N, option_type, american)

    # --- Price Display ---
    st.subheader("Price")
    if method == "Black-Scholes":
        st.metric("Black-Scholes Price", f"${bs_p:.4f}")
    else:
        mc1, mc2 = st.columns(2)
        mc1.metric("CRR Binomial Price", f"${crr_p:.4f}")
        mc2.metric("BS Reference", f"${bs_p:.4f}", delta=f"{crr_p - bs_p:+.4f}")
        if american:
            prem = american_premium(S, K, T, r, sigma, N, option_type)
            st.metric("Early Exercise Premium", f"${prem:.4f}")

    # --- Greeks ---
    st.subheader("Greeks")
    g1, g2, g3, g4, g5 = st.columns(5)
    if method == "Black-Scholes":
        g1.metric("Delta", f"{float(g[delta_key]):.4f}")
        g2.metric("Gamma", f"{float(g['gamma']):.6f}")
        g3.metric("Vega", f"{float(g['vega']):.4f}")
        g4.metric("Theta", f"{float(g[theta_key]):.4f}")
        g5.metric("Rho", f"{float(g[rho_key]):.4f}")
    else:
        g1.metric("Delta", f"{crr_g['delta']:.4f}")
        g2.metric("Gamma", f"{crr_g['gamma']:.6f}")
        g3.metric("Theta", f"{crr_g['theta']:.4f}")
        g4.metric("Vega (BS)", f"{float(g['vega']):.4f}")
        g5.metric("Rho (BS)", f"{float(g[rho_key]):.4f}")

    st.divider()

    # --- Charts ---
    col_a, col_b = st.columns(2)

    with col_a:
        # Payoff Diagram
        st.subheader("Payoff at Expiry")
        s_range = np.linspace(max(S * 0.5, 0.01), S * 1.5, 200)
        if option_type == "call":
            payoff = np.maximum(s_range - K, 0)
        else:
            payoff = np.maximum(K - s_range, 0)
        prices_line = np.array(bs_price(s_range, K, T, r, sigma, option_type), dtype=np.float32)

        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(x=s_range, y=payoff.astype(np.float32),
                                     mode="lines", name="Payoff", line=dict(dash="dash")))
        fig_pay.add_trace(go.Scatter(x=s_range, y=prices_line,
                                     mode="lines", name="Option Value"))
        fig_pay.add_vline(x=K, line_dash="dot", line_color="gray", annotation_text="Strike")
        fig_pay.update_layout(template="plotly_dark", height=350,
                              xaxis_title="Spot Price", yaxis_title="Value")
        st.plotly_chart(fig_pay, use_container_width=True)

    with col_b:
        # Binomial Convergence
        st.subheader("Binomial Convergence")
        n_vals = [10, 20, 50, 100, 200, 500]
        crr_prices = [crr_price(S, K, T, r, sigma, n, option_type, american) for n in n_vals]

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=n_vals, y=crr_prices,
                                      mode="lines+markers", name="CRR"))
        fig_conv.add_hline(y=bs_p, line_dash="dash", line_color="red",
                           annotation_text=f"BS={bs_p:.4f}")
        fig_conv.update_layout(template="plotly_dark", height=350,
                               xaxis_title="Steps (N)", yaxis_title="Price")
        st.plotly_chart(fig_conv, use_container_width=True)

    # --- Greeks Sensitivity ---
    st.subheader("Greeks Sensitivity")
    s_sens = np.linspace(max(S * 0.5, 0.01), S * 1.5, 100)
    g_sens = bs_greeks(s_sens, K, T, r, sigma)

    fig_greeks = make_subplots(rows=2, cols=2,
                               subplot_titles=["Delta", "Gamma", "Vega", "Theta"])

    fig_greeks.add_trace(go.Scatter(x=s_sens, y=np.asarray(g_sens[delta_key], dtype=np.float32),
                                    mode="lines", name="Delta", showlegend=False), row=1, col=1)
    fig_greeks.add_trace(go.Scatter(x=s_sens, y=np.asarray(g_sens["gamma"], dtype=np.float32),
                                    mode="lines", name="Gamma", showlegend=False), row=1, col=2)
    fig_greeks.add_trace(go.Scatter(x=s_sens, y=np.asarray(g_sens["vega"], dtype=np.float32),
                                    mode="lines", name="Vega", showlegend=False), row=2, col=1)
    fig_greeks.add_trace(go.Scatter(x=s_sens, y=np.asarray(g_sens[theta_key], dtype=np.float32),
                                    mode="lines", name="Theta", showlegend=False), row=2, col=2)

    for i in range(1, 3):
        for j in range(1, 3):
            fig_greeks.add_vline(x=K, line_dash="dot", line_color="gray", row=i, col=j)

    fig_greeks.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig_greeks, use_container_width=True)

    # --- 3D Volatility Surface ---
    st.subheader("Price Surface (Strike x Expiry)")
    strikes_3d = np.linspace(S * 0.7, S * 1.3, 20)
    expiries_3d = np.linspace(0.1, 2.0, 15)
    K_grid, T_grid = np.meshgrid(strikes_3d, expiries_3d)
    Z = np.array(bs_price(S, K_grid, T_grid, r, sigma, option_type), dtype=np.float32)

    fig_surf = go.Figure(data=[go.Surface(
        x=strikes_3d, y=expiries_3d, z=Z,
        colorscale="Viridis", showscale=True,
        colorbar=dict(title="Price"),
    )])
    fig_surf.update_layout(
        template="plotly_dark", height=500,
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Expiry (years)",
            zaxis_title="Option Price",
        ),
    )
    st.plotly_chart(fig_surf, use_container_width=True)
