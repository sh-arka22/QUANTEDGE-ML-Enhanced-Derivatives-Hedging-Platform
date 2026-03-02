"""Main Streamlit entry point for the Quantitative Finance Analytics Platform."""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

st.set_page_config(page_title="Quant Platform", layout="wide", page_icon="📊")


def main():
    """Render the main application."""
    st.title("Quantitative Finance Analytics Platform")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        fred_key = st.text_input(
            "FRED API Key (optional)",
            type="password",
            value=os.environ.get("FRED_API_KEY", ""),
            help="Enter to enable macro features. Leave blank to use technical-only mode.",
        )
        if fred_key:
            os.environ["FRED_API_KEY"] = fred_key

        rf_rate = st.number_input(
            "Risk-Free Rate", value=0.02, min_value=0.0, max_value=0.20, step=0.005, format="%.3f"
        )

    # Load data
    from src.data.loaders import get_all_data

    with st.spinner("Loading market data..."):
        data = get_all_data()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Portfolio Analytics",
        "Price Prediction",
        "Derivatives Pricing",
        "Hedging Simulator",
    ])

    with tab1:
        try:
            from src.ui.tab_portfolio import render as render_portfolio
            render_portfolio(data["returns"], data["market_returns"], rf_rate)
        except Exception as e:
            st.error(f"Portfolio tab error: {e}")

    with tab2:
        try:
            from src.ui.tab_prediction import render as render_prediction
            render_prediction(data["prices"], data["market_returns"], data.get("macro_aligned"))
        except Exception as e:
            st.error(f"Prediction tab error: {e}")

    with tab3:
        try:
            from src.ui.tab_derivatives import render as render_derivatives
            render_derivatives()
        except Exception as e:
            st.error(f"Derivatives tab error: {e}")

    with tab4:
        try:
            from src.ui.tab_hedging import render as render_hedging
            render_hedging(data["prices"])
        except Exception as e:
            st.error(f"Hedging tab error: {e}")

    # Footer
    st.divider()
    st.caption("Built with Streamlit | Quantitative Finance Analytics Platform")


if __name__ == "__main__":
    main()
