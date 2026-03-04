"""Tab 2: Price Prediction — AAPL regression + BAC ML classification."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import probplot
import streamlit as st

from src.data import config as cfg


def _render_regression(prices_df: pd.DataFrame, market_returns) -> None:
    """AAPL OLS regression section."""
    from src.analytics.regression import run_regression

    market_df = prices_df[[c for c in cfg.TICKERS["market"] if c in prices_df.columns]]

    with st.spinner("Running AAPL regression..."):
        reg = run_regression(prices_df, market_df)

    metrics = reg["metrics"]
    diag = reg["diagnostics"]

    # Metrics cards
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{metrics['rmse']:.6f}")
    m2.metric("R-squared", f"{metrics['r_squared']:.4f}")
    m3.metric("MAE", f"{metrics['mae']:.6f}")

    st.info("Newey-West HAC standard errors applied (robust to heteroscedasticity and autocorrelation).")

    # Diagnostics
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Durbin-Watson", f"{diag['dw_stat']:.2f}", delta=diag["dw_flag"])
    d2.metric("Breusch-Pagan p", f"{diag['bp_pvalue']:.4f}")
    d3.metric("Jarque-Bera p", f"{diag['jb_pvalue']:.4f}")
    vif_max = max(diag["vif"].values()) if diag["vif"] else 0
    d4.metric("Max VIF", f"{vif_max:.1f}", delta="Flag" if diag["vif_flag"] else "OK",
              delta_color="inverse" if diag["vif_flag"] else "normal")

    # Charts
    col_a, col_b = st.columns(2)

    with col_a:
        # Actual vs Predicted
        y_test = np.asarray(reg["y_test"])
        preds = metrics["predictions"]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=y_test, y=preds, mode="markers", marker=dict(size=3), name="Points"))
        min_v, max_v = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
        fig_pred.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                      mode="lines", line=dict(dash="dash", color="red"), name="Perfect"))
        fig_pred.update_layout(template="plotly_dark", height=350,
                               title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_b:
        # Residuals
        residuals = metrics["residuals"]
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(y=residuals, mode="markers", marker=dict(size=3), name="Residuals"))
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_layout(template="plotly_dark", height=350,
                              title="Residuals", xaxis_title="Index", yaxis_title="Residual")
        st.plotly_chart(fig_res, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        # QQ Plot
        (osm, osr), _ = probplot(residuals)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(size=3), name="Residuals"))
        fig_qq.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[osm.min(), osm.max()],
                                    mode="lines", line=dict(dash="dash", color="red"), name="Normal"))
        fig_qq.update_layout(template="plotly_dark", height=350,
                             title="QQ Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    with col_d:
        # VIF Bar Chart
        vif_df = pd.DataFrame(list(diag["vif"].items()), columns=["Feature", "VIF"])
        fig_vif = go.Figure(go.Bar(x=vif_df["Feature"], y=vif_df["VIF"]))
        fig_vif.add_hline(y=cfg.VIF_THRESHOLD, line_dash="dash", line_color="red",
                          annotation_text=f"Threshold={cfg.VIF_THRESHOLD}")
        fig_vif.update_layout(template="plotly_dark", height=350, title="VIF by Feature")
        st.plotly_chart(fig_vif, use_container_width=True)


def _render_classification(prices_df: pd.DataFrame, macro_df) -> None:
    """BAC ML classification section."""
    from src.analytics.classification import run_classification

    with st.spinner("Running BAC classification..."):
        cls = run_classification(prices_df, macro_df=macro_df)

    metrics = cls["metrics"]

    # Model comparison table
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "Recall": f"{m['recall']:.3f}",
            "F1": f"{m['f1']:.3f}",
            "AUC-ROC": f"{m['roc_auc']:.3f}" if np.isfinite(m['roc_auc']) else "N/A",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # ROC Curves
        fig_roc = go.Figure()
        for name, m in metrics.items():
            if name == "Voting Ensemble":
                continue
            proba = m.get("probabilities")
            if proba is not None and not np.all(np.isnan(proba)):
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(cls["y_test"], proba)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={m['roc_auc']:.2f})",
                ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="gray"), name="Random"))
        fig_roc.update_layout(template="plotly_dark", height=400,
                              title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_b:
        # Feature importance
        importance = cls["feature_importance"]
        fig_imp = go.Figure(go.Bar(
            x=importance["importance"].values[::-1],
            y=importance["feature"].values[::-1],
            orientation="h",
        ))
        fig_imp.update_layout(template="plotly_dark", height=400, title="Feature Importance (RF)")
        st.plotly_chart(fig_imp, use_container_width=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    base_models = {k: v for k, v in metrics.items() if k != "Voting Ensemble"}
    names = list(base_models.keys())
    fig_cm = make_subplots(rows=1, cols=len(names), subplot_titles=names)
    for idx, name in enumerate(names):
        cm = base_models[name]["confusion_matrix"]
        fig_cm.add_trace(go.Heatmap(
            z=cm, text=cm, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
        ), row=1, col=idx + 1)
    fig_cm.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Class distribution
    col_c, col_d = st.columns(2)
    with col_c:
        dist = cls["class_distribution"]
        fig_pie = go.Figure(go.Pie(
            labels=["Down (0)", "Up (1)"],
            values=[dist.get(0, 0), dist.get(1, 0)],
        ))
        fig_pie.update_layout(template="plotly_dark", height=300, title="Training Class Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)


def render(prices_df: pd.DataFrame, market_returns, macro_df) -> None:
    """Render prediction tab with regression and classification sections."""
    st.subheader("AAPL Regression Analysis")
    _render_regression(prices_df, market_returns)

    st.divider()

    st.subheader("BAC Direction Classification")
    _render_classification(prices_df, macro_df)
