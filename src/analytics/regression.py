"""AAPL OLS regression with full diagnostic testing."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import normaltest
from sklearn.linear_model import Ridge
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from src.data import config as cfg


def prepare_features(
    prices_df: pd.DataFrame, market_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build AAPL regression features with chronological train/test split."""
    if "AAPL" not in prices_df.columns:
        raise ValueError("AAPL not found in prices_df columns")

    aapl = prices_df["AAPL"].copy()
    aapl_ret = np.log(aapl / aapl.shift(1))

    # Features
    feat = pd.DataFrame(index=prices_df.index)
    feat["SMA_5"] = aapl.rolling(5).mean()
    feat["SMA_20"] = aapl.rolling(20).mean()
    feat["Lag_1_Return"] = aapl_ret.shift(1)

    # Market lagged returns
    if isinstance(market_df, pd.Series):
        market_df = market_df.to_frame()

    for col in market_df.columns:
        mkt_ret = np.log(market_df[col] / market_df[col].shift(1))
        label = col.replace("^GSPC", "SP500").replace("^IXIC", "NASDAQ")
        feat[f"{label}_Lag1"] = mkt_ret.shift(1)

    # Target: next-day return
    target = aapl_ret.shift(-1)

    # Align and drop NaN
    combined = feat.join(target.rename("target")).dropna()
    if len(combined) < cfg.MIN_PERIODS:
        raise ValueError(f"Insufficient data after dropna: {len(combined)} rows < {cfg.MIN_PERIODS}")

    X = combined.drop(columns=["target"])
    y = combined["target"]

    # Drop zero-variance features
    zero_var = X.columns[X.std() < 1e-10]
    if len(zero_var) > 0:
        warnings.warn(f"Dropping zero-variance features: {list(zero_var)}")
        X = X.drop(columns=zero_var)

    # Chronological 80/20 split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    return X_train, X_test, y_train, y_test


def fit_ols(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit OLS with constant; falls back to Ridge on singular matrix."""
    X_c = sm.add_constant(X_train, has_constant="add")
    try:
        model = sm.OLS(y_train, X_c).fit()
        return model
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix in OLS — falling back to Ridge regression")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        return ridge


def diagnostics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """VIF, Breusch-Pagan, Durbin-Watson, Jarque-Bera diagnostics."""
    result = {}

    # VIF
    X_c = sm.add_constant(X, has_constant="add")
    vif = {}
    feature_names = X.columns.tolist()
    for i, name in enumerate(feature_names):
        try:
            vif[name] = float(variance_inflation_factor(X_c.values, i + 1))  # skip const at 0
        except Exception:
            vif[name] = np.nan
    result["vif"] = vif
    result["vif_flag"] = any(v > cfg.VIF_THRESHOLD for v in vif.values() if np.isfinite(v))

    # Residuals (handle both statsmodels and sklearn)
    if hasattr(model, "resid"):
        resid = model.resid.values if hasattr(model.resid, "values") else np.asarray(model.resid)
        fitted = model.fittedvalues
        if hasattr(fitted, "values"):
            fitted = fitted.values
    else:
        preds = model.predict(X)
        resid = np.asarray(y) - np.asarray(preds)
        fitted = np.asarray(preds)

    # Breusch-Pagan
    try:
        exog = sm.add_constant(X, has_constant="add")
        bp_stat, bp_pval, _, _ = het_breuschpagan(resid, exog)
        result["bp_stat"] = float(bp_stat)
        result["bp_pvalue"] = float(bp_pval)
        result["is_heteroscedastic"] = bp_pval < 0.05
    except Exception:
        result["bp_stat"] = np.nan
        result["bp_pvalue"] = np.nan
        result["is_heteroscedastic"] = False

    # Durbin-Watson
    dw = float(durbin_watson(resid))
    result["dw_stat"] = dw
    if dw < 1.5:
        result["dw_flag"] = "positive autocorrelation"
    elif dw > 2.5:
        result["dw_flag"] = "negative autocorrelation"
    else:
        result["dw_flag"] = "no significant autocorrelation"

    # Jarque-Bera (using scipy normaltest)
    jb_stat, jb_pval = normaltest(resid)
    result["jb_stat"] = float(jb_stat)
    result["jb_pvalue"] = float(jb_pval)
    result["is_normal"] = jb_pval >= 0.05

    return result


def _refit_robust(X_train: pd.DataFrame, y_train: pd.Series):
    """Refit OLS with HC3 robust standard errors."""
    X_c = sm.add_constant(X_train, has_constant="add")
    return sm.OLS(y_train, X_c).fit(cov_type="HC3")


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Out-of-sample RMSE, MAE, R-squared."""
    if hasattr(model, "predict") and hasattr(model, "params"):
        X_c = sm.add_constant(X_test, has_constant="add")
        preds = model.predict(X_c)
    else:
        preds = model.predict(X_test)

    preds = np.asarray(preds)
    actual = np.asarray(y_test)
    residuals = actual - preds

    # Degenerate model check
    if np.std(preds) < 1e-10:
        warnings.warn("Degenerate model: all predictions identical")

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan

    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "r_squared": float(r2),
        "predictions": preds,
        "residuals": residuals,
    }


def run_regression(prices_df: pd.DataFrame, market_df: pd.DataFrame) -> dict:
    """Orchestrator: prepare -> fit -> diagnostics -> evaluate."""
    X_train, X_test, y_train, y_test = prepare_features(prices_df, market_df)

    model = fit_ols(X_train, y_train)
    diag = diagnostics(model, X_train, y_train)

    # Refit with robust SEs if heteroscedastic
    robust_model = None
    if diag["is_heteroscedastic"] and hasattr(model, "resid"):
        robust_model = _refit_robust(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    return {
        "model": model,
        "robust_model": robust_model,
        "diagnostics": diag,
        "metrics": metrics,
        "feature_names": X_train.columns.tolist(),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
