"""Portfolio risk analytics: returns, ratios, CAPM, VaR, CVaR, drawdown."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.config import TRADING_DAYS, VAR_CONFIDENCE


def weighted_returns(returns_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute portfolio return as weighted sum of individual asset returns."""
    if not weights:
        raise ValueError("Weights dict is empty")

    common = [c for c in weights if c in returns_df.columns]
    if not common:
        raise ValueError(f"No matching columns between weights {list(weights)} and returns {list(returns_df.columns)}")

    w = np.array([weights[c] for c in common], dtype=np.float64)
    w_sum = w.sum()
    if abs(w_sum) < 1e-10:
        raise ValueError("Weights sum to zero")
    if abs(w_sum - 1.0) > 1e-6:
        warnings.warn(f"Weights sum to {w_sum:.4f}, normalizing to 1.0")
        w = w / w_sum

    return returns_df[common].values @ w


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float, periods: int = TRADING_DAYS) -> float:
    """Annualized Sharpe ratio."""
    returns = np.asarray(returns, dtype=np.float64)
    daily_rf = risk_free_rate / periods
    excess = returns - daily_rf
    std = excess.std(ddof=1)
    if std < 1e-8:
        return np.nan
    return float((excess.mean() * periods) / (std * np.sqrt(periods)))


def sortino_ratio(
    returns: np.ndarray, risk_free_rate: float, mar: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Annualized Sortino ratio using downside deviation."""
    returns = np.asarray(returns, dtype=np.float64)
    daily_rf = risk_free_rate / periods
    excess = returns - daily_rf
    downside = returns[returns < mar] - mar
    if len(downside) < 5:
        warnings.warn(f"Only {len(downside)} downside observations, returning NaN")
        return np.nan
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std < 1e-8:
        return np.nan
    return float((excess.mean() * periods) / (downside_std * np.sqrt(periods)))


def capm(portfolio_returns: np.ndarray, market_returns: np.ndarray) -> dict:
    """CAPM regression: Rp = alpha + beta * Rm."""
    port = np.asarray(portfolio_returns, dtype=np.float64).ravel()
    mkt = np.asarray(market_returns, dtype=np.float64).ravel()
    min_len = min(len(port), len(mkt))
    port, mkt = port[:min_len], mkt[:min_len]

    nan_result = {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan,
                  "alpha_pvalue": np.nan, "beta_pvalue": np.nan}
    try:
        X = sm.add_constant(mkt)
        model = sm.OLS(port, X).fit()
        return {
            "alpha": float(model.params[0]),
            "beta": float(model.params[1]),
            "r_squared": float(model.rsquared),
            "alpha_pvalue": float(model.pvalues[0]),
            "beta_pvalue": float(model.pvalues[1]),
        }
    except Exception:
        return nan_result


def var_parametric(returns: np.ndarray, confidence: float = VAR_CONFIDENCE) -> float:
    """Parametric VaR assuming normal distribution."""
    returns = np.asarray(returns, dtype=np.float64)
    std = returns.std(ddof=1)
    if std < 1e-10:
        return 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - confidence)
    return float(returns.mean() + z * std)


def var_historical(returns: np.ndarray, confidence: float = VAR_CONFIDENCE) -> float:
    """Historical VaR at given confidence level."""
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 100:
        warnings.warn(f"Only {len(returns)} observations for historical VaR (recommend >= 100)")
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar(returns: np.ndarray, confidence: float = VAR_CONFIDENCE) -> float:
    """Conditional VaR (Expected Shortfall) — mean of returns below VaR."""
    returns = np.asarray(returns, dtype=np.float64)
    var_threshold = var_historical(returns, confidence)
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown."""
    cum = np.asarray(cumulative_returns, dtype=np.float64)
    peak = np.maximum.accumulate(cum)
    drawdowns = (peak - cum) / np.where(peak > 0, peak, 1.0)
    return float(drawdowns.max()) if len(drawdowns) > 0 else 0.0


def risk_summary(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    market_returns: np.ndarray,
    risk_free_rate: float,
) -> dict:
    """Comprehensive portfolio risk summary."""
    port_ret = weighted_returns(returns_df, weights)
    ann_return = float(np.mean(port_ret) * TRADING_DAYS)
    ann_vol = float(np.std(port_ret, ddof=1) * np.sqrt(TRADING_DAYS))

    sharpe = sharpe_ratio(port_ret, risk_free_rate)
    sortino = sortino_ratio(port_ret, risk_free_rate)
    capm_res = capm(port_ret, market_returns)

    var_p = var_parametric(port_ret)
    var_h = var_historical(port_ret)
    cvar_val = cvar(port_ret)

    cum_ret = np.cumprod(1 + port_ret)
    mdd = max_drawdown(cum_ret)

    skew = float(pd.Series(port_ret).skew())
    kurt = float(pd.Series(port_ret).kurtosis())

    # Flag fat-tail risk if parametric and historical VaR diverge significantly
    kurtosis_risk = False
    if abs(var_h) > 1e-10:
        kurtosis_risk = abs(var_p - var_h) / abs(var_h) > 0.2

    return {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "beta": capm_res["beta"],
        "alpha": capm_res["alpha"],
        "alpha_pvalue": capm_res["alpha_pvalue"],
        "beta_pvalue": capm_res["beta_pvalue"],
        "r_squared": capm_res["r_squared"],
        "var_parametric": var_p,
        "var_historical": var_h,
        "cvar": cvar_val,
        "max_drawdown": mdd,
        "skewness": skew,
        "kurtosis": kurt,
        "kurtosis_risk": kurtosis_risk,
        "portfolio_returns": port_ret,
        "cumulative_returns": cum_ret,
    }
