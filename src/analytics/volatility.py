"""JPM volatility modeling: GARCH(1,1), EWMA, forecasting, volatility cones."""

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from src.data import config as cfg


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> dict | None:
    """Fit GARCH(p,q) model on percentage-scaled returns."""
    returns = returns.dropna()
    if len(returns) < cfg.MIN_PERIODS:
        raise ValueError(f"Insufficient data: {len(returns)} < {cfg.MIN_PERIODS}")
    if returns.std() < 1e-8:
        raise ValueError("Zero variance in returns")

    scaled = returns * 100  # arch expects percentage returns

    # Try fitting with default, then alternate starting values on failure
    result = None
    starting_configs = [None, [0.01, 0.05, 0.90]]
    for sv in starting_configs:
        try:
            model = arch_model(scaled, vol="Garch", p=p, q=q, dist="normal", mean="Constant")
            kw = {"disp": "off", "options": {"maxiter": 1000}}
            if sv is not None:
                kw["starting_values"] = np.array([scaled.mean()] + sv)
            result = model.fit(**kw)
            if result.convergence_flag == 0:
                break
        except Exception:
            continue

    if result is None:
        warnings.warn("GARCH convergence failed on all attempts")
        return None

    alpha = float(result.params.get("alpha[1]", 0))
    beta = float(result.params.get("beta[1]", 0))
    omega = float(result.params.get("omega", 0))
    persistence = alpha + beta

    # Long-run variance (annualized vol)
    long_run_var = omega / (1 - persistence) if persistence < 1.0 else np.nan
    long_run_vol = float(np.sqrt(long_run_var) / 100 * np.sqrt(cfg.TRADING_DAYS)) if np.isfinite(long_run_var) else np.nan

    # Conditional volatility: convert from percentage to decimal, then annualize
    cond_vol = result.conditional_volatility / 100 * np.sqrt(cfg.TRADING_DAYS)

    return {
        "params": {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": persistence,
            "long_run_vol": long_run_vol,
        },
        "conditional_vol": cond_vol,
        "igarch_flag": persistence >= 0.999,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "model": result,
    }


def fit_ewma(returns: pd.Series, lambda_: float = cfg.EWMA_LAMBDA) -> pd.Series:
    """EWMA volatility estimate, annualized."""
    returns = returns.dropna()
    if not 0 < lambda_ < 1:
        warnings.warn(f"Lambda {lambda_} outside (0,1), clamping to {cfg.EWMA_LAMBDA}")
        lambda_ = cfg.EWMA_LAMBDA

    daily_vol = returns.pow(2).ewm(alpha=1 - lambda_, adjust=False).mean().pow(0.5)
    return daily_vol * np.sqrt(cfg.TRADING_DAYS)


def forecast_volatility(garch_result, horizon: int = 30) -> pd.Series:
    """Forecast annualized volatility from fitted GARCH model."""
    if garch_result is None:
        return pd.Series(dtype=np.float64)

    model_result = garch_result["model"]
    n = len(model_result.resid)
    horizon = min(horizon, n // 3)
    if horizon < 1:
        warnings.warn("Horizon too small after capping")
        return pd.Series(dtype=np.float64)

    fc = model_result.forecast(horizon=horizon)
    # fc.variance is in percentage-squared; convert to annualized decimal vol
    variance_forecast = fc.variance.dropna().iloc[-1]
    vol_forecast = np.sqrt(variance_forecast) / 100 * np.sqrt(cfg.TRADING_DAYS)

    return pd.Series(
        vol_forecast.values,
        index=range(1, horizon + 1),
        name="forecast_vol",
    )


def volatility_cones(returns: pd.Series, windows: list[int] | None = None) -> pd.DataFrame:
    """Realized volatility statistics across rolling windows."""
    if windows is None:
        windows = [10, 30, 60, 90]

    returns = returns.dropna()
    records = []
    for w in windows:
        if w > len(returns):
            continue
        rolling_vol = returns.rolling(w).std() * np.sqrt(cfg.TRADING_DAYS)
        rolling_vol = rolling_vol.dropna()
        if len(rolling_vol) == 0:
            continue
        records.append({
            "window": w,
            "min": float(rolling_vol.min()),
            "q25": float(rolling_vol.quantile(0.25)),
            "median": float(rolling_vol.median()),
            "q75": float(rolling_vol.quantile(0.75)),
            "max": float(rolling_vol.max()),
        })

    if not records:
        warnings.warn("All windows exceed data length")
        return pd.DataFrame(columns=["window", "min", "q25", "median", "q75", "max"])

    df = pd.DataFrame(records).set_index("window")
    return df


def realized_vs_predicted(
    returns: pd.Series, garch_result: dict | None, window: int = 30
) -> pd.DataFrame:
    """Compare realized rolling vol vs GARCH conditional vol."""
    returns = returns.dropna()
    realized = returns.rolling(window).std() * np.sqrt(cfg.TRADING_DAYS)
    realized = realized.rename("realized")

    if garch_result is not None:
        predicted = garch_result["conditional_vol"]
        predicted = predicted.reindex(realized.index).rename("predicted")
    else:
        predicted = pd.Series(np.nan, index=realized.index, name="predicted")

    return pd.concat([realized, predicted], axis=1).dropna()


def run_volatility(returns_series: pd.Series) -> dict:
    """Orchestrator: GARCH + EWMA + forecast + cones + comparison."""
    returns_series = returns_series.dropna()

    # GARCH(1,1)
    garch = fit_garch(returns_series)

    # EWMA fallback / complement
    ewma_vol = fit_ewma(returns_series)

    # Forecast
    forecast = forecast_volatility(garch, horizon=30) if garch is not None else pd.Series(dtype=np.float64)

    # Volatility cones
    cones = volatility_cones(returns_series)

    # Realized vs predicted comparison
    comparison = realized_vs_predicted(returns_series, garch)

    return {
        "garch": garch,
        "ewma_vol": ewma_vol,
        "forecast": forecast,
        "cones": cones,
        "realized_vs_predicted": comparison,
        "used_ewma_fallback": garch is None,
    }
