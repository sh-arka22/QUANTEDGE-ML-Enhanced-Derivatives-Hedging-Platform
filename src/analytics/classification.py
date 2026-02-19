"""BAC direction classification: Decision Tree, Random Forest, KNN, SVM, Voting Ensemble."""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data import config as cfg


def _compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """RSI indicator using vectorized ewm."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(span=window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=window, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _compute_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_Signal": macd_signal,
        "MACD_Hist": macd_line - macd_signal,
    })


def _compute_bollinger_pctb(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.Series:
    """Bollinger Band %B position."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = upper - lower
    return (prices - lower) / bandwidth.replace(0, 1e-10)


def _compute_stochastic(
    prices: pd.Series, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator (close-price approximation)."""
    low_min = prices.rolling(k_period).min()
    high_max = prices.rolling(k_period).max()
    denom = (high_max - low_min).replace(0, 1e-10)
    stoch_k = 100 * (prices - low_min) / denom
    stoch_d = stoch_k.rolling(d_period).mean()
    return pd.DataFrame({"Stoch_K": stoch_k, "Stoch_D": stoch_d})


def _compute_roc(prices: pd.Series, period: int) -> pd.Series:
    """Rate of Change."""
    shifted = prices.shift(period).replace(0, 1e-10)
    return (prices - prices.shift(period)) / shifted


def prepare_ml_features(
    prices_df: pd.DataFrame, macro_df: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler, list[str]]:
    """Build BAC classification features with chronological split and scaling."""
    if "BAC" not in prices_df.columns:
        raise ValueError("BAC not found in prices_df columns")

    bac = prices_df["BAC"].copy()
    bac_ret = np.log(bac / bac.shift(1))

    # Target: next-day direction (1 if return > 0, else 0)
    target = (bac_ret.shift(-1) > 0).astype(int)

    # Technical features (always available)
    feat = pd.DataFrame(index=prices_df.index)
    feat["SMA_5"] = bac.rolling(5).mean()
    feat["SMA_20"] = bac.rolling(20).mean()
    for lag in range(1, 6):
        feat[f"Lag_{lag}"] = bac_ret.shift(lag)
    feat["RSI_14"] = _compute_rsi(bac, 14)

    # MACD
    macd_df = _compute_macd(bac)
    feat["MACD"] = macd_df["MACD"]
    feat["MACD_Signal"] = macd_df["MACD_Signal"]
    feat["MACD_Hist"] = macd_df["MACD_Hist"]

    # Bollinger %B
    feat["BB_PctB"] = _compute_bollinger_pctb(bac)

    # Stochastic
    stoch_df = _compute_stochastic(bac)
    feat["Stoch_K"] = stoch_df["Stoch_K"]
    feat["Stoch_D"] = stoch_df["Stoch_D"]

    # Rate of Change
    feat["ROC_10"] = _compute_roc(bac, 10)
    feat["ROC_20"] = _compute_roc(bac, 20)

    # Realized volatility
    feat["RVol_20"] = bac_ret.rolling(20).std() * np.sqrt(cfg.TRADING_DAYS)

    # SMA ratio and Z-score
    rolling_std_20 = bac.rolling(20).std()
    feat["SMA_Ratio"] = feat["SMA_5"] / feat["SMA_20"].replace(0, 1e-10)
    feat["Z_Score"] = (bac - feat["SMA_20"]) / rolling_std_20.replace(0, 1e-10)

    # Macro features (if available)
    if macro_df is not None:
        macro = macro_df.copy()
        macro.index = pd.to_datetime(macro.index)
        if macro.index.tz is not None:
            macro.index = macro.index.tz_localize(None)
        macro = macro.sort_index()
        macro = macro[~macro.index.duplicated(keep="last")]

        feat_sorted = feat.sort_index()
        feat_sorted = feat_sorted[~feat_sorted.index.duplicated(keep="last")]
        feat_sorted.index.name = "date"
        macro.index.name = "date"

        # Only include numeric macro columns
        macro_cols = [c for c in macro.columns if c in cfg.FRED_SERIES]
        if macro_cols:
            merged = pd.merge_asof(
                feat_sorted.reset_index(),
                macro[macro_cols].reset_index(),
                on="date",
                direction="backward",
                tolerance=pd.Timedelta("90d"),
            ).set_index("date")
            feat = merged

    # Combine features and target, drop NaN/inf
    combined = feat.join(target.rename("target")).replace([np.inf, -np.inf], np.nan).dropna()

    if len(combined) < cfg.MIN_PERIODS:
        raise ValueError(f"Insufficient data after dropna: {len(combined)} rows < {cfg.MIN_PERIODS}")

    X = combined.drop(columns=["target"])
    y = combined["target"].values

    # Drop zero-variance columns
    variances = X.std()
    zero_var = variances[variances < 1e-10].index.tolist()
    if zero_var:
        warnings.warn(f"Dropping zero-variance features: {zero_var}")
        X = X.drop(columns=zero_var)

    feature_names = X.columns.tolist()

    # Chronological 80/20 split (NO shuffle)
    split = int(len(X) * 0.8)
    X_train, X_test = X.values[:split], X.values[split:]
    y_train, y_test = y[:split], y[split:]

    # RobustScaler fit on train, transform both
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names


def train_models(
    X_train: np.ndarray, y_train: np.ndarray
) -> dict:
    """Train DT, RF, KNN, SVM classifiers with balanced class weights."""
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced",
            oob_score=True, random_state=42, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(
            kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=42,
        ),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def tune_models(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[dict, dict]:
    """Hyperparameter tuning via RandomizedSearchCV with TimeSeriesSplit."""
    cv = TimeSeriesSplit(n_splits=cfg.CV_SPLITS)

    param_grids = {
        "Decision Tree": {
            "model": DecisionTreeClassifier(class_weight="balanced", random_state=42),
            "params": {
                "max_depth": [3, 5, 7, 10],
                "min_samples_leaf": [10, 20, 50],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                class_weight="balanced", oob_score=True, random_state=42, n_jobs=-1
            ),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "max_features": ["sqrt", "log2"],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
        "SVM": {
            "model": SVC(
                kernel="rbf", probability=True,
                class_weight="balanced", random_state=42,
            ),
            "params": {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
            },
        },
    }

    tuned_models = {}
    cv_results = {}

    for name, spec in param_grids.items():
        search = RandomizedSearchCV(
            estimator=spec["model"],
            param_distributions=spec["params"],
            n_iter=min(cfg.TUNING_N_ITER, _param_grid_size(spec["params"])),
            scoring=cfg.TUNING_SCORING,
            cv=cv,
            random_state=42,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        cv_results[name] = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
        }

    return tuned_models, cv_results


def _param_grid_size(params: dict) -> int:
    """Count total combinations in a parameter grid."""
    size = 1
    for v in params.values():
        size *= len(v)
    return size


def build_ensemble(models: dict) -> VotingClassifier:
    """Soft-voting ensemble of RF + KNN + SVM (DT excluded)."""
    estimators = [
        ("rf", models["Random Forest"]),
        ("knn", models["KNN"]),
        ("svm", models["SVM"]),
    ]
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    # Mark as fitted by copying attributes from pre-fitted estimators
    ensemble.estimators_ = [models["Random Forest"], models["KNN"], models["SVM"]]
    le = LabelEncoder()
    le.classes_ = models["Random Forest"].classes_
    ensemble.le_ = le
    ensemble.classes_ = models["Random Forest"].classes_
    return ensemble


def compute_mutual_info(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Mutual information scores between features and target."""
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    df = pd.DataFrame({"feature": feature_names, "mi_score": mi})
    return df.sort_values("mi_score", ascending=False).reset_index(drop=True)


def select_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    mi_df: pd.DataFrame,
    feature_names: list[str],
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop features with MI below threshold; fallback keeps all if none pass."""
    if threshold is None:
        threshold = cfg.MI_THRESHOLD

    keep_mask = mi_df.set_index("feature").loc[feature_names, "mi_score"] >= threshold
    kept_names = [f for f, keep in zip(feature_names, keep_mask) if keep]

    # Fallback: keep all if nothing passes
    if not kept_names:
        return X_train, X_test, feature_names

    col_idx = [feature_names.index(f) for f in kept_names]
    return X_train[:, col_idx], X_test[:, col_idx], kept_names


def evaluate_models(
    models: dict, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Compute accuracy, precision, recall, F1, AUC, confusion matrix per model."""
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)

        # ROC AUC via predict_proba or decision_function
        proba = None
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
        except (AttributeError, IndexError):
            try:
                scores = model.decision_function(X_test)
                auc = float(roc_auc_score(y_test, scores))
            except Exception:
                auc = np.nan

        results[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix(y_test, preds),
            "predictions": preds,
            "probabilities": proba if proba is not None else np.full(len(y_test), np.nan),
        }
    return results


def feature_importance(rf_model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Extract and sort Random Forest feature importances."""
    imp = rf_model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def run_classification(
    prices_df: pd.DataFrame, macro_df: pd.DataFrame | None = None, tune: bool = False
) -> dict:
    """Orchestrator: prepare -> MI select -> train/tune -> ensemble -> evaluate -> importance."""
    X_train, X_test, y_train, y_test, scaler, feat_names = prepare_ml_features(
        prices_df, macro_df
    )

    # Mutual information feature selection
    mi_df = compute_mutual_info(X_train, y_train, feat_names)
    X_train, X_test, feat_names = select_features(X_train, X_test, mi_df, feat_names)

    # Train or tune base models
    cv_results = None
    if tune:
        models, cv_results = tune_models(X_train, y_train)
    else:
        models = train_models(X_train, y_train)

    # Build ensemble from fitted base models
    ensemble = build_ensemble(models)
    models["Voting Ensemble"] = ensemble

    metrics = evaluate_models(models, X_test, y_test)
    importance = feature_importance(models["Random Forest"], feat_names)

    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique.astype(int), counts.astype(int)))

    result = {
        "models": models,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_names": feat_names,
        "scaler": scaler,
        "class_distribution": class_dist,
        "y_test": y_test,
        "X_test": X_test,
        "mutual_info": mi_df,
    }
    if cv_results is not None:
        result["cv_results"] = cv_results

    return result
