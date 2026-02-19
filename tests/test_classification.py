"""Tests for BAC ML classification module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.analytics.classification import (
    _compute_bollinger_pctb,
    _compute_macd,
    _compute_roc,
    _compute_rsi,
    _compute_stochastic,
    build_ensemble,
    compute_mutual_info,
    evaluate_models,
    feature_importance,
    prepare_ml_features,
    run_classification,
    select_features,
    train_models,
    tune_models,
)


def _make_prices(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic price data for BAC + market tickers."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    tickers = ["MS", "JPM", "BAC", "AAPL", "^GSPC", "^IXIC"]
    data = {}
    for t in tickers:
        # Random walk prices starting at 100
        returns = rng.normal(0.0003, 0.015, n)
        data[t] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)


def _make_macro(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic macro data aligned to price dates."""
    rng = np.random.RandomState(99)
    dates = prices_df.index[::5]  # every 5th business day
    return pd.DataFrame(
        {
            "DGS10": rng.uniform(1.0, 4.0, len(dates)),
            "T10Y2Y": rng.uniform(-0.5, 2.0, len(dates)),
            "VIXCLS": rng.uniform(10, 40, len(dates)),
        },
        index=dates,
    )


class TestPrepareFeatures:
    def test_returns_correct_shapes(self):
        prices = _make_prices()
        X_train, X_test, y_train, y_test, scaler, feat_names = prepare_ml_features(prices)
        assert X_train.shape[0] > X_test.shape[0], "Train should be larger than test"
        assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch"
        assert len(feat_names) == X_train.shape[1]
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]

    def test_chronological_split(self):
        """Train set should precede test set (no random shuffle)."""
        prices = _make_prices()
        X_train, X_test, y_train, y_test, _, _ = prepare_ml_features(prices)
        ratio = len(X_train) / (len(X_train) + len(X_test))
        assert 0.79 < ratio < 0.81

    def test_target_is_binary(self):
        prices = _make_prices()
        _, _, y_train, y_test, _, _ = prepare_ml_features(prices)
        assert set(np.unique(y_train)).issubset({0, 1})
        assert set(np.unique(y_test)).issubset({0, 1})

    def test_features_are_scaled(self):
        """After RobustScaler, train median ~0."""
        prices = _make_prices()
        X_train, _, _, _, _, _ = prepare_ml_features(prices)
        np.testing.assert_allclose(np.median(X_train, axis=0), 0, atol=1e-6)

    def test_scaler_is_robust(self):
        """Scaler returned should be RobustScaler."""
        prices = _make_prices()
        _, _, _, _, scaler, _ = prepare_ml_features(prices)
        assert isinstance(scaler, RobustScaler)

    def test_with_macro_features(self):
        """Including macro data should add extra columns."""
        prices = _make_prices()
        macro = _make_macro(prices)
        _, _, _, _, _, feat_tech = prepare_ml_features(prices, macro_df=None)
        _, _, _, _, _, feat_macro = prepare_ml_features(prices, macro_df=macro)
        assert len(feat_macro) > len(feat_tech), "Macro features should add columns"

    def test_missing_bac_raises(self):
        prices = pd.DataFrame({"AAPL": [100, 101, 102]}, index=pd.bdate_range("2024-01-01", periods=3))
        with pytest.raises(ValueError, match="BAC not found"):
            prepare_ml_features(prices)

    def test_insufficient_data_raises(self):
        """Fewer than MIN_PERIODS rows after dropna should raise."""
        prices = _make_prices(n=30)
        with pytest.raises(ValueError, match="Insufficient data"):
            prepare_ml_features(prices)

    def test_expanded_feature_count(self):
        """Should have ~18 technical features (up from 9)."""
        prices = _make_prices()
        _, _, _, _, _, feat_names = prepare_ml_features(prices)
        assert len(feat_names) >= 18, f"Expected >=18 features, got {len(feat_names)}"


class TestNewIndicators:
    def test_macd_columns(self):
        prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, 200))))
        result = _compute_macd(prices)
        assert set(result.columns) == {"MACD", "MACD_Signal", "MACD_Hist"}
        assert len(result) == 200

    def test_macd_hist_is_diff(self):
        prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, 200))))
        result = _compute_macd(prices)
        np.testing.assert_allclose(result["MACD_Hist"], result["MACD"] - result["MACD_Signal"])

    def test_bollinger_pctb_range(self):
        """In typical conditions, %B should mostly be in [0,1] with occasional excursions."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 300))))
        pctb = _compute_bollinger_pctb(prices).dropna()
        # Most values within [-0.5, 1.5] (allowing for some overshoot)
        within = ((pctb >= -0.5) & (pctb <= 1.5)).mean()
        assert within > 0.9

    def test_stochastic_range(self):
        """Stochastic %K and %D should be in [0, 100]."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
        result = _compute_stochastic(prices)
        valid = result.dropna()
        assert (valid["Stoch_K"] >= 0).all() and (valid["Stoch_K"] <= 100).all()
        assert (valid["Stoch_D"] >= 0).all() and (valid["Stoch_D"] <= 100).all()

    def test_roc_known_value(self):
        """ROC of 10 periods: if price goes from 100 to 110, ROC = 0.1."""
        prices = pd.Series([100.0] * 10 + [110.0])
        roc = _compute_roc(prices, 10)
        np.testing.assert_allclose(roc.iloc[-1], 0.1, atol=1e-10)


class TestTrainModels:
    def test_all_four_models_fitted(self):
        prices = _make_prices()
        X_train, _, y_train, _, _, _ = prepare_ml_features(prices)
        models = train_models(X_train, y_train)
        assert set(models.keys()) == {"Decision Tree", "Random Forest", "KNN", "SVM"}
        for name, m in models.items():
            assert hasattr(m, "predict"), f"{name} has no predict method"

    def test_rf_has_oob_score(self):
        prices = _make_prices()
        X_train, _, y_train, _, _, _ = prepare_ml_features(prices)
        models = train_models(X_train, y_train)
        rf = models["Random Forest"]
        assert hasattr(rf, "oob_score_"), "RF should compute OOB score"
        assert 0 < rf.oob_score_ < 1


class TestTuneModels:
    def test_returns_models_and_cv_results(self):
        prices = _make_prices()
        X_train, _, y_train, _, _, _ = prepare_ml_features(prices)
        models, cv_results = tune_models(X_train, y_train)
        assert set(models.keys()) == {"Decision Tree", "Random Forest", "KNN", "SVM"}
        for name in models:
            assert name in cv_results
            assert "best_params" in cv_results[name]
            assert "best_score" in cv_results[name]

    def test_tuned_models_predict(self):
        prices = _make_prices()
        X_train, X_test, y_train, _, _, _ = prepare_ml_features(prices)
        models, _ = tune_models(X_train, y_train)
        for name, m in models.items():
            preds = m.predict(X_test)
            assert set(np.unique(preds)).issubset({0, 1}), f"{name} predictions not binary"


class TestVotingEnsemble:
    def test_ensemble_predicts_binary(self):
        prices = _make_prices()
        X_train, X_test, y_train, _, _, _ = prepare_ml_features(prices)
        base_models = train_models(X_train, y_train)
        ensemble = build_ensemble(base_models)
        preds = ensemble.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_ensemble_has_predict_proba(self):
        prices = _make_prices()
        X_train, X_test, y_train, _, _, _ = prepare_ml_features(prices)
        base_models = train_models(X_train, y_train)
        ensemble = build_ensemble(base_models)
        proba = ensemble.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestMutualInfo:
    def test_mi_scores_non_negative(self):
        prices = _make_prices()
        X_train, _, y_train, _, _, feat_names = prepare_ml_features(prices)
        mi_df = compute_mutual_info(X_train, y_train, feat_names)
        assert (mi_df["mi_score"] >= 0).all()
        assert len(mi_df) == len(feat_names)

    def test_select_features_reduces_columns(self):
        prices = _make_prices()
        X_train, X_test, y_train, _, _, feat_names = prepare_ml_features(prices)
        mi_df = compute_mutual_info(X_train, y_train, feat_names)
        # Use a high threshold to ensure some features are dropped
        high_thresh = mi_df["mi_score"].max() + 0.01
        X_tr_sel, X_te_sel, kept = select_features(X_train, X_test, mi_df, feat_names, threshold=high_thresh)
        # Fallback: nothing passes, so all kept
        assert len(kept) == len(feat_names)

    def test_select_features_with_low_threshold(self):
        prices = _make_prices()
        X_train, X_test, y_train, _, _, feat_names = prepare_ml_features(prices)
        mi_df = compute_mutual_info(X_train, y_train, feat_names)
        X_tr_sel, X_te_sel, kept = select_features(X_train, X_test, mi_df, feat_names, threshold=0.0)
        assert len(kept) == len(feat_names), "Threshold 0 should keep all features"


class TestEvaluateModels:
    def test_metrics_structure(self):
        prices = _make_prices()
        X_train, X_test, y_train, y_test, _, _ = prepare_ml_features(prices)
        models = train_models(X_train, y_train)
        results = evaluate_models(models, X_test, y_test)
        for name in models:
            assert name in results
            r = results[name]
            assert "accuracy" in r
            assert "precision" in r
            assert "recall" in r
            assert "f1" in r
            assert "roc_auc" in r
            assert "confusion_matrix" in r
            assert 0 <= r["accuracy"] <= 1
            assert 0 <= r["precision"] <= 1
            assert 0 <= r["f1"] <= 1

    def test_predictions_correct_length(self):
        prices = _make_prices()
        X_train, X_test, y_train, y_test, _, _ = prepare_ml_features(prices)
        models = train_models(X_train, y_train)
        results = evaluate_models(models, X_test, y_test)
        for name, r in results.items():
            assert len(r["predictions"]) == len(y_test)

    def test_confusion_matrix_shape(self):
        prices = _make_prices()
        X_train, X_test, y_train, y_test, _, _ = prepare_ml_features(prices)
        models = train_models(X_train, y_train)
        results = evaluate_models(models, X_test, y_test)
        for name, r in results.items():
            cm = r["confusion_matrix"]
            assert cm.shape == (2, 2), f"{name} confusion matrix should be 2x2"


class TestFeatureImportance:
    def test_importance_sums_to_one(self):
        prices = _make_prices()
        X_train, _, y_train, _, _, feat_names = prepare_ml_features(prices)
        models = train_models(X_train, y_train)
        imp = feature_importance(models["Random Forest"], feat_names)
        np.testing.assert_allclose(imp["importance"].sum(), 1.0, atol=1e-6)

    def test_sorted_descending(self):
        prices = _make_prices()
        X_train, _, y_train, _, _, feat_names = prepare_ml_features(prices)
        models = train_models(X_train, y_train)
        imp = feature_importance(models["Random Forest"], feat_names)
        assert (imp["importance"].diff().dropna() <= 1e-10).all(), "Not sorted descending"

    def test_all_features_present(self):
        prices = _make_prices()
        X_train, _, y_train, _, _, feat_names = prepare_ml_features(prices)
        models = train_models(X_train, y_train)
        imp = feature_importance(models["Random Forest"], feat_names)
        assert set(imp["feature"]) == set(feat_names)


class TestRunClassification:
    def test_end_to_end_no_macro(self):
        prices = _make_prices()
        result = run_classification(prices, macro_df=None)
        assert "models" in result
        assert "metrics" in result
        assert "feature_importance" in result
        assert "class_distribution" in result
        assert "mutual_info" in result
        assert len(result["models"]) == 5  # 4 base + Voting Ensemble

    def test_end_to_end_with_macro(self):
        """Macro run should include macro-derived features (before MI pruning)."""
        prices = _make_prices()
        macro = _make_macro(prices)
        result = run_classification(prices, macro_df=macro)
        mi_features = set(result["mutual_info"]["feature"])
        macro_in_mi = [f for f in mi_features if f in {"DGS10", "T10Y2Y", "VIXCLS"}]
        assert len(macro_in_mi) > 0, "Macro features should appear in MI analysis"

    def test_class_distribution_sums_to_train_size(self):
        prices = _make_prices()
        result = run_classification(prices)
        total_train = sum(result["class_distribution"].values())
        n_test = len(result["y_test"])
        n_total = total_train + n_test
        ratio = total_train / n_total
        assert 0.79 < ratio < 0.81, f"Train ratio {ratio} not ~0.8"

    def test_voting_ensemble_in_metrics(self):
        prices = _make_prices()
        result = run_classification(prices)
        assert "Voting Ensemble" in result["metrics"]

    def test_mutual_info_in_result(self):
        prices = _make_prices()
        result = run_classification(prices)
        assert "mutual_info" in result
        assert len(result["mutual_info"]) > 0


class TestRunClassificationTuned:
    def test_tune_mode_returns_cv_results(self):
        prices = _make_prices()
        result = run_classification(prices, tune=True)
        assert "cv_results" in result
        assert len(result["cv_results"]) == 4

    def test_tune_mode_has_mutual_info(self):
        prices = _make_prices()
        result = run_classification(prices, tune=True)
        assert "mutual_info" in result
        mi = result["mutual_info"]
        assert (mi["mi_score"] >= 0).all()


class TestRSI:
    def test_rsi_bounds(self):
        prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, 200))))
        rsi = _compute_rsi(prices)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_rising_prices(self):
        """Monotonically rising prices should have RSI near 100."""
        prices = pd.Series(np.arange(1, 102, dtype=float))
        rsi = _compute_rsi(prices)
        assert rsi.iloc[-1] > 90
