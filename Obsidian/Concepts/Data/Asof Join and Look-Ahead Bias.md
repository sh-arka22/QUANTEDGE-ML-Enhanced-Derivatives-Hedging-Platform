---
aliases: [Look-Ahead Bias, Data Alignment, merge_asof]
tags: [data, methodology, first-principles]
---

# Asof Join and Look-Ahead Bias

## First Principles: What Is Look-Ahead Bias?

Look-ahead bias occurs when a model uses information that **wouldn't have been available** at the time of prediction. For example, using Friday's GDP release to predict Wednesday's stock return is cheating — even though both data points exist in your historical dataset.

This is the most dangerous source of inflated backtest performance.

## The Challenge with Macro Data

Macro data arrives at different frequencies than daily equity data:
- **GDP**: Quarterly (released ~1 month after quarter end)
- **CPI**: Monthly
- **VIX**: Daily (but still on a different schedule)

Naively joining by date would either create NaN gaps or misalign timestamps.

## Solution: `pd.merge_asof()` with Backward Direction

In `src/data/loaders.py`, `align_data()`:

```python
pd.merge_asof(equity, macro, on="date", direction="backward", tolerance=pd.Timedelta("90d"))
```

**What this does**: For each equity date, find the **most recent** macro observation that occurred on or before that date (within 90 days). This guarantees you only use macro data that was actually available at the time.

### Additional Safeguards
1. Both DataFrames are sorted by index and deduplicated
2. Sparse macro series (quarterly GDP) are forward-filled after joining
3. Rows where ALL macro columns are NaN are dropped
4. Minimum 100 rows validation after alignment

## The Same Principle Applies to Features

The [[OLS Regression with Diagnostics]] and [[ML Classification Pipeline]] both enforce chronological data handling:
- **Train/test split**: Chronological, not random (80/20 temporal split)
- **Winsorization**: Only uses training set percentiles
- **[[Hyperparameter Tuning (TimeSeriesSplit)]]**: Expanding window CV
- **Feature lags**: All features use `shift(1)` or more — no same-day information

## Connections

- [[Data Ingestion Layer]] — `align_data()` is part of the data pipeline
- [[ML Classification Pipeline]] — Chronological split and forward-fill in classification features
- [[OLS Regression with Diagnostics]] — Chronological split and winsorization
- [[Hyperparameter Tuning (TimeSeriesSplit)]] — Temporal CV to prevent leakage
