---
aliases: [Technical Features, Technical Analysis Features]
tags: [machine-learning, feature-engineering, technical-analysis, first-principles]
---

# Technical Indicators as Features

## First Principles

Technical indicators attempt to extract **patterns from price and volume data** that might predict future moves. While the [[Efficient Market Hypothesis]] suggests these shouldn't work (prices already reflect all information), they're widely used and this project tests them empirically.

## Indicators Used in BAC Classification

### Momentum Indicators

**RSI (Relative Strength Index)** — Window: 14 days
$$RSI = 100 - \frac{100}{1 + RS}, \quad RS = \frac{EMA(gains)}{EMA(losses)}$$
Ranges 0-100. RSI > 70 = overbought, RSI < 30 = oversold. Implemented using vectorized `ewm()`.

**MACD (Moving Average Convergence Divergence)**
- MACD Line = EMA(12) - EMA(26)
- Signal Line = EMA(9) of MACD Line
- Histogram = MACD - Signal

MACD crossovers (line crossing signal) are classic momentum signals.

**Rate of Change (ROC)** — Periods: 10, 20 days
$$ROC = \frac{P_{today} - P_{n\text{ days ago}}}{P_{n\text{ days ago}}}$$
Simple momentum measure — positive ROC means the price is higher than $n$ days ago.

### Mean-Reversion Indicators

**Bollinger Band %B** — Window: 20, 2σ
$$\%B = \frac{P - Lower}{Upper - Lower}$$
Measures where the price sits within the Bollinger Bands. %B > 1 = above upper band (potentially overbought), %B < 0 = below lower band.

**Z-Score** = $(P - SMA_{20}) / \sigma_{20}$ — How many standard deviations the price is from its 20-day mean.

**Stochastic Oscillator** — K period: 14, D period: 3
$$\%K = 100 \times \frac{P - Low_{14}}{High_{14} - Low_{14}}$$
Measures momentum relative to the recent price range.

### Trend Indicators

**SMA Ratio** = $SMA_5 / SMA_{20}$ — Short-term vs medium-term trend. Ratio > 1 = uptrend.

**Lagged Returns** — 1-5 day lags of log returns. Tests whether recent returns predict future direction (momentum or reversal).

### Volatility Indicator

**Realized Volatility (20d)** = Rolling 20-day std × √252. High vol environments may behave differently than low vol.

### Macro Features (Optional)

When a FRED API key is provided, macro features from [[Data Ingestion Layer]] are added: 10Y Treasury, yield curve, VIX, GDP, CPI, USD index, WTI oil. These are aligned via [[Asof Join and Look-Ahead Bias]].

## Connections

- [[ML Classification Pipeline]] — These features feed the classification models
- [[Feature Selection (Mutual Information)]] — Tests which indicators actually carry signal
- [[OLS Regression with Diagnostics]] — Uses simpler features (lagged returns, vol, market return)
- [[Efficient Market Hypothesis]] — The empirical test of whether these indicators predict anything
