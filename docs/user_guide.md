# Quantitative Finance Analytics Platform — User Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Dashboard Overview](#dashboard-overview)
5. [Tab 1: Portfolio Analytics](#tab-1-portfolio-analytics)
6. [Tab 2: Price Prediction](#tab-2-price-prediction)
7. [Tab 3: Derivatives Pricing](#tab-3-derivatives-pricing)
8. [Tab 4: Hedging Simulator](#tab-4-hedging-simulator)
9. [Sidebar Controls](#sidebar-controls)
10. [Running Tests](#running-tests)
11. [Generating Resume Metrics](#generating-resume-metrics)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before running the application, ensure you have the following installed on your machine:

| Requirement | Version | Why |
|-------------|---------|-----|
| Python | 3.11 or higher | Required for modern type hints used throughout the codebase |
| Conda or pip | Any recent | Package and environment management |
| Git | Any | To clone the repository |
| Internet access | Required on first launch | The app downloads 10 years of stock data from Yahoo Finance |

Optional:

| Requirement | Purpose |
|-------------|---------|
| FRED API Key | Enables macroeconomic overlays (yield curve, VIX, GDP, CPI) in the classification model. Free at https://fred.stlouisfed.org/docs/api/api_key.html |
| Docker | Alternative deployment method, no Python setup needed |

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd quant-platform
```

### Step 2: Create a Python Environment

**Using Conda (recommended):**

```bash
conda create -p ./quant python=3.11 -y
conda activate ./quant
```

**Using venv (alternative):**

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs 12 packages: numpy, pandas, yfinance, fredapi, statsmodels, scikit-learn, arch, scipy, streamlit, plotly, matplotlib, seaborn.

### Step 4: Configure FRED API Key (Optional)

If you have a FRED API key, save it so the app picks it up automatically:

```bash
# Create the secrets file (already gitignored, safe to store keys here)
echo 'FRED_API_KEY = "your_key_here"' > .streamlit/secrets.toml
```

Or you can enter the key in the sidebar when the app is running. The app works fully without it — macro features are simply disabled.

---

## Running the Application

### Launch the Dashboard

```bash
streamlit run src/ui/app.py
```

The terminal will show:

```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

Open **http://localhost:8501** in your browser.

### First Launch Behavior

1. The app fetches 10 years of daily price data (2015-2024) for 6 tickers: MS, JPM, BAC, AAPL, ^GSPC, ^IXIC
2. This takes approximately 10-15 seconds on the first load
3. Data is cached for 1 hour — subsequent tab switches and page refreshes are instant
4. If a FRED key is configured, macroeconomic data (treasury yields, VIX, GDP, CPI, USD index, oil) is also fetched and aligned

### Alternative: Run with Docker

```bash
docker build -t quant-platform .
docker run -p 8501:8501 quant-platform

# With FRED key
docker run -p 8501:8501 -e FRED_API_KEY=your_key quant-platform
```

### Stopping the Application

Press `Ctrl+C` in the terminal where Streamlit is running, or close the terminal.

---

## Dashboard Overview

The application has 4 tabs, each focused on a different area of quantitative finance. All tabs share the same underlying market data loaded once at startup.

| Tab | Focus Area | What It Answers |
|-----|-----------|-----------------|
| Portfolio Analytics | Risk measurement | How is our banking portfolio performing? What are the risks? |
| Price Prediction | Statistical modeling | Can we predict AAPL returns? Can we classify BAC direction? |
| Derivatives Pricing | Options valuation | What is this option worth? How do Greeks change with spot? |
| Hedging Simulator | Risk management | How effective is delta hedging? What rebalance band is optimal? |

---

## Tab 1: Portfolio Analytics

### Purpose

Evaluate the risk-return profile of an equally-weighted banking portfolio (Morgan Stanley, JPMorgan Chase, Bank of America) over the 2015-2024 period. This tab answers: **Is this portfolio worth holding? What are the tail risks?**

### What You See

**Key Metrics Row** (top of the tab):

| Metric | What It Means |
|--------|---------------|
| Sharpe Ratio | Return earned per unit of total risk. Higher is better. Above 1.0 is considered good. |
| Sortino Ratio | Like Sharpe, but only penalizes downside volatility. More relevant for loss-averse investors. |
| Beta | Sensitivity to the overall market. Beta > 1 means the portfolio amplifies market moves. |
| Alpha (annualized) | Excess return above what CAPM predicts. Positive alpha = outperformance. |
| Max Drawdown | Worst peak-to-trough loss. Shows the maximum pain an investor would have experienced. |

**Value at Risk Row**:

| Metric | What It Means |
|--------|---------------|
| Parametric VaR (95%) | Assuming normal returns, the worst daily loss with 95% confidence. |
| Historical VaR (95%) | The actual 5th percentile worst daily loss from historical data. |
| CVaR (Expected Shortfall) | Average loss on the worst 5% of days. Always worse than VaR. |

If parametric and historical VaR diverge significantly, a **fat tail warning** appears — meaning the normal distribution underestimates true risk.

**Charts**:

- **Cumulative Returns**: Growth of $1 invested in each stock and the portfolio. The white line is the portfolio. Compare against individual stocks to see diversification benefits.
- **Return Distribution**: Histogram of daily portfolio returns with VaR lines overlaid. The red dashed line is parametric VaR, orange dotted is historical VaR. Returns left of these lines represent tail risk events.
- **Rolling 30-Day Volatility**: Shows how portfolio risk changes over time. Spikes correspond to market stress events (e.g., COVID crash in March 2020).
- **Correlation Matrix**: Heatmap showing how correlated the banking stocks are with each other. High correlation (close to 1.0) means less diversification benefit.

**Expandable Section**: Click "Full Risk Summary" to see every computed metric in a table.

### How to Interpret

- A Sharpe near 0.4-0.5 is typical for a single-sector equity portfolio
- Max Drawdown above 50% reflects the banking sector's sensitivity to financial crises
- High inter-stock correlation (0.7-0.9) is expected for same-sector holdings

---

## Tab 2: Price Prediction

### Purpose

Apply statistical and machine learning models to real market data. The regression section asks: **Can lagged features predict AAPL's next-day return?** The classification section asks: **Can we predict whether BAC goes up or down tomorrow?**

### Section A: AAPL Regression

**What it does**: Fits an Ordinary Least Squares (OLS) regression to predict AAPL's next-day log return using 5 features: SMA_5, SMA_20, lagged AAPL return, lagged S&P 500, lagged NASDAQ.

**Metrics Row**:

| Metric | What It Means |
|--------|---------------|
| RMSE | Root Mean Squared Error — average prediction error magnitude. Lower is better. |
| R-squared | Proportion of variance explained by the model. Near 0 for daily returns is typical. |
| MAE | Mean Absolute Error — average absolute prediction miss. |

**Diagnostics Row** (statistical validity checks):

| Test | What It Checks | Good Result |
|------|---------------|-------------|
| Durbin-Watson | Serial correlation in residuals | Between 1.5 and 2.5 |
| Breusch-Pagan p | Whether error variance is constant | p > 0.05 |
| Jarque-Bera p | Whether residuals are normally distributed | p > 0.05 |
| Max VIF | Multicollinearity among features | Below 10 |

If heteroscedasticity is detected, a blue info banner appears indicating the model uses HC3 robust standard errors.

**Charts**:

- **Actual vs Predicted**: Points should cluster around the diagonal red line. Wide scatter means poor predictive power (expected for daily returns).
- **Residuals**: Should look like random noise centered at zero. Patterns indicate model misspecification.
- **QQ Plot**: Points along the diagonal = residuals are normally distributed. Curved tails = fat tails in returns.
- **VIF Bar Chart**: Bars above the red threshold line indicate features with multicollinearity issues.

### Section B: BAC Classification

**What it does**: Trains 5 machine learning models to predict whether BAC's price goes up (1) or down (0) tomorrow. Uses 19 engineered technical indicators, filtered down to ~8 via mutual information.

**Model Comparison Table**:

| Column | What It Means |
|--------|---------------|
| Accuracy | Percentage of correct predictions |
| Precision | Of predicted "up" days, what fraction actually went up |
| Recall | Of actual "up" days, what fraction did the model catch |
| F1 | Harmonic mean of precision and recall (balanced measure) |
| AUC-ROC | Area under the ROC curve. 0.5 = random guessing, 1.0 = perfect. |

Models included: Decision Tree, Random Forest, KNN, SVM, Voting Ensemble (soft vote of RF+KNN+SVM).

**Charts**:

- **ROC Curves**: Each model's true positive rate vs false positive rate. Curves above the gray diagonal line are better than random. Higher AUC = better.
- **Feature Importance**: Horizontal bar chart from the Random Forest model. Shows which technical indicators have the most predictive power for BAC direction.
- **Confusion Matrices**: For each base model, a 2x2 grid showing true positives, false positives, true negatives, false negatives.
- **Class Distribution**: Pie chart showing the up/down split in training data. Near 50/50 is ideal for balanced modeling.

### How to Interpret

- Daily stock returns are notoriously hard to predict. R-squared near 0 and accuracy near 50% is expected
- The value is in the methodology: proper chronological splits, diagnostic testing, and ensemble construction
- AUC-ROC above 0.5 indicates the models capture some signal beyond random chance

---

## Tab 3: Derivatives Pricing

### Purpose

Interactively price European and American options using two industry-standard models. This tab answers: **What is this option worth? How sensitive is the price to each input? How does the binomial tree converge to Black-Scholes?**

### Input Controls

| Control | What to Set |
|---------|-------------|
| Spot (S) | Current price of the underlying asset |
| Strike (K) | Exercise price of the option |
| Expiry (T) | Time to expiration in years (e.g., 0.25 = 3 months) |
| Risk-Free Rate | Annual risk-free interest rate (e.g., 0.05 = 5%) |
| Volatility | Annualized implied volatility (e.g., 0.20 = 20%) |
| Option Type | Call (right to buy) or Put (right to sell) |
| Pricing Method | Black-Scholes (closed-form) or Binomial CRR (tree-based) |
| Steps (N) | Number of tree steps for CRR (higher = more accurate, slower) |
| American | Toggle American exercise (early exercise allowed) |

### What You See

**Price Display**: The computed option price. If using CRR, also shows the BS reference price and the difference between them.

**Greeks Row**:

| Greek | What It Measures | Practical Meaning |
|-------|-----------------|-------------------|
| Delta | Price sensitivity to spot | If delta=0.6, option gains $0.60 per $1 spot increase |
| Gamma | Rate of change of delta | How quickly delta changes (risk of delta hedging) |
| Vega | Price sensitivity to volatility | Per 1% vol change |
| Theta | Time decay per day | How much value the option loses each calendar day |
| Rho | Price sensitivity to interest rates | Per 1% rate change |

**Charts**:

- **Payoff at Expiry**: Dashed line shows the intrinsic payoff at expiration (hockey stick shape). Solid line shows today's option value. The gap between them is time value.
- **Binomial Convergence**: CRR prices at N=10, 20, 50, 100, 200, 500 steps. The red horizontal line is the exact BS price. As N increases, CRR converges. Useful for understanding the accuracy-speed tradeoff.
- **Greeks Sensitivity**: Four subplots showing how Delta, Gamma, Vega, and Theta change as spot price varies. The gray dotted line marks the strike price. Notice how delta transitions from 0 to 1 (call) around the strike.
- **3D Price Surface**: A 3D mesh showing option price across a grid of strikes (x-axis) and expiries (y-axis). Deep in-the-money options have higher prices; longer expiries have more time value.

### Things to Try

- Set T=0 to see intrinsic value only (Greeks disabled)
- Compare BS vs CRR at N=10 (large difference) vs N=500 (nearly identical)
- Toggle American for a put option to see the early exercise premium
- Move the volatility slider to watch how all Greeks respond

---

## Tab 4: Hedging Simulator

### Purpose

Simulate a delta-neutral hedging strategy on real historical stock prices. This tab answers: **How effective is delta hedging in practice? How do transaction costs erode profits? What rebalance frequency is optimal?**

### Input Controls

| Control | What to Set |
|---------|-------------|
| Underlying | Stock ticker to hedge (AAPL, MS, JPM, BAC) |
| Option Type | Call or Put |
| Position | Short (sold the option, need to hedge) or Long |
| Strike (K) | Option strike price (defaults to current spot) |
| Expiry (T) | Option expiry in years |
| Risk-Free Rate | Annual rate used for BS delta calculation |
| Volatility | Implied vol used for hedging |
| Transaction Cost (bps) | Cost per trade in basis points (5 bps = 0.05%) |
| Rebalance Band | Delta threshold that triggers a rebalance (0.05 = rebalance when net delta exceeds 5 shares) |

Click **"Run Simulation"** to execute.

### What Happens Under the Hood

1. At day 0, the simulator computes the BS delta and buys/sells shares to neutralize the option's delta
2. Each subsequent day, it recalculates delta using the new spot price and remaining time
3. If the net delta exposure exceeds the rebalance band, it trades shares back to neutral
4. Transaction costs are deducted for each trade
5. The simulation tracks share P&L, option P&L, and costs separately

### Summary Metrics

| Metric | What It Means |
|--------|---------------|
| Net P&L | Final profit/loss after all costs |
| Transaction Costs | Total costs paid for rebalancing trades |
| Rebalances | Number of times the hedge was adjusted (and what % of total days) |
| Avg \|Delta\| | Average absolute delta exposure — lower means tighter hedging |
| PnL Sharpe | Risk-adjusted return of the hedging P&L stream |

A warning appears if transaction costs exceed 50% of gross profit.

### Charts

- **Spot Price & Rebalances**: The stock's price path over the simulation period. Orange triangles mark days when rebalancing occurred. The red dotted line shows the strike price.
- **Portfolio Delta**: The option's delta over time. Shows how delta changes as the option moves in/out of the money and approaches expiry.
- **Cumulative P&L Breakdown**: Stacked area chart showing the components: share P&L (from the stock hedge), option P&L, transaction costs (negative), and net P&L (white line). This decomposition shows where money is made and lost.

### Optimal Band Search (Expandable Section)

Click the expander to run simulations across 5 different rebalance bands (0.01, 0.02, 0.05, 0.10, 0.20). Results show:

- **Table**: Net P&L, rebalances, costs, and Sharpe for each band
- **Chart**: Dual-axis bar+line chart — bars show P&L per band, orange line shows rebalance count

**Key insight**: Tighter bands (0.01) mean more rebalances and higher costs but tighter delta control. Wider bands (0.20) mean fewer trades but larger unhedged exposure. The optimal band depends on the cost-accuracy tradeoff.

### Things to Try

- Compare a tight band (0.01) vs wide band (0.20) — watch costs vs P&L
- Try hedging a put vs a call on the same underlying
- Increase transaction costs to 15-20 bps to see cost impact
- Use a shorter expiry (0.1 years) to see near-expiry delta behavior (delta snaps to 0 or 1)

---

## Sidebar Controls

The left sidebar (visible on all tabs) has two controls:

| Control | Purpose | Default |
|---------|---------|---------|
| FRED API Key | Paste your key to enable macro data. Stored in session only. | Empty (from secrets.toml if configured) |
| Risk-Free Rate | Used by the Portfolio Analytics tab for Sharpe/Sortino calculations | 0.020 (2%) |

---

## Running Tests

The project includes 171 automated tests covering every module:

```bash
# Run the full suite (~90 seconds)
pytest tests/ -v

# Run a specific module's tests
pytest tests/test_pricing.py -v
pytest tests/test_volatility.py -v
pytest tests/test_hedging.py -v

# Run a single test by name
pytest tests/test_pricing.py::TestBSPrice::test_hull_call -v
```

All tests use synthetic or deterministic data — no network calls required.

---

## Generating Resume Metrics

The metrics script runs all analytics modules against real market data and compares results to baselines:

```bash
python scripts/generate_metrics.py
```

**Output**:
- Prints ~11 resume-ready bullet points with real numbers to the terminal
- Saves detailed JSON to `results/metrics_report.json`

This script requires internet access (fetches live data from Yahoo Finance).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src'` | Make sure you are running from the project root directory: `cd quant-platform` then `streamlit run src/ui/app.py` |
| App shows "Loading market data..." forever | Check your internet connection. yfinance needs to download stock data on first run. |
| `No data found for this date range` | Yahoo Finance occasionally has outages. Wait a few minutes and retry. |
| Slow first load (10-15 seconds) | Normal behavior. Data is cached for 1 hour after the first fetch. |
| `arch` package fails to install | Run `pip install --upgrade pip setuptools wheel` first, then retry. |
| Port 8501 already in use | Either kill the existing process (`pkill -f streamlit`) or run on a different port: `streamlit run src/ui/app.py --server.port 8502` |
| Charts not rendering | Ensure plotly is installed: `pip install plotly>=5.22`. Try a different browser if the issue persists. |
| `DuplicateWidgetID` error | This was fixed. If you still see it, make sure you have the latest code (`git pull`). |
| Memory issues on Streamlit Cloud | The app is optimized for 1GB RAM: float32 plotting, lazy tab loading, capped 3D surface grids (20x15). |
| FRED features not working | Verify your API key is correct. Enter it in the sidebar or save to `.streamlit/secrets.toml`. |
