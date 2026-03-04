---
aliases: [BSM, Black-Scholes-Merton, BS Model]
tags: [derivatives, pricing, first-principles, quantitative-finance]
---

# Black-Scholes Model

## First Principles: What Problem Does It Solve?

Before Black-Scholes, there was no agreed-upon way to price options. The fundamental question is: **what is the fair price of the right (but not obligation) to buy/sell an asset at a fixed price in the future?**

The key insight is **risk-neutral valuation**: if you can perfectly replicate the option payoff by continuously trading the underlying stock and a risk-free bond, then the option's price must equal the cost of this replicating portfolio — otherwise there's an arbitrage.

## The Core Assumptions

1. The stock price follows **geometric Brownian motion**: $dS = \mu S \, dt + \sigma S \, dW$
2. **No arbitrage** — you can't make risk-free profit above the risk-free rate
3. **Continuous trading** with no transaction costs
4. **Constant** volatility $\sigma$ and risk-free rate $r$
5. The stock pays **no dividends** (the basic model)
6. **European exercise** only (at expiry)

## The Formula

For a European call option:

$$C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$

Where:
$$d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**Intuition**: $N(d_2)$ is the risk-neutral probability the option expires in-the-money. $S \cdot N(d_1)$ is the expected value of receiving the stock, discounted. $K e^{-rT} \cdot N(d_2)$ is the present value of paying the strike.

For a put: use [[Put-Call Parity]] or substitute $N(-d_1)$ and $N(-d_2)$.

## How This Project Implements It

The implementation in `src/pricing/black_scholes.py` is **fully vectorized** using NumPy — it can price entire grids of options simultaneously.

### Key Design Decisions

1. **d1/d2 clamping**: Values are clipped to `[-50, 50]` to prevent numerical overflow in `norm.cdf()` for deep ITM/OTM options.

2. **Expiry handling**: When $T \leq 0$, the function returns intrinsic value directly instead of applying the BS formula (which would divide by zero via $\sqrt{T}$).

3. **Vectorization**: All inputs are broadcast-compatible NumPy arrays, enabling the [[Volatility Surface]] computation and [[Tab 3 - Derivatives Pricing]] 3D price surface to evaluate thousands of (K, T) pairs in one call.

### Functions Provided

- `price(S, K, T, r, sigma, option_type)` — The BS price, fully vectorized
- `greeks(S, K, T, r, sigma)` — All [[Option Greeks]] in a single pass
- `implied_volatility(...)` — Newton-Raphson + bisection fallback (see [[Implied Volatility]])
- `vol_surface(...)` — Grid of IVs across strikes and expiries (see [[Volatility Surface]])

## Connections

- The BS price serves as the **reference benchmark** for the [[Binomial Tree (CRR)]] convergence test
- [[Option Greeks]] are derived as partial derivatives of the BS formula
- [[Implied Volatility]] inverts the BS formula to recover $\sigma$ from market prices
- The [[Delta-Neutral Hedging]] simulator uses BS delta to determine hedge ratios
- [[GARCH(1,1) Model]] output can feed into BS as a time-varying $\sigma$ estimate
