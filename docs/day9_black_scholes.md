# Day 9 — Black-Scholes Pricing, Greeks & Implied Volatility

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/pricing/black_scholes.py` | 164 | BS pricing engine, Greeks, IV solver, vol surface |
| `tests/test_pricing.py` | 152 | 29 tests covering pricing, Greeks, IV, edge cases |

## Module: `src/pricing/black_scholes.py`

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `price()` | `(S, K, T, r, sigma, option_type) -> np.ndarray` | Black-Scholes-Merton call/put price. Fully vectorized — accepts scalars or arrays with broadcasting. |
| `greeks()` | `(S, K, T, r, sigma) -> dict` | All first-order Greeks (delta, gamma, vega, theta, rho) for both calls and puts. |
| `implied_volatility()` | `(price_market, S, K, T, r, option_type) -> float` | Newton-Raphson IV solver with bisection fallback for robustness. |
| `vol_surface()` | `(S, strikes, expiries, r, option_prices, option_type) -> pd.DataFrame` | Implied vol surface from a grid of market prices. |

### Design Decisions

- **Fully vectorized**: All pricing and Greeks use numpy broadcasting — no Python for-loops in core computation.
- **d1/d2 clamping**: Values clamped to [-50, 50] to prevent `exp()` overflow on deep ITM/OTM options.
- **T=0 handling**: Returns intrinsic value `max(S-K, 0)` for calls, `max(K-S, 0)` for puts — never NaN.
- **Near-expiry Greeks**: When T < 1/365, delta snaps to 0 or 1 (ITM/OTM), gamma/theta/vega = 0.
- **IV solver**: Newton-Raphson (fast convergence) with automatic bisection fallback when vega is near zero. Returns `np.nan` on non-convergence.

### Edge Cases Handled

| Edge Case | Behavior |
|-----------|----------|
| T <= 0 (at expiry) | Returns intrinsic value |
| sigma <= 0 | Raises `ValueError` |
| S <= 0 or K <= 0 | Raises `ValueError` |
| Deep ITM/OTM (d1/d2 overflow) | Clamped to [-50, 50] |
| T < 1/365 (near-expiry Greeks) | Delta = 0 or 1, gamma/vega/theta = 0 |
| IV vega near zero | Bisection fallback in [0.001, 5.0] |
| IV non-convergence | Returns `np.nan` |
| IV with price_market <= 0 | Returns `np.nan` |

## Tests: `tests/test_pricing.py`

### Test Breakdown (29 tests)

| Category | Tests | What's Verified |
|----------|-------|-----------------|
| **BS Price** | 12 | Hull textbook value (S=42,K=40 -> 4.76), put-call parity, intrinsic at expiry, vectorized shapes, deep ITM bounds, input validation |
| **Greeks** | 10 | Delta bounds [0,1]/[-1,0], ATM delta ≈ 0.5, gamma/vega positive, theta negative, near-expiry snapping, put-call delta relation |
| **Implied Vol** | 5 | Call/put roundtrip recovery, high vol (σ=1.5), expired -> NaN, zero price -> NaN |
| **Vol Surface** | 2 | Correct shape, NaN on bad prices, IV recovery across grid |

### Key Validations

- **Hull textbook**: `price(42, 40, 0.5, 0.1, 0.2, 'call')` ≈ 4.76 (within 0.02)
- **Put-call parity**: `C - P = S - K*exp(-rT)` within 1e-6
- **IV roundtrip**: `implied_volatility(price(S,K,T,r,σ), S,K,T,r)` recovers σ within 1e-4

## Integration

The module is imported as:
```python
from src.pricing.black_scholes import price, greeks, implied_volatility, vol_surface
```

Used by:
- Day 10 (binomial tree) — for BS convergence comparison
- Day 11 (hedging) — for delta computation in hedging simulation
- Day 13 (Streamlit Tab 3) — derivatives pricing dashboard
- Day 15 (metrics) — can add BS pricing accuracy benchmarks
