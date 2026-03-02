# Day 10 — Vectorized CRR Binomial Tree

## Files Created/Modified

| File | Lines | Purpose |
|------|-------|---------|
| `src/pricing/binomial.py` | 141 | CRR pricing, Greeks, American premium |
| `tests/test_pricing.py` | +98 | 18 new tests (47 total in file) |

## Module: `src/pricing/binomial.py`

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `crr_price()` | `(S, K, T, r, sigma, N, option_type, american) -> float` | CRR binomial tree price. Backward induction loop with vectorized array slicing at each step. |
| `crr_greeks()` | `(S, K, T, r, sigma, N, option_type, american) -> dict` | Delta, gamma, theta from the tree at steps 1 and 2. |
| `american_premium()` | `(S, K, T, r, sigma, N, option_type) -> float` | Early exercise premium = American price - European price. |

### CRR Parameterization
```
dt = T / N
u  = exp(sigma * sqrt(dt))
d  = 1 / u
p  = (exp(r * dt) - d) / (u - d)
```

### Vectorization Strategy

- **Terminal prices**: Single vectorized expression `S * u^(N - 2*j)` for `j = 0..N`
- **Backward induction**: One Python loop over time steps, but each step uses vectorized array slicing: `V = disc * (p * V[:-1] + (1-p) * V[1:])`
- **American early exercise**: Vectorized comparison `np.maximum(V, exercise)` at each step
- No nested loops — the inner operation on `N+1` element arrays is fully vectorized

### Design Decisions

- **N capped at MAX_TREE_STEPS=1000**: Prevents memory issues on Streamlit Cloud (1GB limit). Warns, doesn't error.
- **Greeks from tree**: Delta/gamma computed from step-1 and step-2 values stored during backward pass, not via finite differences on price().
- **T=0**: Returns intrinsic value for `crr_price()`, returns NaN for `crr_greeks()`.
- **Risk-neutral probability check**: Raises ValueError if p is outside (0,1) — catches degenerate parameter combinations.

### Edge Cases Handled

| Edge Case | Behavior |
|-----------|----------|
| T <= 0 | Returns intrinsic (price), NaN (Greeks) |
| sigma <= 0 | Raises `ValueError` |
| N < 1 | Raises `ValueError` |
| N > 1000 (MAX_TREE_STEPS) | Warns and caps |
| p outside (0,1) | Raises `ValueError` |

## Tests Added (18 new tests)

| Category | Tests | What's Verified |
|----------|-------|-----------------|
| **CRR Price** | 10 | BS convergence (call+put at N=500 within 0.5%), intrinsic at T=0, American put >= European, American call ≈ European (no dividend), input validation, N capping, deep ITM put, Hull example |
| **CRR Greeks** | 5 | Delta bounds [0,1]/[-1,0], gamma positive, delta matches BS at N=500, T=0 -> NaN |
| **American Premium** | 3 | Put premium > 0 for ITM, call premium ≈ 0 (no dividend), premium increases with rate |

### Key Validations

- **BS convergence**: `crr_price(100,100,1,0.05,0.2, N=500, 'call')` matches `bs.price()` within 0.5%
- **American >= European**: Always holds for puts (early exercise value)
- **Hull example**: American put S=50,K=52,T=2,r=0.05,sigma=0.3 in range [7.0, 9.0]
- **CRR delta matches BS delta**: Within 0.01 at N=500

## Integration

```python
from src.pricing.binomial import crr_price, crr_greeks, american_premium
```

Used by:
- Day 11 (hedging) — optional American option hedging
- Day 13 (Streamlit Tab 3) — binomial pricing + convergence chart vs BS
- Day 15 (metrics) — can benchmark CRR convergence rate
