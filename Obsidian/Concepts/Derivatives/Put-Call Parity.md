---
aliases: [PCP]
tags: [derivatives, no-arbitrage, first-principles]
---

# Put-Call Parity

## First Principles

Put-call parity is a **no-arbitrage relationship** between the prices of European calls and puts with the same strike and expiry:

$$C - P = S - K e^{-rT}$$

**Intuition**: Holding a call and selling a put (with the same K, T) has the same payoff as holding the stock and owing $K$ at expiry. Since the payoffs are identical, the prices must be equal — otherwise arbitrageurs would exploit the mismatch.

## Why It Matters in This Project

1. **Model Validation**: The [[Black-Scholes Model]] implementation satisfies put-call parity by construction, since calls and puts are priced from the same $d_1, d_2$ values.

2. **American Options**: Put-call parity does **not** hold for American options because early exercise creates value asymmetry. This is why the [[Binomial Tree (CRR)]] `american_premium()` function only measures a meaningful premium for puts (not calls on non-dividend stocks).

3. **Pricing the Put**: Given a call price, you can derive the put price (or vice versa) without re-running the full pricing model.

## Connections

- [[Black-Scholes Model]] — PCP is satisfied by construction in BS
- [[Binomial Tree (CRR)]] — American exercise breaks PCP for puts
- [[Option Greeks]] — Delta of call minus delta of put equals 1 (another PCP implication)
