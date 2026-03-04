---
aliases: [CRR Model, Cox-Ross-Rubinstein, Binomial Pricing]
tags: [derivatives, pricing, lattice-methods, first-principles]
---

# Binomial Tree (CRR)

## First Principles: The Discrete Approach

The Cox-Ross-Rubinstein (CRR) binomial tree is a **discrete-time model** for option pricing. Instead of assuming the stock price evolves continuously (as in [[Black-Scholes Model]]), we model it as taking one of two possible moves at each time step: **up** by factor $u$ or **down** by factor $d$.

This is conceptually simpler and more flexible than BS — it can price **American options** (early exercise) and options with path-dependent features.

## How the Tree Works

### Step 1: Build the Price Tree
Divide the option's life $T$ into $N$ equal time steps of length $\Delta t = T/N$.

At each step, the stock can go:
- **Up**: $S \to S \cdot u$ where $u = e^{\sigma\sqrt{\Delta t}}$
- **Down**: $S \to S \cdot d$ where $d = 1/u$ (CRR symmetry)

After $N$ steps, there are $N+1$ possible terminal prices: $S \cdot u^{N-2j}$ for $j = 0, 1, \ldots, N$.

### Step 2: Risk-Neutral Probability
The probability of an up-move under the risk-neutral measure:
$$p = \frac{e^{r\Delta t} - d}{u - d}$$

This must satisfy $0 < p < 1$ for no-arbitrage.

### Step 3: Backward Induction
At terminal nodes, option value = intrinsic value: $\max(S_T - K, 0)$ for calls.

Then roll backward: at each node, the option value is the discounted expected value under risk-neutral probabilities:
$$V = e^{-r\Delta t}[p \cdot V_{up} + (1-p) \cdot V_{down}]$$

For **American options**, at each node we also check if early exercise is optimal:
$$V = \max(V_{continuation}, V_{exercise})$$

### Step 4: Convergence
As $N \to \infty$, the CRR price converges to the [[Black-Scholes Model]] price for European options. This project demonstrates this visually in [[Tab 3 - Derivatives Pricing]].

## How This Project Implements It

In `src/pricing/binomial.py`:

### `crr_price()` — Vectorized Backward Induction
- Terminal payoffs are computed as a NumPy array for all $N+1$ nodes simultaneously
- The backward induction loop is vectorized **within each step**: `V = disc * (p * V[:-1] + (1-p) * V[1:])` — this avoids Python loops over nodes
- Memory is $O(N)$ not $O(N^2)$ since only one level of the tree is stored at a time
- `MAX_TREE_STEPS = 1000` cap prevents excessive computation

### `crr_greeks()` — Numerical Greeks from Tree Values
Instead of analytical formulas, Greeks are computed from the first few tree nodes:
- **Delta**: $\frac{f_u - f_d}{S_u - S_d}$ (first-order finite difference)
- **Gamma**: Second-order finite difference using $f_{uu}, f_{ud}, f_{dd}$
- **Theta**: $\frac{f_{ud} - f_0}{2\Delta t \cdot 365}$ (time decay per calendar day)

This provides an independent check against the analytical [[Option Greeks]] from BS.

### `american_premium()` — Early Exercise Value
Computes the difference between American and European prices to quantify the value of early exercise. For calls on non-dividend-paying stocks, this premium should be zero (it's never optimal to exercise early). For puts, the premium is positive — especially deep ITM.

## Connections

- [[Black-Scholes Model]] — CRR converges to BS; BS provides the reference price
- [[Option Greeks]] — CRR Greeks (numerical) vs BS Greeks (analytical) comparison
- [[Tab 3 - Derivatives Pricing]] — Convergence chart shows CRR approaching BS as N grows
- [[Put-Call Parity]] — American put premium breaks the European put-call parity relationship
