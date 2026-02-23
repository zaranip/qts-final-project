# Trading Strategy Ideas

## 1. Minimum Spanning Tree (MST) Topology Trading

**Intuition:** Construct the MST of the asset correlation/distance matrix using Kruskal's algorithm. Assets at the periphery (leaf nodes) are most independent; assets at the center (hubs) are most connected. When hub-to-leaf return spreads diverge, mean-revert the relationship.

**Mathematical Theory:**

Compute the Mantegna distance metric from pairwise correlations rho_ij:

    d_ij = sqrt(2 * (1 - rho_ij))

This satisfies metric axioms: d in [0, 2], symmetry, triangle inequality. Apply Kruskal's algorithm: sort all N*(N-1)/2 edges by d_ij ascending, greedily add edges that do not form a cycle until N-1 edges are selected.

Centrality on the resulting tree:
- Degree centrality: k_i = number of MST edges incident to node i
- Betweenness centrality: B_i = sum_{s,t} sigma_{st}(i) / sigma_{st}

**Signal:** Recompute MST on a rolling 252-day return window. When the spread between a hub node (top-3 betweenness) and a leaf node (degree=1) in the same subtree exceeds 2 historical sigma, mean-revert: long the laggard, short the leader. Hold across 10+ assets (currencies, commodities, or equity sectors) simultaneously.

**Key Papers:**
- Mantegna, R. (1999). *Hierarchical Structure in Financial Markets.* European Physical Journal B, 11(1), 193–197.
- Onnela, J.P., Chakraborti, A., Kaski, K. & Kertesz, J. (2003). *Dynamics of Market Correlations: Taxonomy and Portfolio Analysis.* Physical Review E, 68, 056110.
- Pozzi, F., Di Matteo, T. & Aste, T. (2013). *Spread of Risk Across Financial Markets.* Scientific Reports, 3, 1665.

---

## 2. Crypto Perpetual Futures Funding Rate Carry

**Intuition:** Perpetual futures use a funding rate mechanism: longs pay shorts when the market is in contango (funding > 0). A delta-neutral cash-and-carry trade — long spot, short perpetual — earns the funding rate as income when rates are persistently positive.

**Mathematical Theory:**

Perpetual contract funding rate (8-hour interval):

    FR(t) = clamp(premium_index(t) + clamp(rate - premium_index(t), -0.05%, 0.05%), -0.075%, 0.075%)

where premium_index(t) = (F_perp(t) - S(t)) / S(t). Annualized funding yield:

    annualized_carry = FR * 3 * 365

Portfolio carries positions in N cryptocurrencies. Total annualized P&L:

    PnL = sum_i notional_i * FR_i * 3 * 365 - sum_i transaction_costs_i

Basis risk: model the spread basis(t) = S(t) - F_perp(t) as AR(1):

    basis(t) = phi * basis(t-1) + epsilon(t)

Position sizing: inverse volatility of basis changes.

**Signal:** Enter long spot + short perp for top-5 coins by 7-day average annualized funding rate, when rate > 10% annualized. Hold until funding rate drops below 5% or basis diverges > 3 sigma.

**Key Papers:**
- Liu, Y., Tsyvinski, A. & Wu, X. (2022). *Common Risk Factors in Cryptocurrency.* Journal of Finance, 77(2), 1133–1177.
- Alexander, C. & Heck, D. (2020). *Price Discovery in Bitcoin: The Impact of Unregulated Markets.* Journal of Financial Stability, 50, 100776.
- Deribit Research. (2020). *Perpetual Swaps: The Mechanics of Crypto Derivatives Funding Rates.* Deribit Insights.

---

## 3. Cross-Sectional Option-Implied Skewness

**Intuition:** Stocks with high option-implied skewness (lottery-like payoffs) are overpriced and earn low future returns — investors overpay for positive skewness. Sort stocks by model-free implied skewness; short high-skewness stocks, long low-skewness stocks.

**Mathematical Theory:**

Model-free implied skewness (Bakshi, Kapadia & Madan 2003):

    SKEW_i(t) = [e^{r*tau} * W(t,tau) - 3*mu_i * e^{r*tau} * V(t,tau) + 2*mu_i^3] / [e^{r*tau} * V(t,tau) - mu_i^2]^{3/2}

where V, W are cubic contracts extracted from OTM option prices via Breeden-Litzenberger:

    q(K) = e^{r*tau} * d^2 C(K) / dK^2   (risk-neutral density from option surface)

Fama-MacBeth cross-sectional regression at each month t:

    R_i(t+1) = a_t + b_t * SKEW_i(t) + c_t * size_i + d_t * B/M_i + epsilon_i

Empirically, b_t < 0: high-skewness stocks underperform by approximately 10% per year (Conrad et al. 2013).

**Signal:** Monthly rebalance. Compute 30-day model-free implied skewness for all optionable stocks. Long bottom-quintile skewness, short top-quintile. Require at least 5 OTM puts and 5 OTM calls per stock. Hold 30+ positions per leg.

**Key Papers:**
- Bakshi, G., Kapadia, N. & Madan, D. (2003). *Stock Return Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options.* Review of Financial Studies, 16(1), 101–143.
- Boyer, B., Mitton, T. & Vorkink, K. (2010). *Expected Idiosyncratic Skewness.* Review of Financial Studies, 23(1), 169–202.
- Conrad, J., Dittmar, R. & Ghysels, E. (2013). *Ex Ante Skewness and Expected Stock Returns.* Journal of Finance, 68(1), 85–124.
