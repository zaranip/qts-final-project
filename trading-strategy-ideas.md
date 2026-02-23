# Trading Strategy Ideas

---

## 1. Minimum Spanning Tree (MST) Topology Trading

**Intuition:** Construct the MST of an asset universe using pairwise return correlations. The
MST encodes the dominant structural relationships in the market. When the *effective resistance
distance* between two connected assets increases — meaning their relationship has become more
diffuse — the pair is likely to mean-revert back toward the topology-implied equilibrium.
The signal is continuous and derived from the spectral properties of the MST Laplacian, not
from rank sorting.

**Mathematical Theory:**

Compute the Mantegna distance metric from pairwise Pearson correlations rho_ij estimated
over a rolling 252-day window:

    d_ij = sqrt(2 * (1 - rho_ij))

This satisfies metric axioms (d in [0, 2], symmetry, triangle inequality). Apply Kruskal's
algorithm: sort all N*(N-1)/2 pairs by d_ij ascending, greedily add edges that do not form
a cycle until N-1 edges are chosen. The result is the MST T*.

Construct the graph Laplacian of T*:

    L_ii = deg(i),    L_ij = -1 if (i,j) in T*,    L_ij = 0 otherwise

The *effective resistance* (Kirchhoff distance) between nodes i and j on the tree is:

    R_ij = (e_i - e_j)^T * L^+ * (e_i - e_j)

where L^+ is the Moore-Penrose pseudoinverse of L. For a tree, this simplifies to the sum
of edge weights along the unique path from i to j:

    R_ij = sum_{e in path(i,j)} d_e

R_ij is a proper metric and encodes structural distance beyond mere pairwise correlation.

The *Fiedler value* lambda_2 (second-smallest eigenvalue of L) measures the algebraic
connectivity of the MST — how "tightly" the market is coupled:

    lambda_2 = min_{x: x perp 1, ||x||=1}  x^T L x

When lambda_2 falls sharply, the market is fragmenting; when it rises, linkages are tightening.

**Trading Signal:**

For each edge (i, j) in the current MST, track the *rolling effective resistance* R_ij(t)
and its historical mean mu_R and standard deviation sigma_R over the past 60 windows:

    z_ij(t) = (R_ij(t) - mu_R_ij) / sigma_R_ij

A positive z_ij means the pair's tree-distance has expanded — they have become more
structurally decoupled than usual — which predicts reversion.

The *continuous* position weight for pair (i, j) is:

    signal_ij(t) = -z_ij(t) * exp(-R_ij(t) / R_bar)

where R_bar is the cross-sectional mean effective resistance. The exponential decay
discounts pairs that are structurally far apart even in normal times (less reliable
mean-reversion). For each pair, go long the lagging asset and short the leading one,
with dollar allocation proportional to |signal_ij(t)|, normalized to target portfolio
volatility sigma_target:

    w_ij(t) = sigma_target * signal_ij(t) / sigma_ij(t)

where sigma_ij(t) is the realized volatility of the spread r_i - beta_ij * r_j.

Additionally, scale down all positions when lambda_2(t) < lambda_2_bar - 1.5*sigma_{lambda_2}
(market fragmenting — mean-reversion less reliable).

Recompute MST and effective resistances weekly. Hold 10+ simultaneous pairs spanning
multiple asset classes (equity sectors, currencies, commodities).

**Key Papers:**
- Mantegna, R. (1999). *Hierarchical Structure in Financial Markets.* European Physical Journal B, 11(1), 193-197.
- Onnela, J.P., Chakraborti, A., Kaski, K. & Kertesz, J. (2003). *Dynamics of Market Correlations: Taxonomy and Portfolio Analysis.* Physical Review E, 68, 056110.
- Klein, D. & Randic, M. (1993). *Resistance Distance.* Journal of Mathematical Chemistry, 12(1), 81-95.
- Pozzi, F., Di Matteo, T. & Aste, T. (2013). *Spread of Risk Across Financial Markets.* Scientific Reports, 3, 1665.

---

## 2. Crypto Perpetual Futures Funding Rate Carry

**Intuition:** Perpetual futures funding rates embed a time-varying risk premium paid by
directional speculators to hedgers. The observable rate FR(t) is a noisy mixture of a
persistent structural component (the "true" carry premium) and transient noise from
short-term crowding. A Kalman filter decomposes these two components in real time. We
size positions proportional to the filtered persistent signal — not by ranking coins or
using raw thresholds — so the carry exposure scales continuously with the estimated
signal strength.

**Mathematical Theory:**

Model the observed 8-hour funding rate FR_i(t) for coin i as a state-space system. The
latent persistent carry premium mu_i(t) evolves as a random walk with drift:

    mu_i(t) = mu_i(t-1) + eta_i(t),       eta_i ~ N(0, sigma_eta^2)

The observed funding rate includes transient noise v_i(t):

    FR_i(t) = mu_i(t) + v_i(t),           v_i ~ N(0, sigma_v^2)

This is the standard local-level (Kalman) model. The Kalman filter recursion gives the
optimal (minimum MSE) estimate of mu_i(t) given FR_i(1), ..., FR_i(t):

    Prediction:
        mu_hat_i(t|t-1) = mu_hat_i(t-1|t-1)
        P(t|t-1) = P(t-1|t-1) + sigma_eta^2

    Update (Kalman gain K_i(t)):
        K_i(t) = P(t|t-1) / (P(t|t-1) + sigma_v^2)
        mu_hat_i(t|t) = mu_hat_i(t|t-1) + K_i(t) * (FR_i(t) - mu_hat_i(t|t-1))
        P(t|t) = (1 - K_i(t)) * P(t|t-1)

Estimate the noise variances sigma_eta^2 and sigma_v^2 by maximum likelihood over a
rolling 180-day training window:

    log L = -T/2 * log(2*pi) - 1/2 * sum_t [log(F_t) + v_t^2 / F_t]

where F_t = P(t|t-1) + sigma_v^2 is the innovation variance and v_t = FR_i(t) - mu_hat_i(t|t-1).

The *signal-to-noise ratio* (SNR) at each period:

    SNR_i(t) = mu_hat_i(t|t) / sqrt(P(t|t))

This is the t-statistic of the Kalman estimate of the persistent carry. It is high when
the filter is confident the structural funding rate is elevated.

**Trading Signal:**

The continuous position weight for coin i at time t is:

    w_i(t) = tanh(SNR_i(t) / c) / sigma_basis_i(t)

where sigma_basis_i(t) = std(FR_i(t) - mu_hat_i(t|t), 30-period rolling) measures
basis noise, and c is a scaling constant chosen so that tanh saturates at 3-sigma events.
The tanh function provides a bounded, smooth, monotone mapping from signal to position —
avoiding the discontinuity of threshold rules and the hard bucketing of quintiles.

The position is: long spot_i, short perp_i, with notional = w_i(t) * total_capital.
Positions are opened/closed continuously as SNR_i(t) rises and falls.

Portfolio-level funding P&L over period [t, t+1]:

    PnL(t) = sum_i w_i(t) * FR_i(t+1) * notional_i - transaction_costs_i

Risk control: if the Kalman innovation |FR_i(t) - mu_hat_i(t|t-1)| > 4*sqrt(F_t)
(outlier funding spike), treat as a structural break — reset the filter and reduce
position to 25% until re-estimation converges.

**Key Papers:**
- Liu, Y., Tsyvinski, A. & Wu, X. (2022). *Common Risk Factors in Cryptocurrency.* Journal of Finance, 77(2), 1133-1177.
- Harvey, A., Ruiz, E. & Shephard, N. (1994). *Multivariate Stochastic Variance Models.* Review of Economic Studies, 61(2), 247-264.
- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods.* Oxford University Press.
- Alexander, C. & Heck, D. (2020). *Price Discovery in Bitcoin: The Impact of Unregulated Markets.* Journal of Financial Stability, 50, 100776.

---

## 3. Cross-Sectional Option-Implied Skewness

**Intuition:** Stocks with high option-implied risk-neutral skewness are overpriced by
investors with a preference for lottery payoffs. Rather than sorting into buckets, we
extract the *full risk-neutral density* for each stock from its option surface via
Breeden-Litzenberger, compute model-free skewness, and then construct a mean-variance
optimal portfolio where expected excess returns are a linear function of the skewness
signal. The optimal weights follow from inverting the sample covariance matrix — a
continuous, risk-aware allocation that fully uses the cross-sectional signal strength.

**Mathematical Theory:**

**Step 1 — Extract the Risk-Neutral Density.**
For each stock i with options expiring at horizon tau, the risk-neutral density q_i(K)
is recovered from call prices C_i(K) via Breeden-Litzenberger (1978):

    q_i(K) = e^{r*tau} * d^2 C_i(K) / dK^2

In practice, fit a smoothing spline to the observed implied volatility smile
sigma_iv(K, tau) in strike space, convert to call prices, then differentiate twice
numerically to obtain q_i(K) on a fine strike grid [K_min, K_max].

**Step 2 — Compute Model-Free Moments.**
The risk-neutral mean, variance, and third central moment are:

    mu_i^Q    = e^{r*tau} - 1        (risk-neutral mean; trivially = risk-free rate)
    V_i(t,tau) = integral (ln(K/F_i))^2 * q_i(K) dK
    W_i(t,tau) = integral (ln(K/F_i))^3 * q_i(K) dK

Following Bakshi, Kapadia & Madan (2003), the model-free implied skewness is:

    SKEW_i(t) = [e^{r*tau} * W_i - 3*mu_i * e^{r*tau} * V_i + 2*mu_i^3]
                / [e^{r*tau} * V_i - mu_i^2]^{3/2}

This is the risk-neutral third standardized moment, estimated entirely from the cross-
section of option prices with no model assumptions.

**Step 3 — Estimate Expected Returns via Rolling Fama-MacBeth.**
At each month t, run the cross-sectional regression:

    R_i(t+1) = a_t + b_t * SKEW_i(t) + g_t * log(ME_i) + d_t * B/M_i + epsilon_i(t)

Collect the time series {b_t}. The Fama-MacBeth estimate of the skewness premium:

    b_hat = (1/T) * sum_t b_t,     SE(b_hat) = std(b_t) / sqrt(T)

Use the rolling 24-month estimated b_hat(t) as a real-time forecast of the reward to
shorting skewness. The skewness-implied expected excess return for stock i at time t:

    mu_i_hat(t) = b_hat(t) * SKEW_i(t)

**Step 4 — Mean-Variance Optimal Portfolio.**
Given N stocks with expected excess returns vector mu_hat and covariance matrix Sigma
(estimated via Ledoit-Wolf shrinkage to correct for estimation error):

    Sigma_shrunk = (1 - alpha) * Sigma_sample + alpha * Sigma_target

The unconstrained mean-variance optimal weight vector (maximizing Sharpe ratio):

    w*(t) = (1 / (2*lambda)) * Sigma_shrunk^{-1} * mu_hat(t)

where lambda is the risk-aversion parameter chosen to target annualized portfolio
volatility sigma_target:

    lambda = (w*' * Sigma_shrunk * w*)^{1/2} / sigma_target   (solved iteratively)

Apply dollar-neutrality constraint: w -> w - mean(w) * 1, so the portfolio is long-short.

Unlike quintile sorting, each stock's weight is a continuous function of its skewness
score, its covariance with all other stocks, and the current estimated premium b_hat(t).
Stocks with high SKEW receive large short weights; their magnitude scales with signal
confidence (b_hat / SE(b_hat)) and is dampened by covariance with existing positions.

**Key Papers:**
- Bakshi, G., Kapadia, N. & Madan, D. (2003). *Stock Return Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options.* Review of Financial Studies, 16(1), 101-143.
- Breeden, D. & Litzenberger, R. (1978). *Prices of State-Contingent Claims Implicit in Option Prices.* Journal of Business, 51(4), 621-651.
- Boyer, B., Mitton, T. & Vorkink, K. (2010). *Expected Idiosyncratic Skewness.* Review of Financial Studies, 23(1), 169-202.
- Conrad, J., Dittmar, R. & Ghysels, E. (2013). *Ex Ante Skewness and Expected Stock Returns.* Journal of Finance, 68(1), 85-124.
- Ledoit, O. & Wolf, M. (2004). *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.* Journal of Multivariate Analysis, 88(2), 365-411.
