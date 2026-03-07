from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, cast

import nasdaqdatalink
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dotenv import load_dotenv


Edge = Tuple[int, int, float]


@dataclass
class StrategySimulationResult:
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    drawdown: pd.Series
    turnover: pd.Series
    trade_count: int
    average_holding_period: float


def fetch_etf_prices(
    tickers: Sequence[str],
    start_date,
    end_date,
) -> pd.DataFrame:
    """Fetch adjusted close prices from NASDAQ Data Link QUOTEMEDIA/PRICES."""
    load_dotenv()
    api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if api_key is None:
        raise ValueError("NASDAQ_DATA_LINK_API_KEY not found in environment.")
    setattr(nasdaqdatalink.ApiConfig, "api_key", api_key)

    start_str = pd.to_datetime(start_date, errors="raise").date().isoformat()
    end_str = pd.to_datetime(end_date, errors="raise").date().isoformat()

    collected: List[pd.DataFrame] = []
    for ticker in tickers:
        # Required fetch pattern.
        df = nasdaqdatalink.get_table(
            "QUOTEMEDIA/PRICES",
            ticker=ticker,
            date={"gte": start_str, "lte": end_str},
            paginate=True,
        )
        if df.empty:
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        small = df[["date", "ticker", price_col]].rename(columns={price_col: "close"})
        collected.append(small)

    if not collected:
        raise ValueError("No data fetched from NASDAQ Data Link for requested tickers and date range.")

    stacked = pd.concat(collected, ignore_index=True)
    prices = (
        stacked.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )
    return prices


def load_or_fetch_prices(
    tickers: Sequence[str],
    start_date,
    end_date,
    data_dir: str,
) -> pd.DataFrame:
    """Load cached CSVs from data_dir; fetch and cache missing tickers."""
    os.makedirs(data_dir, exist_ok=True)
    start_ts = pd.Timestamp(pd.to_datetime(start_date, errors="raise"))
    end_ts = pd.Timestamp(pd.to_datetime(end_date, errors="raise"))

    cached_frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for ticker in tickers:
        cache_path = os.path.join(data_dir, f"{ticker}.csv")
        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            cached = cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)]
            if not cached.empty:
                cached_frames.append(cached.assign(ticker=ticker))
                continue
        missing.append(ticker)

    if missing:
        fetched = fetch_etf_prices(missing, start_date, end_date)
        for ticker in missing:
            if ticker not in fetched.columns:
                continue
            single = fetched[[ticker]].dropna().reset_index().rename(columns={ticker: "close"})
            single["ticker"] = ticker
            single.to_csv(os.path.join(data_dir, f"{ticker}.csv"), index=False)
            cached_frames.append(single)

    if not cached_frames:
        raise ValueError("No cached or fetched data available.")

    pooled = pd.concat(cached_frames, ignore_index=True)
    prices = (
        pooled.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )
    prices = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices


def compute_rolling_correlations(
    returns: pd.DataFrame,
    window: int = 252,
) -> dict[object, pd.DataFrame]:
    """Compute rolling Pearson correlation matrix for each date."""
    clean = returns.dropna(how="all")
    out: dict[object, pd.DataFrame] = {}
    for idx in range(window - 1, len(clean)):
        date_raw = pd.to_datetime(clean.index[idx], errors="coerce")
        if pd.isna(date_raw):
            continue
        date = str(date_raw)
        sample = clean.iloc[idx - window + 1 : idx + 1]
        out[date] = sample.corr()
    return out


def mantegna_distance(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute Mantegna distance: d_ij = sqrt(2 * (1 - rho_ij))."""
    rho = correlation_matrix.clip(-1.0, 1.0)
    dist = np.sqrt(2.0 * (1.0 - rho)).copy()
    arr = dist.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    dist = pd.DataFrame(arr, index=dist.index, columns=dist.columns)
    return dist


def build_mst_kruskal(distance_matrix: pd.DataFrame, asset_names: Sequence[str]) -> List[Edge]:
    """Build MST using Kruskal's algorithm over the distance matrix."""
    names = list(asset_names)
    n = len(names)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    edges: List[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = names[i], names[j]
            weight = float(distance_matrix.loc[ni, nj])
            if np.isfinite(weight):
                edges.append((i, j, weight))
    edges.sort(key=lambda x: x[2])

    mst: List[Edge] = []
    for i, j, w in edges:
        if union(i, j):
            mst.append((i, j, w))
            if len(mst) == n - 1:
                break

    if len(mst) != n - 1:
        raise ValueError("Unable to construct MST. Distance matrix may be disconnected.")
    return mst


def compute_graph_laplacian(mst_edges: Sequence[Edge], n_assets: int) -> NDArray[np.float64]:
    """Build unweighted graph Laplacian from MST edges."""
    adjacency = np.zeros((n_assets, n_assets), dtype=float)
    for i, j, _ in mst_edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    degrees = adjacency.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency
    return laplacian


def compute_effective_resistance(laplacian: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute all-pairs effective resistance via pseudoinverse of Laplacian."""
    lplus = np.linalg.pinv(laplacian)
    diag = np.diag(lplus)
    resist = diag[:, None] + diag[None, :] - 2.0 * lplus
    resist = np.maximum(resist, 0.0)
    return resist


def compute_fiedler_value(laplacian: NDArray[np.float64]) -> float:
    """Compute second-smallest Laplacian eigenvalue (algebraic connectivity)."""
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(np.real(eigenvalues))
    if len(eigenvalues) < 2:
        return 0.0
    return float(eigenvalues[1])


def generate_trading_signals(
    effective_resistances: Dict[pd.Timestamp, pd.DataFrame],
    lookback: int = 60,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Generate edge-level momentum signals from rolling resistance z-scores.

    A positive z-score (resistance above its rolling mean) indicates the pair
    has drifted apart topologically, and empirically this drift persists over
    the subsequent week — a momentum effect.  The exponential damping term
    moderates signals for pairs that are already far apart in the MST."""
    dates = sorted(effective_resistances.keys())
    if not dates:
        return {}

    assets = [str(c) for c in effective_resistances[dates[0]].columns]
    pairs = [(i, j) for i in assets for j in assets if i < j]

    history = {pair: [] for pair in pairs}
    output: Dict[pd.Timestamp, pd.DataFrame] = {}

    for date in dates:
        current_r = effective_resistances[date]
        mat = pd.DataFrame(np.nan, index=pd.Index(assets), columns=pd.Index(assets), dtype=float)

        pair_values = [float(current_r.loc[i, j]) for i, j in pairs]
        r_bar = float(np.nanmean(pair_values)) if pair_values else np.nan

        for i, j in pairs:
            rij = float(current_r.loc[i, j])
            series = history[(i, j)]
            if len(series) >= lookback:
                sample = np.array(series[-lookback:], dtype=float)
                mu = float(np.nanmean(sample))
                sigma = max(float(np.nanstd(sample, ddof=1)), 1e-6)
                if np.isfinite(r_bar) and r_bar > 0:
                    z = float(np.clip((rij - mu) / sigma, -3.0, 3.0))
                    signal = z * np.exp(-rij / r_bar)
                    mat.loc[i, j] = signal
                    mat.loc[j, i] = -signal
            series.append(rij)

        arr = mat.to_numpy(copy=True)
        np.fill_diagonal(arr, 0.0)
        mat = pd.DataFrame(arr, index=mat.index, columns=mat.columns)
        output[date] = mat
    return output

def compute_lambda_scale(
    lambda_series: pd.Series,
    mode: str = "small_only",   # "small_only", "large_only", "two_sided", "none"
    min_periods: int = 20,
    z_threshold: float = 2.0,
    shrink_small: float = 0.5,
    shrink_large: float = 0.5,
) -> pd.DataFrame:
    """
    Compute risk-control scaling based on Fiedler value (lambda_2).

    Parameters
    ----------
    lambda_series : pd.Series
        Time series of Fiedler values indexed by rebalance date.
    mode : str
        "small_only" : reduce exposure when lambda is unusually small
        "large_only" : reduce exposure when lambda is unusually large
        "two_sided"  : reduce exposure on both tails
        "none"       : no lambda-based scaling
    min_periods : int
        Minimum history before activating thresholds.
    z_threshold : float
        Number of std devs used to define unusual lambda.
    shrink_small : float
        Multiplicative scale when lambda is too small.
    shrink_large : float
        Multiplicative scale when lambda is too large.

    Returns
    -------
    pd.DataFrame with columns:
        lambda, mean, std, lower, upper, scale
    """
    lam = lambda_series.astype(float).copy()

    mean = lam.expanding(min_periods=min_periods).mean()
    std = lam.expanding(min_periods=min_periods).std(ddof=1).fillna(0.0)

    lower = mean - z_threshold * std
    upper = mean + z_threshold * std

    scale = pd.Series(1.0, index=lam.index, dtype=float)

    valid = mean.notna() & std.notna()

    if mode == "small_only":
        scale[valid & (lam < lower)] = shrink_small

    elif mode == "large_only":
        scale[valid & (lam > upper)] = shrink_large

    elif mode == "two_sided":
        scale[valid & (lam < lower)] = shrink_small
        scale[valid & (lam > upper)] = np.minimum(
            scale[valid & (lam > upper)],
            shrink_large
        )

    elif mode == "none":
        pass

    else:
        raise ValueError(f"Unsupported lambda scaling mode: {mode}")

    out = pd.DataFrame(
        {
            "lambda": lam,
            "mean": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
            "scale": scale,
        }
    )
    return out


def compute_pair_weights(
    signals: Dict[pd.Timestamp, pd.DataFrame],
    returns: pd.DataFrame,
    sigma_target: float = 0.10,
    fiedler_values: Optional[pd.Series] = None,
    mst_by_date: Optional[Dict[pd.Timestamp, List[Edge]]] = None,
    asset_names: Optional[List[str]] = None,
    max_pairs: int = 15,
    beta_window: int = 60,
    leader_window: int = 20,
    max_pair_weight: float = 1.5,
    max_gross_leverage: float = 4.0,
    lambda_risk_mode: str = "small_only",   # NEW
    lambda_min_periods: int = 20,           # NEW
    lambda_z_threshold: float = 2.0,        # NEW
    lambda_shrink_small: float = 0.5,       # NEW
    lambda_shrink_large: float = 0.5,       # NEW
) -> Dict[str, Dict[pd.Timestamp, pd.Series] | pd.Series]:
    """Compute pair and aggregate asset weights using volatility-targeted sizing.

    When mst_by_date is provided, only pairs along MST edges are considered
    as trade candidates. This preserves the topological signal and avoids
    dilution from unrelated pair correlations.
    """
    dates = sorted(signals.keys())
    if not dates:
        return {"pair_weights": {}, "asset_weights": {}, "lambda2_scale": pd.Series(dtype=float)}

    assets = [str(c) for c in signals[dates[0]].columns]
    if asset_names is None:
        asset_names = assets
    pair_weights_out: Dict[pd.Timestamp, pd.Series] = {}
    asset_weights_out: Dict[pd.Timestamp, pd.Series] = {}

    if fiedler_values is None:
        lambda_series = pd.Series(1.0, index=dates, dtype=float)
        lambda_diag = pd.DataFrame(
            {
                "lambda": lambda_series,
                "mean": np.nan,
                "std": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "scale": 1.0,
            },
            index=dates,
        )
    else:
        lambda_series = fiedler_values.reindex(dates).astype(float)
        lambda_diag = compute_lambda_scale(
            lambda_series=lambda_series,
            mode=lambda_risk_mode,
            min_periods=lambda_min_periods,
            z_threshold=lambda_z_threshold,
            shrink_small=lambda_shrink_small,
            shrink_large=lambda_shrink_large,
        )

    lambda_scale = lambda_diag["scale"].reindex(dates).fillna(1.0)
    for date in dates:
        if date not in returns.index:
            continue
        loc = returns.index.get_loc(date)
        if isinstance(loc, slice):
            loc = loc.start
        start_beta = max(0, loc - beta_window + 1)
        start_leader = max(0, loc - leader_window + 1)
        hist_beta = returns.iloc[start_beta : loc + 1]
        hist_leader = returns.iloc[start_leader : loc + 1]

        sig = signals[date]

        # Restrict candidates to MST edges if tree is available
        if mst_by_date is not None and date in mst_by_date:
            mst_edges = mst_by_date[date]
            allowed_pairs = set()
            for i_idx, j_idx, _ in mst_edges:
                a_name = asset_names[i_idx]
                b_name = asset_names[j_idx]
                allowed_pairs.add((min(a_name, b_name), max(a_name, b_name)))
        else:
            allowed_pairs = None  # no filter

        candidates: List[Tuple[Tuple[str, str], float]] = []
        for i in assets:
            for j in assets:
                if i < j:
                    if allowed_pairs is not None and (i, j) not in allowed_pairs:
                        continue
                    s = float(sig.loc[i, j])
                    if np.isfinite(s):
                        candidates.append(((i, j), s))

        if not candidates:
            continue
        candidates.sort(key=lambda x: abs(x[1]), reverse=True)
        top_k = min(max_pairs, len(candidates))
        chosen = candidates[:top_k]

        pair_vals: Dict[Tuple[str, str], float] = {}
        asset_w = pd.Series(0.0, index=assets)
        for (a, b), signal in chosen:
            ra = hist_beta[a].dropna()
            rb = hist_beta[b].dropna()
            joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
            if len(joined) < 20:
                continue

            beta_denom = float(np.var(joined.iloc[:, 1].values, ddof=1))
            if beta_denom <= 0:
                continue
            beta = float(np.cov(joined.iloc[:, 0].values, joined.iloc[:, 1].values, ddof=1)[0, 1] / beta_denom)

            spread = joined.iloc[:, 0] - beta * joined.iloc[:, 1]
            sigma_ij = float(spread.std(ddof=1) * np.sqrt(252.0))
            # Floor spread vol to prevent weight explosion when pairs co-move perfectly
            sigma_ij = max(sigma_ij, 0.01)
            if not np.isfinite(sigma_ij):
                continue

            lead_lag = hist_leader[[a, b]].dropna()
            if lead_lag.empty:
                continue
            cum_a = float((1.0 + lead_lag[a]).prod() - 1.0)
            cum_b = float((1.0 + lead_lag[b]).prod() - 1.0)

            if cum_a <= cum_b:
                lagging, leading = a, b
            else:
                lagging, leading = b, a

            w = float((sigma_target * signal / sigma_ij) * lambda_scale.loc[date])
            # Cap individual pair weight to prevent concentration
            w = float(np.clip(w, -max_pair_weight, max_pair_weight))
            pair_vals[(lagging, leading)] = w
            asset_w[lagging] += w
            asset_w[leading] -= w

        if pair_vals:
            # Rescale if gross leverage exceeds cap
            gross = float(asset_w.abs().sum())
            if gross > max_gross_leverage:
                scale = max_gross_leverage / gross
                asset_w *= scale
                pair_vals = {k: v * scale for k, v in pair_vals.items()}

            pair_index = pd.MultiIndex.from_tuples(pair_vals.keys(), names=["long", "short"])
            pair_series = pd.Series(pair_vals, index=pair_index, dtype=float)
            pair_weights_out[date] = pair_series
            asset_weights_out[date] = asset_w

    return {
        "pair_weights": pair_weights_out,
        "asset_weights": asset_weights_out,
        "lambda2_scale": lambda_scale,
        "lambda2_diag": lambda_diag,
    }

def simulate_strategy(
    prices: pd.DataFrame,
    signals: Dict[pd.Timestamp, pd.DataFrame],
    weights: Dict[str, Dict[pd.Timestamp, pd.Series] | pd.Series],
    transaction_cost_bps: float = 0,
) -> StrategySimulationResult:
    """Run daily strategy simulation from rebalance weights and prices."""
    returns = prices.pct_change().fillna(0.0)
    asset_weights = weights.get("asset_weights", {})
    if not isinstance(asset_weights, dict):
        raise ValueError("weights['asset_weights'] must be a date-to-Series dictionary.")

    daily = pd.Series(0.0, index=returns.index, dtype=float)
    turnover = pd.Series(0.0, index=returns.index, dtype=float)

    current = pd.Series(0.0, index=returns.columns, dtype=float)
    trade_count = 0
    open_days: Dict[Tuple[str, str], int] = {}
    completed_holds: List[int] = []

    pair_weights = weights.get("pair_weights", {})
    if not isinstance(pair_weights, dict):
        pair_weights = {}

    for date in returns.index:
        if date in asset_weights:
            new_w = asset_weights[date].reindex(returns.columns).fillna(0.0)
            delta = new_w - current
            turnover.loc[date] = float(np.abs(delta).sum())
            cost = float((transaction_cost_bps / 10000.0) * np.abs(delta).sum())
            current = new_w

            now_pairs = pair_weights.get(date, pd.Series(dtype=float))
            if isinstance(now_pairs, pd.Series) and not now_pairs.empty:
                active = {
                    (str(k[0]), str(k[1]))
                    for k in now_pairs.index.tolist()
                    if isinstance(k, tuple) and len(k) == 2
                }
                ended = [k for k in open_days if k not in active]
                for k in ended:
                    completed_holds.append(open_days[k])
                    del open_days[k]
                for k in active:
                    if k not in open_days:
                        open_days[k] = 0
                        trade_count += 1

            daily.loc[date] = float(current.dot(returns.loc[date]) - cost)
        else:
            daily.loc[date] = float(current.dot(returns.loc[date]))

        for k in list(open_days):
            open_days[k] += 1

    for k in list(open_days):
        completed_holds.append(open_days[k])

    cumulative = (1.0 + daily).cumprod() - 1.0
    running_max = (1.0 + cumulative).cummax()
    drawdown = (1.0 + cumulative) / running_max - 1.0

    avg_holding = float(np.mean(completed_holds)) if completed_holds else 0.0
    return StrategySimulationResult(
        daily_returns=daily,
        cumulative_returns=cumulative,
        drawdown=drawdown,
        turnover=turnover,
        trade_count=trade_count,
        average_holding_period=avg_holding,
    )


def compute_performance_metrics(
    returns: pd.Series,
    rf_rate: float = 0.02,
    trade_count: Optional[int] = None,
    average_holding_period: Optional[float] = None,
) -> pd.Series:
    """Compute strategy performance and risk metrics."""
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("Return series is empty.")

    ann_factor = 252.0
    total_periods = len(r)
    cum_return = float((1.0 + r).prod() - 1.0)
    ann_return = float((1.0 + cum_return) ** (ann_factor / total_periods) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(ann_factor))

    rf_daily = rf_rate / ann_factor
    excess = r - rf_daily
    sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(ann_factor)) if ann_vol > 0 else np.nan

    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(ann_factor)) if len(downside) > 1 else np.nan
    sortino = float((excess.mean() * ann_factor) / downside_vol) if downside_vol and downside_vol > 0 else np.nan

    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    var_95 = float(np.quantile(r, 0.05))
    cvar_95 = float(r[r <= var_95].mean())
    win_rate = float((r > 0).mean())

    metrics = pd.Series(
        {
            "Cumulative Return": cum_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Maximum Drawdown": max_drawdown,
            "VaR 95%": var_95,
            "CVaR 95%": cvar_95,
            "Win Rate": win_rate,
            "Number of Trades": float(trade_count) if trade_count is not None else np.nan,
            "Average Holding Period": float(average_holding_period)
            if average_holding_period is not None
            else np.nan,
        }
    )
    return metrics


def run_weekly_mst_pipeline(
    prices: pd.DataFrame,
    corr_window: int = 252,
    resistance_lookback: int = 60,
) -> Dict[str, object]:
    """Convenience pipeline to build weekly MST, resistance, and fiedler series."""
    returns = prices.pct_change().dropna()
    rolling_corrs = compute_rolling_correlations(returns, window=corr_window)
    sorted_keys = sorted(rolling_corrs.keys(), key=lambda x: pd.Timestamp(str(x)))
    weekly_dates = [pd.Timestamp(str(d)) for d in sorted_keys if pd.Timestamp(str(d)).weekday() == 4]

    resistances: Dict[pd.Timestamp, pd.DataFrame] = {}
    mst_by_date: Dict[pd.Timestamp, List[Edge]] = {}
    fiedler = {}

    assets = [str(c) for c in prices.columns]
    for date in weekly_dates:
        date_str = str(date)
        if date_str == "NaT":
            continue
        current_date = cast(pd.Timestamp, pd.Timestamp(date_str))
        corr = rolling_corrs[date_str].reindex(index=assets, columns=assets)
        dist = mantegna_distance(corr)
        mst = build_mst_kruskal(dist, assets)
        lap = compute_graph_laplacian(mst, len(assets))
        eff = compute_effective_resistance(lap)

        resistances[current_date] = pd.DataFrame(eff, index=pd.Index(assets), columns=pd.Index(assets))
        mst_by_date[current_date] = mst
        fiedler[current_date] = compute_fiedler_value(lap)

    fiedler_series = pd.Series(fiedler).sort_index()
    signals = generate_trading_signals(resistances, lookback=resistance_lookback)

    return {
        "returns": returns,
        "rolling_correlations": rolling_corrs,
        "weekly_dates": weekly_dates,
        "mst_by_date": mst_by_date,
        "effective_resistances": resistances,
        "fiedler_values": fiedler_series,
        "signals": signals,
    }
