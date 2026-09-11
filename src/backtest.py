"""Calendar-time overlapping-portfolio backtest.

Why calendar-time rather than averaging event-window CARs: earnings cluster into
four short seasons a year. If you average CARs across events, you implicitly
assume the events are independent draws, and they are not - hundreds of calls in
the same fortnight share one macro shock. That dependence makes naive event-study
t-stats far too confident.

The calendar-time approach sidesteps it. Every day, hold every position whose
call was within the last H trading days, equally weighted. The result is a single
daily return series whose statistical properties are honest, and which is also
just... what the strategy would actually have returned. Turnover, capacity and
costs all become measurable rather than hypothetical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import stats as st


@dataclass
class BacktestResult:
    daily: pd.DataFrame            # date, ret_gross, ret_net, turnover, n_long, n_short
    metrics: Dict[str, float]
    positions: pd.DataFrame        # date x ticker weights, for inspection
    spec: Dict[str, object]


def build_weights(
    signals: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    hold_days: int = 63,
    quantile: float = 0.2,
    long_only: bool = False,
) -> pd.DataFrame:
    """Map ranked signals onto a daily weight matrix.

    `signals` needs: ticker, entry_date, tone_rank (0-1, higher = tone improved).
    """
    td = pd.DatetimeIndex(sorted(pd.to_datetime(trading_days).unique()))
    tickers = sorted(signals["ticker"].unique())
    pos = pd.DataFrame(0.0, index=td, columns=tickers)

    sig = signals.dropna(subset=["tone_rank"]).copy()
    sig["entry_date"] = pd.to_datetime(sig["entry_date"])

    for _, row in sig.iterrows():
        rank = row["tone_rank"]
        if rank >= 1 - quantile:
            side = 1.0
        elif rank <= quantile and not long_only:
            side = -1.0
        else:
            continue

        start = td.searchsorted(row["entry_date"], side="left")
        if start >= len(td):
            continue
        end = min(start + hold_days, len(td))
        # Accumulate: a name with two qualifying calls inside one holding window
        # legitimately gets double weight, and normalisation below handles it.
        pos.iloc[start:end, pos.columns.get_loc(row["ticker"])] += side

    return pos


def normalise(pos: pd.DataFrame) -> pd.DataFrame:
    """Scale each side to 100% gross, so the book is always fully invested.

    Long and short legs are normalised independently. Normalising the combined
    book instead would let a day with 40 longs and 3 shorts sit net-long by
    accident, and the resulting market exposure would show up as alpha.
    """
    longs = pos.clip(lower=0)
    shorts = pos.clip(upper=0)
    ls = longs.sum(axis=1).replace(0, np.nan)
    ss = shorts.sum(axis=1).abs().replace(0, np.nan)
    out = longs.div(ls, axis=0).fillna(0.0) + shorts.div(ss, axis=0).fillna(0.0)
    return out


def run(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    hold_days: int = 63,
    quantile: float = 0.2,
    cost_bps: float = 10.0,
    long_only: bool = False,
    market_col: str = "MKT",
) -> BacktestResult:
    px = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets = px.pct_change()
    td = rets.index

    raw = build_weights(signals, td, hold_days, quantile, long_only)
    w = normalise(raw)

    cols = [c for c in w.columns if c in rets.columns and c != market_col]
    w, r = w[cols], rets[cols]

    # Position held from the close of day t earns day t+1's return. Using day t's
    # own return would be the single-day lookahead that flatters every naive
    # backtest in this genre.
    gross = (w.shift(1) * r).sum(axis=1, skipna=True)

    turnover = (w - w.shift(1)).abs().sum(axis=1) / 2.0
    cost = turnover * (cost_bps / 10_000.0)
    net = gross - cost

    daily = pd.DataFrame(
        {
            "ret_gross": gross,
            "ret_net": net,
            "turnover": turnover,
            "n_long": (w > 0).sum(axis=1),
            "n_short": (w < 0).sum(axis=1),
        }
    )
    # Trim the warm-up before any position exists; leading zeros would otherwise
    # depress volatility and inflate the Sharpe.
    active = daily[(daily["n_long"] + daily["n_short"]) > 0]
    daily = daily.loc[active.index.min() : active.index.max()] if len(active) else daily

    metrics = summarise(daily, market_col_returns=rets[market_col] if market_col in rets else None)
    spec = {
        "hold_days": hold_days,
        "quantile": quantile,
        "cost_bps": cost_bps,
        "long_only": long_only,
    }
    return BacktestResult(daily=daily, metrics=metrics, positions=w, spec=spec)


def summarise(
    daily: pd.DataFrame, market_col_returns: Optional[pd.Series] = None
) -> Dict[str, float]:
    net = daily["ret_net"].dropna()
    if len(net) < 30:
        return {"error": "insufficient backtest history", "n_days": len(net)}

    ann_ret = float((1 + net).prod() ** (st.TRADING_DAYS / len(net)) - 1)
    ann_vol = float(net.std(ddof=1) * np.sqrt(st.TRADING_DAYS))
    curve = (1 + net).cumprod()
    dd = float((curve / curve.cummax() - 1).min())
    nw = st.newey_west_tstat(net.values)

    out = {
        "n_days": len(net),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe_net": st.sharpe(net.values),
        "sharpe_gross": st.sharpe(daily["ret_gross"].dropna().values),
        "max_drawdown": dd,
        "nw_t_stat": nw["t_stat"],
        "avg_daily_turnover": float(daily["turnover"].mean()),
        "avg_n_long": float(daily["n_long"].mean()),
        "avg_n_short": float(daily["n_short"].mean()),
        "hit_rate_daily": float((net > 0).mean()),
    }

    # Cost sensitivity: the level of costs at which the edge disappears. If this
    # is under ~15bps the strategy is a fee study, not a strategy.
    gross_mu = daily["ret_gross"].dropna().mean()
    tno = daily["turnover"].mean()
    out["breakeven_cost_bps"] = float(gross_mu / tno * 10_000) if tno > 0 else float("inf")

    if market_col_returns is not None:
        mk = market_col_returns.reindex(net.index).dropna()
        common = net.index.intersection(mk.index)
        if len(common) > 60:
            b = float(np.cov(net.loc[common], mk.loc[common], ddof=1)[0, 1] / mk.loc[common].var(ddof=1))
            out["market_beta"] = b
    return out


def decile_spread(signals: pd.DataFrame, y: str = "car_drift", n_bins: int = 5) -> pd.DataFrame:
    """Mean forward drift by signal quintile - the monotonicity check.

    A real signal should be roughly monotone across bins. A hump, or a result
    driven entirely by one extreme bin, usually means a handful of outliers
    rather than a relationship.
    """
    d = signals.dropna(subset=["tone_score", y]).copy()
    if len(d) < n_bins * 10:
        return pd.DataFrame()
    d["bin"] = pd.qcut(d["tone_score"], n_bins, labels=False, duplicates="drop") + 1
    g = d.groupby("bin")[y]
    out = pd.DataFrame(
        {"n": g.size(), "mean": g.mean(), "median": g.median(), "std": g.std()}
    ).reset_index()
    out["t_stat"] = out["mean"] / (out["std"] / np.sqrt(out["n"]))
    return out
