"""Event-study machinery: abnormal returns around each call.

Two windows, kept strictly separate, because conflating them is the classic way
to claim an edge that does not exist:

  ANNOUNCEMENT (0, +1)   the immediate repricing. Any signal correlates with
                         this, because the signal is derived from the same call
                         the market just read. This window is measured and
                         REPORTED but never traded - we enter after it closes.

  DRIFT (+2, +63)        the part that would still be available to someone who
                         read the transcript carefully overnight. This is the
                         only window the strategy claims.

Abnormal return uses a market model estimated on (-250, -31): far enough back
that the estimation window cannot overlap the event, long enough that beta is
not noise. Firms without at least MIN_EST_OBS clean observations are dropped
rather than assigned beta = 1, since a silently defaulted beta would push their
abnormal returns toward raw returns and let market moves masquerade as alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

EST_START, EST_END = -250, -31
ANNOUNCE_WINDOW = (0, 1)
DRIFT_WINDOW = (2, 63)
MIN_EST_OBS = 100


@dataclass
class EventPanel:
    events: pd.DataFrame          # one row per call, with CARs attached
    dropped: pd.DataFrame         # what fell out and why


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide matrix of simple daily returns, dates x tickers."""
    px = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return px.pct_change()


def _market_model(
    stock: np.ndarray, mkt: np.ndarray
) -> Optional[tuple[float, float]]:
    """OLS of stock on market. Returns (alpha, beta) or None if degenerate."""
    ok = np.isfinite(stock) & np.isfinite(mkt)
    if ok.sum() < MIN_EST_OBS:
        return None
    x, y = mkt[ok], stock[ok]
    var = x.var()
    if var <= 0:
        return None
    beta = float(np.cov(x, y, ddof=1)[0, 1] / (var * len(x) / (len(x) - 1)))
    alpha = float(y.mean() - beta * x.mean())
    return alpha, beta


def run(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market_col: str = "MKT",
) -> EventPanel:
    """Compute announcement and drift CARs for every event.

    `events` needs ticker and call_date. Everything else is carried through.
    """
    rets = to_returns(prices)
    if market_col not in rets.columns:
        raise ValueError(f"price data has no {market_col!r} series for the market model")
    mkt = rets[market_col]
    dates = rets.index

    keep_rows, drop_rows = [], []

    for _, ev in events.iterrows():
        tkr, call_date = ev["ticker"], pd.Timestamp(ev["call_date"])
        if tkr not in rets.columns:
            drop_rows.append({**ev.to_dict(), "reason": "no_price_history"})
            continue

        # Event day 0 = first trading day on or after the call.
        pos = dates.searchsorted(call_date, side="left")
        if pos >= len(dates):
            drop_rows.append({**ev.to_dict(), "reason": "call_after_price_history"})
            continue

        est_lo, est_hi = pos + EST_START, pos + EST_END
        if est_lo < 0:
            drop_rows.append({**ev.to_dict(), "reason": "insufficient_estimation_window"})
            continue
        drift_hi = pos + DRIFT_WINDOW[1]
        if drift_hi >= len(dates):
            drop_rows.append({**ev.to_dict(), "reason": "truncated_drift_window"})
            continue

        s = rets[tkr]
        fit = _market_model(
            s.iloc[est_lo:est_hi].to_numpy(), mkt.iloc[est_lo:est_hi].to_numpy()
        )
        if fit is None:
            drop_rows.append({**ev.to_dict(), "reason": "estimation_failed"})
            continue
        alpha, beta = fit

        def car(lo: int, hi: int) -> float:
            sl = slice(pos + lo, pos + hi + 1)
            ar = s.iloc[sl].to_numpy() - (alpha + beta * mkt.iloc[sl].to_numpy())
            return float(np.nansum(ar))

        keep_rows.append(
            {
                **ev.to_dict(),
                "alpha": alpha,
                "beta": beta,
                "car_announce": car(*ANNOUNCE_WINDOW),
                "car_drift": car(*DRIFT_WINDOW),
                "event_idx": pos,
            }
        )

    kept = pd.DataFrame(keep_rows)
    dropped = pd.DataFrame(drop_rows)
    return EventPanel(events=kept, dropped=dropped)


def cross_sectional_regression(
    panel: pd.DataFrame,
    y: str = "car_drift",
    x: str = "tone_score",
    controls: Optional[list[str]] = None,
) -> Dict[str, float]:
    """Regress drift on the signal, controlling for the announcement reaction.

    Controlling for `car_announce` is the test that matters. Without it, a
    positive coefficient could just mean "tone tracks the earnings surprise, and
    good surprises drift" - which is post-earnings-announcement drift, a
    forty-year-old published anomaly, not a finding of yours. With it, the
    coefficient answers the only interesting question: does the *language*
    predict anything the *price reaction* did not already say?
    """
    import statsmodels.api as sm

    # `controls or [...]` would be wrong here: an explicitly empty list means
    # "run this uncontrolled", not "use the default". Conflating the two also
    # silently regresses car_announce on itself in the placebo spec.
    if controls is None:
        controls = ["car_announce"]

    # Never let the dependent variable in as its own regressor, and keep the
    # column list unique so the design matrix cannot pick up a duplicate.
    cols: list[str] = []
    for c in [x] + list(controls):
        if c != y and c in panel.columns and c not in cols:
            cols.append(c)
    if x not in cols:
        return {"n": 0, "error": f"regressor {x!r} unavailable (missing, or equal to y)"}

    d = panel[[y] + cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 30:
        return {"n": len(d), "error": "too few observations"}

    X = sm.add_constant(d[cols])
    # Clustering by month: earnings arrive in seasonal bursts, so residuals of
    # calls in the same window share a common shock. Plain OLS standard errors
    # would be far too small and every t-stat would look impressive.
    groups = panel.loc[d.index, "call_date"].dt.to_period("M").astype(str)
    fit = sm.OLS(d[y], X).fit(cov_type="cluster", cov_kwds={"groups": groups})

    return {
        "n": int(fit.nobs),
        "coef": float(fit.params[x]),
        "t_stat": float(fit.tvalues[x]),
        "p_value": float(fit.pvalues[x]),
        "r2": float(fit.rsquared),
        "controls": ", ".join(cols[1:]) or "none",
    }
