"""Turn raw per-call tone scores into the signal actually being tested.

The whole thesis lives in this file, in one line: we never use a raw tone level.
We use the *change* against the same company's own trailing four calls.

Why that matters. Raw hedging density is dominated by two things that have
nothing to do with the future: who the CFO is (some people hedge constantly),
and what industry it is (biotech talks about uncertainty because the business is
uncertain). Both are persistent, both are known to the market, and neither
predicts anything. Differencing within firm removes both by construction - a
fixed effect subtracted rather than estimated.

Every function here is written so that a row dated t uses only rows dated < t.
`audit_lookahead` re-checks that property on the finished panel rather than
trusting the implementation, because this is the failure mode that produces a
beautiful backtest and a wrong answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rubric import AXES, AXIS_SIGN

MIN_HISTORY = 4  # trailing calls required before a firm produces a signal


def build_panel(scores: pd.DataFrame) -> pd.DataFrame:
    """Assemble the per-call feature panel.

    Parameters
    ----------
    scores
        One row per call: ticker, call_date, fiscal_period, and one column per
        rubric axis.

    Returns
    -------
    The same rows plus, for each axis, `d_<axis>` (change vs trailing mean) and
    `z_<axis>` (that change scaled by the firm's own trailing dispersion), plus
    a `tone_delta` composite. Rows without MIN_HISTORY priors get NaN and are
    dropped by `investable()`.
    """
    missing = [a for a in AXES if a not in scores.columns]
    if missing:
        raise ValueError(f"scores frame missing axis columns: {missing}")

    df = scores.copy()
    df["call_date"] = pd.to_datetime(df["call_date"])
    df = df.sort_values(["ticker", "call_date"]).reset_index(drop=True)

    if df.duplicated(["ticker", "call_date"]).any():
        dupes = df[df.duplicated(["ticker", "call_date"], keep=False)]
        raise ValueError(
            f"duplicate ticker/call_date rows would double-count a firm:\n{dupes.head()}"
        )

    g = df.groupby("ticker", group_keys=False)

    for axis in AXES:
        # shift(1) is what makes this point-in-time: the window ends at the
        # PREVIOUS call, so the current call is never in its own baseline.
        prior = g[axis].shift(1)
        base_mean = prior.groupby(df["ticker"]).transform(
            lambda s: s.rolling(MIN_HISTORY, min_periods=MIN_HISTORY).mean()
        )
        base_std = prior.groupby(df["ticker"]).transform(
            lambda s: s.rolling(MIN_HISTORY, min_periods=MIN_HISTORY).std(ddof=1)
        )
        delta = df[axis] - base_mean
        df[f"d_{axis}"] = delta
        # A firm with a genuinely constant axis has std ~0; dividing by it would
        # manufacture enormous z-scores from rounding noise. Floor it.
        df[f"z_{axis}"] = delta / base_std.clip(lower=0.02)

    # Composite: sign-adjust each axis so positive always means "tone
    # deteriorated", then average. Equal weights, deliberately - fitting weights
    # on the same sample that evaluates them is how you get an in-sample mirage,
    # and with six axes the estimation error would swamp any real weighting gain.
    z_cols = [f"z_{a}" for a in AXES]
    signed = pd.DataFrame(
        {f"z_{a}": df[f"z_{a}"] * AXIS_SIGN[a] for a in AXES}, index=df.index
    )
    df["tone_delta"] = signed[z_cols].mean(axis=1, skipna=False)

    # Convention for the rest of the pipeline: HIGHER tone_score = tone IMPROVED
    # = the leg we go long. Flipping here keeps every downstream sort ascending.
    df["tone_score"] = -df["tone_delta"]
    df["n_prior_calls"] = g.cumcount()
    return df


def investable(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop calls that cannot produce a signal, for reasons stated in `why`."""
    out = panel.copy()
    out["why_dropped"] = np.where(
        out["n_prior_calls"] < MIN_HISTORY,
        "insufficient_history",
        np.where(out["tone_score"].isna(), "nan_score", ""),
    )
    return out[out["why_dropped"] == ""].drop(columns="why_dropped").reset_index(drop=True)


def add_entry_dates(panel: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Attach the first date the signal is actually tradeable.

    Earnings calls are held after the close (or before the open). Assuming you
    can trade at the close of the call date is the single most common way this
    genre of backtest leaks - it hands you the announcement reaction, which is
    the part that is unambiguously already priced.

    We enter at the close of the NEXT trading day: one full session after the
    transcript is public. That is conservative and defensible.
    """
    td = pd.DatetimeIndex(sorted(pd.to_datetime(trading_days).unique()))
    out = panel.copy()
    out["call_date"] = pd.to_datetime(out["call_date"])
    # searchsorted 'right' -> strictly after the call date; +0 lands on the next
    # session, and we take one more to sit at the following close.
    idx = td.searchsorted(out["call_date"].values, side="right")
    entry_idx = np.minimum(idx, len(td) - 1)
    out["entry_date"] = td[entry_idx]
    out.loc[idx >= len(td), "entry_date"] = pd.NaT
    return out.dropna(subset=["entry_date"]).reset_index(drop=True)


def cross_sectional_rank(panel: pd.DataFrame, cohort: str = "M") -> pd.DataFrame:
    """Rank signals against contemporaneous peers, not against the full history.

    Earnings arrive in bursts. Comparing a February call's score to an August
    call's score mixes in whatever drifted between them; ranking within a
    monthly cohort keeps the comparison local in time. The cohort is defined on
    `entry_date`, so it only ever contains information already public.
    """
    out = panel.copy()
    out["cohort"] = out["entry_date"].dt.to_period(cohort)
    grp = out.groupby("cohort")["tone_score"]
    out["cohort_n"] = grp.transform("size")
    # pct rank in [0,1]; NaN where the cohort is too thin to rank meaningfully
    out["tone_rank"] = grp.rank(pct=True, method="average")
    out.loc[out["cohort_n"] < 10, "tone_rank"] = np.nan
    return out


def audit_lookahead(panel: pd.DataFrame, raw_scores: pd.DataFrame) -> list[str]:
    """Independently re-derive each firm's baseline and check for leakage.

    This does NOT reuse `build_panel`'s logic - it recomputes the trailing mean
    from the raw scores with an explicit date filter. If the two disagree, the
    vectorised version leaked, and we would rather fail the run than ship a
    number nobody can trust.
    """
    problems: list[str] = []
    raw = raw_scores.copy()
    raw["call_date"] = pd.to_datetime(raw["call_date"])

    if (panel["entry_date"] <= panel["call_date"]).any():
        n = int((panel["entry_date"] <= panel["call_date"]).sum())
        problems.append(f"{n} rows enter on or before the call date")

    sample = panel.dropna(subset=["tone_score"]).sample(
        min(len(panel.dropna(subset=["tone_score"])), 200), random_state=0
    )
    for _, row in sample.iterrows():
        hist = raw[(raw["ticker"] == row["ticker"]) & (raw["call_date"] < row["call_date"])]
        hist = hist.sort_values("call_date").tail(MIN_HISTORY)
        if len(hist) < MIN_HISTORY:
            problems.append(
                f"{row['ticker']} @ {row['call_date'].date()} scored with only {len(hist)} priors"
            )
            continue
        for axis in AXES:
            expected = row[axis] - hist[axis].mean()
            if not np.isclose(expected, row[f"d_{axis}"], atol=1e-8, equal_nan=True):
                problems.append(
                    f"{row['ticker']} @ {row['call_date'].date()} d_{axis} "
                    f"{row[f'd_{axis}']:.6f} != recomputed {expected:.6f}"
                )
                break
    return problems
