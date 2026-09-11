"""End-to-end orchestration: transcripts -> scores -> signal -> study -> backtest."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import backtest, eventstudy, features
from . import stats as st
from .providers import PRICE_PROVIDERS, TRANSCRIPT_PROVIDERS, Universe
from .rubric import AXES
from .scoring import lexical
from .scoring.llm import get_scorer


@dataclass
class RunResult:
    scores: pd.DataFrame
    panel: pd.DataFrame
    events: pd.DataFrame
    dropped: pd.DataFrame
    regressions: Dict[str, dict]
    backtests: Dict[str, backtest.BacktestResult]
    deciles: pd.DataFrame
    audit: List[str]
    trials: st.TrialLog
    meta: Dict[str, object] = field(default_factory=dict)


def score_transcripts(
    transcripts: pd.DataFrame, scorer_kind: str, cache_path: Path, model: str, verbose: bool = True
) -> pd.DataFrame:
    scorer = get_scorer(scorer_kind, cache_path, model)
    rows = []
    total = len(transcripts)
    for i, r in enumerate(transcripts.itertuples(), 1):
        if verbose and (i % 200 == 0 or i == total):
            print(f"  scored {i}/{total}")
        s = scorer.score_text(r.text)
        rows.append(
            {
                "ticker": r.ticker,
                "call_date": r.call_date,
                "fiscal_period": r.fiscal_period,
                **s.as_dict(),
            }
        )
    return pd.DataFrame(rows)


def apply_universe(df: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    """Keep only calls made while the firm was actually in the index.

    When `universe.pit` is False this is a no-op, and the flag propagates to the
    report so the survivorship caveat is printed rather than forgotten.
    """
    if not universe.pit:
        return df
    keep = [universe.contains(r.ticker, pd.Timestamp(r.call_date)) for r in df.itertuples()]
    return df.loc[np.array(keep)].reset_index(drop=True)


def run(
    transcripts: pd.DataFrame,
    prices: pd.DataFrame,
    universe: Universe,
    scorer_kind: str = "lexical",
    cache_path: Path = Path("data/llm_cache.sqlite"),
    model: str = "claude-sonnet-5",
    hold_days: int = 63,
    quantile: float = 0.2,
    cost_bps: float = 10.0,
    verbose: bool = True,
    scores: Optional[pd.DataFrame] = None,
) -> RunResult:
    trials = st.TrialLog()

    transcripts = transcripts.copy()
    transcripts["call_date"] = pd.to_datetime(transcripts["call_date"])
    transcripts = apply_universe(transcripts, universe)

    # Some vendor transcripts carry prepared remarks only. Evasion and pushback
    # are measured exclusively in the Q&A, so scoring those calls would return a
    # complete-looking six-axis result with a third of it computed from empty
    # text - and a plain average would then place them alongside fully scored
    # calls as though the two were comparable. Excluding them costs sample size
    # and is the honest trade; the count is carried into the report so the
    # exclusion is disclosed rather than assumed.
    n_before = len(transcripts)
    usable = transcripts["text"].apply(lexical.has_usable_qa)
    dropped_no_qa = transcripts.loc[~usable, ["ticker", "fiscal_period"]]
    transcripts = transcripts[usable].reset_index(drop=True)
    if verbose and len(dropped_no_qa):
        print(
            f"[0/6] excluded {len(dropped_no_qa)} of {n_before} calls with no usable Q&A"
        )

    if scores is None:
        if verbose:
            print(f"[1/6] scoring {len(transcripts)} transcripts with '{scorer_kind}'")
        scores = score_transcripts(transcripts, scorer_kind, cache_path, model, verbose)
    else:
        # Pre-computed scores must be narrowed to the same surviving calls, or
        # the exclusion above is undone silently by the scores frame.
        scores = scores.copy()
        scores["call_date"] = pd.to_datetime(scores["call_date"])
        keys = set(zip(transcripts["ticker"], transcripts["call_date"]))
        scores = scores[
            [k in keys for k in zip(scores["ticker"], scores["call_date"])]
        ].reset_index(drop=True)

    if verbose:
        print("[2/6] building within-firm differenced features")
    panel = features.build_panel(scores)
    panel = features.investable(panel)

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    trading_days = pd.DatetimeIndex(sorted(prices["date"].unique()))
    panel = features.add_entry_dates(panel, trading_days)
    panel = features.cross_sectional_rank(panel)

    if verbose:
        print("[3/6] auditing for lookahead")
    audit = features.audit_lookahead(panel, scores)
    if audit:
        raise RuntimeError(
            "LOOKAHEAD AUDIT FAILED - refusing to report results:\n  "
            + "\n  ".join(audit[:10])
        )

    if verbose:
        print("[4/6] event study")
    ep = eventstudy.run(panel, prices)
    if ep.events.empty:
        raise RuntimeError(
            "no events survived the event study; check price coverage. "
            f"drop reasons: {ep.dropped['reason'].value_counts().to_dict() if not ep.dropped.empty else 'n/a'}"
        )

    # --- regressions ------------------------------------------------------
    # The level spec is not a throwaway. If tone LEVEL predicts as well as the
    # differenced signal, the within-firm story is wrong and the project's
    # central claim collapses. Reporting it either way is the point.
    if verbose:
        print("[5/6] cross-sectional regressions")
    regressions: Dict[str, dict] = {}

    ev = ep.events.copy()
    ev["tone_level"] = ev[list(AXES)].mean(axis=1)

    specs = {
        "differenced_signal": ("tone_score", ["car_announce"]),
        "differenced_signal_uncontrolled": ("tone_score", []),
        "raw_level_signal": ("tone_level", ["car_announce"]),
        "announcement_window_placebo": ("tone_score", []),
    }
    for name, (xcol, ctrls) in specs.items():
        ycol = "car_announce" if name == "announcement_window_placebo" else "car_drift"
        trials.record(name)
        regressions[name] = eventstudy.cross_sectional_regression(ev, y=ycol, x=xcol, controls=ctrls)

    # Per-axis attribution: which of the six axes is doing the work?
    for axis in AXES:
        col = f"z_{axis}"
        if col in ev.columns:
            trials.record(f"axis::{axis}")
            regressions[f"axis::{axis}"] = eventstudy.cross_sectional_regression(
                ev, y="car_drift", x=col, controls=["car_announce"]
            )

    if verbose:
        print("[6/6] backtests")
    backtests: Dict[str, backtest.BacktestResult] = {}
    grid = {
        "base": dict(hold_days=hold_days, quantile=quantile, cost_bps=cost_bps),
        "no_costs": dict(hold_days=hold_days, quantile=quantile, cost_bps=0.0),
        "short_hold": dict(hold_days=21, quantile=quantile, cost_bps=cost_bps),
        "long_only": dict(hold_days=hold_days, quantile=quantile, cost_bps=cost_bps, long_only=True),
        "wide_quantile": dict(hold_days=hold_days, quantile=0.33, cost_bps=cost_bps),
    }
    for name, kw in grid.items():
        trials.record(f"backtest::{name}")
        backtests[name] = backtest.run(ev, prices, **kw)

    deciles = backtest.decile_spread(ev)

    return RunResult(
        scores=scores,
        panel=panel,
        events=ev,
        dropped=ep.dropped,
        regressions=regressions,
        backtests=backtests,
        deciles=deciles,
        audit=audit,
        trials=trials,
        meta={
            "scorer": scorer_kind,
            "model": model if scorer_kind == "llm" else None,
            "point_in_time_universe": universe.pit,
            "n_transcripts": len(transcripts),
            "n_transcripts_before_qa_filter": n_before,
            "dropped_no_qa": len(dropped_no_qa),
            "dropped_no_qa_list": [
                f"{r.ticker} {r.fiscal_period}" for r in dropped_no_qa.itertuples()
            ],
            "n_events": len(ev),
            "n_dropped": len(ep.dropped),
        },
    )
