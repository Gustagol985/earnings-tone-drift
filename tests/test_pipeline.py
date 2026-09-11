"""Tests aimed at the failure modes that would produce a wrong *answer*.

Coverage of plumbing is cheap and mostly uninteresting. What is worth testing
here is the small set of mistakes that turn this project from research into a
misleading chart: lookahead in the baseline, entering on the call date itself,
scoring that ignores the text, and statistics that call noise a discovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest, eventstudy, features  # noqa: E402
from src.providers import ProviderError, Universe  # noqa: E402
from src import stats as st  # noqa: E402
from src.rubric import AXES, ToneScores  # noqa: E402
from src.scoring import lexical  # noqa: E402


# --------------------------------------------------------------------------
# rubric
# --------------------------------------------------------------------------
def test_tone_scores_reject_out_of_range():
    with pytest.raises(ValueError):
        ToneScores(1.4, 0.5, 0.5, 0.5, 0.5, 0.5)


# --------------------------------------------------------------------------
# universe file — hand-edited, so its failure modes are silent
# --------------------------------------------------------------------------
def _write_universe(tmp_path, body: str):
    p = tmp_path / "universe.csv"
    p.write_text("ticker,start_date,end_date\n" + body, encoding="utf-8")
    return p


def test_comment_lines_are_not_tickers(tmp_path):
    """Provenance notes in the file must not become companies."""
    p = _write_universe(
        tmp_path,
        "# --- still listed ---\nAAPL,1900-01-01,\n"
        "# --- delisted (SVB failure) ---\nSIVB,1900-01-01,2023-03-15\n",
    )
    u = Universe.from_csv(p)
    assert sorted(u.members["ticker"]) == ["AAPL", "SIVB"]


def test_delisted_firm_drops_out_after_end_date(tmp_path):
    p = _write_universe(tmp_path, "AAPL,1900-01-01,\nSIVB,1900-01-01,2023-03-15\n")
    u = Universe.from_csv(p)
    assert u.contains("SIVB", pd.Timestamp("2023-01-10"))
    assert not u.contains("SIVB", pd.Timestamp("2023-06-01"))
    assert u.contains("AAPL", pd.Timestamp("2025-01-01"))


def test_blank_end_date_means_still_a_member(tmp_path):
    p = _write_universe(tmp_path, "AAPL,1900-01-01,\n")
    u = Universe.from_csv(p)
    assert u.members["end_date"].isna().all()
    assert u.contains("AAPL", pd.Timestamp("2030-01-01"))


def test_malformed_date_is_rejected_not_coerced(tmp_path):
    """A typo'd date must fail loudly — silently becoming NaT would promote a
    delisted firm back to 'survived forever'."""
    p = _write_universe(tmp_path, "SIVB,1900-01-01,15/03/2023\n")
    with pytest.raises(ProviderError, match="end_date"):
        Universe.from_csv(p)


def test_duplicate_ticker_is_rejected(tmp_path):
    p = _write_universe(tmp_path, "AAPL,1900-01-01,\nAAPL,1900-01-01,\n")
    with pytest.raises(ProviderError, match="duplicate"):
        Universe.from_csv(p)


def test_real_universe_file_is_valid():
    """Guards the actual file the study will run on."""
    p = Path(__file__).resolve().parents[1] / "data" / "real" / "universe.csv"
    if not p.exists():
        pytest.skip("no real universe file yet")
    u = Universe.from_csv(p)
    assert u.pit
    assert 20 <= len(u.members) <= 60, f"unexpected universe size: {len(u.members)}"
    delisted = u.members.dropna(subset=["end_date"])
    assert len(delisted) >= 3, "universe has too few delisted names to control survivorship"


# --------------------------------------------------------------------------
# lexical scorer
# --------------------------------------------------------------------------
def test_hedging_responds_to_hedged_language():
    flat = "Revenue was 400 million. Margin was 22 percent. " * 30
    hedged = (
        "We think revenue could be roughly 400 million, though it is hard to say. "
        "We believe margin may improve, potentially, but it is too early to tell. "
    ) * 30
    assert lexical.score_text(hedged).hedging > lexical.score_text(flat).hedging


def test_commitment_requires_number_and_horizon():
    vague = "We expect continued momentum going forward. " * 30
    committed = "We expect revenue growth of 8 to 10 percent in Q3. " * 30
    assert lexical.score_text(committed).commitment > lexical.score_text(vague).commitment
    assert lexical.score_text(vague).commitment < 0.2


def test_prepared_remarks_only_transcript_is_flagged_unusable():
    """A call with no Q&A cannot be scored on evasion or pushback."""
    prepared = "CEO: Revenue grew 10 percent in the quarter. " * 200
    text = prepared + "\n\nOperator: This concludes today's call.\n"
    assert not lexical.has_usable_qa(text)


def test_normal_transcript_is_usable():
    text = (
        "CEO: " + "Revenue grew 10 percent in the quarter. " * 200 + "\n\n"
        "Operator: We will now take our first question.\n\n"
        "Tom Becker: Can you size the margin impact?\n\n"
        "CEO: " + "It was two points from mix. " * 40 + "\n\n"
        "Tom Becker: And the phasing from here?\n\n"
        "CEO: " + "Second half weighted. " * 40 + "\n\n"
        "Priya Nandan: What are you assuming for pricing?\n\n"
        "CEO: " + "Low single digits. " * 40 + "\n"
    )
    assert lexical.has_usable_qa(text)


def test_qa_section_is_split_off_prepared_remarks():
    text = (
        "CEO: Revenue grew 10 percent.\n\n"
        "Operator: We will now begin the question-and-answer session.\n\n"
        "Analyst: Can you size that?\n"
    )
    prepared, qa = lexical.split_sections(text)
    assert "Revenue grew" in prepared and "Can you size that" in qa


def test_curly_apostrophes_score_the_same_as_straight():
    """Vendors ship typographic punctuation; a lexicon must not care which."""
    straight = (
        "CEO: We'll have to see. It's possible margins improve, but it's too "
        "early to tell. We don't break that out. " * 20
    )
    curly = straight.replace("'", "’")
    assert lexical.score_text(curly).as_dict() == lexical.score_text(straight).as_dict()


def test_normalise_punctuation_folds_typographic_forms():
    assert lexical.normalise_punctuation("we’ll “yes” — no") == 'we\'ll "yes" - no'


def test_scores_are_deterministic():
    t = "We think margin could improve. " * 50
    assert lexical.score_text(t).as_dict() == lexical.score_text(t).as_dict()


# --------------------------------------------------------------------------
# features - the lookahead-critical logic
# --------------------------------------------------------------------------
def _scores_frame(n_firms: int = 3, n_calls: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_firms):
        for q in range(n_calls):
            rows.append(
                {
                    "ticker": f"T{f}",
                    "call_date": pd.Timestamp("2020-01-15") + pd.Timedelta(days=91 * q),
                    "fiscal_period": f"2020Q{q % 4 + 1}",
                    **{a: float(rng.uniform(0.1, 0.9)) for a in AXES},
                }
            )
    return pd.DataFrame(rows)


def test_baseline_excludes_the_current_call():
    """The trailing mean must be built from PRIOR calls only.

    If the current call leaked into its own baseline, the delta would shrink
    toward zero by construction and, worse, would embed contemporaneous info.
    """
    sc = _scores_frame()
    panel = features.build_panel(sc)
    row = panel[panel["ticker"] == "T0"].iloc[5]
    prior = sc[(sc["ticker"] == "T0") & (sc["call_date"] < row["call_date"])]
    expected = row["hedging"] - prior.sort_values("call_date").tail(4)["hedging"].mean()
    assert np.isclose(row["d_hedging"], expected)


def test_first_calls_produce_no_signal():
    panel = features.build_panel(_scores_frame())
    early = panel[panel["n_prior_calls"] < features.MIN_HISTORY]
    assert early["tone_score"].isna().all()
    assert features.investable(panel)["n_prior_calls"].min() >= features.MIN_HISTORY


def test_duplicate_calls_are_rejected():
    sc = pd.concat([_scores_frame(1, 6), _scores_frame(1, 6)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        features.build_panel(sc)


def test_entry_is_strictly_after_the_call():
    """Entering on the call date would capture the announcement move itself."""
    panel = features.investable(features.build_panel(_scores_frame()))
    td = pd.bdate_range("2019-01-01", "2024-01-01")
    panel = features.add_entry_dates(panel, td)
    assert (panel["entry_date"] > panel["call_date"]).all()


def test_audit_catches_injected_lookahead():
    """Corrupt a delta and confirm the auditor refuses to pass it."""
    sc = _scores_frame()
    panel = features.investable(features.build_panel(sc))
    td = pd.bdate_range("2019-01-01", "2024-01-01")
    panel = features.add_entry_dates(panel, td)
    assert features.audit_lookahead(panel, sc) == []

    tampered = panel.copy()
    tampered.loc[tampered.index[0], "d_hedging"] += 0.5
    assert features.audit_lookahead(tampered, sc) != []


def test_zscore_survives_a_constant_axis():
    """A firm with zero dispersion must not generate an infinite z-score."""
    sc = _scores_frame(1, 8)
    sc["hedging"] = 0.5
    panel = features.build_panel(sc)
    assert np.isfinite(panel["z_hedging"].dropna()).all()


# --------------------------------------------------------------------------
# event study
# --------------------------------------------------------------------------
def _price_frame(days: int = 800, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=days)
    frames = []
    mkt = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, days)))
    for t in ["T0", "T1", "T2"]:
        p = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, days)))
        frames.append(pd.DataFrame({"ticker": t, "date": dates, "close": p, "adj_close": p}))
    frames.append(pd.DataFrame({"ticker": "MKT", "date": dates, "close": mkt, "adj_close": mkt}))
    return pd.concat(frames, ignore_index=True)


def test_events_without_estimation_window_are_dropped_not_defaulted():
    prices = _price_frame()
    events = pd.DataFrame(
        {"ticker": ["T0"], "call_date": [pd.Timestamp("2018-02-01")]}  # too early
    )
    ep = eventstudy.run(events, prices)
    assert ep.events.empty
    assert ep.dropped["reason"].iloc[0] == "insufficient_estimation_window"


def test_abnormal_return_removes_market_move():
    """A pure-beta stock with no idiosyncratic move should show ~zero CAR."""
    dates = pd.bdate_range("2018-01-01", periods=800)
    rng = np.random.default_rng(3)
    mret = rng.normal(0.0003, 0.01, 800)
    mkt = 100 * np.exp(np.cumsum(mret))
    stock = 100 * np.exp(np.cumsum(1.5 * mret))  # beta 1.5, zero idio
    prices = pd.concat(
        [
            pd.DataFrame({"ticker": "T0", "date": dates, "close": stock, "adj_close": stock}),
            pd.DataFrame({"ticker": "MKT", "date": dates, "close": mkt, "adj_close": mkt}),
        ],
        ignore_index=True,
    )
    ev = pd.DataFrame({"ticker": ["T0"], "call_date": [dates[400]]})
    out = eventstudy.run(ev, prices).events
    assert abs(out["beta"].iloc[0] - 1.5) < 0.05
    assert abs(out["car_drift"].iloc[0]) < 0.01


def test_uncontrolled_spec_is_actually_uncontrolled():
    """Regression guard for the `controls or [...]` bug: an explicitly empty
    control list must mean no controls, not the default."""
    rng = np.random.default_rng(0)
    n = 300
    panel = pd.DataFrame(
        {
            "car_drift": rng.normal(0, 0.1, n),
            "car_announce": rng.normal(0, 0.05, n),
            "tone_score": rng.normal(0, 1, n),
            "call_date": pd.bdate_range("2020-01-01", periods=n),
        }
    )
    out = eventstudy.cross_sectional_regression(panel, controls=[])
    assert out["controls"] == "none"
    # And the placebo must not regress the dependent variable on itself.
    pl = eventstudy.cross_sectional_regression(panel, y="car_announce", controls=[])
    assert "error" not in pl and pl["controls"] == "none"


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------
def test_weights_are_dollar_neutral_when_both_legs_exist():
    pos = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [1.0, 0.0], "C": [-1.0, -1.0]},
        index=pd.bdate_range("2020-01-01", periods=2),
    )
    w = backtest.normalise(pos)
    assert np.isclose(w.iloc[0].sum(), 0.0)
    assert np.isclose(w.iloc[0].clip(lower=0).sum(), 1.0)
    assert np.isclose(w.iloc[0].clip(upper=0).sum(), -1.0)


def test_costs_reduce_returns():
    sig = pd.DataFrame(
        {
            "ticker": ["T0", "T1", "T2"],
            "entry_date": [pd.Timestamp("2019-01-07")] * 3,
            "tone_rank": [0.95, 0.5, 0.05],
        }
    )
    prices = _price_frame()
    free = backtest.run(sig, prices, cost_bps=0.0)
    dear = backtest.run(sig, prices, cost_bps=50.0)
    assert dear.daily["ret_net"].sum() < free.daily["ret_net"].sum()


def test_position_earns_next_day_return_not_same_day():
    """Guards the one-day lookahead that flatters naive backtests."""
    sig = pd.DataFrame(
        {"ticker": ["T0"], "entry_date": [pd.Timestamp("2019-01-07")], "tone_rank": [0.95]}
    )
    res = backtest.run(sig, _price_frame(), cost_bps=0.0, long_only=True)
    first_active = res.positions.index[(res.positions != 0).any(axis=1)][0]
    # No return is booked on the very first day the position exists.
    assert np.isclose(res.daily.loc[first_active, "ret_gross"], 0.0) or np.isnan(
        res.daily.loc[first_active, "ret_gross"]
    )


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def test_newey_west_widens_se_under_autocorrelation():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 0.01, 2000)
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    ar += 0.0005
    naive_t = ar.mean() / (ar.std(ddof=1) / np.sqrt(len(ar)))
    nw = st.newey_west_tstat(ar)
    assert abs(nw["t_stat"]) < abs(naive_t)


def test_bootstrap_ci_brackets_zero_for_noise():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.01, 1000)
    out = st.stationary_bootstrap_sharpe(noise, n_boot=300)
    assert out["ci_low"] < 0 < out["ci_high"]


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(9)
    r = rng.normal(0.0006, 0.01, 1200)
    v = 1.5e-4
    few = st.deflated_sharpe(r, n_trials=2, sr_variance=v)["dsr"]
    many = st.deflated_sharpe(r, n_trials=500, sr_variance=v)["dsr"]
    assert many < few


def test_benjamini_hochberg_rejects_lone_marginal_result():
    p = {f"spec{i}": 0.6 for i in range(19)}
    p["lucky"] = 0.04
    out = st.benjamini_hochberg(p)
    assert not out["lucky"]["survives_fdr"]
