"""Tests for the resumable download.

This runs unattended for weeks against a hard daily quota, so the properties
that matter are not "does it fetch" but "does it lose or repeat work". Every
test here fakes the API and checks bookkeeping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import ingest  # noqa: E402


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "real"
    d.mkdir()
    pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "start_date": ["1900-01-01"] * 2, "end_date": [None, None]}
    ).to_csv(d / "universe.csv", index=False)
    return d


def _fake_api(monkeypatch, calls: list, empty_for=(), fail_after=None):
    """Replace both endpoints with counters, so budget logic is observable."""

    def earnings(ticker, api_key):
        calls.append(("earnings", ticker))
        if fail_after is not None and len(calls) > fail_after:
            raise ingest.RateLimited("hit the cap")
        return {"2023Q1": "2023-04-20", "2023Q2": "2023-07-20"}

    def transcript(ticker, quarter, api_key):
        calls.append(("transcript", ticker, quarter))
        if fail_after is not None and len(calls) > fail_after:
            raise ingest.RateLimited("hit the cap")
        if (ticker, quarter) in empty_for:
            return None
        return f"Operator: welcome. CEO: results for {quarter}. " * 40

    monkeypatch.setattr(ingest, "fetch_earnings_dates", earnings)
    monkeypatch.setattr(ingest, "fetch_transcript", transcript)


def test_budget_is_respected(data_dir, monkeypatch):
    calls: list = []
    _fake_api(monkeypatch, calls)
    ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=3, api_key="x", sleep=0)
    assert len(calls) == 3


def test_resume_does_not_refetch(data_dir, monkeypatch):
    """The core property: a second run must not pay for the first run's work."""
    calls: list = []
    _fake_api(monkeypatch, calls)

    ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=2, api_key="x", sleep=0)
    first = list(calls)
    assert first == [("earnings", "AAA"), ("earnings", "BBB")]

    # New day — reset the spend ledger, keep the fetched-work ledger.
    state = ingest.State(data_dir / "ingest_state.json")
    state.spend = {}
    state.save()

    calls.clear()
    ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=4, api_key="x", sleep=0)
    assert all(kind != "earnings" for kind, *_ in calls), "refetched already-known dates"
    assert len(calls) == 4


def test_same_day_rerun_is_free(data_dir, monkeypatch):
    calls: list = []
    _fake_api(monkeypatch, calls)
    ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=2, api_key="x", sleep=0)
    n = len(calls)
    ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=2, api_key="x", sleep=0)
    assert len(calls) == n, "second run the same day spent more requests"


def test_transcripts_persist_across_runs(data_dir, monkeypatch):
    calls: list = []
    _fake_api(monkeypatch, calls)
    for _ in range(4):
        st = ingest.State(data_dir / "ingest_state.json")
        st.spend = {}
        st.save()
        ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=2, api_key="x", sleep=0)

    df = pd.read_parquet(data_dir / "transcripts.parquet")
    assert len(df) == 4  # 2 tickers x 2 quarters
    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert df["call_date"].notna().all(), "call dates missing — event study impossible"
    assert not df.duplicated(["ticker", "fiscal_period"]).any()


def test_empty_quarters_are_not_retried(data_dir, monkeypatch):
    """A quarter with no transcript must be remembered, or it eats the quota daily."""
    calls: list = []
    _fake_api(monkeypatch, calls, empty_for={("AAA", "2023Q1")})
    for _ in range(4):
        st = ingest.State(data_dir / "ingest_state.json")
        st.spend = {}
        st.save()
        ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=2, api_key="x", sleep=0)

    calls.clear()
    st = ingest.State(data_dir / "ingest_state.json")
    st.spend = {}
    st.save()
    ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=5, api_key="x", sleep=0)
    assert calls == [], "retried a quarter already known to be empty"


def test_rate_limit_mid_run_keeps_completed_work(data_dir, monkeypatch):
    calls: list = []
    _fake_api(monkeypatch, calls, fail_after=3)
    ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=25, api_key="x", sleep=0)

    state = json.loads((data_dir / "ingest_state.json").read_text())
    assert state["earnings"] == {"AAA": "done", "BBB": "done"}
    assert len(state["transcripts"]) == 1, "work completed before the limit was lost"
    assert (data_dir / "transcripts.parquet").exists()


def test_phase_two_waits_for_phase_one(data_dir, monkeypatch):
    """Transcripts without dates are useless — don't buy them first."""
    calls: list = []
    _fake_api(monkeypatch, calls)
    ingest.run(data_dir, "2023-01-01", "2023-06-30", budget=1, api_key="x", sleep=0)
    assert all(kind == "earnings" for kind, *_ in calls)


def test_template_refuses_to_clobber(data_dir, capsys):
    p = data_dir / "universe.csv"
    before = p.read_text()
    assert ingest.write_template(p) == 1
    assert p.read_text() == before


def test_comment_lines_are_not_tickers(tmp_path):
    """The shipped template contains '#' notes — they must not become symbols."""
    p = tmp_path / "universe.csv"
    ingest.write_template(p)
    assert ingest.load_universe(p) == ["AAPL", "MSFT"]


def test_empty_universe_fails_loudly(tmp_path):
    p = tmp_path / "universe.csv"
    p.write_text("ticker,start_date,end_date\n# nothing yet\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        ingest.load_universe(p)


def test_append_is_atomic(tmp_path):
    p = tmp_path / "t.parquet"
    ingest.append_rows(p, [{"ticker": "A", "call_date": pd.Timestamp("2020-01-01"),
                            "fiscal_period": "2020Q1", "text": "x"}], ingest.TRANSCRIPT_COLS)
    ingest.append_rows(p, [{"ticker": "B", "call_date": pd.Timestamp("2020-01-02"),
                            "fiscal_period": "2020Q1", "text": "y"}], ingest.TRANSCRIPT_COLS)
    assert len(pd.read_parquet(p)) == 2
    assert not list(tmp_path.glob("*.tmp.parquet")), "temp file left behind"
