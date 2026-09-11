"""Resumable, budget-aware transcript download for the AlphaVantage free tier.

The free tier allows 25 requests per day. A 40-name x 20-quarter study needs
~840 requests, so the download runs for weeks. That makes three properties
non-negotiable, and they are the entire reason this script exists:

  RESUMABLE   Every successful fetch is written to disk and recorded in a state
              file before the next one starts. Kill it, lose power, hit the
              rate limit - rerunning picks up exactly where it stopped.

  BUDGETED    It stops itself after N requests instead of discovering the limit
              by getting rejected. A rejected request still costs you a slot.

  IDEMPOTENT  Re-running on the same day does nothing except report progress.
              You can put it on a scheduler and forget about it.

Two phases, because AlphaVantage splits the data across two endpoints:

  Phase 1  EARNINGS       one request per ticker, returns every historical
                          quarter's report date. Needed because the transcript
                          endpoint does NOT tell you when the call happened,
                          and without a date there is no event study.

  Phase 2  TRANSCRIPT     one request per ticker-quarter. The bulk of the spend.

Phase 1 runs first and completely. A transcript with no matching date is
unusable, so there is no point spending 800 requests on transcripts before
confirming the dates exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

BASE = "https://www.alphavantage.co/query"
TRANSCRIPT_COLS = ["ticker", "call_date", "fiscal_period", "text"]


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
class State:
    """Tracks what has already been fetched, so nothing is paid for twice.

    Entries are recorded for empty results too, marked "empty" rather than
    "done". Without that distinction a quarter with no transcript would be
    retried every single run, and on a 25/day budget a handful of those would
    eat the entire allowance forever.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = {}
        self.earnings: Dict[str, str] = raw.get("earnings", {})
        self.transcripts: Dict[str, str] = raw.get("transcripts", {})
        self.spend: Dict[str, int] = raw.get("spend", {})
        # AlphaVantage's quota resets on a rolling ~24h window from the first
        # request of a batch, not at local midnight. Tracking the real clock
        # time lets the script say when to come back instead of leaving the
        # user to guess from a refusal that claims "0 used today".
        self.first_request_at: Optional[str] = raw.get("first_request_at")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "earnings": self.earnings,
                    "transcripts": self.transcripts,
                    "spend": self.spend,
                    "first_request_at": self.first_request_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def spent_today(self) -> int:
        return self.spend.get(date.today().isoformat(), 0)

    def charge(self, n: int = 1) -> None:
        today = date.today().isoformat()
        self.spend[today] = self.spend.get(today, 0) + n

    def note_request(self) -> None:
        """Stamp the start of a quota window on the first request of a batch."""
        if self.first_request_at is None:
            self.first_request_at = datetime.now().isoformat(timespec="seconds")

    def window_opens_at(self) -> Optional[datetime]:
        """When the rolling quota should next allow requests, if known."""
        if not self.first_request_at:
            return None
        return datetime.fromisoformat(self.first_request_at) + timedelta(hours=24)

    def clear_window(self) -> None:
        self.first_request_at = None


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def append_rows(path: Path, rows: List[dict], cols: List[str]) -> None:
    """Append to a parquet file, rewriting it atomically.

    Parquet has no true append, so this reads, concatenates and rewrites via a
    temp file. At a few hundred rows that is cheap, and writing to a temp file
    first means an interruption mid-write cannot corrupt the store you have
    spent weeks filling.
    """
    if not rows:
        return
    new = pd.DataFrame(rows, columns=cols)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    new.to_parquet(tmp, index=False)
    tmp.replace(path)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
class RateLimited(RuntimeError):
    pass


def _get(params: dict, api_key: str, timeout: int = 30) -> dict:
    r = requests.get(BASE, params={**params, "apikey": api_key}, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    # AlphaVantage signals the daily cap with a 200 and a prose message rather
    # than a 429, so this has to be detected by content, not status code.
    msg = js.get("Information") or js.get("Note") or js.get("Error Message")
    if msg:
        if "rate limit" in str(msg).lower() or "premium" in str(msg).lower():
            raise RateLimited(str(msg))
        raise RuntimeError(f"AlphaVantage error: {msg}")
    return js


def fetch_earnings_dates(ticker: str, api_key: str) -> Dict[str, str]:
    """Map fiscal period -> report date, e.g. {'2023Q1': '2023-04-27'}."""
    js = _get({"function": "EARNINGS", "symbol": ticker}, api_key)
    out: Dict[str, str] = {}
    for q in js.get("quarterlyEarnings") or []:
        ending, reported = q.get("fiscalDateEnding"), q.get("reportedDate")
        if not ending or not reported:
            continue
        ts = pd.Timestamp(ending)
        out[f"{ts.year}Q{ts.quarter}"] = reported
    return out


def fetch_transcript(ticker: str, quarter: str, api_key: str) -> Optional[str]:
    js = _get(
        {"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": ticker, "quarter": quarter},
        api_key,
    )
    items = js.get("transcript") or []
    if not items:
        return None
    return "\n\n".join(
        f"{seg.get('speaker', 'Unknown')}: {seg.get('content', '')}" for seg in items
    )


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
def quarters_between(start: str, end: str) -> List[str]:
    rng = pd.period_range(start=start, end=end, freq="Q")
    return [f"{p.year}Q{p.quarter}" for p in rng]


def load_universe(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(
            f"no universe file at {path}\n"
            "Create it first — see `python -m scripts.ingest template --help`."
        )
    # comment="#" matters: the template ships with explanatory '#' lines, and
    # without this pandas reads them as tickers and the downloader spends real
    # requests asking AlphaVantage about a symbol named '#'.
    df = pd.read_csv(path, comment="#")
    if "ticker" not in df.columns:
        raise SystemExit(f"{path} needs at least a 'ticker' column")
    tickers = (
        df["ticker"].astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    )
    if not tickers:
        raise SystemExit(f"{path} has no tickers in it yet — add some rows.")
    return sorted(tickers)


# --------------------------------------------------------------------------
# main run
# --------------------------------------------------------------------------
def run(
    data_dir: Path,
    start: str,
    end: str,
    budget: int,
    api_key: str,
    sleep: float = 1.0,
) -> int:
    state = State(data_dir / "ingest_state.json")
    tickers = load_universe(data_dir / "universe.csv")
    quarters = quarters_between(start, end)
    dates_path = data_dir / "earnings_dates.json"
    tx_path = data_dir / "transcripts.parquet"

    # If a full 24h has elapsed since this window opened, start a fresh one.
    opens = state.window_opens_at()
    if opens is not None and datetime.now() >= opens:
        state.clear_window()
        opens = None

    already = state.spent_today()
    remaining = max(budget - already, 0)
    print(f"budget {budget}/day · already used today: {already} · remaining: {remaining}")
    if opens is not None and already == 0:
        print(
            f"note: quota window opened {state.first_request_at}; "
            f"the vendor may not reset until {opens:%H:%M on %d %b}"
        )
    if remaining == 0:
        print("Nothing to do today. Run again tomorrow.")
        report_progress(state, tickers, quarters)
        return 0

    all_dates: Dict[str, Dict[str, str]] = (
        json.loads(dates_path.read_text(encoding="utf-8")) if dates_path.exists() else {}
    )
    used = 0

    # -- phase 1: earnings dates -------------------------------------------
    todo_dates = [t for t in tickers if state.earnings.get(t) != "done"]
    if todo_dates:
        print(f"\n[phase 1] earnings dates — {len(todo_dates)} tickers remaining")
    for tkr in todo_dates:
        if used >= remaining:
            break
        try:
            all_dates[tkr] = fetch_earnings_dates(tkr, api_key)
        except RateLimited as exc:
            print(f"  rate limit reached: {exc}")
            state.charge(used)
            state.save()
            _flush_dates(dates_path, all_dates)
            report_progress(state, tickers, quarters)
            return 0
        used += 1
        state.earnings[tkr] = "done"
        _flush_dates(dates_path, all_dates)
        state.save()
        print(f"  {tkr}: {len(all_dates[tkr])} quarters dated  ({used}/{remaining})")
        time.sleep(sleep)

    # Transcripts without dates are unusable, so don't start phase 2 early.
    if any(state.earnings.get(t) != "done" for t in tickers):
        state.charge(used)
        state.save()
        print("\nPhase 1 not finished — stopping before transcripts.")
        report_progress(state, tickers, quarters)
        return 0

    # -- phase 2: transcripts ----------------------------------------------
    pending = [
        (t, q)
        for t in tickers
        for q in quarters
        if state.transcripts.get(f"{t}:{q}") is None
        # Skip quarters the earnings endpoint never reported — no date, no event.
        and q in all_dates.get(t, {})
    ]
    if pending:
        print(f"\n[phase 2] transcripts — {len(pending)} ticker-quarters remaining")

    for tkr, q in pending:
        if used >= remaining:
            break
        try:
            text = fetch_transcript(tkr, q, api_key)
        except RateLimited as exc:
            print(f"  rate limit reached: {exc}")
            break
        used += 1
        if text:
            append_rows(
                tx_path,
                [
                    {
                        "ticker": tkr,
                        "call_date": pd.Timestamp(all_dates[tkr][q]),
                        "fiscal_period": q,
                        "text": text,
                    }
                ],
                TRANSCRIPT_COLS,
            )
            state.transcripts[f"{tkr}:{q}"] = "done"
            print(f"  {tkr} {q}: {len(text.split()):>6} words  ({used}/{remaining})")
        else:
            state.transcripts[f"{tkr}:{q}"] = "empty"
            print(f"  {tkr} {q}: no transcript available  ({used}/{remaining})")
        state.save()
        time.sleep(sleep)

    state.charge(used)
    state.save()
    print(f"\nUsed {used} requests this run.")
    report_progress(state, tickers, quarters)
    return 0


def _flush_dates(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def report_progress(state: State, tickers: List[str], quarters: List[str]) -> None:
    dated = sum(1 for t in tickers if state.earnings.get(t) == "done")
    got = sum(1 for v in state.transcripts.values() if v == "done")
    empty = sum(1 for v in state.transcripts.values() if v == "empty")
    target = len(tickers) * len(quarters)
    left = max(target - len(state.transcripts), 0) + (len(tickers) - dated)

    print("\n" + "=" * 58)
    print("PROGRESS")
    print("=" * 58)
    print(f"  tickers dated       : {dated}/{len(tickers)}")
    print(f"  transcripts stored  : {got}")
    print(f"  quarters with none  : {empty}")
    print(f"  requests remaining  : ~{left}")
    if left:
        print(f"  days left at 25/day : ~{-(-left // 25)}")
    else:
        print("  DOWNLOAD COMPLETE — you can run the study.")
    print("=" * 58)


def probe(
    ticker: str,
    quarter: Optional[str],
    api_key: str,
    skip_dates: bool = False,
    sleep: float = 2.0,
) -> int:
    """Test one ticker directly, bypassing the saved state.

    Exists because the two endpoints can disagree: a delisted firm may return
    no earnings dates while still having transcripts, or vice versa. The main
    loop skips any quarter with no date, so it can never discover that on its
    own. This costs 1-2 requests and answers the question definitively.
    """
    print(f"probing {ticker}...")
    dates: Dict[str, str] = {}
    if not skip_dates:
        try:
            dates = fetch_earnings_dates(ticker, api_key)
        except RateLimited as exc:
            print(f"  rate limited: {exc}")
            return 1
        print(f"  EARNINGS endpoint: {len(dates)} quarters")
        if dates:
            newest = sorted(dates)[-1]
            print(f"    newest: {newest} reported {dates[newest]}")

    if quarter is None:
        print("\n(no --quarter given, so the transcript endpoint was not tested)")
        return 0

    # AlphaVantage enforces roughly one request per second on free keys, and
    # returns the daily-limit message for a burst violation too — so without
    # this pause the second call looks like an exhausted quota when it isn't.
    if not skip_dates:
        time.sleep(sleep)

    try:
        text = fetch_transcript(ticker, quarter, api_key)
    except RateLimited as exc:
        print(f"  rate limited: {exc}")
        return 1
    if text:
        print(f"  TRANSCRIPT endpoint: {len(text.split())} words for {quarter}")
        print(f"    opens: {text[:160]!r}")
        if not dates:
            print(
                "\n  NOTE: transcripts exist but earnings dates do not.\n"
                "  The main loop skips quarters with no date, so this ticker\n"
                "  would be silently dropped. Worth handling explicitly."
            )
    else:
        print(f"  TRANSCRIPT endpoint: nothing for {quarter}")
    return 0


def write_template(path: Path) -> int:
    """Write a starter universe.csv the user fills in by hand.

    Deliberately includes a delisted example row. The end_date column is the
    survivorship-bias control: leaving it blank for a firm that was actually
    removed silently claims it survived, which is the single most common way a
    study like this produces a result that isn't real.
    """
    if path.exists():
        print(f"{path} already exists — not overwriting.")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ticker,start_date,end_date\n"
        "AAPL,1900-01-01,\n"
        "MSFT,1900-01-01,\n"
        "# end_date blank = still a member today.\n"
        "# Fill it in for any firm that left the index, e.g.:\n"
        "# FRC,1900-01-01,2023-05-04\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")
    print("Edit it: one row per company, and fill end_date for anything delisted.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="write a starter universe.csv")
    t.add_argument("--data-dir", type=Path, default=Path("data/real"))

    r = sub.add_parser("run", help="download today's allowance")
    r.add_argument("--data-dir", type=Path, default=Path("data/real"))
    r.add_argument("--start", default="2019-01-01")
    r.add_argument("--end", default="2025-12-31")
    r.add_argument("--budget", type=int, default=25, help="requests per day")
    r.add_argument("--sleep", type=float, default=1.0)

    p = sub.add_parser(
        "probe",
        help="spend 1-2 requests testing one ticker, without touching saved state",
    )
    p.add_argument("ticker")
    p.add_argument("--quarter", default=None, help="e.g. 2022Q4; omit to test dates only")
    p.add_argument(
        "--transcript-only",
        action="store_true",
        help="skip the earnings-date call — costs 1 request instead of 2",
    )

    s = sub.add_parser("status", help="show progress without spending requests")
    s.add_argument("--data-dir", type=Path, default=Path("data/real"))
    s.add_argument("--start", default="2019-01-01")
    s.add_argument("--end", default="2025-12-31")

    a = ap.parse_args(argv)

    if a.cmd == "template":
        return write_template(a.data_dir / "universe.csv")

    if a.cmd == "probe":
        key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if not key:
            print("ALPHAVANTAGE_API_KEY is not set.", file=sys.stderr)
            return 1
        return probe(a.ticker, a.quarter, key, skip_dates=a.transcript_only)

    if a.cmd == "status":
        st = State(a.data_dir / "ingest_state.json")
        report_progress(
            st, load_universe(a.data_dir / "universe.csv"), quarters_between(a.start, a.end)
        )
        return 0

    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        print("ALPHAVANTAGE_API_KEY is not set.", file=sys.stderr)
        print("Get a free key at https://www.alphavantage.co/support/#api-key", file=sys.stderr)
        return 1
    return run(a.data_dir, a.start, a.end, a.budget, key, a.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
