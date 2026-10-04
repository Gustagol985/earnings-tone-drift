"""Data providers: transcripts, prices, and the point-in-time universe.

Every provider returns the same shape, so the rest of the pipeline never learns
which vendor the data came from:

  transcripts: ticker, call_date, fiscal_period, text
  prices:      ticker, date, close, adj_close  (plus a 'MKT' pseudo-ticker)

On survivorship bias - the single most common way a study like this produces a
fake result. If you build the universe from *today's* index membership, you have
selected companies that survived the sample. Failures are precisely the cases
where deteriorating tone should predict the worst returns, so dropping them
removes the observations most favourable to the hypothesis... which sounds
conservative but is not: it also removes the bankruptcies whose -90% returns
would otherwise be attributed to whatever signal happened to flag them.

`Universe` therefore refuses to run in point-in-time mode without a membership
file carrying join/leave dates, and stamps `pit=False` on the output when you
override it, so `report.py` can print the caveat instead of you remembering to.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

TRANSCRIPT_COLS = ["ticker", "call_date", "fiscal_period", "text"]
PRICE_COLS = ["ticker", "date", "close", "adj_close"]


class ProviderError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
@dataclass
class Universe:
    members: pd.DataFrame  # ticker, start_date, end_date (NaT = still a member)
    pit: bool

    @classmethod
    def from_csv(cls, path: Path) -> "Universe":
        """Load a point-in-time membership file.

        The parsing here is deliberately strict. A universe file is hand-edited
        and its failure mode is silent: a stray line becomes a ticker that
        matches no price data, a date that fails to parse becomes a string that
        compares wrong, and either way the study runs to completion on a
        universe nobody intended. Better to refuse the file than to report a
        result derived from it.
        """
        # comment="#" is load-bearing: these files carry provenance notes for
        # every delisting, and without it each note is read as a company.
        df = pd.read_csv(path, comment="#", skipinitialspace=True)
        missing = {"ticker", "start_date", "end_date"} - set(df.columns)
        if missing:
            raise ProviderError(f"universe file missing columns: {sorted(missing)}")

        df["ticker"] = df["ticker"].astype(str).str.strip()
        df = df[df["ticker"].str.len() > 0].reset_index(drop=True)
        if df.empty:
            raise ProviderError(f"{path} contains no tickers")

        for col in ("start_date", "end_date"):
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")

        # start_date is mandatory; end_date is legitimately empty for firms that
        # are still members, so only a malformed one is an error.
        bad_start = df[df["start_date"].isna()]
        if not bad_start.empty:
            raise ProviderError(
                "unparseable start_date (expected YYYY-MM-DD) for: "
                f"{bad_start['ticker'].tolist()}"
            )
        raw_end = pd.read_csv(path, comment="#", skipinitialspace=True)["end_date"]
        raw_end = raw_end.reindex(df.index)
        bad_end = df[df["end_date"].isna() & raw_end.notna()]
        if not bad_end.empty:
            raise ProviderError(
                "unparseable end_date (expected YYYY-MM-DD) for: "
                f"{bad_end['ticker'].tolist()}"
            )

        if df["ticker"].duplicated().any():
            dupes = df.loc[df["ticker"].duplicated(), "ticker"].tolist()
            raise ProviderError(f"duplicate tickers in universe file: {dupes}")

        return cls(members=df, pit=True)

    @classmethod
    def from_ticker_list(cls, tickers: Iterable[str]) -> "Universe":
        """Escape hatch. Produces a survivorship-biased universe on purpose,
        flagged as such so downstream output carries the warning."""
        df = pd.DataFrame(
            {
                "ticker": list(tickers),
                "start_date": pd.Timestamp("1900-01-01"),
                "end_date": pd.NaT,
            }
        )
        return cls(members=df, pit=False)

    def active_on(self, when: pd.Timestamp) -> List[str]:
        m = self.members
        live = (m["start_date"] <= when) & (m["end_date"].isna() | (m["end_date"] >= when))
        return sorted(m.loc[live, "ticker"].tolist())

    def contains(self, ticker: str, when: pd.Timestamp) -> bool:
        return ticker in set(self.active_on(when))


# --------------------------------------------------------------------------
# Transcript providers
# --------------------------------------------------------------------------
class LocalTranscripts:
    """Read whatever `ingest` already wrote. Used by every rerun after the first."""

    name = "local"

    def __init__(self, path: Path) -> None:
        self.path = path

    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        if not self.path.exists():
            raise ProviderError(f"no local transcript store at {self.path}")
        df = pd.read_parquet(self.path)
        df["call_date"] = pd.to_datetime(df["call_date"])
        mask = (
            df["ticker"].isin(tickers)
            & (df["call_date"] >= pd.Timestamp(start))
            & (df["call_date"] <= pd.Timestamp(end))
        )
        return df.loc[mask, TRANSCRIPT_COLS].reset_index(drop=True)


class AlphaVantageTranscripts:
    """https://www.alphavantage.co/documentation/#earnings-call-transcript

    Free tier is 25 requests/day, which is a real constraint: one request is one
    ticker-quarter, so a 50-name x 20-quarter study is 1000 requests. Plan on
    either the paid tier or a long, resumable ingest - hence the parquet append
    and the skip-if-present logic in `ingest_transcripts`.
    """

    name = "alphavantage"
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None, sleep: float = 0.9) -> None:
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ProviderError("ALPHAVANTAGE_API_KEY not set")
        self.sleep = sleep

    def _quarters(self, start: str, end: str) -> List[str]:
        rng = pd.period_range(start=start, end=end, freq="Q")
        return [f"{p.year}Q{p.quarter}" for p in rng]

    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        rows = []
        for tkr in tickers:
            for q in self._quarters(start, end):
                params = {
                    "function": "EARNINGS_CALL_TRANSCRIPT",
                    "symbol": tkr,
                    "quarter": q,
                    "apikey": self.api_key,
                }
                r = requests.get(self.BASE, params=params, timeout=30)
                r.raise_for_status()
                js = r.json()
                if "Information" in js or "Note" in js:
                    raise ProviderError(
                        f"AlphaVantage rate limit hit: {js.get('Information') or js.get('Note')}"
                    )
                items = js.get("transcript") or []
                if not items:
                    continue
                text = "\n\n".join(
                    f"{seg.get('speaker','Unknown')}: {seg.get('content','')}"
                    for seg in items
                )
                rows.append(
                    {
                        "ticker": tkr,
                        "call_date": pd.NaT,  # filled from the earnings calendar
                        "fiscal_period": q,
                        "text": text,
                    }
                )
                time.sleep(self.sleep)
        return pd.DataFrame(rows, columns=TRANSCRIPT_COLS)


class FMPTranscripts:
    """Financial Modeling Prep. Returns the call date directly, which is why it
    is the better choice if you are paying for one of the two."""

    name = "fmp"
    BASE = "https://financialmodelingprep.com/api/v3/earning_call_transcript"

    def __init__(self, api_key: Optional[str] = None, sleep: float = 0.3) -> None:
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ProviderError("FMP_API_KEY not set")
        self.sleep = sleep

    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        rows = []
        for tkr in tickers:
            for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
                r = requests.get(
                    f"{self.BASE}/{tkr}",
                    params={"year": year, "apikey": self.api_key},
                    timeout=30,
                )
                r.raise_for_status()
                for item in r.json() or []:
                    rows.append(
                        {
                            "ticker": tkr,
                            "call_date": pd.to_datetime(item["date"]),
                            "fiscal_period": f"{item['year']}Q{item['quarter']}",
                            "text": item.get("content", ""),
                        }
                    )
                time.sleep(self.sleep)
        df = pd.DataFrame(rows, columns=TRANSCRIPT_COLS)
        if not df.empty:
            df = df[
                (df["call_date"] >= pd.Timestamp(start))
                & (df["call_date"] <= pd.Timestamp(end))
            ]
        return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Price providers
# --------------------------------------------------------------------------
class LocalPrices:
    name = "local"

    def __init__(self, path: Path) -> None:
        self.path = path

    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        if not self.path.exists():
            raise ProviderError(f"no local price store at {self.path}")
        df = pd.read_parquet(self.path)
        df["date"] = pd.to_datetime(df["date"])
        keep = set(tickers) | {"MKT"}
        mask = (
            df["ticker"].isin(keep)
            & (df["date"] >= pd.Timestamp(start))
            & (df["date"] <= pd.Timestamp(end))
        )
        return df.loc[mask, PRICE_COLS].reset_index(drop=True)


class YahooPrices:
    """Daily total-return prices via yfinance. No API key, no quota.

    `auto_adjust=True` returns prices adjusted for both splits and dividends,
    so `adj_close` here is a genuine total return rather than the price-only
    approximation the Stooq path provided. For a 63-day drift study the
    dividend component is small, but it is no longer an acknowledged gap.

    SPY stands in for the market factor and is stored under the 'MKT'
    pseudo-ticker the event study expects.
    """

    name = "yahoo"
    MARKET_PROXY = "SPY"

    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "yfinance not installed. `pip install yfinance`, or use "
                "--prices local if prices are already cached."
            ) from exc

        symbols = sorted(set(tickers) | {self.MARKET_PROXY})
        raw = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise ProviderError("Yahoo returned no price data")

        # One ticker yields flat columns, several yield a MultiIndex. Normalise
        # to a ticker-keyed frame so the rest of the function is shape-agnostic.
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            close.columns = [symbols[0]]

        frames = []
        missing = []
        for sym in symbols:
            if sym not in close.columns:
                missing.append(sym)
                continue
            s = close[sym].dropna()
            if s.empty:
                missing.append(sym)
                continue
            out_name = "MKT" if sym == self.MARKET_PROXY else sym
            frames.append(
                pd.DataFrame(
                    {
                        "ticker": out_name,
                        "date": pd.to_datetime(s.index),
                        "close": s.to_numpy(),
                        "adj_close": s.to_numpy(),
                    }
                )
            )

        if missing:
            # Surfaced rather than swallowed: a firm silently absent from the
            # price panel is dropped from the study with no trace in the report.
            print(f"  WARNING: no Yahoo price data for {sorted(missing)}")
        if not frames:
            raise ProviderError("Yahoo returned no usable series")
        return pd.concat(frames, ignore_index=True)[PRICE_COLS]


class StooqPrices:
    """Free daily OHLC, no key required. Adequate for a study at daily frequency.

    NOTE (Oct 2026): Stooq now serves a bot-protection page instead of CSV on
    every URL variant, so this path no longer works programmatically. Kept in
    case it is restored; use `yahoo` instead. Stooq history was also only
    split-adjusted, not dividend-adjusted, which `yahoo` fixes.
    """

    name = "stooq"
    BASE = "https://stooq.com/q/d/l/"

    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        frames = []
        for tkr in list(tickers) + ["MKT"]:
            sym = "spy.us" if tkr == "MKT" else f"{tkr.lower()}.us"
            r = requests.get(
                self.BASE,
                params={"s": sym, "d1": start.replace("-", ""), "d2": end.replace("-", ""), "i": "d"},
                timeout=30,
            )
            r.raise_for_status()
            if r.text.startswith("<") or "Date" not in r.text[:200]:
                continue
            from io import StringIO

            d = pd.read_csv(StringIO(r.text), parse_dates=["Date"])
            if d.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "ticker": tkr,
                        "date": d["Date"],
                        "close": d["Close"],
                        "adj_close": d["Close"],
                    }
                )
            )
            time.sleep(0.3)
        if not frames:
            raise ProviderError("Stooq returned no usable data")
        return pd.concat(frames, ignore_index=True)[PRICE_COLS]


TRANSCRIPT_PROVIDERS = {
    "local": LocalTranscripts,
    "alphavantage": AlphaVantageTranscripts,
    "fmp": FMPTranscripts,
}
PRICE_PROVIDERS = {"local": LocalPrices, "yahoo": YahooPrices, "stooq": StooqPrices}
