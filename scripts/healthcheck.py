"""Inspect downloaded data for problems. Free, offline, run as often as you like.

`ingest status` answers "is it downloading?". This answers the harder question:
"is what arrived actually usable?" Those fail independently — a vendor can
return 200 OK with a paywall notice, a truncated stub, or the same boilerplate
for every quarter, and the downloader will happily store all of it.

Every check here has a stated pass condition, so a green run means something
specific rather than "no exception was raised". Checks are advisory: they print
WARN or FAIL and keep going, because during a five-week download most failures
are simply "not enough data yet" and shouldn't stop you looking at the rest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.providers import Universe  # noqa: E402
from src.scoring import lexical  # noqa: E402

# A real earnings call runs roughly 5,000-12,000 words. Much below that and you
# are probably looking at a stub, a paywall notice, or prepared remarks with the
# Q&A missing - and Q&A is where two of the six rubric axes live.
MIN_PLAUSIBLE_WORDS = 2000
SHORT_WORDS = 3500

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "


class Report:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    def render(self) -> int:
        print("\n" + "=" * 72)
        print("DATA HEALTH CHECK")
        print("=" * 72)
        for status, name, detail in self.rows:
            print(f"[{status}] {name}")
            if detail:
                for line in detail.splitlines():
                    print(f"         {line}")
        n_fail = sum(1 for s, _, _ in self.rows if s == FAIL)
        n_warn = sum(1 for s, _, _ in self.rows if s == WARN)
        print("=" * 72)
        print(f"{len(self.rows)} checks · {n_fail} failed · {n_warn} warnings")
        print("=" * 72)
        return 1 if n_fail else 0


def check_download_progress(data_dir: Path, rep: Report) -> None:
    state_path = data_dir / "ingest_state.json"
    if not state_path.exists():
        rep.add(FAIL, "download has started", "no ingest_state.json — run `ingest run`")
        return
    st = json.loads(state_path.read_text(encoding="utf-8"))
    dated = len(st.get("earnings", {}))
    tx = sum(1 for v in st.get("transcripts", {}).values() if v == "done")
    empty = sum(1 for v in st.get("transcripts", {}).values() if v == "empty")
    rep.add(OK, "download has started", f"{dated} tickers dated · {tx} transcripts · {empty} empty")

    # An empty rate far above ~20% usually means the vendor's coverage is thinner
    # than the plan assumed, and the sample will end up smaller than intended.
    total = tx + empty
    if total >= 20:
        rate = empty / total
        if rate > 0.35:
            rep.add(
                WARN,
                "vendor coverage",
                f"{rate:.0%} of attempted quarters returned nothing.\n"
                "Expect a smaller sample than planned.",
            )
        else:
            rep.add(OK, "vendor coverage", f"{rate:.0%} of quarters empty")


def check_universe(data_dir: Path, rep: Report) -> Universe | None:
    path = data_dir / "universe.csv"
    if not path.exists():
        rep.add(FAIL, "universe file", "missing")
        return None
    try:
        u = Universe.from_csv(path)
    except Exception as exc:
        rep.add(FAIL, "universe file parses", str(exc))
        return None
    delisted = u.members["end_date"].notna().sum()
    rep.add(
        OK,
        "universe file parses",
        f"{len(u.members)} companies · {delisted} with a removal date",
    )
    if delisted == 0:
        rep.add(
            WARN,
            "survivorship control",
            "no delisted companies — results will be survivor-biased",
        )
    return u


def check_transcripts(data_dir: Path, universe: Universe | None, rep: Report) -> pd.DataFrame | None:
    path = data_dir / "transcripts.parquet"
    if not path.exists():
        rep.add(
            WARN,
            "transcripts downloaded",
            "none yet — phase 2 has not started (this is normal early on)",
        )
        return None
    df = pd.read_parquet(path)
    df["call_date"] = pd.to_datetime(df["call_date"])
    rep.add(OK, "transcripts downloaded", f"{len(df)} calls across {df['ticker'].nunique()} firms")

    if df.duplicated(["ticker", "fiscal_period"]).any():
        d = df[df.duplicated(["ticker", "fiscal_period"], keep=False)]
        rep.add(FAIL, "no duplicates", f"{len(d)} duplicated ticker/quarter rows")
    else:
        rep.add(OK, "no duplicates")

    if df["call_date"].isna().any():
        rep.add(FAIL, "every call has a date", f"{df['call_date'].isna().sum()} missing")
    elif (df["call_date"] > pd.Timestamp.today()).any():
        rep.add(FAIL, "no future-dated calls", "some call dates are in the future")
    else:
        rep.add(
            OK,
            "call dates sane",
            f"{df['call_date'].min().date()} to {df['call_date'].max().date()}",
        )

    words = df["text"].str.split().str.len()
    tiny = int((words < MIN_PLAUSIBLE_WORDS).sum())
    short = int((words < SHORT_WORDS).sum())
    detail = (
        f"median {int(words.median())} words · "
        f"range {int(words.min())}-{int(words.max())}"
    )
    if tiny:
        rep.add(
            FAIL,
            "transcripts are full length",
            detail + f"\n{tiny} under {MIN_PLAUSIBLE_WORDS} words — likely stubs, not transcripts",
        )
    elif short:
        rep.add(WARN, "transcripts are full length", detail + f"\n{short} on the short side")
    else:
        rep.add(OK, "transcripts are full length", detail)

    # Two of the six rubric axes (evasion, analyst pushback) are measured only in
    # the Q&A. Some vendor transcripts ship prepared remarks alone, which scores
    # those axes on empty text and returns a confident-looking number anyway.
    usable = df["text"].apply(lexical.has_usable_qa)
    frac = float(usable.mean())
    bad = df.loc[~usable, ["ticker", "fiscal_period"]]
    listing = ", ".join(f"{r.ticker} {r.fiscal_period}" for r in bad.itertuples())
    detail = f"{frac:.0%} of calls usable"
    if listing:
        detail += f"\nprepared-remarks only: {listing}\nexclude these before the final run"
    if frac < 0.5:
        rep.add(FAIL, "Q&A usable for scoring", detail)
    elif frac < 0.95:
        rep.add(WARN, "Q&A usable for scoring", detail)
    else:
        rep.add(OK, "Q&A usable for scoring", detail)

    # Identical text across quarters means the vendor is serving a placeholder.
    dupe_text = int(df["text"].duplicated().sum())
    if dupe_text:
        rep.add(FAIL, "transcripts are distinct", f"{dupe_text} byte-identical texts")
    else:
        rep.add(OK, "transcripts are distinct")

    if universe is not None:
        listed = set(universe.members["ticker"])
        stray = sorted(set(df["ticker"]) - listed)
        if stray:
            rep.add(FAIL, "all firms are in the universe", f"unexpected: {stray}")
        else:
            rep.add(OK, "all firms are in the universe")

        # The point-in-time file can claim a control the data does not deliver.
        delisted = universe.members.dropna(subset=["end_date"])["ticker"].tolist()
        absent = [t for t in delisted if t not in set(df["ticker"])]
        if delisted and len(absent) == len(delisted):
            rep.add(
                FAIL,
                "delisted firms actually present",
                f"none of {delisted} returned data.\n"
                "The universe claims point-in-time construction but the sample\n"
                "is survivor-only. This must be stated in the writeup.",
            )
        elif absent:
            rep.add(WARN, "delisted firms actually present", f"missing: {absent}")
        elif delisted:
            rep.add(OK, "delisted firms actually present", f"{len(delisted)} contributing")

    return df


def check_scoring(df: pd.DataFrame | None, rep: Report) -> None:
    """Score a sample and confirm the numbers move.

    A scorer that returns near-identical scores for every call is worse than a
    broken one: the pipeline runs, the report renders, and every result is noise
    around a constant. Cheap to check, and it only needs a handful of calls.
    """
    if df is None or len(df) < 5:
        rep.add(WARN, "scorer produces varying scores", "not enough transcripts yet")
        return
    sample = df.sample(min(len(df), 25), random_state=0)
    scores = pd.DataFrame([lexical.score_text(t).as_dict() for t in sample["text"]])

    flat = [c for c in scores.columns if scores[c].std() < 0.01]
    spread = " · ".join(f"{c}={scores[c].mean():.2f}" for c in scores.columns)
    if flat:
        rep.add(
            FAIL,
            "scorer produces varying scores",
            f"no variation in: {flat}\nthose axes cannot carry signal",
        )
    else:
        rep.add(OK, "scorer produces varying scores", spread)


def check_ready_to_run(df: pd.DataFrame | None, rep: Report) -> None:
    """Can the study run yet? Needs 5+ calls per firm: 4 to build a baseline,
    1 to measure against it."""
    if df is None:
        rep.add(WARN, "enough history to run the study", "no transcripts yet")
        return
    per = df.groupby("ticker").size()
    ready = int((per >= 5).sum())
    events = int((per[per >= 5] - 4).sum())
    detail = f"{ready} firms have 5+ calls · ~{events} usable events so far"
    if events < 100:
        rep.add(WARN, "enough history to run the study", detail + "\nkeep downloading")
    else:
        rep.add(OK, "enough history to run the study", detail)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/real"))
    a = ap.parse_args(argv)

    rep = Report()
    check_download_progress(a.data_dir, rep)
    universe = check_universe(a.data_dir, rep)
    df = check_transcripts(a.data_dir, universe, rep)
    check_scoring(df, rep)
    check_ready_to_run(df, rep)
    return rep.render()


if __name__ == "__main__":
    raise SystemExit(main())
