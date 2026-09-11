"""Command line entry point.

  python -m src.cli synth      generate the synthetic validation world
  python -m src.cli validate   prove the pipeline on planted-signal + null worlds
  python -m src.cli run        run the study on whatever data is in --data-dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import report
from .pipeline import run as run_pipeline
from .providers import PRICE_PROVIDERS, TRANSCRIPT_PROVIDERS, Universe


def _load(data_dir: Path, tp: str, pp: str, tickers, start: str, end: str):
    tprov = (
        TRANSCRIPT_PROVIDERS[tp](data_dir / "transcripts.parquet")
        if tp == "local"
        else TRANSCRIPT_PROVIDERS[tp]()
    )
    pprov = (
        PRICE_PROVIDERS[pp](data_dir / "prices.parquet")
        if pp == "local"
        else PRICE_PROVIDERS[pp]()
    )
    uni_path = data_dir / "universe.csv"
    if uni_path.exists():
        universe = Universe.from_csv(uni_path)
    else:
        if not tickers:
            raise SystemExit(
                f"no universe file at {uni_path} and no --tickers given. "
                "A point-in-time membership file (ticker,start_date,end_date) is "
                "strongly preferred; --tickers produces a survivorship-biased run."
            )
        universe = Universe.from_ticker_list(tickers)

    all_tickers = sorted(universe.members["ticker"].unique())
    transcripts = tprov.fetch(all_tickers, start, end)
    prices = pprov.fetch(all_tickers, start, end)
    return transcripts, prices, universe


def cmd_run(a) -> int:
    transcripts, prices, universe = _load(
        a.data_dir, a.transcripts, a.prices, a.tickers, a.start, a.end
    )
    if transcripts.empty:
        print("no transcripts fetched — nothing to do", file=sys.stderr)
        return 1
    print(f"loaded {len(transcripts)} transcripts, {len(prices)} price rows")

    res = run_pipeline(
        transcripts=transcripts,
        prices=prices,
        universe=universe,
        scorer_kind=a.scorer,
        cache_path=a.data_dir / "llm_cache.sqlite",
        model=a.model,
        hold_days=a.hold_days,
        quantile=a.quantile,
        cost_bps=a.cost_bps,
    )
    path = report.save(res, a.out, title=a.title)
    png = report.equity_curve_png(res, a.out)
    print(f"\nreport: {path}")
    if png:
        print(f"chart:  {png}")
    return 0


def cmd_synth(a) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.make_synthetic import generate

    info = generate(a.out, a.firms, a.quarters, a.effect, a.seed)
    for k, v in info.items():
        print(f"{k:>14}: {v}")
    return 0


def cmd_validate(a) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.validate import main as validate_main

    return validate_main(a.out, scorer=a.scorer)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="earnings-tone-drift", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the study")
    r.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    r.add_argument("--out", type=Path, default=Path("output"))
    r.add_argument("--transcripts", choices=list(TRANSCRIPT_PROVIDERS), default="local")
    r.add_argument("--prices", choices=list(PRICE_PROVIDERS), default="local")
    r.add_argument("--scorer", choices=["lexical", "llm"], default="lexical")
    r.add_argument("--model", default="claude-sonnet-5")
    r.add_argument("--tickers", nargs="*", default=None)
    r.add_argument("--start", default="2015-01-01")
    r.add_argument("--end", default="2025-12-31")
    r.add_argument("--hold-days", type=int, default=63)
    r.add_argument("--quantile", type=float, default=0.2)
    r.add_argument("--cost-bps", type=float, default=10.0)
    r.add_argument("--title", default="Earnings Call Tone Drift")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("synth", help="generate synthetic data")
    s.add_argument("--out", type=Path, default=Path("data/synthetic"))
    s.add_argument("--firms", type=int, default=60)
    s.add_argument("--quarters", type=int, default=24)
    s.add_argument("--effect", type=float, default=0.04)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_synth)

    v = sub.add_parser("validate", help="prove the pipeline on known worlds")
    v.add_argument("--out", type=Path, default=Path("output/validation"))
    v.add_argument("--scorer", choices=["lexical", "llm"], default="lexical")
    v.set_defaults(func=cmd_validate)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
