"""Prove the pipeline before trusting it on real money questions.

Three checks, each of which can fail the build:

  A. SIGNAL WORLD  - drift was planted proportional to the tone innovation.
                     The differenced signal must find it.
  B. NULL WORLD    - identical generator, effect = 0. The pipeline must find
                     NOTHING. This is the more important test: any analysis
                     pipeline with enough freedom will produce a positive result
                     on noise, and the only way to know yours does not is to
                     feed it noise on purpose.
  C. FIXED-EFFECT  - in the signal world, the raw tone LEVEL must be markedly
     DISCRIMINATION   weaker than the differenced signal, because the generator
                      makes level mostly firm personality. This is what validates
                      the project's actual thesis rather than just its plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_synthetic import generate  # noqa: E402
from src import report  # noqa: E402
from src import stats as st  # noqa: E402
from src.pipeline import run as run_pipeline  # noqa: E402
from src.providers import Universe  # noqa: E402

SIGNAL_T_MIN = 2.5      # planted effect must clear this
NULL_T_MAX = 2.5        # null world must stay under this
LEVEL_RATIO_MAX = 0.6   # |t_level| must be < this * |t_differenced|
DSR_MIN = 0.95          # deflated Sharpe: pass in signal world, fail in null


def _run_world(tag: str, out: Path, effect: float, scorer: str, seed: int):
    data_dir = out / f"data_{tag}"
    print(f"\n=== {tag.upper()} WORLD (planted effect = {effect}) ===")
    info = generate(data_dir, n_firms=60, n_quarters=24, effect=effect, seed=seed)
    print(f"  generated {info['n_calls']} calls across {info['n_firms']} firms, "
          f"{info['n_delisted']} delisted")

    transcripts = pd.read_parquet(data_dir / "transcripts.parquet")
    prices = pd.read_parquet(data_dir / "prices.parquet")
    universe = Universe.from_csv(data_dir / "universe.csv")

    res = run_pipeline(
        transcripts=transcripts,
        prices=prices,
        universe=universe,
        scorer_kind=scorer,
        cache_path=data_dir / "llm_cache.sqlite",
        verbose=True,
    )
    report.save(res, out / f"report_{tag}", title=f"Validation — {tag} world")
    report.equity_curve_png(res, out / f"report_{tag}")
    return res


def main(out: Path = Path("output/validation"), scorer: str = "lexical") -> int:
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    sig = _run_world("signal", out, effect=0.04, scorer=scorer, seed=42)
    null = _run_world("null", out, effect=0.0, scorer=scorer, seed=7)

    t_sig = sig.regressions["differenced_signal"]["t_stat"]
    t_lvl = sig.regressions["raw_level_signal"]["t_stat"]
    t_null = null.regressions["differenced_signal"]["t_stat"]
    sharpe_sig = sig.backtests["base"].metrics.get("sharpe_net", float("nan"))
    sharpe_null = null.backtests["base"].metrics.get("sharpe_net", float("nan"))

    print("\n" + "=" * 68)
    print("VALIDATION RESULTS")
    print("=" * 68)

    print(f"\nA. Signal world detects the planted effect")
    print(f"   differenced signal t-stat = {t_sig:+.2f}   (need > {SIGNAL_T_MIN})")
    print(f"   base backtest net Sharpe  = {sharpe_sig:+.2f}")
    if not (t_sig > SIGNAL_T_MIN):
        failures.append(f"signal world t={t_sig:.2f} did not clear {SIGNAL_T_MIN}")

    print(f"\nB. Null world stays null")
    print(f"   differenced signal t-stat = {t_null:+.2f}   (need |t| < {NULL_T_MAX})")
    print(f"   base backtest net Sharpe  = {sharpe_null:+.2f}")
    if abs(t_null) >= NULL_T_MAX:
        failures.append(
            f"null world t={t_null:.2f} exceeded {NULL_T_MAX} — pipeline manufactures signal"
        )

    print(f"\nC. Differencing beats raw level (the actual thesis)")
    print(f"   differenced t = {t_sig:+.2f}   raw level t = {t_lvl:+.2f}")
    print(f"   ratio |level|/|diff| = {abs(t_lvl)/max(abs(t_sig),1e-9):.2f}   "
          f"(need < {LEVEL_RATIO_MAX})")
    if abs(t_lvl) >= LEVEL_RATIO_MAX * abs(t_sig):
        failures.append(
            f"raw level t={t_lvl:.2f} too close to differenced t={t_sig:.2f}; "
            "within-firm differencing is not doing the work it claims to"
        )

    print(f"\nD. Deflated Sharpe separates skill from a lucky search")
    dsrs = {}
    for tag, res in (("signal", sig), ("null", null)):
        net = res.backtests["base"].daily["ret_net"].dropna().values
        v = st.trial_sharpe_variance(
            [b.daily["ret_net"].dropna().values for b in res.backtests.values()]
        )
        dsrs[tag] = st.deflated_sharpe(net, n_trials=len(res.trials), sr_variance=v)["dsr"]
    print(f"   signal world DSR = {dsrs['signal']:.3f}   (need > {DSR_MIN})")
    print(f"   null   world DSR = {dsrs['null']:.3f}   (need < {DSR_MIN})")
    print(f"   (null world's raw Sharpe was {sharpe_null:+.2f} — a positive-looking")
    print(f"    equity curve on pure noise, which is exactly what deflation is for)")
    if not (dsrs["signal"] > DSR_MIN):
        failures.append(f"signal world DSR {dsrs['signal']:.3f} below {DSR_MIN}")
    if dsrs["null"] >= DSR_MIN:
        failures.append(
            f"null world DSR {dsrs['null']:.3f} above {DSR_MIN} — deflation is not "
            "penalising a search over noise"
        )

    print("\n" + "=" * 68)
    if failures:
        print("VALIDATION FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VALIDATION PASSED — pipeline finds planted signal, stays quiet on noise,")
    print("and the within-firm differencing is demonstrably what carries the result.")
    print(f"\nreports written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
