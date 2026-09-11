"""Statistical honesty utilities.

A backtest with a positive Sharpe and no discussion of multiple testing is not
evidence. This module supplies the three corrections that make the difference
between a result and a coincidence:

* Newey-West t-stats, because overlapping 63-day holding periods induce
  autocorrelation that inflates naive t-stats by roughly sqrt(overlap).
* A stationary bootstrap CI on the Sharpe ratio, because the point estimate on
  ~5 years of daily data has a standard error around 0.45 - wide enough that
  "Sharpe 0.8" and "Sharpe 0" are frequently the same experiment.
* The deflated Sharpe ratio, which asks the question a portfolio manager will
  ask out loud: how many variants did you try before this one worked?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as sps

TRADING_DAYS = 252


@dataclass
class TrialLog:
    """Counts every distinct specification evaluated during a session.

    Whatever this number is, it belongs in the writeup. Reporting the best of 20
    configurations as though it were the only one tried is the difference
    between research and a sales pitch.
    """

    trials: List[str] = field(default_factory=list)

    def record(self, name: str) -> None:
        self.trials.append(name)

    def __len__(self) -> int:
        return len(self.trials)


def newey_west_tstat(returns: Sequence[float], lags: int | None = None) -> Dict[str, float]:
    """t-stat of the mean, robust to autocorrelation and heteroskedasticity."""
    r = np.asarray([x for x in returns if np.isfinite(x)], dtype=float)
    n = len(r)
    if n < 30:
        return {"mean": float("nan"), "t_stat": float("nan"), "n": n, "lags": 0}
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))  # Newey-West 1994 rule

    e = r - r.mean()
    gamma0 = float(e @ e / n)
    var = gamma0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)  # Bartlett kernel
        gamma = float(e[L:] @ e[:-L] / n)
        var += 2.0 * w * gamma
    se = np.sqrt(max(var, 1e-18) / n)
    return {
        "mean": float(r.mean()),
        "t_stat": float(r.mean() / se),
        "n": n,
        "lags": lags,
    }


def sharpe(returns: Sequence[float], periods: int = TRADING_DAYS) -> float:
    r = np.asarray([x for x in returns if np.isfinite(x)], dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def stationary_bootstrap_sharpe(
    returns: Sequence[float],
    n_boot: int = 2000,
    mean_block: int = 20,
    seed: int = 0,
    periods: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Politis-Romano stationary bootstrap CI for the Sharpe ratio.

    Geometric block lengths preserve the serial dependence that overlapping
    holding periods create; an iid bootstrap would understate the CI badly here.
    """
    r = np.asarray([x for x in returns if np.isfinite(x)], dtype=float)
    n = len(r)
    if n < 60:
        return {"sharpe": sharpe(r, periods), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = int(rng.integers(n))
            else:
                i = (i + 1) % n
        out[b] = sharpe(r[idx], periods)
    out = out[np.isfinite(out)]
    return {
        "sharpe": sharpe(r, periods),
        "ci_low": float(np.percentile(out, 2.5)),
        "ci_high": float(np.percentile(out, 97.5)),
        "p_sharpe_le_0": float((out <= 0).mean()),
    }


def trial_sharpe_variance(
    trial_returns: Sequence[Sequence[float]], periods: int = TRADING_DAYS
) -> float:
    """Variance of the per-period Sharpe ratio ACROSS the strategies tried.

    This is the quantity `deflated_sharpe` needs, and getting it wrong is easy:
    it is the dispersion of results across the configurations you searched, not
    the dispersion of one strategy's returns over time. The time-series version
    is roughly an order of magnitude larger and drives the deflated Sharpe to
    zero for every strategy, significant or not.

    With only a handful of variants this estimate is itself noisy, which is a
    genuine limitation - the honest reading is "indicative", not "exact".
    """
    srs = [sharpe(r, periods) / np.sqrt(periods) for r in trial_returns]
    srs = [s for s in srs if np.isfinite(s)]
    if len(srs) < 2:
        return float("nan")
    return float(np.var(srs, ddof=1))


def deflated_sharpe(
    returns: Sequence[float],
    n_trials: int,
    sr_variance: float,
    periods: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Bailey & Lopez de Prado's deflated Sharpe ratio.

    Interpretation: the probability that the observed Sharpe reflects real skill
    rather than being the best of `n_trials` attempts, given how much the trials
    varied and how non-normal the returns are. Below ~0.95, the result should
    not be called significant no matter how good the equity curve looks.

    Parameters
    ----------
    sr_variance
        Variance of the per-period Sharpe across trials, from
        `trial_sharpe_variance`.
    """
    r = np.asarray([x for x in returns if np.isfinite(x)], dtype=float)
    n = len(r)
    if n < 60 or n_trials < 1 or not np.isfinite(sr_variance) or sr_variance <= 0:
        return {
            "dsr": float("nan"),
            "sr_threshold_ann": float("nan"),
            "n_trials": n_trials,
            "note": "needs >=60 obs, >=1 trial, and a positive cross-trial variance",
        }

    sr = sharpe(r, periods) / np.sqrt(periods)  # per-period
    g3 = float(sps.skew(r))
    g4 = float(sps.kurtosis(r, fisher=False))

    # Expected maximum Sharpe from n_trials draws of a zero-skill strategy.
    e = np.euler_gamma
    if n_trials == 1:
        sr0 = 0.0
    else:
        z1 = sps.norm.ppf(1 - 1.0 / n_trials)
        z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = np.sqrt(sr_variance) * ((1 - e) * z1 + e * z2)

    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    dsr = float(sps.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))
    return {
        "dsr": dsr,
        "sr_threshold_ann": float(sr0 * np.sqrt(periods)),
        "n_trials": n_trials,
        "skew": g3,
        "kurtosis": g4,
    }


def benjamini_hochberg(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    """FDR control across the specifications tested."""
    items = [(k, v) for k, v in p_values.items() if np.isfinite(v)]
    m = len(items)
    if m == 0:
        return {}
    items.sort(key=lambda kv: kv[1])
    out, max_sig = {}, -1
    for i, (k, p) in enumerate(items, start=1):
        if p <= alpha * i / m:
            max_sig = i
    for i, (k, p) in enumerate(items, start=1):
        out[k] = {
            "p": p,
            "q": min(p * m / i, 1.0),
            "survives_fdr": i <= max_sig,
        }
    return out
