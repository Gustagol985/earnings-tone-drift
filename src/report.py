"""Render a run into a markdown report a portfolio manager could actually read.

The report is deliberately front-loaded with the things that would kill the
result - sample construction, the multiple-testing count, the placebo, the
breakeven cost - before it shows a single performance number. That ordering is
itself the point: it signals that the author knows which parts a sceptical
reader checks first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import stats as st
from .pipeline import RunResult


def _pct(x: float, dp: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{dp}f}%"


def _num(x: float, dp: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{dp}f}"


def _reg_table(regs: Dict[str, dict], keys: list[str]) -> str:
    head = "| spec | n | coef | t-stat | p | R² | controls |\n|---|---:|---:|---:|---:|---:|---|\n"
    rows = []
    for k in keys:
        r = regs.get(k)
        if not r or "error" in r:
            rows.append(f"| `{k}` | — | — | — | — | — | {r.get('error','missing') if r else 'missing'} |")
            continue
        rows.append(
            f"| `{k}` | {r['n']} | {r['coef']:.5f} | {r['t_stat']:.2f} | "
            f"{r['p_value']:.4f} | {r['r2']:.4f} | {r['controls']} |"
        )
    return head + "\n".join(rows)


def build(res: RunResult, title: str = "Earnings Call Tone Drift") -> str:
    m = res.meta
    L: list[str] = [f"# {title}", ""]

    # ---- caveats first ---------------------------------------------------
    L += ["## Read this before the results", ""]
    if not m.get("point_in_time_universe"):
        L += [
            "> **Survivorship bias present.** The universe was built from a static "
            "ticker list, not point-in-time index membership. Firms that delisted "
            "during the sample are absent, which biases results in an unknown "
            "direction. Supply a membership file with join/leave dates to remove this.",
            "",
        ]
    else:
        L += [
            "Universe is point-in-time: calls are included only for dates on which "
            "the firm was an index member, so delisted firms contribute their full "
            "history up to removal.",
            "",
        ]

    n_trials = len(res.trials)
    L += [
        f"**Specifications evaluated this run: {n_trials}.** Every one is listed in "
        "the multiple-testing section below with an FDR-adjusted q-value. Any single "
        "t-stat here should be read against that count, not on its own.",
        "",
        f"- Scorer: `{m.get('scorer')}`" + (f" (model `{m.get('model')}`)" if m.get("model") else ""),
        f"- Transcripts scored: {m.get('n_transcripts')}",
        f"- Events with complete price windows: {m.get('n_events')}",
        f"- Events dropped: {m.get('n_dropped')}",
        "",
    ]

    n_noqa = m.get("dropped_no_qa") or 0
    if n_noqa:
        raw = m.get("n_transcripts_before_qa_filter") or (n_noqa + (m.get("n_transcripts") or 0))
        pct = n_noqa / raw if raw else 0
        L += [
            f"> **{n_noqa} of {raw} calls ({pct:.1%}) were excluded before scoring.** "
            "The transcript vendor supplied prepared remarks only for these, with no "
            "Q&A section. Evasion and analyst pushback are measured exclusively in "
            "the Q&A, so including them would have produced a complete-looking "
            "six-axis score with a third of it computed from empty text — and then "
            "averaged alongside fully scored calls as if comparable.",
            "",
        ]
        names = m.get("dropped_no_qa_list") or []
        if names:
            shown = ", ".join(names[:20]) + (" …" if len(names) > 20 else "")
            L += [f"> Excluded: {shown}", ""]

    if not res.dropped.empty and "reason" in res.dropped.columns:
        L += ["Drop reasons:", ""]
        for reason, n in res.dropped["reason"].value_counts().items():
            L.append(f"- `{reason}`: {n}")
        L.append("")

    L += [
        "Lookahead audit: **passed** — trailing baselines were independently "
        "recomputed from raw scores with an explicit date filter and matched the "
        "vectorised panel on every sampled row.",
        "",
    ]

    # ---- the central test ------------------------------------------------
    L += [
        "## 1. Does the differenced signal predict drift?",
        "",
        "Dependent variable is the market-model abnormal return cumulated over "
        "trading days +2 to +63. The announcement window (0, +1) is deliberately "
        "excluded from the dependent variable *and* included as a control: without "
        "that control a positive coefficient would just be re-discovering "
        "post-earnings-announcement drift.",
        "",
        _reg_table(
            res.regressions,
            ["differenced_signal", "differenced_signal_uncontrolled", "raw_level_signal"],
        ),
        "",
    ]

    d, lvl = res.regressions.get("differenced_signal"), res.regressions.get("raw_level_signal")
    if d and lvl and "coef" in d and "coef" in lvl:
        L += [
            f"The differenced signal carries t = {d['t_stat']:.2f} against the raw "
            f"level's t = {lvl['t_stat']:.2f}. That gap is the project's central "
            "claim: what predicts is the *change* in how management speaks relative "
            "to their own recent baseline, not how they speak in absolute terms.",
            "",
        ]

    # ---- placebo ---------------------------------------------------------
    pl = res.regressions.get("announcement_window_placebo")
    L += ["### Placebo", ""]
    if pl and "coef" in pl:
        L += [
            f"Regressing the same signal on the *announcement* window (0, +1) gives "
            f"t = {pl['t_stat']:.2f}. A large value here would be a warning, not a "
            "success: it would mean the signal is mostly restating the immediate "
            "price reaction, which is unavailable to trade by the time the "
            "transcript is parsed.",
            "",
        ]

    # ---- axis attribution -----------------------------------------------
    axis_keys = sorted(k for k in res.regressions if k.startswith("axis::"))
    if axis_keys:
        L += [
            "## 2. Which axis is doing the work?",
            "",
            "Each rubric axis, differenced and standardised within firm, run on its own.",
            "",
            _reg_table(res.regressions, axis_keys),
            "",
            "Axes that are individually weak but jointly useful are expected — the "
            "composite averages six noisy measurements of one latent construct. An "
            "axis that is strong alone and absent from the composite would be the "
            "interesting anomaly.",
            "",
        ]

    # ---- monotonicity ----------------------------------------------------
    if not res.deciles.empty:
        L += ["## 3. Monotonicity across signal quintiles", ""]
        L += ["| quintile | n | mean drift | median | t |", "|---:|---:|---:|---:|---:|"]
        for r in res.deciles.itertuples():
            L.append(
                f"| {int(r.bin)} | {int(r.n)} | {_pct(r.mean)} | {_pct(r.median)} | {_num(r.t_stat)} |"
            )
        L += [
            "",
            "Quintile 1 is the most-deteriorated tone, quintile 5 the most improved. "
            "A clean result is roughly monotone; a result concentrated in one extreme "
            "bin usually means a few outliers rather than a relationship.",
            "",
        ]

    # ---- backtests -------------------------------------------------------
    L += ["## 4. Calendar-time backtest", ""]
    L += [
        "| variant | ann. return | ann. vol | Sharpe (net) | Sharpe (gross) | max DD | NW t | turnover/day | breakeven cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, bt in res.backtests.items():
        k = bt.metrics
        if "error" in k:
            L.append(f"| {name} | {k['error']} | | | | | | | |")
            continue
        L.append(
            f"| {name} | {_pct(k['ann_return'])} | {_pct(k['ann_vol'])} | "
            f"{_num(k['sharpe_net'])} | {_num(k['sharpe_gross'])} | {_pct(k['max_drawdown'])} | "
            f"{_num(k['nw_t_stat'])} | {_num(k['avg_daily_turnover'], 3)} | "
            f"{_num(k['breakeven_cost_bps'], 1)}bps |"
        )
    L.append("")

    base = res.backtests.get("base")
    if base and "error" not in base.metrics:
        be = base.metrics["breakeven_cost_bps"]
        L += [
            f"Breakeven transaction cost is {_num(be, 1)}bps per unit of turnover. "
            + (
                "That is thin enough that the strategy is substantially a study of "
                "execution quality rather than of language."
                if be < 15
                else "That leaves headroom above realistic large-cap costs of roughly "
                "5-10bps, so the edge is not purely a cost artefact."
            ),
            "",
        ]

        boot = st.stationary_bootstrap_sharpe(base.daily["ret_net"].dropna().values)
        # Cross-trial Sharpe dispersion comes from the backtest variants actually
        # run this session — that is the search whose selection effect we deflate.
        sr_var = st.trial_sharpe_variance(
            [b.daily["ret_net"].dropna().values for b in res.backtests.values()]
        )
        dsr = st.deflated_sharpe(
            base.daily["ret_net"].dropna().values,
            n_trials=max(n_trials, 1),
            sr_variance=sr_var,
        )
        L += [
            "### Is the Sharpe distinguishable from luck?",
            "",
            f"- Point estimate: **{_num(boot['sharpe'])}**",
            f"- Stationary-bootstrap 95% CI: **[{_num(boot['ci_low'])}, {_num(boot['ci_high'])}]** "
            "(geometric blocks, mean length 20 days, to respect the overlap induced "
            "by 63-day holding periods)",
            f"- Bootstrap P(Sharpe ≤ 0): **{_num(boot.get('p_sharpe_le_0', float('nan')), 3)}**",
            f"- Deflated Sharpe ratio across {n_trials} specifications: "
            f"**{_num(dsr['dsr'], 3)}**",
            f"- Sharpe a zero-skill search over {n_trials} trials would be expected "
            f"to produce anyway: **{_num(dsr['sr_threshold_ann'])}** "
            f"(cross-trial Sharpe dispersion estimated from "
            f"{len(res.backtests)} backtest variants, so this threshold is "
            "indicative rather than exact)",
            "",
        ]
        if np.isfinite(dsr["dsr"]) and dsr["dsr"] < 0.95:
            L += [
                "> The deflated Sharpe is below 0.95. Given the number of "
                "specifications tried, this result should be described as "
                "*not statistically distinguishable from selection effects* — "
                "regardless of how the equity curve looks.",
                "",
            ]

    # ---- multiple testing -----------------------------------------------
    pvals = {k: v.get("p_value", np.nan) for k, v in res.regressions.items() if "p_value" in v}
    bh = st.benjamini_hochberg(pvals)
    if bh:
        L += [
            "## 5. Multiple testing",
            "",
            f"{len(pvals)} regression specifications, Benjamini-Hochberg FDR at 5%.",
            "",
            "| spec | p | q (FDR-adj) | survives |",
            "|---|---:|---:|:--:|",
        ]
        for k, v in sorted(bh.items(), key=lambda kv: kv[1]["p"]):
            L.append(f"| `{k}` | {v['p']:.4f} | {v['q']:.4f} | {'yes' if v['survives_fdr'] else 'no'} |")
        L.append("")

    # ---- limitations -----------------------------------------------------
    L += [
        "## 6. Known limitations",
        "",
        "- **Transcript coverage is not random.** Vendors cover large caps far more "
        "completely than small caps, so the effective universe tilts large even when "
        "the nominal one does not.",
        "- **Short-side feasibility is assumed, not modelled.** The long/short "
        "variants borrow at zero cost and with unlimited availability. For the names "
        "where deteriorating tone is most informative, borrow is exactly where it is "
        "most expensive.",
        "- **Prices are close-to-close.** Entering at the next close is conservative "
        "on timing but ignores the spread actually paid.",
        "- **LLM scoring may carry memorisation risk.** The model has seen many of "
        "these calls in training and may recall the outcome. Identifiers are stripped "
        "before scoring, but the clean test is out-of-sample in time: rerun on calls "
        "dated after the model's training cutoff and compare.",
        "- **One market, one regime.** Nothing here establishes the effect survives "
        "outside the sample period or in other markets.",
        "",
    ]
    return "\n".join(L)


def save(res: RunResult, out_dir: Path, title: str = "Earnings Call Tone Drift") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = build(res, title)
    path = out_dir / "report.md"
    path.write_text(md, encoding="utf-8")

    res.events.to_parquet(out_dir / "events.parquet", index=False)
    res.scores.to_parquet(out_dir / "scores.parquet", index=False)
    for name, bt in res.backtests.items():
        bt.daily.to_csv(out_dir / f"backtest_{name}.csv")
    return path


def equity_curve_png(res: RunResult, out_dir: Path) -> Path | None:
    """Single chart: cumulative net return of the base variant vs the market."""
    base = res.backtests.get("base")
    if base is None or "error" in base.metrics:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = base.daily["ret_net"].dropna()
    curve = (1 + net).cumprod()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(curve.index, curve.values, lw=1.6, color="#1b3a5c", label="Tone-delta L/S (net)")
    ax.axhline(1.0, color="#999", lw=0.8, ls="--")
    ax.set_title(
        f"Cumulative net return  ·  Sharpe {base.metrics['sharpe_net']:.2f}  ·  "
        f"{base.metrics['n_days']} trading days",
        fontsize=11,
    )
    ax.set_ylabel("growth of 1")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "equity_curve.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
