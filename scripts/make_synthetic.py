"""Generate a synthetic world with a KNOWN answer, to validate the pipeline.

This is the part of the project that makes the rest trustworthy, so it is worth
being precise about what it proves.

The generator plants a specific causal structure:

    s[i,t] = 0.5 * s[i,t-1] + e[i,t]        latent state, AR(1)
    text[i,t] is written to reflect  f[i] + s[i,t]
    future drift[i,t] = effect * e[i,t]  +  noise

Note carefully what depends on what:

* The TEXT reflects the firm fixed effect f[i] plus the state. Firm f[i] is a
  persistent speaking style - some CFOs always hedge - and it is large relative
  to the state, exactly as in real transcripts.
* The DRIFT depends only on e[i,t], the *innovation* in the state. Not on the
  level, and not at all on f[i].

So a naive strategy that ranks firms on raw tone LEVEL is sorting mostly on
f[i], which by construction predicts nothing. Only the within-firm differenced
signal recovers e[i,t]. If the pipeline is built correctly, the level signal
should come out near zero and the differenced signal should come out strong -
and `validate.py` checks precisely that.

Run with --effect 0 to produce a null world. A pipeline that finds a signal
there is broken, and that test is worth more than any positive result.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --- sentence pools -------------------------------------------------------
# Phrasing is chosen so the lexical scorer's patterns genuinely fire; the LLM
# scorer reads the same text and should reach similar conclusions.

RETRO = [
    "Revenue in the quarter was {rev} million, up {g} percent versus last year.",
    "We reported operating margin of {m} percent during the period.",
    "Gross profit grew {g} percent in the quarter.",
    "Free cash flow came in at {rev} million for the period.",
    "Segment revenue was up {g} percent versus prior year.",
    "We delivered {g} percent growth in our core business during the period.",
]

FWD_COMMITTED = [
    "We expect revenue growth of {g} to {g2} percent in Q{q}.",
    "We are guiding to {m} percent operating margin for the full year.",
    "We will invest {rev} million in capacity by the end of Q{q}.",
    "Our target is {g} percent margin expansion in FY{fy}.",
    "We plan to deliver {rev} million of cost savings in the second half.",
]

FWD_VAGUE = [
    "We feel good about the back half.",
    "We expect continued momentum going forward.",
    "We anticipate the trends we have seen will broadly persist.",
    "We will continue to focus on operational excellence.",
    "Our outlook remains constructive.",
]

HEDGED = [
    "We think demand could improve, though it is hard to say at this point.",
    "Roughly speaking, we believe the trend may continue, but it is too early to tell.",
    "It is possible that we see some degree of recovery, we will have to see.",
    "We feel the environment might stabilise, approximately in line with our plan.",
    "We are not prepared to quantify that, but it could be somewhere around flat.",
    "That may or may not persist; we think it is difficult to predict.",
]

EXTERNAL = [
    "The macro environment created meaningful headwinds during the period.",
    "FX and currency movements weighed on reported results.",
    "Industry-wide supply chain disruption affected the quarter.",
    "Consumer weakness and broader market conditions were a factor.",
    "The interest rate environment remains a headwind for our category.",
]

INTERNAL = [
    "We underexecuted on the rollout and we own that.",
    "This was a self-inflicted issue and our responsibility.",
    "We fell short on execution in the quarter.",
    "Frankly, our mistake was misestimating the ramp.",
]

Q_OPEN = [
    "Can you walk us through the drivers of the margin change?",
    "How should we think about the phasing of growth from here?",
    "What are you assuming for pricing in the guidance?",
    "Can you size the contribution from the new product?",
    "What gives you confidence in the second half acceleration?",
    "How much of the growth was volume versus price?",
]

Q_PUSH = [
    "Just to follow up, that doesn't square with what you guided last quarter.",
    "Coming back to that, can you be more specific on the number?",
    "Again, but you said the ramp would be complete by now. How do you reconcile that?",
    "To be clear, I'm still not clear on the magnitude. With respect, can you quantify?",
]

A_DIRECT = [
    "Yes. Volume was {g} percent of the growth and price was the balance.",
    "Sure. The margin change was {m} points from mix and the rest from cost.",
    "It was {rev} million in the quarter, and we expect a similar figure next quarter.",
    "We are assuming {g} percent pricing in the guide, no change to that assumption.",
]

A_EVASIVE = [
    "What I would say is, stepping back, we feel good about the overall trajectory.",
    "We don't break that out, but more broadly the business is performing.",
    "I'd point you to our prior commentary. We're not going to quantify that today.",
    "Let's take that offline. No change to our overall framework.",
    "We'll come back to you on the specifics.",
]

SPEAKERS_MGMT = ["John Reeves - CEO", "Maria Alvarez - CFO"]
SPEAKERS_ANALYST = [
    "Tom Becker - Wexford Securities",
    "Priya Nandan - Halloran Capital",
    "Dan Osei - Corvid Research",
    "Elena Ruiz - Bramwell & Co",
]


def _fmt(rng: random.Random, s: str) -> str:
    return s.format(
        rev=rng.randint(80, 900),
        g=rng.randint(1, 18),
        g2=rng.randint(19, 26),
        m=rng.randint(8, 34),
        q=rng.randint(1, 4),
        fy=rng.randint(24, 27),
    )


def _pick(rng: random.Random, pool: List[str], n: int) -> List[str]:
    return [_fmt(rng, rng.choice(pool)) for _ in range(max(n, 0))]


def make_transcript(tone: float, rng: random.Random) -> str:
    """Write a transcript whose language reflects `tone` (higher = healthier).

    `tone` is squashed to a 0-1 intensity that drives the mix of sentence types.
    """
    t = 1.0 / (1.0 + np.exp(-tone))  # logistic squash to (0,1)

    n_hedge = int(round(2 + 10 * (1 - t)))
    n_committed = int(round(1 + 8 * t))
    n_vague = int(round(1 + 7 * (1 - t)))
    n_retro = int(round(4 + 6 * (1 - t)))
    n_ext = int(round(1 + 6 * (1 - t)))
    n_int = int(round(1 + 4 * t))

    body = (
        _pick(rng, RETRO, n_retro)
        + _pick(rng, HEDGED, n_hedge)
        + _pick(rng, FWD_COMMITTED, n_committed)
        + _pick(rng, FWD_VAGUE, n_vague)
        + _pick(rng, EXTERNAL, n_ext)
        + _pick(rng, INTERNAL, n_int)
    )
    rng.shuffle(body)

    lines = [
        "Operator: Good morning and welcome to the quarterly earnings conference call.",
        f"{SPEAKERS_MGMT[0]}: Thank you, operator, and good morning everyone.",
        " ".join(body[: len(body) // 2]),
        f"{SPEAKERS_MGMT[1]}: Turning to the financials.",
        " ".join(body[len(body) // 2 :]),
        "",
        "Operator: Thank you. We will now begin the question-and-answer session.",
    ]

    n_q = rng.randint(6, 10)
    n_evasive = int(round(n_q * (1 - t) * 0.8))
    n_push = int(round(n_q * (1 - t) * 0.7))
    evasive_slots = set(rng.sample(range(n_q), min(n_evasive, n_q)))
    push_slots = set(rng.sample(range(n_q), min(n_push, n_q)))

    for i in range(n_q):
        analyst = rng.choice(SPEAKERS_ANALYST)
        lines.append(f"{analyst}: {_fmt(rng, rng.choice(Q_OPEN))}")
        pool = A_EVASIVE if i in evasive_slots else A_DIRECT
        lines.append(f"{rng.choice(SPEAKERS_MGMT)}: {_fmt(rng, rng.choice(pool))}")
        if i in push_slots:
            lines.append(f"{analyst}: {_fmt(rng, rng.choice(Q_PUSH))}")
            lines.append(f"{rng.choice(SPEAKERS_MGMT)}: {_fmt(rng, rng.choice(A_EVASIVE))}")

    lines.append("Operator: This concludes today's conference call.")
    return "\n".join(lines)


def generate(
    out_dir: Path,
    n_firms: int = 60,
    n_quarters: int = 24,
    effect: float = 0.04,
    seed: int = 42,
    start: str = "2017-01-05",
) -> dict:
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = [f"SYN{i:03d}" for i in range(n_firms)]
    # Firm speaking style. Deliberately 1.5x the state's scale: raw tone LEVEL is
    # mostly personality, which is the whole reason differencing is required.
    firm_fx = rng.normal(0, 1.5, size=n_firms)

    # --- calendar ---------------------------------------------------------
    all_days = pd.bdate_range(start=start, periods=int(n_quarters * 63 + 400))
    call_offsets = {}
    for qi in range(n_quarters):
        # Calls cluster in a ~3 week reporting season, as in reality.
        base = 300 + qi * 63
        call_offsets[qi] = base

    # --- latent states and transcripts ------------------------------------
    rows, truth = [], []
    state = np.zeros(n_firms)
    for qi in range(n_quarters):
        innov = rng.normal(0, 0.7, size=n_firms)
        state = 0.5 * state + innov
        for fi, tkr in enumerate(tickers):
            day_idx = call_offsets[qi] + int(rng.integers(0, 18))
            call_date = all_days[min(day_idx, len(all_days) - 100)]
            tone_level = firm_fx[fi] + state[fi]
            text = make_transcript(tone_level, prng)
            rows.append(
                {
                    "ticker": tkr,
                    "call_date": call_date,
                    "fiscal_period": f"{call_date.year}Q{((qi % 4) + 1)}",
                    "text": text,
                }
            )
            truth.append(
                {
                    "ticker": tkr,
                    "call_date": call_date,
                    "true_state": state[fi],
                    "true_innovation": innov[fi],
                    "firm_effect": firm_fx[fi],
                }
            )

    transcripts = pd.DataFrame(rows)
    truth_df = pd.DataFrame(truth)

    # --- prices -----------------------------------------------------------
    n_days = len(all_days)
    mkt_ret = rng.normal(0.0003, 0.010, size=n_days)
    betas = rng.uniform(0.6, 1.6, size=n_firms)

    idio = rng.normal(0.0, 0.018, size=(n_days, n_firms))
    # Plant the drift: each event's innovation adds a constant daily increment
    # across its (+2, +63) window. Spreading it rather than dropping it in one
    # day is what makes it a *drift* the strategy must hold through.
    for r in truth_df.itertuples():
        fi = tickers.index(r.ticker)
        start_idx = all_days.searchsorted(r.call_date) + 2
        end_idx = min(start_idx + 62, n_days)
        if start_idx >= n_days:
            continue
        per_day = effect * r.true_innovation / max(end_idx - start_idx, 1)
        idio[start_idx:end_idx, fi] += per_day

    stock_ret = betas[None, :] * mkt_ret[:, None] + idio
    px = 100 * np.exp(np.cumsum(stock_ret, axis=0))
    mkt_px = 100 * np.exp(np.cumsum(mkt_ret))

    frames = [
        pd.DataFrame(
            {"ticker": t, "date": all_days, "close": px[:, i], "adj_close": px[:, i]}
        )
        for i, t in enumerate(tickers)
    ]
    frames.append(
        pd.DataFrame({"ticker": "MKT", "date": all_days, "close": mkt_px, "adj_close": mkt_px})
    )
    prices = pd.concat(frames, ignore_index=True)

    # --- universe (with real delistings, so survivorship logic is exercised)
    uni_rows = []
    n_dead = max(1, n_firms // 10)
    dead = set(prng.sample(tickers, n_dead))
    for t in tickers:
        end = all_days[int(rng.integers(int(n_days * 0.6), n_days - 1))] if t in dead else pd.NaT
        uni_rows.append({"ticker": t, "start_date": all_days[0], "end_date": end})
    universe = pd.DataFrame(uni_rows)

    transcripts.to_parquet(out_dir / "transcripts.parquet", index=False)
    prices.to_parquet(out_dir / "prices.parquet", index=False)
    truth_df.to_parquet(out_dir / "ground_truth.parquet", index=False)
    universe.to_csv(out_dir / "universe.csv", index=False)

    return {
        "n_firms": n_firms,
        "n_calls": len(transcripts),
        "effect": effect,
        "n_delisted": len(dead),
        "date_range": f"{all_days[0].date()} to {all_days[-1].date()}",
        "out_dir": str(out_dir),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--firms", type=int, default=60)
    ap.add_argument("--quarters", type=int, default=24)
    ap.add_argument(
        "--effect",
        type=float,
        default=0.04,
        help="planted drift per unit of tone innovation. 0 = null world.",
    )
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    info = generate(a.out, a.firms, a.quarters, a.effect, a.seed)
    for k, v in info.items():
        print(f"{k:>14}: {v}")
