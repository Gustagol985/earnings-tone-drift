# Earnings Call Tone Drift

Does a **change** in how management speaks on an earnings call — relative to how
that same management team spoke on their own last four calls — predict abnormal
returns over the following quarter, after controlling for the market's immediate
reaction to the call?

The pipeline scores every transcript on six axes of speaking *manner*,
differences each axis within firm, runs an event study on the resulting signal,
and backtests it in calendar time with transaction costs.

---

## Why this isn't another sentiment backtest

Most earnings-call NLP projects score sentiment and find that positive calls are
followed by positive returns. That result is real and uninteresting: sentiment is
nearly collinear with the earnings surprise, and post-earnings-announcement
drift has been documented since Ball & Brown in 1968. Rediscovering it is not a
finding.

Three design choices separate this from that project:

**1. Manner, not mood.** The rubric ([`src/rubric.py`](src/rubric.py)) scores
hedging density, guidance commitment, answer evasion, analyst pushback, forward
orientation, and blame externalisation. None of these ask whether the news was
good.

**2. Within-firm differencing.** No raw score is ever used. Every axis is
differenced against the same firm's trailing four calls and standardised by that
firm's own dispersion. A CFO who always hedges carries no information; a CFO who
hedges much more than they did last quarter might. This subtracts the firm fixed
effect by construction rather than estimating it.

**3. The announcement window is excluded and controlled for.** The dependent
variable is the abnormal return from day +2 to +63. The (0, +1) reaction is
measured, reported, used as a regression control — and never traded. Any signal
derived from a call correlates with that call's own price reaction; the only
interesting question is what survives once you take it out.

---

## The pipeline validates itself before it reports anything

Running this on real data and getting a positive number proves nothing, because
no one — including the author — knows the right answer. So the pipeline is first
run against synthetic worlds where the answer *is* known.

[`scripts/make_synthetic.py`](scripts/make_synthetic.py) generates firms whose
transcripts reflect a persistent speaking style **plus** a latent state, while
the planted future drift depends **only on the innovation in that state** and not
at all on the style. That structure makes raw tone level a deliberate trap: it is
dominated by personality, which predicts nothing.

```bash
python -m src.cli validate
```

Four gates, all of which must pass:

| | check | result |
|---|---|---|
| **A** | Signal world: differenced signal must find the planted effect (t > 2.5) | **t = +3.72** ✓ |
| **B** | Null world (effect = 0): pipeline must find nothing (\|t\| < 2.5) | **t = +1.56** ✓ |
| **C** | Differencing must beat raw level (\|t_level\| < 0.6 × \|t_diff\|) | **−1.71 vs +3.72, ratio 0.46** ✓ |
| **D** | Deflated Sharpe > 0.95 in signal world, < 0.95 in null | **0.994 vs 0.309** ✓ |

Gate **B** is the one that matters most. Any analysis pipeline with enough
researcher degrees of freedom will produce a positive result on noise; the only
way to know yours doesn't is to feed it noise on purpose. Note that the null
world's backtest still posts a **+0.42 Sharpe** and a rising equity curve — on
data with zero planted signal. That is the entire argument for reporting deflated
Sharpe ratios rather than equity curves.

Gate **C** validates the thesis rather than the plumbing: in a world built so
that only the innovation predicts, the differenced signal finds it and the raw
level does not.

---

## Guardrails against fooling yourself

| risk | how it's handled | where |
|---|---|---|
| Lookahead in the baseline | Trailing window is independently recomputed from raw scores with an explicit date filter and compared to the vectorised panel. Mismatch **raises** rather than warns. | `features.audit_lookahead` |
| Trading the announcement move | Entry is the close of the next trading session, never the call date. | `features.add_entry_dates` |
| Same-day return leak | Weights are lagged one day before multiplying by returns. | `backtest.run` |
| Survivorship bias | Point-in-time universe with join/leave dates; static ticker lists are permitted but stamp a warning banner on the report. | `providers.Universe` |
| Event clustering | Calendar-time portfolios, plus month-clustered standard errors in the cross-section. | `backtest`, `eventstudy` |
| Overlapping holding periods | Newey–West t-stats; stationary bootstrap with geometric blocks. | `stats` |
| Multiple testing | Every specification is counted in a `TrialLog`; report shows BH-adjusted q-values and a deflated Sharpe. | `stats`, `report` |
| Silently defaulted beta | Events without ≥100 clean estimation-window observations are dropped with a reason, never assigned β = 1. | `eventstudy.run` |
| LLM memorising outcomes | Tickers, quarters and years are stripped before scoring. | `LLMScorer.sanitise` |
| LLM not earning its cost | A free regex baseline scores the same rubric; both are runnable and comparable. | `scoring/lexical.py` |

---

## Setting up on a new machine

The repository carries the collected transcripts, so a fresh clone can run the
full study immediately — no re-downloading, no API key needed.

**Windows**

```bash
git clone https://github.com/<you>/earnings-tone-drift.git
cd earnings-tone-drift
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest tests/ -q          # expect 42 passed
.venv\Scripts\python.exe -m src.cli validate          # expect 4 gates passed
```

**macOS / Linux**

```bash
git clone https://github.com/<you>/earnings-tone-drift.git
cd earnings-tone-drift
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m src.cli validate
```

If the tests and the four gates pass, the machine is set up correctly.

### Continuing the download elsewhere

`data/real/ingest_state.json` is committed, so the downloader knows exactly what
has already been fetched. On a new machine, set the key and carry on — it will
not re-fetch anything:

```bash
setx ALPHAVANTAGE_API_KEY "your-key"     # Windows; reopen the terminal after
export ALPHAVANTAGE_API_KEY="your-key"   # macOS / Linux
.venv/bin/python -m scripts.ingest run
```

The free tier allows 25 requests per rolling 24 hours. Running on two machines
the same day shares one quota, so alternate rather than doubling up — and
`git pull` before each run, `git push` after, or the two copies of the state
file will diverge.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

Validate the pipeline (no API keys, no network, ~2 minutes):

```bash
python -m src.cli validate
```

Run on real data:

```bash
python -m src.cli run --data-dir data/real --transcripts fmp --prices stooq --scorer llm
```

Requires `FMP_API_KEY` (or `ALPHAVANTAGE_API_KEY`) and `ANTHROPIC_API_KEY`.
Put a point-in-time membership file at `data/real/universe.csv` with columns
`ticker,start_date,end_date`; without it you must pass `--tickers`, and the
report will carry a survivorship-bias banner.

Use `--scorer lexical` to run the whole study with no LLM spend at all — and
run it both ways, because the difference between them is a result in itself.

Tests:

```bash
python -m pytest tests/ -q
```

---

## Layout

```
src/rubric.py           the six axes, and the prompt shared by both scorers
src/scoring/lexical.py  deterministic regex scorer — the baseline to beat
src/scoring/llm.py      Claude scorer, cached on (model, rubric, transcript)
src/providers.py        FMP / AlphaVantage / Stooq adapters, PIT universe
src/features.py         within-firm differencing + the lookahead auditor
src/eventstudy.py       market-model abnormal returns, clustered regressions
src/backtest.py         calendar-time overlapping portfolios with costs
src/stats.py            Newey–West, stationary bootstrap, deflated Sharpe, BH
src/report.py           markdown report — caveats first, results second
scripts/make_synthetic.py   synthetic worlds with a known answer
scripts/validate.py         the four gates
```

---

## What this does not establish

The validation proves the *machinery* is sound. It says nothing about whether
the effect exists in real markets — that is what running it on real transcripts
is for, and a null result there is a perfectly good outcome to report.

Specific limitations, stated rather than buried: transcript vendor coverage
tilts large-cap; short-side borrow cost and availability are assumed away;
returns are close-to-close and ignore the spread actually paid; Stooq's
`adj_close` is split- but not fully dividend-adjusted; and the LLM scorer may
carry memorisation risk on calls inside its training window, for which the clean
test is a rerun restricted to calls dated after the model's cutoff.
