"""The tone rubric.

This is the heart of the project's claim to originality, so it is worth being
explicit about the design.

Naive earnings-call NLP scores *sentiment* ("was this call positive?"). That is
almost entirely priced: sentiment correlates ~1:1 with the earnings surprise
itself, which the market absorbs in seconds. There is no drift left in it.

What we score instead is the *manner* of speaking, along six axes chosen because
each one plausibly leaks information that a headline EPS number does not carry:

  hedging          - density of epistemic hedges ("we think", "roughly", "may")
  commitment       - forward statements that carry a number and a date, vs vague
  evasion          - questions asked in Q&A that never receive a direct answer
  pushback         - analysts repeating, pressing, or expressing skepticism
  forward_ratio    - share of management talk oriented forward vs backward
  externalisation  - blame placed on outside forces vs owned internally

Every axis is defined so that HIGHER = MORE NEGATIVE for the firm, except
`commitment` and `forward_ratio`, which are inverted in `features.py` so the
composite has a single sign convention.

Crucially, none of these are used at their raw level. `features.py` differences
each axis against the same company's own trailing four calls. A CFO who always
hedges tells you nothing; a CFO who hedges much more than they did last quarter
might. That within-firm differencing is the actual hypothesis under test.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Dict

# Axis name -> +1 if a higher raw score is bad news for the firm, -1 if good.
AXIS_SIGN: Dict[str, int] = {
    "hedging": +1,
    "commitment": -1,
    "evasion": +1,
    "pushback": +1,
    "forward_ratio": -1,
    "externalisation": +1,
}

AXES = tuple(AXIS_SIGN.keys())


@dataclass(frozen=True)
class ToneScores:
    """One call, scored. All axes are on a 0-1 scale."""

    hedging: float
    commitment: float
    evasion: float
    pushback: float
    forward_ratio: float
    externalisation: float

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, (int, float)):
                raise TypeError(f"{f.name} must be numeric, got {type(v).__name__}")
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{f.name}={v} outside [0, 1]")

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


# Sent to the LLM scorer. Kept in this module so the lexical and LLM backends
# are demonstrably scoring the same construct.
RUBRIC_PROMPT = """\
You are scoring the *manner* of management's speech on an earnings call, not \
whether the news was good or bad. Ignore the actual financial results entirely. \
Two calls reporting identical numbers should receive different scores only if \
management *spoke* differently.

Score each axis from 0.0 to 1.0:

hedging: density of epistemic hedging in management's own statements - "we \
think", "roughly", "somewhere around", "it's possible", "hard to say". 0.0 = \
management speaks in flat declaratives. 1.0 = nearly every claim is qualified.

commitment: how load-bearing the forward-looking statements are. A forward \
statement counts as committed when it carries BOTH a quantity and a timeframe \
("we expect 8-10% growth in Q3"). 0.0 = all forward talk is directionally vague \
("we feel good about the back half"). 1.0 = forward talk is consistently \
specific and dated.

evasion: of the direct questions asked in Q&A, what share never receive a \
responsive answer - management redirects, answers a different question, or \
defers. 0.0 = every question answered head-on. 1.0 = systematic non-answers.

pushback: how hard analysts press. Score higher for repeat questions on the same \
topic, explicit skepticism, follow-ups that restate the question, or analysts \
noting an inconsistency. 0.0 = deferential Q&A. 1.0 = adversarial.

forward_ratio: share of management's prepared remarks oriented to future \
performance rather than explaining the quarter just ended. 0.0 = entirely \
retrospective. 1.0 = entirely prospective.

externalisation: when explaining shortfalls or weakness, how much is attributed \
to forces outside the firm's control (macro, weather, FX, "the environment") \
versus owned as execution. 0.0 = fully owned. 1.0 = fully externalised. If \
nothing negative is discussed, score 0.5.

Return ONLY a JSON object with exactly these six keys and float values, no \
prose, no markdown fence:
{"hedging": 0.0, "commitment": 0.0, "evasion": 0.0, "pushback": 0.0, \
"forward_ratio": 0.0, "externalisation": 0.0}
"""
