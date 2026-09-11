"""Deterministic, offline, zero-cost tone scorer.

This exists for three reasons, in increasing order of importance:

1. It lets the whole pipeline run without an LLM API key.
2. It is fully reproducible, so pipeline tests are not flaky.
3. It is the BASELINE THE LLM HAS TO BEAT.

Point 3 is the one that matters at interview. "I used an LLM" is not a finding.
"I used an LLM and it added 0.3 Sharpe over a regex baseline that costs nothing"
is a finding, and so is "it didn't, so the LLM is not earning its API bill here."
`compare_backends` in stats.py is what adjudicates that.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ..rubric import ToneScores

# --- lexicons -------------------------------------------------------------
# Deliberately small and readable. A bigger lexicon is not obviously better;
# it is just harder to explain when someone asks what the score means.

HEDGE_TERMS = [
    r"\bwe think\b", r"\bwe believe\b", r"\bwe feel\b", r"\bprobably\b",
    r"\broughly\b", r"\bapproximately\b", r"\bsomewhere (?:around|in)\b",
    r"\bsort of\b", r"\bkind of\b", r"\bmore or less\b", r"\bit'?s possible\b",
    r"\bhard to say\b", r"\bdifficult to predict\b", r"\bmay\b", r"\bmight\b",
    r"\bcould\b", r"\bpotentially\b", r"\bsome degree\b", r"\bat this point\b",
    r"\btoo early to\b", r"\bwe'?ll have to see\b", r"\bnot prepared to\b",
]

EVASION_TERMS = [
    r"\bwe don'?t (?:break|disclose|guide)\b", r"\bnot going to comment\b",
    r"\bwe'?ll come back to (?:you|that)\b", r"\btake that offline\b",
    r"\bwhat i would say is\b", r"\bstepping back\b", r"\bmore broadly\b",
    r"\bi'?d point you to\b", r"\bno change to our\b", r"\bwe'?re not\b.{0,20}\bquantify\b",
]

PUSHBACK_TERMS = [
    r"\bjust to (?:follow up|clarify|push)\b", r"\bcoming back to\b",
    r"\bi'?m still not\b", r"\bthat doesn'?t\b.{0,30}\b(?:square|square with|add up)\b",
    r"\bcan you be more specific\b", r"\bwith respect,?\b", r"\bto be clear\b",
    r"\bbut you (?:said|guided)\b", r"\bhow do you reconcile\b", r"\bagain,?\b",
]

EXTERNAL_TERMS = [
    r"\bmacro(?:economic)?\b", r"\bthe environment\b", r"\bheadwind", r"\bfx\b",
    r"\bcurrency\b", r"\bweather\b", r"\bsupply chain\b", r"\bindustry-wide\b",
    r"\bconsumer (?:weakness|softness)\b", r"\bmarket conditions\b",
    r"\btariff", r"\binterest rate environment\b", r"\bgeopolit",
]

INTERNAL_TERMS = [
    r"\bwe (?:mis|under)(?:executed|estimated|invested|performed)\b",
    r"\bour (?:mistake|error|miss|shortfall|execution)\b",
    r"\bwe fell short\b", r"\bexecution\b", r"\bwe own\b", r"\bwe did not execute\b",
    r"\bself-inflicted\b", r"\bour responsibility\b", r"\bwe (?:could|should) have\b",
    r"\bwe (?:were|are) (?:too )?(?:slow|late)\b", r"\bwe missed\b",
    r"\bnot satisfied with\b", r"\bdisappointed (?:in|with) our\b",
    r"\bwe need to (?:do better|improve)\b", r"\bon us\b", r"\bwe take responsibility\b",
]

# Minimum attribution hits before the external/internal ratio means anything.
# Internal-blame language is genuinely rare on earnings calls, so a ratio from
# one or two matches would read as "totally externalising" on nearly every call.
MIN_ATTRIBUTION_HITS = 4

FORWARD_TERMS = [
    r"\bwill\b", r"\bexpect\b", r"\banticipate\b", r"\bgoing forward\b",
    r"\bnext (?:quarter|year)\b", r"\bfy2\d\b", r"\bguid(?:e|ance|ing)\b",
    r"\boutlook\b", r"\bplan to\b", r"\btarget\b",
]

BACKWARD_TERMS = [
    r"\bin the quarter\b", r"\blast quarter\b", r"\bdelivered\b", r"\bwe reported\b",
    r"\bcame in\b", r"\bduring the period\b", r"\bversus (?:last|prior) year\b",
    r"\bgrew\b", r"\bdeclined\b", r"\bwas up\b", r"\bwas down\b",
]

# A committed forward statement needs a magnitude AND a horizon in the same
# sentence. This is the pattern that separates "we expect 8-10% growth in Q3"
# from "we feel good about the back half".
# Guidance is routinely quantified without digits - "low double digits" is every
# bit as committed as "11%". Matching only numerals classified almost all real
# guidance as vague, which pinned this axis near zero on live transcripts.
MAGNITUDE = re.compile(
    r"(\d+(?:\.\d+)?\s?(?:%|percent|bps|basis points)"
    r"|\$\s?\d+(?:\.\d+)?\s?(?:m|b|mm|bn|million|billion|trillion)?"
    r"|\b(?:low|mid|high)[- ](?:to[- ](?:low|mid|high)[- ])?(?:single|double)[- ]digits?"
    r"|\b(?:single|double)[- ]digits?"
    r"|\b(?:roughly |about |approximately )?flat\b"
    r"|\bin line with (?:last|the prior)\b"
    r"|\b\d+(?:\.\d+)?\s?(?:to|-|and)\s?\d+(?:\.\d+)?\b)",
    re.I,
)
# Companies on non-calendar fiscal years name the quarter by month ("our
# December quarter"), which a numbered-quarter pattern alone never catches.
HORIZON = re.compile(
    r"\b(q[1-4]|(?:first|second|third|fourth) quarter|fy\s?2\d|next (?:year|quarter)|"
    r"full[- ]year|second half|back half|by (?:year|the end)|"
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december) quarter|(?:this|current|coming|upcoming) (?:quarter|year))\b",
    re.I,
)
FORWARD_VERB = re.compile(
    r"\b(expect|anticipate|guid\w+|will|target|plan|project|see)\b", re.I
)

_SENT = re.compile(r"(?<=[.!?])\s+")


def _compile(terms: List[str]) -> List[re.Pattern]:
    return [re.compile(t, re.I) for t in terms]


_HEDGE = _compile(HEDGE_TERMS)
_EVADE = _compile(EVASION_TERMS)
_PUSH = _compile(PUSHBACK_TERMS)
_EXT = _compile(EXTERNAL_TERMS)
_INT = _compile(INTERNAL_TERMS)
_FWD = _compile(FORWARD_TERMS)
_BWD = _compile(BACKWARD_TERMS)


def _hits(patterns: List[re.Pattern], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def _saturate(rate: float, scale: float) -> float:
    """Map an unbounded per-1k-word rate onto [0, 1].

    Linear rescaling would let one pathological transcript dominate the
    cross-section, so we use a saturating curve: doubling the hedge rate of an
    already-hedgy call moves the score much less than doubling a flat one.
    """
    return float(rate / (rate + scale)) if rate > 0 else 0.0


# A prepared-remarks monologue runs many hundreds of words; a Q&A turn is a
# few dozen to a couple hundred. That gap is the most format-independent signal
# available, and it is what the structural detector keys on.
MONOLOGUE_WORDS = 400

_SPEAKER_BLOCK = re.compile(r"(?:\A|\n)\s*([^:\n]{2,60}?):\s", re.M)

_QA_MARKER = re.compile(
    r"(question[- ]and[- ]answer session|q\s?&\s?a session|"
    r"(?:we|I)'?ll now (?:take|open|begin)|now begin the question|"
    r"first question comes from|we'?ll (?:now )?open the (?:call|line)s? (?:up )?(?:to|for) question)",
    re.I,
)


def _blocks(text: str) -> List[Tuple[int, str, str]]:
    """Parse `Speaker: body` blocks into (offset, speaker, body)."""
    marks = list(_SPEAKER_BLOCK.finditer(text))
    out: List[Tuple[int, str, str]] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.start(1), m.group(1).strip(), text[m.end() : end]))
    return out


def split_transcript(text: str) -> Tuple[str, str, str]:
    """Split into (prepared remarks, Q&A, method_used).

    Four strategies, most explicit first. The method is returned rather than
    hidden because the last one is a guess: a 50/50 character split lands
    mid-sentence and silently assigns half the prepared remarks to the Q&A.
    Scores computed on that are not wrong-looking, just wrong - so callers need
    to be able to see it happened and exclude those calls.
    """
    blocks = _blocks(text)

    if blocks:
        words = [len(b.split()) for _, _, b in blocks]
        longest = max(range(len(words)), key=words.__getitem__)

        # 1. An explicit transition phrase, but only after the main monologue -
        #    the opening preamble often says "we'll open the call to questions"
        #    while still being prepared remarks.
        after = blocks[longest][0]
        m = _QA_MARKER.search(text, after)
        if m:
            nxt = next((off for off, _, _ in blocks if off > m.start()), None)
            if nxt is not None:
                return text[:nxt], text[nxt:], "marker"

        # 2. The first Operator turn after the longest monologue. Operator
        #    greetings at the very top are excluded by that ordering.
        for off, who, _ in blocks:
            if off > blocks[longest][0] and re.fullmatch(r"operator", who, re.I):
                return text[:off], text[off:], "operator"

        # 3. Structural: Q&A begins after the last long monologue.
        last_long = max(
            (i for i, w in enumerate(words) if w >= MONOLOGUE_WORDS), default=None
        )
        if last_long is not None and last_long + 1 < len(blocks):
            off = blocks[last_long + 1][0]
            return text[:off], text[off:], "monologue"

    mid = len(text) // 2
    return text[:mid], text[mid:], "midpoint"


def split_sections(text: str) -> Tuple[str, str]:
    """Backwards-compatible two-value form of `split_transcript`."""
    prepared, qa, _ = split_transcript(text)
    return prepared, qa


# Some vendor transcripts carry prepared remarks only, with the Q&A missing
# entirely. Two of the six axes live exclusively in the Q&A, so scoring such a
# call yields a full-looking six-axis result of which a third is measured on
# nothing. Calls failing this check should be excluded, not silently scored.
# The question count does the discriminating work: on real data every call that
# failed this test had *zero* questions, while every call that passed had at
# least four and over a thousand words of Q&A. The word floor is only a guard
# against a stray question mark in a closing remark - set it high enough to look
# reasonable for full-length transcripts and it silently starts rejecting
# shorter ones, which is how a threshold tuned on one corpus corrupts another.
MIN_QA_QUESTIONS = 3
MIN_QA_WORDS = 100


def has_usable_qa(text: str) -> bool:
    """Whether a transcript contains enough Q&A to score evasion and pushback."""
    _, qa, _ = split_transcript(text)
    return qa.count("?") >= MIN_QA_QUESTIONS and len(qa.split()) >= MIN_QA_WORDS


def _analyst_turns(qa: str) -> List[str]:
    """Best-effort extraction of analyst turns from a Q&A section."""
    turns = re.split(r"\n(?=[A-Z][a-zA-Z.\- ]{2,40}:)", qa)
    return [t for t in turns if "?" in t]


def _management(prepared: str) -> set:
    """Whoever delivers the prepared remarks is management, by definition.

    Deriving this from the transcript beats a hardcoded name list: it works for
    any company, survives executive turnover, and needs no maintenance.
    """
    who = {s.lower() for _, s, _ in _blocks(prepared)}
    who.add("operator")
    return who


def _followup_rate(prepared: str, qa: str) -> Optional[float]:
    """Share of analyst turns that are follow-ups rather than opening questions.

    This replaces phrase-matching for pushback, and is a better measure of the
    same thing. An analyst who is satisfied moves on; an analyst who takes a
    second and third turn on the same topic is pressing, whatever words they
    use to do it. Being structural rather than lexical, it also transfers
    across companies and transcript vendors instead of encoding one house style.

    Returns None when the Q&A has too few turns to be meaningful, so the caller
    can decline to score rather than emit a number built on two observations.
    """
    mgmt = _management(prepared)
    turns = [(who, body) for _, who, body in _blocks(qa)]
    analyst = [(who, body) for who, body in turns if who.lower() not in mgmt]
    if len(analyst) < 3:
        return None

    seen: set = set()
    followups = 0
    for who, _ in analyst:
        key = who.lower()
        if key in seen:
            followups += 1
        seen.add(key)
    return followups / len(analyst)


# Vendors ship typographic punctuation - 86% of transcripts in the current
# corpus use curly apostrophes. A pattern written as `we'?ll` matches "well" and
# "we'll" but not "we’ll", so those phrases are silently never found. The
# measured effect here is small because most patterns make the apostrophe
# optional, but it is invisible when it does bite: no error, just a lexicon
# quietly failing to fire.
_PUNCT_FIXES = {
    "’": "'", "‘": "'",           # curly single quotes
    "“": '"', "”": '"',           # curly double quotes
    "–": "-", "—": "-",           # en / em dash
    " ": " ", "…": "...",         # nbsp, ellipsis
}


def normalise_punctuation(text: str) -> str:
    """Fold typographic punctuation to ASCII so patterns match either form."""
    for src, dst in _PUNCT_FIXES.items():
        text = text.replace(src, dst)
    return text


def score_text(text: str) -> ToneScores:
    """Score one transcript. Pure function of the text - no network, no state."""
    text = normalise_punctuation(text)
    prepared, qa = split_sections(text)
    n_words = max(len(text.split()), 1)
    per_1k = 1000.0 / n_words

    hedging = _saturate(_hits(_HEDGE, prepared) * (1000.0 / max(len(prepared.split()), 1)), 6.0)

    # commitment: of forward-looking sentences, what share are load-bearing?
    fwd_sents = [s for s in _SENT.split(prepared) if FORWARD_VERB.search(s)]
    if fwd_sents:
        committed = sum(
            1 for s in fwd_sents if MAGNITUDE.search(s) and HORIZON.search(s)
        )
        commitment = committed / len(fwd_sents)
    else:
        commitment = 0.0

    evasion = _saturate(_hits(_EVADE, qa) * (1000.0 / max(len(qa.split()), 1)), 3.0)

    # Structural follow-up rate first; fall back to phrase-matching only when
    # speaker parsing fails (some vendors ship transcripts with no attribution).
    rate = _followup_rate(prepared, qa)
    if rate is not None:
        pushback = min(rate, 1.0)
    else:
        turns = _analyst_turns(qa)
        if turns:
            pressed = sum(1 for t in turns if _hits(_PUSH, t) > 0)
            pushback = min(pressed / len(turns), 1.0)
        else:
            pushback = _saturate(_hits(_PUSH, qa) * per_1k, 3.0)

    f, b = _hits(_FWD, prepared), _hits(_BWD, prepared)
    forward_ratio = f / (f + b) if (f + b) else 0.5

    # A ratio built from one or two hits is noise wearing a decimal point, and
    # because internal-attribution language is rare it would resolve to ~1.0 on
    # almost every call. Below a minimum count, decline to score.
    e, i = _hits(_EXT, text), _hits(_INT, text)
    externalisation = e / (e + i) if (e + i) >= MIN_ATTRIBUTION_HITS else 0.5

    return ToneScores(
        hedging=round(hedging, 6),
        commitment=round(commitment, 6),
        evasion=round(evasion, 6),
        pushback=round(pushback, 6),
        forward_ratio=round(forward_ratio, 6),
        externalisation=round(externalisation, 6),
    )
