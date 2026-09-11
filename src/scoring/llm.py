"""LLM tone scorer (Claude), with on-disk caching.

Design notes worth defending in an interview:

* **Caching is keyed on (model, rubric hash, transcript hash).** Change the
  rubric and every score is recomputed; rerun the pipeline unchanged and it
  costs nothing. Without this, iterating on the backtest means paying for
  thousands of calls each time.

* **Long transcripts are chunked and averaged, not truncated.** Truncating to
  the first N tokens would systematically drop the Q&A section, which is where
  `evasion` and `pushback` live - i.e. truncation would silently null out two of
  six axes rather than just adding noise.

* **The model never sees the ticker, the date, or the results.** It is asked to
  score speech patterns only. This is a lookahead guard: a model that knows it
  is reading NVDA's Q3 FY24 call can recall what the stock did next, and that
  memory would contaminate the score with information unavailable at the time.
  `sanitise()` strips identifying detail before the text is sent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from ..rubric import AXES, RUBRIC_PROMPT, ToneScores

DEFAULT_MODEL = "claude-opus-5"
MAX_CHARS_PER_CHUNK = 60_000

# Scoring is a bounded rubric task, not open-ended reasoning, so it runs at low
# effort. Across a thousand transcripts that choice is most of the bill; raise it
# with --effort if a scorer-quality comparison shows it earns the spend.
DEFAULT_EFFORT = "low"

# Thinking is on by default and is billed against max_tokens along with the
# answer. A tight cap sized for ~100 tokens of JSON would truncate mid-object,
# so leave real headroom.
MAX_OUTPUT_TOKENS = 2000

# Constraining the response to the schema is what makes the parse reliable.
# Without it the model can wrap the JSON in prose or a fence, and the retry loop
# below would be papering over a formatting problem rather than a real failure.
SCORE_SCHEMA = {
    "type": "object",
    "properties": {a: {"type": "number"} for a in AXES},
    "required": list(AXES),
    "additionalProperties": False,
}


class LLMScorer:
    def __init__(
        self,
        cache_path: Path,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 4,
        effort: str = DEFAULT_EFFORT,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_retries = max_retries
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        self._rubric_hash = hashlib.sha256(RUBRIC_PROMPT.encode()).hexdigest()[:12]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS scores ("
            "  key TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._db.commit()

    # -- caching ----------------------------------------------------------
    def _key(self, text: str) -> str:
        # Effort belongs in the key: the same transcript scored at low and high
        # effort is two different measurements, and silently serving one for the
        # other would quietly invalidate a scorer comparison.
        h = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:24]
        return f"{self.model}:{self.effort}:{self._rubric_hash}:{h}"

    def _get(self, key: str) -> Optional[ToneScores]:
        row = self._db.execute("SELECT payload FROM scores WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        return ToneScores(**json.loads(row[0]))

    def _put(self, key: str, scores: ToneScores) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO scores VALUES (?,?,?)",
            (key, json.dumps(scores.as_dict()), time.time()),
        )
        self._db.commit()

    # -- lookahead guard --------------------------------------------------
    @staticmethod
    def sanitise(text: str) -> str:
        """Strip identifiers so the model cannot recognise the call.

        Imperfect by nature - a model may still infer the company from context -
        but it removes the easy giveaways. The honest framing for a reader is
        that this reduces, not eliminates, memorisation risk; the definitive
        control is running the study on calls after the model's training cutoff,
        which `report.py` reports separately.
        """
        text = re.sub(r"\((?:NYSE|NASDAQ|LSE|B3)\s?:\s?[A-Z.]{1,6}\)", "(TICKER)", text)
        text = re.sub(r"\b(?:Q[1-4])\s?(?:FY)?\s?20\d{2}\b", "the quarter", text, flags=re.I)
        text = re.sub(r"\bfiscal (?:year )?20\d{2}\b", "the fiscal year", text, flags=re.I)
        text = re.sub(r"\b20[0-2]\d\b", "20XX", text)
        return text

    # -- scoring ----------------------------------------------------------
    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic  # imported lazily so the lexical path needs no SDK
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "anthropic SDK not installed. `pip install anthropic`, or run "
                    "with --scorer lexical to stay offline."
                ) from exc
            if not self.api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Set it, or run with "
                    "--scorer lexical."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @staticmethod
    def _chunks(text: str) -> List[str]:
        if len(text) <= MAX_CHARS_PER_CHUNK:
            return [text]
        paras, out, buf = text.split("\n\n"), [], ""
        for p in paras:
            if len(buf) + len(p) > MAX_CHARS_PER_CHUNK and buf:
                out.append(buf)
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            out.append(buf)
        return out

    def _call_once(self, chunk: str) -> ToneScores:
        client = self._ensure_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=[
                # Cached: the rubric is byte-identical on every call, and only
                # the transcript below it varies. Cache reads bill at ~0.1x, so
                # this is free money across a few thousand calls.
                {
                    "type": "text",
                    "text": RUBRIC_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCORE_SCHEMA},
            },
            messages=[{"role": "user", "content": f"<transcript>\n{chunk}\n</transcript>"}],
        )
        # A refusal is a successful HTTP 200 with an empty or partial content
        # list, so indexing content[0] without this check raises IndexError and
        # burns the retry budget on an error that will never resolve.
        if resp.stop_reason == "refusal":
            raise RuntimeError(
                "scoring request was declined by safety classifiers "
                f"(category={getattr(resp.stop_details, 'category', None)})"
            )
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError(f"no text block in response (stop_reason={resp.stop_reason})")
        data = json.loads(text)
        return ToneScores(**{a: float(data[a]) for a in AXES})

    def score_text(self, text: str) -> ToneScores:
        clean = self.sanitise(text)
        key = self._key(clean)
        cached = self._get(key)
        if cached is not None:
            return cached

        parts: List[ToneScores] = []
        for chunk in self._chunks(clean):
            last_err: Optional[Exception] = None
            for attempt in range(self.max_retries):
                try:
                    parts.append(self._call_once(chunk))
                    break
                except Exception as exc:  # network, rate limit, or bad JSON
                    last_err = exc
                    time.sleep(min(2**attempt, 30))
            else:
                raise RuntimeError(f"scoring failed after retries: {last_err}")

        merged = ToneScores(
            **{a: sum(getattr(p, a) for p in parts) / len(parts) for a in AXES}
        )
        self._put(key, merged)
        return merged


def get_scorer(
    kind: str, cache_path: Path, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT
):
    """Return an object exposing `.score_text(str) -> ToneScores`."""
    if kind == "lexical":
        from . import lexical

        class _Lex:
            name = "lexical"

            @staticmethod
            def score_text(text: str) -> ToneScores:
                return lexical.score_text(text)

        return _Lex()
    if kind == "llm":
        s = LLMScorer(cache_path=cache_path, model=model, effort=effort)
        s.name = "llm"  # type: ignore[attr-defined]
        return s
    raise ValueError(f"unknown scorer {kind!r}; expected 'lexical' or 'llm'")
