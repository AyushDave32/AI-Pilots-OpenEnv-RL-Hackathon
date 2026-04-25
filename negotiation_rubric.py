# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Negotiation rubric for the PharmaNegotiate Supply Chain RL Environment.

Scores the agent's daily `negotiation_message` using an LLM-as-Judge approach.
The judge evaluates 3 criteria that require genuine language understanding:

  1. RELATIONSHIP   — did the agent reference past history / reliability?
  2. CONCRETE_OFFER — did the agent make a specific payment/quantity commitment?
  3. TONE           — is the message professionally written?

Scoring:
    total_score = checks_passed / 3  always in {0.0, 0.33, 0.67, 1.0}
    checks_needed varies by phase (easy=1, medium=2, hard=3) — controls
    how many checks are required to earn the full negotiation bonus.

Primary scorer  : LLM Judge — same model as inference.py
                  (meta-llama/Llama-3.3-70B-Instruct via HuggingFace Router)
                  Reads HF_TOKEN and API_BASE_URL from environment — same
                  vars already required to run inference.py.
Fallback scorer : Keyword + regex matching (when HF_TOKEN not set or API down)
Cache           : MD5 hash of message → same message never scored twice

Why LLM Judge and not keyword matching or sentiment analysis:
  - Sentiment analysis only detects emotion, not negotiation content
  - Keywords can be gamed ("partnership Rs 1000" → instant 1.0 score)
  - LLM Judge understands meaning — any wording, any industry, any currency
  - Same model doing the task also judges it — self-consistent design
"""

import hashlib
import json
import logging
import os
import re
from typing import TypedDict

logger = logging.getLogger(__name__)


# ── Return type ───────────────────────────────────────────────────────────────

class NegotiationResult(TypedDict):
    relationship_referenced: bool
    concrete_offer_made: bool
    tone_appropriate: bool
    checks_passed: int
    checks_needed: int
    total_score: float    # checks_passed / 3  (0.0, 0.33, 0.67, 1.0)
    reason: str           # LLM explanation of the scores
    scorer_used: str      # "llm" | "keyword_fallback"


# ── LLM Judge configuration ───────────────────────────────────────────────────
# Mirrors inference.py exactly — same model, same API, same token.
# No new environment variables needed.

JUDGE_API_KEY  = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
JUDGE_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
JUDGE_MODEL    = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")


# ── Score cache ───────────────────────────────────────────────────────────────
# Training runs the same messages many times — caching avoids redundant API
# calls and makes training significantly faster.

_score_cache: dict[str, NegotiationResult] = {}


def _cache_key(message: str) -> str:
    return hashlib.md5(message.strip().encode()).hexdigest()


# ── LLM Judge prompt ──────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert business negotiation evaluator. "
    "Your job is to score supplier negotiation messages on exactly 3 criteria. "
    "You must return ONLY a valid JSON object — no explanation outside the JSON."
)

_JUDGE_USER_TEMPLATE = """\
Evaluate this supplier negotiation message and return a JSON score.

MESSAGE:
\"\"\"{message}\"\"\"

CONTEXT:
- Day {day} of 30 in a supply chain episode
- Supplier trust score: {trust_score:.2f} / 1.00
- Crisis period active: {crisis_active}
- Difficulty phase: {phase}

SCORING CRITERIA — answer 0 or 1 for each:

1. "relationship" (0 or 1)
   Does the message clearly reference past order history, payment reliability,
   track record, long-term commitment, or the ongoing business relationship?
   1 = yes, clearly mentioned | 0 = not mentioned at all

2. "concrete_offer" (0 or 1)
   Does the message make a SPECIFIC commitment?
   Examples: payment amount, advance/upfront payment, prepayment offer,
   quantity guarantee, payment timeline.
   1 = specific commitment present | 0 = vague or no commitment

3. "tone" (0 or 1)
   Is the message written in a professional business tone?
   Must be: respectful, coherent, at least 20 characters, no threats or demands.
   1 = professional | 0 = threatening / rude / too short / gibberish

Return ONLY this JSON, nothing else:
{{
  "relationship": <0 or 1>,
  "concrete_offer": <0 or 1>,
  "tone": <0 or 1>,
  "reason": "<one sentence explaining the scores>"
}}"""


# ── LLM Judge call ────────────────────────────────────────────────────────────

def _call_llm_judge(message: str, state: dict) -> "NegotiationResult | None":
    """
    Call the LLM judge. Returns NegotiationResult on success, None on failure.
    Uses temperature=0 for deterministic scoring.
    """
    if not JUDGE_API_KEY:
        logger.warning("LLM judge skipped — HF_TOKEN not set. Using keyword fallback.")
        return None

    try:
        from openai import OpenAI  # already in pyproject.toml

        client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)

        user_prompt = _JUDGE_USER_TEMPLATE.format(
            message=message,
            day=state.get("day", 1),
            trust_score=state.get("trust_score", 0.5),
            crisis_active=state.get("crisis_active", False),
            phase=state.get("phase", "easy"),
        )

        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,    # deterministic — same message always same score
            max_tokens=200,
            timeout=15,       # fail fast if API is slow or unavailable
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON robustly — LLM sometimes wraps in markdown fences
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logger.warning("LLM judge returned non-JSON: %s", raw[:200])
            return None

        parsed = json.loads(json_match.group())

        relationship   = bool(int(parsed.get("relationship",   0)))
        concrete_offer = bool(int(parsed.get("concrete_offer", 0)))
        tone           = bool(int(parsed.get("tone",           0)))
        reason         = str(parsed.get("reason", ""))

        checks_passed = sum([relationship, concrete_offer, tone])
        total_score   = round(checks_passed / 3, 4)

        return NegotiationResult(
            relationship_referenced=relationship,
            concrete_offer_made=concrete_offer,
            tone_appropriate=tone,
            checks_passed=checks_passed,
            checks_needed=3,      # overwritten by caller with phase-specific value
            total_score=total_score,
            reason=reason,
            scorer_used="llm",
        )

    except Exception as exc:
        logger.warning(
            "LLM judge failed (%s: %s). Falling back to keyword scorer. "
            "Check HF_TOKEN is set and API_BASE_URL is reachable.",
            type(exc).__name__, exc,
        )
        return None


# ── Keyword fallback scorer ───────────────────────────────────────────────────
# Activated automatically when the LLM judge is unavailable.
# Keeps training alive — never crashes. Failure is always logged, never silent.

_RELATIONSHIP_KEYWORDS = [
    "past orders", "history", "loyalty", "partnership", "years", "months",
    "track record", "zero defaults", "long-term", "trusted partner",
    "consistent", "reliable", "always paid", "previous order", "our relationship",
    "good standing", "spotless record", "never defaulted", "timely payments",
    "commitment", "dependable", "established", "ongoing relationship",
]

_CONCRETE_OFFER_PATTERNS = [
    r"[Rr]s\.?\s*\d+",
    r"₹\s*\d+",
    r"\$\s*\d+",
    r"€\s*\d+",
    r"\d+\s*units?\b",
    r"advance payment",
    r"upfront",
    r"immediate payment",
    r"immediate transfer",
    r"within \d+ day",
    r"prepay",
    r"full payment",
    r"bank transfer",
    r"commit to",
    r"guarantee \d+",
]

_INAPPROPRIATE_PHRASES = [
    "you must", "or else", "demand that", "unacceptable",
    "ridiculous", "terrible", "worst", "useless", "incompetent",
]


def _keyword_fallback(message: str, state: dict, checks_needed: int) -> NegotiationResult:
    """Keyword-based scorer used when the LLM judge is unavailable."""
    if not message or not message.strip():
        return NegotiationResult(
            relationship_referenced=False, concrete_offer_made=False,
            tone_appropriate=False, checks_passed=0,
            checks_needed=checks_needed, total_score=0.0,
            reason="Empty message.", scorer_used="keyword_fallback",
        )

    msg_lower = message.lower()
    day = state.get("day", 1)

    # Check 1 — relationship referenced
    relationship = any(kw in msg_lower for kw in _RELATIONSHIP_KEYWORDS)
    # Looser check on early days — agent has no history yet
    if not relationship and day < 5:
        relationship = any(w in msg_lower for w in ["partner", "together", "mutual", "future"])

    # Check 2 — concrete offer made
    concrete_offer = any(re.search(p, message) for p in _CONCRETE_OFFER_PATTERNS)

    # Check 3 — tone appropriate
    has_bad    = any(p in msg_lower for p in _INAPPROPRIATE_PHRASES)
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", message))
    too_short  = len(message.strip()) < 20
    tone       = not has_bad and caps_words <= 1 and not too_short

    checks_passed = sum([relationship, concrete_offer, tone])
    total_score   = round(checks_passed / 3, 4)

    return NegotiationResult(
        relationship_referenced=relationship,
        concrete_offer_made=concrete_offer,
        tone_appropriate=tone,
        checks_passed=checks_passed,
        checks_needed=checks_needed,
        total_score=total_score,
        reason="Scored by keyword fallback (LLM judge unavailable).",
        scorer_used="keyword_fallback",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def score_negotiation(
    message: str,
    state: dict,
    phase: str = "hard",
    checks_needed: int = 3,
    adaptive_bonus: float = 0.0,
) -> NegotiationResult:
    """
    Score the agent's negotiation message to the supplier.

    Primary:  LLM Judge (meta-llama/Llama-3.3-70B-Instruct via HuggingFace Router)
    Fallback: Keyword matching (when HF_TOKEN not set or API unavailable)
    Cache:    Same message is never scored twice — instant on repeat calls

    Args:
        message:        The agent's negotiation_message string from its action.
        state:          Current environment state dict with keys:
                            day, phase, trust_score, crisis_active,
                            current_stock, budget_remaining.
        phase:          "easy" | "medium" | "hard"
        checks_needed:  Checks required for full bonus (easy=1, medium=2, hard=3).
                        Caller passes RUBRIC_CHECKS_NEEDED[phase].
        adaptive_bonus: Float 0.0–0.15; supplier raises bar when agent performs
                        well (Theme 4 adaptive difficulty). Used by environment
                        to adjust neg_bonus — not used internally here.

    Returns:
        NegotiationResult TypedDict with individual check booleans,
        total_score, reason, and scorer_used.
    """
    # Empty message — instant zero, no LLM call needed
    if not message or not message.strip():
        return NegotiationResult(
            relationship_referenced=False, concrete_offer_made=False,
            tone_appropriate=False, checks_passed=0,
            checks_needed=checks_needed, total_score=0.0,
            reason="Empty message — no negotiation attempted.",
            scorer_used="llm",
        )

    # Cache check — same message never scored twice
    key = _cache_key(message)
    if key in _score_cache:
        cached = _score_cache[key].copy()
        cached["checks_needed"] = checks_needed   # update phase-specific field
        return cached

    # Try LLM judge first
    result = _call_llm_judge(message, state)

    # Fall back to keywords if LLM unavailable
    if result is None:
        result = _keyword_fallback(message, state, checks_needed)
    else:
        result["checks_needed"] = checks_needed

    # Store in cache
    _score_cache[key] = result

    return result
