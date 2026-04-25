# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Negotiation rubric for the PharmaNegotiate Supply Chain RL Environment.

Scores the agent's daily `negotiation_message` to GloveMaker Industries on
three criteria. Phase-aware: checks_needed varies by phase so the rubric
scales with curriculum difficulty.

    easy   → 1 check needed for full negotiation bonus
    medium → 2 checks needed
    hard   → all 3 checks needed

total_score = checks_passed / 3  (always denominator 3, in {0.0, 0.33, 0.67, 1.0})
The environment uses checks_needed to decide how much of total_score earns reward.

Returns a TypedDict so the environment can access fields as dict keys:
    neg_result["total_score"]
    neg_result["checks_passed"]
"""

import re
from typing import List, TypedDict


# ── Return type ───────────────────────────────────────────────────────────────

class NegotiationResult(TypedDict):
    relationship_referenced: bool
    concrete_offer_made: bool
    tone_appropriate: bool
    checks_passed: int
    checks_needed: int
    total_score: float  # checks_passed / 3, always in {0.0, 0.33, 0.67, 1.0}


# ── Keyword / pattern banks ───────────────────────────────────────────────────

RELATIONSHIP_KEYWORDS: List[str] = [
    "past orders", "history", "loyalty", "partnership", "years", "months",
    "track record", "zero defaults", "long-term", "trusted partner",
    "consistent", "reliable", "always paid", "previous order", "our relationship",
    "account", "established", "ongoing", "valued", "committed",
]

# Looser keywords used on early days (day < 5) when agent has no real history yet
_EARLY_RELATIONSHIP_KEYWORDS: List[str] = [
    "partner", "together", "mutual", "future", "collaboration",
]

CONCRETE_OFFER_PATTERNS: List[str] = [
    r"[Rr]s\.?\s*[\d,]+",      # Rs amount — e.g. "Rs 24,000" or "Rs24000"
    r"₹\s*[\d,]+",             # Rupee symbol
    r"\d[\d,]*\s*units?\b",    # quantity — "100 units"
    r"\d[\d,]*\s*boxes?\b",    # quantity — "120 boxes"
    r"advance payment",
    r"upfront",
    r"immediate payment",
    r"within \d+ day",
    r"guarantee \d+",
    r"commit to",
    r"faster payment",
    r"prepay",
]

# Phrases that signal an inappropriate / unprofessional tone
_INAPPROPRIATE_PHRASES: List[str] = [
    "you must",
    "or else",
    "demand that",
    "unacceptable",
    "ridiculous",
    "terrible",
    "worst",
    "useless",
]


# ── Individual check helpers ──────────────────────────────────────────────────

def _check_relationship_referenced(message: str, day: int) -> bool:
    msg_lower = message.lower()
    if any(kw in msg_lower for kw in RELATIONSHIP_KEYWORDS):
        return True
    # On early days (no real history yet) accept weaker relational language
    if day < 5:
        return any(kw in msg_lower for kw in _EARLY_RELATIONSHIP_KEYWORDS)
    return False


def _check_concrete_offer_made(message: str) -> bool:
    return any(re.search(pat, message) for pat in CONCRETE_OFFER_PATTERNS)


def _check_tone_appropriate(message: str) -> bool:
    """Professional tone: no aggressive phrases, no ALL-CAPS shouting, not too short."""
    if len(message.strip()) < 20:
        return False
    msg_lower = message.lower()
    if any(phrase in msg_lower for phrase in _INAPPROPRIATE_PHRASES):
        return False
    # Count standalone ALL-CAPS words (3+ letters) — more than 1 signals shouting
    caps_words = re.findall(r'\b[A-Z]{3,}\b', message)
    if len(caps_words) > 1:
        return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def score_negotiation(
    message: str,
    state: dict,
    phase: str = "hard",
    checks_needed: int = 3,
    adaptive_bonus: float = 0.0,
) -> NegotiationResult:
    """
    Score the agent's negotiation message to GloveMaker Industries.

    Args:
        message:       The raw negotiation_message string from the agent's action.
        state:         Current environment state dict with keys:
                           day, phase, trust_score, crisis_active,
                           current_stock, budget_remaining.
        phase:         Current difficulty phase ("easy" | "medium" | "hard").
        checks_needed: How many of the 3 checks must pass for full reward.
                       Caller passes RUBRIC_CHECKS_NEEDED[phase].
        adaptive_bonus: Float 0.0–0.15 added to score when agent is performing
                        well (Theme 4 adaptive difficulty). Unused internally here
                        — it is used by the environment to adjust neg_bonus.

    Returns:
        NegotiationResult TypedDict.
    """
    day = state.get("day", 1)

    # Empty or trivially short message → immediate zero
    if not message or not message.strip():
        return NegotiationResult(
            relationship_referenced=False,
            concrete_offer_made=False,
            tone_appropriate=False,
            checks_passed=0,
            checks_needed=checks_needed,
            total_score=0.0,
        )

    relationship_referenced = _check_relationship_referenced(message, day)
    concrete_offer_made = _check_concrete_offer_made(message)
    tone_appropriate = _check_tone_appropriate(message)

    checks_passed = sum([relationship_referenced, concrete_offer_made, tone_appropriate])

    # total_score always /3 — environment multiplies by bonus cap & phase weight separately
    total_score = round(checks_passed / 3, 4)

    return NegotiationResult(
        relationship_referenced=relationship_referenced,
        concrete_offer_made=concrete_offer_made,
        tone_appropriate=tone_appropriate,
        checks_passed=checks_passed,
        checks_needed=checks_needed,
        total_score=total_score,
    )
