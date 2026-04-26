"""
15 tests for negotiation_rubric.score_negotiation().
All must pass before Person B signs off.

Run: python -m pytest test_negotiation_rubric.py -v
"""

import pytest
from negotiation_rubric import score_negotiation

# ── Shared state fixtures ─────────────────────────────────────────────────────

BASE_STATE = {
    "day": 10, "phase": "hard", "trust_score": 0.6,
    "crisis_active": False, "current_stock": 100, "budget_remaining": 5000,
}
CRISIS_STATE = {**BASE_STATE, "day": 21, "crisis_active": True}
EARLY_STATE  = {**BASE_STATE, "day": 3}


# ── Empty / trivially bad messages ───────────────────────────────────────────

def test_empty_message():
    r = score_negotiation("", BASE_STATE)
    assert r["total_score"] == 0.0
    assert not r["relationship_referenced"]
    assert not r["concrete_offer_made"]
    assert not r["tone_appropriate"]


def test_whitespace_only():
    r = score_negotiation("   ", BASE_STATE)
    assert r["total_score"] == 0.0


def test_too_short():
    r = score_negotiation("send boxes", BASE_STATE)
    assert not r["tone_appropriate"]


# ── Tone check failures ───────────────────────────────────────────────────────

def test_all_caps_fails_tone():
    r = score_negotiation(
        "As a partner, SEND 100 UNITS NOW with Rs 5000 advance payment", BASE_STATE
    )
    assert not r["tone_appropriate"]


def test_you_must_fails_tone():
    r = score_negotiation(
        "You must send 100 units. Rs 5000 payment. Our long-term partnership requires it.",
        BASE_STATE,
    )
    assert not r["tone_appropriate"]


# ── Relationship reference ────────────────────────────────────────────────────

def test_partnership_keyword():
    r = score_negotiation(
        "As a long-term partner, we would like to place an order today.", BASE_STATE
    )
    assert r["relationship_referenced"]


def test_zero_defaults_keyword():
    r = score_negotiation(
        "Our company has zero defaults on all past orders with your team.", BASE_STATE
    )
    assert r["relationship_referenced"]


# ── Concrete offer ────────────────────────────────────────────────────────────

def test_currency_amount_concrete():
    r = score_negotiation(
        "We offer advance payment of $24000 to secure this order.", BASE_STATE
    )
    assert r["concrete_offer_made"]


def test_rupee_symbol_concrete():
    r = score_negotiation(
        "Immediate transfer of ₹15000 upon confirmation.", BASE_STATE
    )
    assert r["concrete_offer_made"]


def test_advance_payment_concrete():
    r = score_negotiation(
        "We are happy to provide advance payment for this order as a trusted partner.",
        BASE_STATE,
    )
    assert r["concrete_offer_made"]
    assert r["relationship_referenced"]


def test_within_days_concrete():
    r = score_negotiation(
        "We can settle payment within 2 days of delivery as per our track record.",
        BASE_STATE,
    )
    assert r["concrete_offer_made"]
    assert r["relationship_referenced"]


# ── Score boundary tests ──────────────────────────────────────────────────────

def test_all_three_pass_score_1():
    msg = (
        "Dear supplier — our company has maintained zero defaults across all past orders. "
        "We request 120 units and offer immediate advance payment of $24,000."
    )
    r = score_negotiation(msg, BASE_STATE)
    assert r["total_score"] == 1.0
    assert r["checks_passed"] == 3


def test_two_pass_score_067():
    # concrete_offer + tone pass; no relationship reference
    msg = "We offer advance payment of Rs 10,000. Please process our order promptly as usual."
    r = score_negotiation(msg, BASE_STATE)
    assert r["checks_passed"] == 2
    assert abs(r["total_score"] - 0.6667) < 0.01


def test_one_pass_score_033():
    # relationship passes; tone fails ("or else"); no concrete offer
    msg = "Our long-term partnership history speaks for itself, or else we find another supplier."
    r = score_negotiation(msg, BASE_STATE)
    assert r["relationship_referenced"]
    assert not r["tone_appropriate"]
    assert not r["concrete_offer_made"]
    assert r["checks_passed"] == 1
    assert abs(r["total_score"] - 0.3333) < 0.01


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_early_day_loose_relationship():
    msg = "We look forward to building a mutual partnership with your team."
    r = score_negotiation(msg, EARLY_STATE)
    assert r["relationship_referenced"]  # loose check applies for day < 5


def test_crisis_day_full_message():
    msg = (
        "Dear supplier — as a consistent long-term partner with zero payment defaults, "
        "we formally request priority allocation of 120 units during this crisis period. "
        "We offer immediate advance payment of $24,000."
    )
    r = score_negotiation(msg, CRISIS_STATE, phase="hard", checks_needed=3)
    assert r["total_score"] == 1.0
    assert r["checks_passed"] == 3
