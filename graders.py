# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Episode graders for the PharmaNegotiate Supply Chain RL Environment.

Each grader evaluates a completed phase and returns a score in [0.0, 1.0].

Scoring formula (weighted sum):
    service_score  — weight 0.30  (fulfilled / demand vs phase target)
    profit_score   — weight 0.25  (gross revenue vs phase max)
    cost_score     — weight 0.20  (total order cost vs phase max)
    spoilage_score — weight 0.15  (spoilage rate; 20%+ spoilage → 0)
    validity_score — weight 0.10  (valid actions / total actions)

Phase targets:
    easy   — SL target 95%, max revenue Rs 25,000, max cost Rs 15,000
    medium — SL target 85%, max revenue Rs 35,000, max cost Rs 25,000
    hard   — SL target 75%, max revenue Rs 30,000, max cost Rs 30,000
"""

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class PhaseHistory:
    """Accumulated history for a single difficulty phase."""

    phase: Literal["easy", "medium", "hard"]
    demand_history: List[float]
    fulfilled_history: List[float]
    total_cost: float               # total ordering cost (fixed + unit cost)
    valid_actions: int
    total_actions: int
    spoilage_history: List[float] = field(default_factory=list)   # units spoiled each day
    revenue_history: List[float] = field(default_factory=list)    # (sell_price - unit_cost) * fulfilled each day


def grade_phase(history: PhaseHistory) -> float:
    """
    Score a completed phase on a 0.0–1.0 scale.

    Args:
        history: PhaseHistory collected during an episode.

    Returns:
        Float in [0.0, 1.0] — higher is better.
    """
    phase = history.phase

    # ── Service score ─────────────────────────────────────────────────────────
    target = {"easy": 0.95, "medium": 0.85, "hard": 0.75}[phase]
    total_demand    = max(sum(history.demand_history), 1.0)
    total_fulfilled = sum(history.fulfilled_history)
    avg_sl          = total_fulfilled / total_demand
    service_score   = min(avg_sl / target, 1.0)

    # ── Profit score ──────────────────────────────────────────────────────────
    max_revenue = {"easy": 25_000.0, "medium": 35_000.0, "hard": 30_000.0}[phase]
    total_revenue = sum(history.revenue_history)
    profit_score  = max(0.0, min(total_revenue / max_revenue, 1.0))

    # ── Cost score ────────────────────────────────────────────────────────────
    max_cost   = {"easy": 15_000.0, "medium": 25_000.0, "hard": 30_000.0}[phase]
    cost_score = max(0.0, 1.0 - history.total_cost / max_cost)

    # ── Spoilage score ────────────────────────────────────────────────────────
    total_spoiled  = sum(history.spoilage_history)
    spoilage_rate  = total_spoiled / total_demand
    # 0% spoilage → 1.0; 20%+ spoilage → 0.0
    spoilage_score = max(0.0, 1.0 - spoilage_rate * 5.0)

    # ── Validity score ────────────────────────────────────────────────────────
    validity_score = history.valid_actions / max(history.total_actions, 1)

    score = (
        0.30 * service_score
        + 0.25 * profit_score
        + 0.20 * cost_score
        + 0.15 * spoilage_score
        + 0.10 * validity_score
    )
    return round(max(0.0, min(1.0, score)), 4)


def grade_easy_phase(history: PhaseHistory) -> float:
    """Grade an easy-phase history. Enforces phase='easy'."""
    return grade_phase(
        PhaseHistory(
            phase="easy",
            demand_history=history.demand_history,
            fulfilled_history=history.fulfilled_history,
            spoilage_history=history.spoilage_history,
            revenue_history=history.revenue_history,
            total_cost=history.total_cost,
            valid_actions=history.valid_actions,
            total_actions=history.total_actions,
        )
    )


def grade_medium_phase(history: PhaseHistory) -> float:
    """Grade a medium-phase history. Enforces phase='medium'."""
    return grade_phase(
        PhaseHistory(
            phase="medium",
            demand_history=history.demand_history,
            fulfilled_history=history.fulfilled_history,
            spoilage_history=history.spoilage_history,
            revenue_history=history.revenue_history,
            total_cost=history.total_cost,
            valid_actions=history.valid_actions,
            total_actions=history.total_actions,
        )
    )


def grade_hard_phase(history: PhaseHistory) -> float:
    """Grade a hard-phase history. Enforces phase='hard'."""
    return grade_phase(
        PhaseHistory(
            phase="hard",
            demand_history=history.demand_history,
            fulfilled_history=history.fulfilled_history,
            spoilage_history=history.spoilage_history,
            revenue_history=history.revenue_history,
            total_cost=history.total_cost,
            valid_actions=history.valid_actions,
            total_actions=history.total_actions,
        )
    )
