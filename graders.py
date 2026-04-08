# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graders for the Adaptive Supply Chain RL Environment.

Each grader evaluates a completed phase and returns a score in [0.0, 1.0].

Scoring formula (weighted sum):
    service_score  = min(avg_service_level / target_service_level, 1.0)   — weight 0.5
    cost_score     = max(0.0, 1.0 - (total_cost / max_allowed_cost))      — weight 0.3
    validity_score = valid_actions / total_actions                          — weight 0.2

Phase targets:
    easy   — service target 95%, max cost $800
    medium — service target 85%, max cost $1200
    hard   — service target 75%, max cost $1500
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class PhaseHistory:
    """Accumulated history for a single difficulty phase."""

    phase: Literal["easy", "medium", "hard"]
    demand_history: List[float]
    fulfilled_history: List[float]
    total_cost: float
    valid_actions: int
    total_actions: int


def grade_phase(history: PhaseHistory) -> float:
    """
    Score a completed phase on a 0.0–1.0 scale.

    Args:
        history: Phase history collected during an episode.

    Returns:
        Float in [0.0, 1.0] — higher is better.
    """
    target = {"easy": 0.95, "medium": 0.85, "hard": 0.75}[history.phase]
    max_cost = {"easy": 800.0, "medium": 1200.0, "hard": 1500.0}[history.phase]

    total_demand = sum(history.demand_history)
    total_fulfilled = sum(history.fulfilled_history)

    avg_sl = total_fulfilled / max(total_demand, 1.0)
    service_score = min(avg_sl / target, 1.0)

    cost_score = max(0.0, 1.0 - (history.total_cost / max_cost))

    validity_score = history.valid_actions / max(history.total_actions, 1)

    score = 0.5 * service_score + 0.3 * cost_score + 0.2 * validity_score
    return round(score, 4)


def grade_easy_phase(history: PhaseHistory) -> float:
    """
    Grade an easy-phase history.

    Enforces phase="easy" regardless of what was recorded in the history.
    """
    return grade_phase(
        PhaseHistory(
            phase="easy",
            demand_history=history.demand_history,
            fulfilled_history=history.fulfilled_history,
            total_cost=history.total_cost,
            valid_actions=history.valid_actions,
            total_actions=history.total_actions,
        )
    )


def grade_medium_phase(history: PhaseHistory) -> float:
    """
    Grade a medium-phase history.

    Enforces phase="medium" regardless of what was recorded in the history.
    """
    return grade_phase(
        PhaseHistory(
            phase="medium",
            demand_history=history.demand_history,
            fulfilled_history=history.fulfilled_history,
            total_cost=history.total_cost,
            valid_actions=history.valid_actions,
            total_actions=history.total_actions,
        )
    )


def grade_hard_phase(history: PhaseHistory) -> float:
    """
    Grade a hard-phase history.

    Enforces phase="hard" regardless of what was recorded in the history.
    """
    return grade_phase(
        PhaseHistory(
            phase="hard",
            demand_history=history.demand_history,
            fulfilled_history=history.fulfilled_history,
            total_cost=history.total_cost,
            valid_actions=history.valid_actions,
            total_actions=history.total_actions,
        )
    )
