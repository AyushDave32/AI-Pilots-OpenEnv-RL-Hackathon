# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Adaptive Supply Chain RL Environment.

Simulates a warehouse manager making daily inventory ordering decisions
over a 30-day episode under uncertain demand and variable supplier lead times.
"""

from typing import List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class SupplyChainAction(Action):
    """Action for the supply chain environment."""

    action_type: Literal["order", "emergency_restock", "hold"] = Field(
        ..., description="Type of action: place a regular order, emergency restock, or hold"
    )
    quantity: Optional[int] = Field(
        default=None,
        description="Units to order (required for 'order' and 'emergency_restock'; None for 'hold')",
    )


class PendingOrder(BaseModel):
    """A pending order that has been placed but not yet arrived."""

    quantity: int = Field(..., description="Number of units ordered")
    arrives_in_days: int = Field(..., description="Days until this order arrives")


class SupplyChainObservation(Observation):
    """Observation from the supply chain environment.

    Note: `done`, `reward`, and `metadata` are inherited from Observation base class.
    Do NOT redefine them here.
    """

    day: int = Field(..., description="Current day (1–30)")
    current_stock: int = Field(..., description="Units currently in the warehouse")
    demand_forecast: float = Field(..., description="Forecasted demand for today")
    forecast_noise: Literal["low", "medium", "high"] = Field(
        ..., description="Uncertainty level of the forecast"
    )
    pending_orders: List[PendingOrder] = Field(
        default_factory=list, description="Orders placed but not yet arrived"
    )
    last_7_day_service_level: float = Field(
        default=1.0, description="Fraction of demand fulfilled over last 7 days"
    )
    holding_cost_per_unit: float = Field(
        default=0.5, description="Cost to hold one unit for one day"
    )
    stockout_penalty: float = Field(
        default=50.0, description="Penalty per stockout event"
    )
    budget_remaining: float = Field(..., description="Remaining budget in dollars")
    supplier_status: Literal["normal", "delayed"] = Field(
        ..., description="Current supplier reliability status"
    )
    current_phase: Literal["easy", "medium", "hard"] = Field(
        ..., description="Current curriculum difficulty phase"
    )
    prompt: str = Field(
        ..., description="Natural language prompt for LLM-based agents"
    )

    # Live grading fields (serialized directly — metadata is stripped by the framework)
    phase_score: float = Field(
        default=0.0, description="Live grader score for the current phase (0.0–1.0)"
    )
    actual_demand: float = Field(
        default=0.0, description="Actual demand sampled this step (0 on reset)"
    )
    actual_fulfilled: float = Field(
        default=0.0, description="Units actually fulfilled this step (0 on reset)"
    )
