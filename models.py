# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the PharmaNegotiate Supply Chain RL Environment.

Simulates MediStock Pvt. Ltd., a surgical gloves distributor in Mumbai,
managing perishable pharmaceutical inventory with supplier negotiation
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
    sell_price: float = Field(
        default=265.0,
        description="Rs per box sold to hospitals today; affects demand via price elasticity",
    )
    negotiation_message: str = Field(
        default="",
        description="Natural language message to GloveMaker Industries; empty = score 0",
    )


class StockBatch(BaseModel):
    """A batch of stock with its own expiry date (for FEFO inventory management)."""

    quantity: int = Field(..., description="Number of units in this batch")
    expires_on_day: int = Field(..., description="Episode day on which this batch spoils if unsold")
    arrived_on_day: int = Field(..., description="Episode day this batch arrived (for logging)")


class PendingOrder(BaseModel):
    """A pending order that has been placed but not yet arrived."""

    quantity: int = Field(..., description="Number of units ordered")
    arrives_in_days: int = Field(..., description="Days until this order arrives")


class SupplyChainObservation(Observation):
    """Observation from the PharmaNegotiate supply chain environment.

    Note: `done`, `reward`, and `metadata` are inherited from Observation base class.
    Do NOT redefine them here.
    """

    # --- CORE FIELDS ---
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
    budget_remaining: float = Field(..., description="Remaining budget in Rs")
    supplier_status: Literal["normal", "delayed"] = Field(
        ..., description="Current supplier reliability status"
    )
    current_phase: Literal["easy", "medium", "hard"] = Field(
        ..., description="Current curriculum difficulty phase"
    )
    prompt: str = Field(
        ..., description="Natural language prompt for LLM-based agents"
    )

    # --- INVENTORY FIELDS ---
    days_until_nearest_expiry: int = Field(
        default=999, description="Days until the nearest-expiring batch expires (999 if no stock)"
    )
    expiring_soon_qty: int = Field(
        default=0, description="Units expiring within 3 days"
    )
    expiry_warning: str = Field(
        default="No stock on hand", description="Human-readable expiry urgency message"
    )
    batch_count: int = Field(
        default=0, description="Number of distinct live stock batches"
    )
    units_spoiled_today: int = Field(
        default=0, description="Units that expired and were lost this step"
    )

    # --- MARKET FIELDS ---
    market_price: float = Field(
        default=265.0, description="Today's market price (Rs/box)"
    )
    last_sell_price: float = Field(
        default=265.0, description="The sell price the agent used yesterday"
    )

    # --- SUPPLIER RELATIONSHIP FIELDS ---
    trust_score: float = Field(
        default=0.8, description="Observable trust score 0.0–1.0"
    )
    supplier_last_message: str = Field(
        default="", description="One-sentence supplier response from previous step"
    )
    lead_time_accuracy: str = Field(
        default="on time", description="'on time' | '1 day late' | 'X days late'"
    )
    emergency_surcharge_rate: float = Field(
        default=2.5, description="Emergency restock surcharge multiplier — reveals loyalty tier"
    )
    proactive_discount_offered: bool = Field(
        default=False, description="True if supplier proactively offered a discount today"
    )
    recent_neg_scores: List[float] = Field(
        default_factory=list, description="Last 3 negotiation scores (0.0–1.0)"
    )

    # --- EPISODE STATE FIELDS ---
    crisis_active: bool = Field(
        default=False, description="True on days 21–25 (factory fire crisis)"
    )
    phase_score: float = Field(
        default=0.0, description="Live grader score for the current phase (0.0–1.0)"
    )
    actual_demand: float = Field(
        default=0.0, description="Actual demand sampled this step (0 on reset)"
    )
    actual_fulfilled: float = Field(
        default=0.0, description="Units actually fulfilled this step (0 on reset)"
    )
