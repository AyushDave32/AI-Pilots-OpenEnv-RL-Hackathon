# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Adaptive Supply Chain RL Environment.

General-purpose perishable goods inventory management over a 30-day episode.
Domain-agnostic: works for pharmaceuticals, food logistics, electronics, or any
other industry with perishable inventory and supplier relationships.

The agent must:
  1. Buy-side decisions — order type, quantity, timing (FEFO, expiry-aware)
  2. Sell-side decisions — set daily sell price via price elasticity
  3. Relationship management — write daily supplier negotiation messages
  4. World modeling — infer supplier's hidden loyalty tier from observable signals

The supplier maintains hidden state (loyalty tier, supplier mood, order
regularity) that drives costs and crisis allocation.
On days 21–25 a supply disruption reduces capacity to 30% — only loyal
customers get full allocation.
"""

import math
import uuid
from typing import List

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PendingOrder, StockBatch, SupplyChainAction, SupplyChainObservation
except (ImportError, ModuleNotFoundError):
    from models import PendingOrder, StockBatch, SupplyChainAction, SupplyChainObservation

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SHELF_LIFE_DAYS = 15
UNIT_COST_STANDARD = 2
FIXED_ORDER_COST = 20
STOCKOUT_PENALTY = 50.0
SPOILAGE_PENALTY_PER_UNIT = 20.0
OVERSTOCK_THRESHOLD = 300
OPTIMAL_STOCK_RANGE = (50, 300)
PRICE_ELASTICITY = 1.5
NEG_EPISODE_CAP = 30.0
CRISIS_DAYS = set(range(21, 26))

MARKET_PRICE = {"easy": 265.0, "medium": 285.0, "hard": 310.0}
BASE_DEMAND = {"easy": 80.0, "medium": 100.0, "hard": 90.0}

SURCHARGE = {"gold": 2.5, "silver": 3.0, "bronze": 4.0}
LEAD_MODIFIER = {"gold": -1, "silver": 0, "bronze": 2}
CRISIS_FILL = {"gold": 1.0, "silver": 0.8, "bronze": 0.5}

SUPPLIER_MESSAGES = {
    "gold": "Your order of {qty} units is confirmed and prioritised — delivery in {lt} days.",
    "silver": "Order received. Expected delivery in {lt} days.",
    "bronze": "We'll process your order. Lead times may extend to {lt} days.",
    "bronze_crisis": "We regret we can only fulfil {actual} of your {qty} units at this time.",
    "gold_discount": "As a valued partner, we are offering a 5% discount on this order.",
    "no_order": "No order placed today. We look forward to your next order.",
}

RUBRIC_CHECKS_NEEDED = {"easy": 1, "medium": 2, "hard": 3}


class AscAgentUnderDemandUncertainityRlEnvironment(Environment):
    """
    Adaptive Supply Chain RL Environment.

    The agent manages perishable goods inventory with:
    - FEFO (First Expired First Out) batch-level inventory tracking
    - Sell-side price elasticity (agent sets daily customer sell price)
    - Supplier hidden state (loyalty tier: bronze/silver/gold)
    - LLM-native negotiation action (scored by rubric, drives loyalty)
    - Day 21–25 factory crisis (capacity 30%; tier determines allocation)

    Episode flow:
        reset() → step() × 30 → done=True (day > 30)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._episode_id: str = str(uuid.uuid4())
        self._day: int = 1
        self._done: bool = False
        self._rng: np.random.Generator = np.random.default_rng()

        self._forced_phase = None
        self._stock_batches: List[StockBatch] = []
        self._pending_orders: List[PendingOrder] = []
        self._budget: float = 10000.0

        self._demand_history: List[float] = []
        self._fulfilled_history: List[float] = []
        self._neg_score_history: List[float] = []

        self._neg_bonus_total: float = 0.0
        self._last_sell_price: float = MARKET_PRICE["easy"]
        self._last_7_day_service_level: float = 1.0

        # Supplier hidden state — never exposed directly in observation
        self._loyalty_tier: str = "gold"
        self._supplier_mood: float = 0.8
        self._order_regularity: float = 0.5
        self._trust_score: float = 0.8
        self._consecutive_holds: int = 0
        self._last_lead_promised: int = 3
        self._supplier_neg_threshold_bonus: float = 0.0

        self._initialized: bool = False

        # Last-step observable supplier signals (built in step, read in _build_observation)
        self._supplier_last_message: str = ""
        self._lead_time_accuracy: str = "on time"
        self._proactive_discount: bool = False
        self._last_units_spoiled: int = 0

        # Phase tracking for live grading
        self._phase_demand: List[float] = []
        self._phase_fulfilled: List[float] = []
        self._phase_spoilage: List[float] = []
        self._phase_revenue: List[float] = []
        self._phase_total_cost: float = 0.0
        self._phase_valid_actions: int = 0
        self._phase_total_actions: int = 0

    # -----------------------------------------------------------------------
    # Environment interface
    # -----------------------------------------------------------------------

    def reset(self, seed=None, task=None, **kwargs) -> SupplyChainObservation:
        """Reset the environment for a new episode."""
        task_to_phase = {
            "easy_phase_inventory": "easy",
            "medium_phase_inventory": "medium",
            "hard_phase_inventory": "hard",
        }
        self._forced_phase = task_to_phase.get(task) if task else None

        self._rng = np.random.default_rng(seed)
        self._day = 1
        self._budget = 10000.0
        self._episode_id = str(uuid.uuid4())
        self._done = False

        self._stock_batches = []
        self._pending_orders = []
        self._demand_history = []
        self._fulfilled_history = []
        self._neg_score_history = []

        self._neg_bonus_total = 0.0
        self._last_sell_price = MARKET_PRICE["easy"]
        self._last_7_day_service_level = 1.0

        # Supplier hidden state
        self._loyalty_tier = "gold"
        self._supplier_mood = 0.8
        self._order_regularity = 0.5
        self._trust_score = 0.8
        self._consecutive_holds = 0
        self._last_lead_promised = 3
        self._supplier_neg_threshold_bonus = 0.0

        # Observable supplier signals reset
        self._supplier_last_message = "Welcome. We look forward to a productive partnership."
        self._lead_time_accuracy = "on time"
        self._proactive_discount = False
        self._last_units_spoiled = 0

        # Phase trackers
        self._phase_demand = []
        self._phase_fulfilled = []
        self._phase_spoilage = []
        self._phase_revenue = []
        self._phase_total_cost = 0.0
        self._phase_valid_actions = 0
        self._phase_total_actions = 0

        self._initialized = True

        # Initial stock — expires day 16 to create urgency entering medium phase
        init_qty = int(self._rng.integers(150, 301))
        self._stock_batches.append(
            StockBatch(quantity=init_qty, expires_on_day=16, arrived_on_day=0)
        )

        return self._build_observation(reward=0.0)

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """Execute one day in the supply chain simulation."""

        # STEP 1 — Guard
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if not self._initialized:
            self.reset()

        # STEP 2 — Validate action
        malformed = False
        reward = 0.0

        phase = self._current_phase()

        if action.action_type in ("order", "emergency_restock"):
            if action.quantity is None or action.quantity <= 0:
                malformed = True
                reward -= 10.0
                action = SupplyChainAction(
                    action_type="hold",
                    quantity=None,
                    sell_price=action.sell_price if action.sell_price and action.sell_price > 0 else MARKET_PRICE[phase],
                    negotiation_message=action.negotiation_message,
                )

        if action.sell_price is None or action.sell_price <= 0:
            malformed = True
            reward -= 10.0
            action = SupplyChainAction(
                action_type=action.action_type,
                quantity=action.quantity,
                sell_price=MARKET_PRICE[phase],
                negotiation_message=action.negotiation_message,
            )

        self._phase_total_actions += 1
        if not malformed:
            self._phase_valid_actions += 1

        # STEP 3 — Determine phase and crisis state
        crisis_active = self._day in CRISIS_DAYS

        # At phase boundaries, reset loyalty tier to phase default
        if self._day == 8 and not self._forced_phase:
            self._loyalty_tier = "silver"
        if self._day == 16 and not self._forced_phase:
            self._loyalty_tier = "bronze"

        # STEP 4 — Spoilage check (BEFORE arrivals and demand)
        expired = [b for b in self._stock_batches if b.expires_on_day <= self._day]
        units_spoiled = sum(b.quantity for b in expired)
        self._stock_batches = [b for b in self._stock_batches if b.expires_on_day > self._day]
        reward -= SPOILAGE_PENALTY_PER_UNIT * units_spoiled
        self._last_units_spoiled = units_spoiled

        # STEP 5 — Process pending order arrivals
        still_pending = []
        for order in self._pending_orders:
            order = PendingOrder(
                quantity=order.quantity,
                arrives_in_days=order.arrives_in_days - 1,
            )
            if order.arrives_in_days <= 0:
                new_batch = StockBatch(
                    quantity=order.quantity,
                    expires_on_day=self._day + SHELF_LIFE_DAYS,
                    arrived_on_day=self._day,
                )
                self._stock_batches.append(new_batch)
            else:
                still_pending.append(order)
        self._pending_orders = still_pending

        # STEP 6 — Compute loyalty tier from hidden state
        if self._order_regularity > 0.70 and self._trust_score > 0.75:
            self._loyalty_tier = "gold"
        elif self._order_regularity > 0.40 or self._trust_score > 0.50:
            self._loyalty_tier = "silver"
        else:
            self._loyalty_tier = "bronze"

        # Adaptive difficulty: agent performing well → supplier raises bar
        if self._last_7_day_service_level > 0.85 and self._day > 7:
            self._supplier_neg_threshold_bonus = min(
                0.15, self._supplier_neg_threshold_bonus + 0.01
            )

        # STEP 7 — Process agent's order
        my_actual = 0
        order_cost = 0.0
        lead_time_promised = 0

        if action.action_type == "order":
            base_lt_map = {
                "easy": 3,
                "medium": int(self._rng.integers(2, 6)),
                "hard": int(self._rng.integers(2, 11)),
            }
            base_lt = base_lt_map[phase]
            lead_time_promised = max(1, base_lt + LEAD_MODIFIER[self._loyalty_tier])
            my_actual = action.quantity
            order_cost = FIXED_ORDER_COST + my_actual * UNIT_COST_STANDARD
            reward -= order_cost
            self._budget -= order_cost
            self._phase_total_cost += order_cost
            self._consecutive_holds = 0
            self._order_regularity = min(1.0, self._order_regularity + 0.03)

        elif action.action_type == "emergency_restock":
            surcharge_rate = SURCHARGE[self._loyalty_tier]
            lead_time_promised = 1
            my_actual = action.quantity
            order_cost = FIXED_ORDER_COST + my_actual * UNIT_COST_STANDARD * surcharge_rate
            reward -= order_cost
            self._budget -= order_cost
            self._phase_total_cost += order_cost
            self._consecutive_holds = 0
            self._order_regularity = max(0.0, self._order_regularity - 0.05)

        else:  # hold
            self._consecutive_holds += 1
            if self._consecutive_holds >= 3:
                self._order_regularity = max(0.0, self._order_regularity - 0.02)

        if self._budget < 0:
            reward -= 100.0
            self._order_regularity = max(0.0, self._order_regularity - 0.10)

        # STEP 8 — Apply crisis allocation
        requested_qty = my_actual
        if crisis_active and my_actual > 0:
            fill_rate = CRISIS_FILL[self._loyalty_tier]
            my_actual = int(my_actual * fill_rate)

        # STEP 9 — Add order to pending with loyalty-adjusted lead time
        if my_actual > 0:
            if self._loyalty_tier == "bronze":
                lt_actual = lead_time_promised + int(self._rng.integers(0, 3))
            else:
                lt_actual = lead_time_promised
            self._last_lead_promised = lead_time_promised
            self._pending_orders.append(
                PendingOrder(quantity=my_actual, arrives_in_days=lt_actual)
            )

        # STEP 10 — Sample actual demand with price elasticity
        base = BASE_DEMAND[phase]
        market_price = MARKET_PRICE[phase]
        noise_pct = {"easy": 0.10, "medium": 0.25, "hard": 0.40}[phase]
        noise = self._rng.uniform(-noise_pct, noise_pct)

        if phase == "medium":
            seasonal = 1.0 + 0.3 * math.sin(math.pi * (self._day - 8) / 7)
            base *= seasonal

        if phase == "hard" and self._rng.random() < 0.2:
            base *= float(self._rng.uniform(1.5, 2.5))

        sell_price = max(action.sell_price, 1.0)
        price_ratio = market_price / sell_price
        demand_factor = price_ratio ** PRICE_ELASTICITY

        if self._supplier_neg_threshold_bonus > 0.05:
            demand_factor *= float(self._rng.uniform(0.9, 1.1))

        actual_demand = max(0, int(base * (1 + noise) * demand_factor))

        # STEP 11 — Fulfill demand using FEFO (First Expired First Out)
        remaining_demand = actual_demand
        for batch in sorted(self._stock_batches, key=lambda b: b.expires_on_day):
            if remaining_demand <= 0:
                break
            take = min(batch.quantity, remaining_demand)
            batch.quantity -= take
            remaining_demand -= take
        self._stock_batches = [b for b in self._stock_batches if b.quantity > 0]

        units_fulfilled = actual_demand - remaining_demand
        stockout_occurred = remaining_demand > 0
        current_stock = sum(b.quantity for b in self._stock_batches)

        # STEP 12 — Sell-side profit and penalties
        gross_profit = 3.0 * units_fulfilled
        reward += gross_profit
        if stockout_occurred:
            reward -= STOCKOUT_PENALTY
        excess = max(0, current_stock - OVERSTOCK_THRESHOLD)
        reward -= 0.5 * excess
        if OPTIMAL_STOCK_RANGE[0] <= current_stock <= OPTIMAL_STOCK_RANGE[1]:
            reward += 5.0

        # STEP 13 — Sell price regularity effect on order_regularity
        price_deviation = abs(sell_price - market_price) / market_price
        if price_deviation <= 0.20:
            self._order_regularity = min(1.0, self._order_regularity + 0.01)

        # STEP 14 — Negotiation scoring
        try:
            try:
                from negotiation_rubric import score_negotiation
            except ImportError:
                from ..negotiation_rubric import score_negotiation  # type: ignore
            checks_needed = RUBRIC_CHECKS_NEEDED[phase]
            neg_result = score_negotiation(
                message=action.negotiation_message,
                state=self._get_state_dict(),
                phase=phase,
                checks_needed=checks_needed,
                adaptive_bonus=self._supplier_neg_threshold_bonus,
            )
            neg_score = neg_result["total_score"]
        except Exception:
            neg_score = 0.0

        neg_bonus = min(neg_score * 10.0, max(0.0, NEG_EPISODE_CAP - self._neg_bonus_total))
        self._neg_bonus_total += neg_bonus
        reward += neg_bonus

        if neg_score >= (RUBRIC_CHECKS_NEEDED[phase] / 3):
            self._trust_score = min(1.0, self._trust_score + 0.03)
            self._order_regularity = min(1.0, self._order_regularity + 0.01)
        else:
            self._trust_score = max(0.0, self._trust_score - 0.01)

        self._neg_score_history.append(neg_score)

        # STEP 15 — Generate supplier response message
        proactive_discount = False
        if crisis_active and self._loyalty_tier == "bronze" and my_actual < requested_qty:
            supplier_msg = SUPPLIER_MESSAGES["bronze_crisis"].format(
                actual=my_actual, qty=requested_qty
            )
        elif self._loyalty_tier == "gold" and my_actual > 0 and self._rng.random() < 0.25:
            supplier_msg = SUPPLIER_MESSAGES["gold_discount"]
            proactive_discount = True
        elif my_actual > 0:
            template = SUPPLIER_MESSAGES[self._loyalty_tier]
            supplier_msg = template.format(qty=my_actual, lt=self._last_lead_promised)
        else:
            supplier_msg = SUPPLIER_MESSAGES["no_order"]

        # Lead time accuracy signal based on loyalty tier
        if self._loyalty_tier == "gold":
            lead_time_accuracy = "on time"
        elif self._loyalty_tier == "silver":
            lead_time_accuracy = "on time" if self._rng.random() > 0.15 else "1 day late"
        else:
            days_late = int(self._rng.integers(0, 3))
            lead_time_accuracy = "on time" if days_late == 0 else f"{days_late} day(s) late"

        self._supplier_last_message = supplier_msg
        self._lead_time_accuracy = lead_time_accuracy
        self._proactive_discount = proactive_discount

        # STEP 16 — Update histories and rolling service level
        self._demand_history.append(float(actual_demand))
        self._fulfilled_history.append(float(units_fulfilled))
        self._last_sell_price = action.sell_price

        self._phase_demand.append(float(actual_demand))
        self._phase_fulfilled.append(float(units_fulfilled))
        self._phase_spoilage.append(float(units_spoiled))
        self._phase_revenue.append(float(gross_profit))

        d7 = self._demand_history[-7:]
        f7 = self._fulfilled_history[-7:]
        self._last_7_day_service_level = sum(f7) / max(sum(d7), 1)

        # STEP 17 — Compute expiry info for observation (handled in _build_observation)

        # STEP 18 — Increment day and check done
        self._day += 1
        self._done = self._day > 30

        # STEP 19 — Build metadata and return observation
        metadata = {
            "actual_demand": actual_demand,
            "units_fulfilled": units_fulfilled,
            "units_spoiled": units_spoiled,
            "stockout_occurred": stockout_occurred,
            "neg_score": neg_score,
            "neg_bonus": neg_bonus,
            "crisis_active": crisis_active,
            "loyalty_tier": self._loyalty_tier,
            "action_malformed": malformed,
            "phase_score": 0.0,
            "reward_components": {
                "gross_profit": gross_profit,
                "spoilage_penalty": -SPOILAGE_PENALTY_PER_UNIT * units_spoiled,
                "stockout_penalty": -STOCKOUT_PENALTY if stockout_occurred else 0.0,
                "overstock_penalty": -0.5 * excess,
                "order_cost": -order_cost,
                "efficiency_bonus": (
                    5.0 if OPTIMAL_STOCK_RANGE[0] <= current_stock <= OPTIMAL_STOCK_RANGE[1] else 0.0
                ),
                "negotiation_bonus": neg_bonus,
            },
        }
        return self._build_observation(reward=reward, metadata=metadata)

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self._day)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _current_phase(self) -> str:
        if self._forced_phase:
            return self._forced_phase
        if self._day <= 7:
            return "easy"
        elif self._day <= 15:
            return "medium"
        return "hard"

    def _get_state_dict(self) -> dict:
        return {
            "day": self._day,
            "phase": self._current_phase(),
            "trust_score": self._trust_score,
            "crisis_active": self._day in CRISIS_DAYS,
            "current_stock": sum(b.quantity for b in self._stock_batches),
            "budget_remaining": self._budget,
        }

    def _compute_phase_score(self) -> float:
        target = {"easy": 0.95, "medium": 0.85, "hard": 0.75}[self._current_phase()]
        max_cost = {"easy": 15000.0, "medium": 25000.0, "hard": 30000.0}[self._current_phase()]
        total_demand = max(sum(self._phase_demand), 1.0)
        total_fulfilled = sum(self._phase_fulfilled)
        service_score = min(total_fulfilled / total_demand / target, 1.0)
        cost_score = max(0.0, 1.0 - self._phase_total_cost / max_cost)
        validity_score = self._phase_valid_actions / max(self._phase_total_actions, 1)
        return round(0.5 * service_score + 0.3 * cost_score + 0.2 * validity_score, 4)

    def _build_observation(self, reward: float, metadata: dict = None) -> SupplyChainObservation:
        phase = self._current_phase()
        market_price = MARKET_PRICE[phase]
        current_stock = sum(b.quantity for b in self._stock_batches)
        crisis_active = (self._day - 1) in CRISIS_DAYS  # use previous day since day already incremented in step

        # Expiry info
        if self._stock_batches:
            nearest = min(self._stock_batches, key=lambda b: b.expires_on_day)
            days_until = nearest.expires_on_day - (self._day - 1 if self._day > 1 else 1)
            expiring_soon_qty = sum(
                b.quantity for b in self._stock_batches
                if b.expires_on_day - (self._day - 1 if self._day > 1 else 1) <= 3
            )
            if days_until <= 2:
                expiry_warning = f"URGENT: {expiring_soon_qty} units expire in {days_until} day(s)!"
            elif days_until <= 5:
                expiry_warning = f"Warning: {expiring_soon_qty} units expire in {days_until} days"
            else:
                expiry_warning = "No imminent expiry"
        else:
            days_until = 999
            expiring_soon_qty = 0
            expiry_warning = "No stock on hand"

        # For reset (day=1), crisis_active is False
        if self._day == 1:
            crisis_active = False

        # Pending summary for prompt
        if self._pending_orders:
            parts = [
                f"{po.quantity} units in {po.arrives_in_days} day{'s' if po.arrives_in_days != 1 else ''}"
                for po in self._pending_orders
            ]
            pending_summary = ", ".join(parts)
        else:
            pending_summary = "none"

        supplier_status = "normal"
        if self._loyalty_tier == "bronze" and self._rng.random() < 0.30:
            supplier_status = "delayed"

        forecast_noise_map = {"easy": "low", "medium": "medium", "hard": "high"}
        forecast_noise = forecast_noise_map[phase]

        base_demand = BASE_DEMAND[phase]
        if phase == "medium":
            seasonal = 1.0 + 0.3 * math.sin(math.pi * max(0, self._day - 8) / 7)
            demand_forecast = base_demand * seasonal
        else:
            demand_forecast = base_demand

        recent_neg_scores = self._neg_score_history[-3:]
        emergency_surcharge_rate = SURCHARGE[self._loyalty_tier]
        phase_score = self._compute_phase_score()

        prompt = (
            f"ADAPTIVE SUPPLY CHAIN — Day {self._day} of 30\n"
            "═══════════════════════════════════════════════════════════\n\n"
            "INVENTORY STATUS\n"
            f"  Current stock         : {current_stock} units across {len(self._stock_batches)} batches\n"
            f"  Nearest expiry        : {days_until} days ({expiring_soon_qty} units)\n"
            f"  {expiry_warning}\n\n"
            "MARKET CONDITIONS\n"
            f"  Market price today    : {market_price}/unit\n"
            f"  Your sell price       : {self._last_sell_price}/unit  (what you charged yesterday)\n"
            f"  Demand forecast       : ~{demand_forecast:.0f} units ({forecast_noise} uncertainty)\n"
            "  Price elasticity      : 1.5x shift per 10% price change vs market\n\n"
            "SUPPLIER RELATIONSHIP\n"
            f"  Trust score           : {self._trust_score:.2f} / 1.00\n"
            f"  Supplier's message    : \"{self._supplier_last_message}\"\n"
            f"  Last order accuracy   : {self._lead_time_accuracy}\n"
            f"  Emergency surcharge   : {emergency_surcharge_rate}x unit cost  (signals loyalty tier)\n"
            f"  Discount offered      : {self._proactive_discount}\n"
            f"  Recent negotiation scores: {recent_neg_scores}   (last 3 days, 0.0–1.0)\n\n"
            "SUPPLY CHAIN\n"
            f"  Pending orders        : {pending_summary}\n"
            f"  Supplier status       : {supplier_status}\n"
            f"  Crisis active         : {crisis_active}\n\n"
            "FINANCIALS\n"
            f"  Budget remaining      : {self._budget:.0f}\n"
            f"  Last 7-day service SL : {self._last_7_day_service_level:.0%}\n"
            f"  Unit cost (standard)  : {UNIT_COST_STANDARD}/unit | Emergency: {emergency_surcharge_rate * UNIT_COST_STANDARD:.1f}/unit\n"
            f"  Fixed order cost      : {FIXED_ORDER_COST} per order\n\n"
            "EPISODE\n"
            f"  Current phase         : {phase}\n"
            f"  Day                   : {self._day} of 30\n\n"
            "═══════════════════════════════════════════════════════════\n"
            "Make four decisions today:\n"
            "1. action_type: \"order\" | \"emergency_restock\" | \"hold\"\n"
            "2. quantity: <int> (required if ordering, null for hold)\n"
            "3. sell_price: <float> (price per unit — affects demand you receive)\n"
            "4. negotiation_message: <str> (professional message to your supplier)\n\n"
            "Respond with exactly one JSON:\n"
            "{\"action_type\": \"...\", \"quantity\": ..., \"sell_price\": ..., \"negotiation_message\": \"...\"}"
        )

        return SupplyChainObservation(
            day=self._day,
            current_stock=current_stock,
            demand_forecast=round(demand_forecast, 2),
            forecast_noise=forecast_noise,
            pending_orders=list(self._pending_orders),
            last_7_day_service_level=round(self._last_7_day_service_level, 4),
            holding_cost_per_unit=0.5,
            stockout_penalty=STOCKOUT_PENALTY,
            budget_remaining=round(self._budget, 2),
            supplier_status=supplier_status,
            current_phase=phase,
            prompt=prompt,
            # New inventory fields
            days_until_nearest_expiry=days_until,
            expiring_soon_qty=expiring_soon_qty,
            expiry_warning=expiry_warning,
            batch_count=len(self._stock_batches),
            units_spoiled_today=self._last_units_spoiled,
            # Market fields
            market_price=market_price,
            last_sell_price=self._last_sell_price,
            # Supplier relationship signals
            trust_score=round(self._trust_score, 4),
            supplier_last_message=self._supplier_last_message,
            lead_time_accuracy=self._lead_time_accuracy,
            emergency_surcharge_rate=emergency_surcharge_rate,
            proactive_discount_offered=self._proactive_discount,
            recent_neg_scores=recent_neg_scores,
            # Episode state
            crisis_active=crisis_active,
            # Live grading
            phase_score=phase_score,
            actual_demand=0.0,
            actual_fulfilled=0.0,
            reward=reward,
            done=self._done,
            metadata=metadata or {
                "episode_id": self._episode_id,
                "day": self._day,
                "phase": phase,
            },
        )
