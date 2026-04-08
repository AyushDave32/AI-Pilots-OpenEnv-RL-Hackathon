# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Adaptive Supply Chain RL Environment — Curriculum Wrapper (Approach C).

A single environment that runs a 30-day episode through 3 difficulty phases
(easy → medium → hard). Difficulty auto-promotes when the agent's rolling
7-day service level exceeds 90% for 7+ consecutive days in a phase.

Phase specs:
  Easy   — stable ~80/day demand, fixed 3-day lead time, ±10% forecast noise
  Medium — seasonal wave (peak day 15), 2–5 day random lead time, ±25% noise
  Hard   — random spikes + baseline, 2–10 day lead time + delays, ±40% noise

Reward (per step):
  + fulfilled_units × 3.0          (sell at $3/unit — $1 margin over $2 unit cost)
  − 50.0  if stockout occurred
  − 0.5 × max(0, stock − 200)      (overstock penalty)
  − (20 + qty × 2) for order
  − (20 + qty × 6) for emergency_restock
  + 5.0   if stock in [50, 200]    (efficiency bonus)
  − 10.0  if action was malformed
"""

from uuid import uuid4

import numpy as np
from pydantic import ValidationError

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PendingOrder, SupplyChainAction, SupplyChainObservation
except (ImportError, ModuleNotFoundError):
    from models import PendingOrder, SupplyChainAction, SupplyChainObservation


class AscAgentUnderDemandUncertainityRlEnvironment(Environment):
    """
    Adaptive Supply Chain RL Environment with curriculum difficulty progression.

    The agent acts as a warehouse manager placing daily inventory orders over a
    30-day episode. Demand is uncertain, lead times vary, and performance is
    measured by service level (fulfilled / total demand).

    Episode flow:
        reset() → step() × 30 → done=True

    Phases promote when service_level > 90% for 7+ consecutive days:
        easy → medium → hard
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ------------------------------------------------------------------ #
    # Cost / penalty / reward constants (fixed across all phases)          #
    # ------------------------------------------------------------------ #
    FULFILLMENT_REWARD_PER_UNIT = 3.0   # sells at $3/unit, buys at $2/unit → $1 margin incentivises ordering
    ORDER_FIXED_COST = 20.0
    ORDER_UNIT_COST = 2.0
    EMERGENCY_FIXED_COST = 20.0
    EMERGENCY_UNIT_COST = 6.0
    HOLDING_COST_PER_UNIT = 0.5
    STOCKOUT_PENALTY = 50.0
    OVERSTOCK_PENALTY_PER_UNIT = 0.5
    OVERSTOCK_THRESHOLD = 200
    EFFICIENCY_BONUS = 5.0
    EFFICIENCY_MIN = 50
    EFFICIENCY_MAX = 200
    MALFORMED_PENALTY = 10.0

    # Phase grading constants (mirrored from graders.py for live metadata)
    _PHASE_TARGET_SL = {"easy": 0.95, "medium": 0.85, "hard": 0.75}
    _PHASE_MAX_COST  = {"easy": 800.0, "medium": 1200.0, "hard": 1500.0}

    def __init__(self):
        self._episode_id: str = str(uuid4())
        self._day: int = 1
        self._done: bool = False
        self._rng: np.random.Generator = np.random.default_rng()

        # Inventory state
        self._stock: int = 200
        self._budget: float = 1000.0
        self._pending_orders: list[PendingOrder] = []

        # Episode-level history (full 30 days)
        self._demand_history: list[float] = []
        self._fulfilled_history: list[float] = []

        # Curriculum state
        self._phase: str = "easy"
        self._days_in_phase: int = 0
        self._last_7_day_service_level: float = 1.0
        self._supplier_status: str = "normal"

        # Phase-level tracking for live grader score in metadata
        self._phase_demand: list[float] = []
        self._phase_fulfilled: list[float] = []
        self._phase_total_cost: float = 0.0
        self._phase_valid_actions: int = 0
        self._phase_total_actions: int = 0

        # Last step actuals for metadata exposure
        self._last_actual_demand: float = 0.0
        self._last_actual_fulfilled: float = 0.0

    # ------------------------------------------------------------------ #
    # Environment interface                                                 #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, task: str | None = None, **kwargs) -> SupplyChainObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Optional RNG seed for reproducibility.
            task: Optional starting phase. Accepted:
                  "easy" | "easy_phase_inventory"
                  "medium" | "medium_phase_inventory"
                  "hard"  | "hard_phase_inventory"
                  Defaults to "easy" (full curriculum).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        self._episode_id = str(uuid4())
        self._day = 1
        self._done = False

        # Randomize starting stock: uniform [150, 300]
        self._stock = int(self._rng.integers(150, 301))
        self._budget = 1000.0
        self._pending_orders = []
        self._demand_history = []
        self._fulfilled_history = []
        self._last_7_day_service_level = 1.0
        self._supplier_status = "normal"
        self._last_actual_demand = 0.0
        self._last_actual_fulfilled = 0.0

        # Starting phase
        if task in ("medium", "medium_phase_inventory"):
            self._phase = "medium"
        elif task in ("hard", "hard_phase_inventory"):
            self._phase = "hard"
        else:
            self._phase = "easy"

        self._days_in_phase = 0
        self._reset_phase_trackers()

        return self._make_observation(reward=0.0)

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """
        Execute one day in the supply chain simulation.

        Steps:
          1.  Validate action  (−10 for malformed, treat as hold)
          2.  Process pending order arrivals
          3.  Place order / emergency restock; deduct cost
          4.  Sample actual demand (phase-dependent noise)
          5.  Fulfill demand: fulfilled = min(stock, demand)
          6.  Update stock; record stockout
          7.  Compute multi-component step reward
          8.  Append to demand/fulfilled history (episode + phase)
          9.  Recompute last_7_day_service_level
          10. Check phase promotion (days_in_phase ≥ 7 AND sl > 0.90)
          11. Increment day; done = (day > 30)
          12. Return SupplyChainObservation
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # 1. Validate action
        action_malformed = False
        try:
            if action.action_type in ("order", "emergency_restock"):
                if action.quantity is None or action.quantity <= 0:
                    action_malformed = True
                    action = SupplyChainAction(action_type="hold")
        except (ValidationError, Exception):
            action_malformed = True
            action = SupplyChainAction(action_type="hold")

        self._phase_total_actions += 1
        if not action_malformed:
            self._phase_valid_actions += 1

        reward = 0.0
        if action_malformed:
            reward -= self.MALFORMED_PENALTY

        # 2. Process pending order arrivals
        arrived = 0
        remaining_orders: list[PendingOrder] = []
        for po in self._pending_orders:
            new_eta = po.arrives_in_days - 1
            if new_eta <= 0:
                arrived += po.quantity
            else:
                remaining_orders.append(PendingOrder(quantity=po.quantity, arrives_in_days=new_eta))
        self._stock += arrived
        self._pending_orders = remaining_orders

        # 3. Place order / emergency restock
        order_cost = 0.0
        if action.action_type == "order" and not action_malformed:
            order_cost = self.ORDER_FIXED_COST + action.quantity * self.ORDER_UNIT_COST
            self._budget -= order_cost
            reward -= order_cost
            self._phase_total_cost += order_cost
            lead_time = self._sample_lead_time()
            self._pending_orders.append(
                PendingOrder(quantity=action.quantity, arrives_in_days=lead_time)
            )
        elif action.action_type == "emergency_restock" and not action_malformed:
            order_cost = self.EMERGENCY_FIXED_COST + action.quantity * self.EMERGENCY_UNIT_COST
            self._budget -= order_cost
            reward -= order_cost
            self._phase_total_cost += order_cost
            self._stock += action.quantity  # arrives immediately

        # 4. Sample actual demand
        demand = self._sample_demand()

        # 5. Fulfill demand
        fulfilled = float(min(self._stock, demand))

        # 6. Update stock; track stockout
        stockout_occurred = demand > self._stock
        self._stock = max(0, self._stock - int(np.ceil(demand)))

        # 7. Compute reward components
        # Core signal: reward fulfilling demand (prevents hold-only degenerate policy)
        reward += fulfilled * self.FULFILLMENT_REWARD_PER_UNIT

        if stockout_occurred:
            reward -= self.STOCKOUT_PENALTY

        excess = max(0, self._stock - self.OVERSTOCK_THRESHOLD)
        reward -= self.OVERSTOCK_PENALTY_PER_UNIT * excess

        if self.EFFICIENCY_MIN <= self._stock <= self.EFFICIENCY_MAX:
            reward += self.EFFICIENCY_BONUS

        # 8. Append history
        self._demand_history.append(demand)
        self._fulfilled_history.append(fulfilled)
        self._phase_demand.append(demand)
        self._phase_fulfilled.append(fulfilled)
        self._last_actual_demand = demand
        self._last_actual_fulfilled = fulfilled

        # 9. Recompute 7-day service level
        last7_demand = sum(self._demand_history[-7:])
        last7_fulfilled = sum(self._fulfilled_history[-7:])
        self._last_7_day_service_level = (
            last7_fulfilled / last7_demand if last7_demand > 0 else 1.0
        )

        # 10. Phase promotion (days_in_phase increments after history update)
        self._days_in_phase += 1
        if self._days_in_phase >= 7 and self._last_7_day_service_level > 0.90:
            if self._phase == "easy":
                self._phase = "medium"
                self._days_in_phase = 0
                self._reset_phase_trackers()
            elif self._phase == "medium":
                self._phase = "hard"
                self._days_in_phase = 0
                self._reset_phase_trackers()
            # hard → no further promotion

        # Update supplier status
        self._supplier_status = self._sample_supplier_status()

        # 11. Increment day; check done
        self._day += 1
        self._done = self._day > 30

        return self._make_observation(reward=reward)

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self._day)

    # ------------------------------------------------------------------ #
    # Phase tracking helpers                                               #
    # ------------------------------------------------------------------ #

    def _reset_phase_trackers(self) -> None:
        """Reset per-phase grading accumulators on phase transition."""
        self._phase_demand = []
        self._phase_fulfilled = []
        self._phase_total_cost = 0.0
        self._phase_valid_actions = 0
        self._phase_total_actions = 0

    def _compute_phase_score(self) -> float:
        """Compute live grader score for the current phase (0.0–1.0)."""
        target = self._PHASE_TARGET_SL[self._phase]
        max_cost = self._PHASE_MAX_COST[self._phase]

        total_demand = sum(self._phase_demand)
        total_fulfilled = sum(self._phase_fulfilled)

        avg_sl = total_fulfilled / max(total_demand, 1.0)
        service_score = min(avg_sl / target, 1.0)
        cost_score = max(0.0, 1.0 - (self._phase_total_cost / max_cost))
        validity_score = self._phase_valid_actions / max(self._phase_total_actions, 1)

        return round(0.5 * service_score + 0.3 * cost_score + 0.2 * validity_score, 4)

    # ------------------------------------------------------------------ #
    # Phase-dependent simulation helpers                                   #
    # ------------------------------------------------------------------ #

    def _sample_demand(self) -> float:
        """Sample actual demand based on current phase."""
        if self._phase == "easy":
            base = 80.0
            noise = self._rng.uniform(-base * 0.10, base * 0.10)
            return max(0.0, base + noise)

        elif self._phase == "medium":
            # Seasonal wave peaking around day 15 of the phase
            peak_offset = np.pi * self._days_in_phase / 14.0
            base = 80.0 + 40.0 * np.sin(peak_offset)
            noise = self._rng.uniform(-base * 0.25, base * 0.25)
            return max(0.0, base + noise)

        else:  # hard
            base = 80.0
            if self._rng.random() < 0.20:  # random spike ~20% of days
                base += float(self._rng.uniform(100.0, 200.0))
            noise = self._rng.uniform(-base * 0.40, base * 0.40)
            return max(0.0, base + noise)

    def _sample_lead_time(self) -> int:
        """Sample supplier lead time based on current phase."""
        if self._phase == "easy":
            return 3
        elif self._phase == "medium":
            return int(self._rng.integers(2, 6))  # 2–5 days
        else:  # hard
            base = int(self._rng.integers(2, 11))  # 2–10 days
            if self._rng.random() < 0.20:
                base += int(self._rng.integers(1, 5))  # additional random delay
            return base

    def _sample_supplier_status(self) -> str:
        if self._phase == "hard" and self._rng.random() < 0.30:
            return "delayed"
        return "normal"

    def _compute_demand_forecast(self) -> float:
        """Return the forecast mean for the current day (no noise applied)."""
        if self._phase == "easy":
            return 80.0
        elif self._phase == "medium":
            peak_offset = np.pi * self._days_in_phase / 14.0
            return 80.0 + 40.0 * np.sin(peak_offset)
        else:
            return 80.0  # hard: baseline only (spikes are unforecastable)

    # ------------------------------------------------------------------ #
    # Observation construction                                             #
    # ------------------------------------------------------------------ #

    def _make_observation(self, reward: float) -> SupplyChainObservation:
        noise_map = {"easy": "low", "medium": "medium", "hard": "high"}
        forecast = self._compute_demand_forecast()
        forecast_noise = noise_map[self._phase]

        if self._pending_orders:
            parts = [
                f"{po.quantity} units in {po.arrives_in_days} day{'s' if po.arrives_in_days != 1 else ''}"
                for po in self._pending_orders
            ]
            pending_summary = ", ".join(parts)
        else:
            pending_summary = "none"

        prompt = (
            "You are a warehouse manager. Here is today's situation:\n"
            f"- Day: {self._day} of 30\n"
            f"- Current stock: {self._stock} units\n"
            f"- Demand forecast: ~{forecast:.0f} units today ({forecast_noise} uncertainty)\n"
            f"- Orders arriving soon: {pending_summary}\n"
            f"- Last 7-day service level: {self._last_7_day_service_level:.0%}\n"
            f"- Budget remaining: ${self._budget:.0f}\n"
            f"- Supplier status: {self._supplier_status}\n"
            f"- Current difficulty: {self._phase}\n\n"
            "Respond with exactly one JSON action:\n"
            '{"action": "order", "quantity": <int>}\n'
            '{"action": "emergency_restock", "quantity": <int>}\n'
            '{"action": "hold"}'
        )

        phase_score = self._compute_phase_score()

        return SupplyChainObservation(
            day=self._day,
            current_stock=self._stock,
            demand_forecast=forecast,
            forecast_noise=forecast_noise,
            pending_orders=list(self._pending_orders),
            last_7_day_service_level=self._last_7_day_service_level,
            holding_cost_per_unit=self.HOLDING_COST_PER_UNIT,
            stockout_penalty=self.STOCKOUT_PENALTY,
            budget_remaining=self._budget,
            supplier_status=self._supplier_status,
            current_phase=self._phase,
            prompt=prompt,
            phase_score=phase_score,
            actual_demand=round(self._last_actual_demand, 2),
            actual_fulfilled=round(self._last_actual_fulfilled, 2),
            reward=reward,
            done=self._done,
            metadata={
                "episode_id": self._episode_id,
                "day": self._day,
                "phase": self._phase,
                "days_in_phase": self._days_in_phase,
                "service_level_7d": round(self._last_7_day_service_level, 4),
                "phase_score": phase_score,          # live grader score 0.0–1.0
                "actual_demand": round(self._last_actual_demand, 2),
                "actual_fulfilled": round(self._last_actual_fulfilled, 2),
                "phase_total_cost": round(self._phase_total_cost, 2),
                "phase_valid_actions": self._phase_valid_actions,
                "phase_total_actions": self._phase_total_actions,
            },
        )
