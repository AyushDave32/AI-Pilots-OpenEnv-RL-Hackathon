# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptive Supply Chain RL Environment — Python client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PendingOrder, SupplyChainAction, SupplyChainObservation


class AscAgentUnderDemandUncertainityRlEnv(
    EnvClient[SupplyChainAction, SupplyChainObservation, State]
):
    """
    Client for the Adaptive Supply Chain RL Environment.

    Maintains a persistent WebSocket connection to the server for low-latency
    multi-step interactions.

    Example:
        >>> with AscAgentUnderDemandUncertainityRlEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.current_stock)
        ...     result = env.step(SupplyChainAction(action_type="order", quantity=100))
        ...     print(result.observation.reward)

    Example with Docker:
        >>> env = AscAgentUnderDemandUncertainityRlEnv.from_docker_image("asc-supply-chain:latest")
        >>> try:
        ...     result = env.reset()
        ...     result = env.step(SupplyChainAction(action_type="hold"))
        ... finally:
        ...     env.close()
    """

    def _step_payload(self, action: SupplyChainAction) -> Dict:
        """Convert SupplyChainAction to JSON payload."""
        return {
            "action_type": action.action_type,
            "quantity": action.quantity,
            "sell_price": action.sell_price,
            "negotiation_message": action.negotiation_message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SupplyChainObservation]:
        """Parse server response into StepResult[SupplyChainObservation]."""
        obs_data = payload.get("observation", {})

        # done and reward may be at top level or nested in obs_data
        done = payload.get("done", obs_data.get("done", False))
        reward = payload.get("reward", obs_data.get("reward", 0.0))

        # Reconstruct pending_orders as PendingOrder objects
        raw_orders = obs_data.get("pending_orders", [])
        pending_orders = [
            PendingOrder(
                quantity=po["quantity"],
                arrives_in_days=po["arrives_in_days"],
            )
            if isinstance(po, dict)
            else po
            for po in raw_orders
        ]

        observation = SupplyChainObservation(
            # Existing fields
            day=obs_data.get("day", 1),
            current_stock=obs_data.get("current_stock", 0),
            demand_forecast=obs_data.get("demand_forecast", 0.0),
            forecast_noise=obs_data.get("forecast_noise", "low"),
            pending_orders=pending_orders,
            last_7_day_service_level=obs_data.get("last_7_day_service_level", 1.0),
            holding_cost_per_unit=obs_data.get("holding_cost_per_unit", 0.5),
            stockout_penalty=obs_data.get("stockout_penalty", 50.0),
            budget_remaining=obs_data.get("budget_remaining", 0.0),
            supplier_status=obs_data.get("supplier_status", "normal"),
            current_phase=obs_data.get("current_phase", "easy"),
            prompt=obs_data.get("prompt", ""),
            # New inventory fields
            days_until_nearest_expiry=obs_data.get("days_until_nearest_expiry", 999),
            expiring_soon_qty=obs_data.get("expiring_soon_qty", 0),
            expiry_warning=obs_data.get("expiry_warning", "No stock on hand"),
            batch_count=obs_data.get("batch_count", 0),
            units_spoiled_today=obs_data.get("units_spoiled_today", 0),
            # Market fields
            market_price=obs_data.get("market_price", 265.0),
            last_sell_price=obs_data.get("last_sell_price", 265.0),
            # Supplier relationship signals
            trust_score=obs_data.get("trust_score", 0.8),
            supplier_last_message=obs_data.get("supplier_last_message", ""),
            lead_time_accuracy=obs_data.get("lead_time_accuracy", "on time"),
            emergency_surcharge_rate=obs_data.get("emergency_surcharge_rate", 2.5),
            proactive_discount_offered=obs_data.get("proactive_discount_offered", False),
            recent_neg_scores=obs_data.get("recent_neg_scores", []),
            # Episode state
            crisis_active=obs_data.get("crisis_active", False),
            # Live grading
            phase_score=obs_data.get("phase_score", 0.0),
            actual_demand=obs_data.get("actual_demand", 0.0),
            actual_fulfilled=obs_data.get("actual_fulfilled", 0.0),
            done=done,
            reward=reward,
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
