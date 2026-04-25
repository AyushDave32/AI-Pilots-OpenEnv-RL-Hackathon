# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline inference script for the Adaptive Supply Chain RL Environment.

Uses the OpenAI-compatible API client. Configure via environment variables:

  HF_TOKEN       Your Hugging Face API key (required)
  API_BASE_URL   LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier (default: meta-llama/Llama-3.3-70B-Instruct)
  ENV_URL        Supply chain server URL (default: live HF Space)

Runs 3 fixed-difficulty episodes (easy → medium → hard) and emits structured
[START] / [STEP] / [END] logs compatible with the OpenEnv evaluator.

Reproducibility:
  - numpy seed: 42
  - episode seed: 0
  - LLM temperature: 0.0

Usage:
  HF_TOKEN=hf_... python inference.py
"""

import json
import os
import re
import sys
from typing import List, Optional

import numpy as np
from openai import OpenAI

from asc_agent_under_demand_uncertainity_rl_env import (
    AscAgentUnderDemandUncertainityRlEnv,
    SupplyChainAction,
)
from graders import PhaseHistory, grade_easy_phase, grade_hard_phase, grade_medium_phase

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)
EPISODE_SEED = 0

# ── API configuration ────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
if not API_KEY:
    sys.exit("Error: HF_TOKEN is not set.\nRun: export HF_TOKEN=hf_your_token_here")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

BENCHMARK = "asc_agent_under_demand_uncertainity_rl_env"
SUCCESS_SCORE_THRESHOLD = 0.5

llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

GRADERS = {
    "easy":   grade_easy_phase,
    "medium": grade_medium_phase,
    "hard":   grade_hard_phase,
}

TASK_IDS = {
    "easy":   "easy_phase_inventory",
    "medium": "medium_phase_inventory",
    "hard":   "hard_phase_inventory",
}


# ── Structured logging ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Action parsing ────────────────────────────────────────────────────────────

def parse_action(content: str) -> tuple[SupplyChainAction, bool]:
    """
    Extract a SupplyChainAction from the LLM's text response.

    Returns (action, is_valid). Falls back to hold on any parse failure.
    LLM is prompted to output a 4-field JSON:
      {"action_type": "order", "quantity": 100, "sell_price": 268.0,
       "negotiation_message": "..."}
    """
    try:
        # Match outermost JSON object (handles nested braces in negotiation_message)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return SupplyChainAction(action_type="hold"), False

        data = json.loads(match.group())
        action_type = data.get("action_type", data.get("action", "hold"))
        quantity    = data.get("quantity")
        sell_price  = float(data.get("sell_price", 265.0))
        neg_message = str(data.get("negotiation_message", ""))

        if action_type in ("order", "emergency_restock") and isinstance(quantity, (int, float)) and quantity > 0:
            return SupplyChainAction(
                action_type=action_type,
                quantity=int(quantity),
                sell_price=sell_price,
                negotiation_message=neg_message,
            ), True
        elif action_type == "hold":
            return SupplyChainAction(
                action_type="hold",
                sell_price=sell_price,
                negotiation_message=neg_message,
            ), True
        else:
            return SupplyChainAction(action_type="hold"), False

    except (json.JSONDecodeError, ValueError, KeyError):
        return SupplyChainAction(action_type="hold"), False


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env: AscAgentUnderDemandUncertainityRlEnv, difficulty: str) -> float:
    """
    Run a 30-step episode at the given difficulty.

    Emits [START], one [STEP] per day, then [END] with grader score.
    Returns the grader score in [0.0, 1.0].
    """
    task_id = TASK_IDS[difficulty]
    result = env.reset(task=task_id, seed=EPISODE_SEED)
    obs = result.observation

    demand_history: List[float] = []
    fulfilled_history: List[float] = []
    spoilage_history: List[float] = []
    revenue_history: List[float] = []
    total_cost: float = 0.0
    valid_actions: int = 0
    total_actions: int = 0
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    done = result.done

    try:
        step = 0
        while not done:
            step += 1
            error: Optional[str] = None

            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": obs.prompt}],
                    temperature=0.0,
                    max_tokens=400,   # large enough for 4-field JSON with negotiation message
                )
                llm_text = response.choices[0].message.content or ""
            except Exception as exc:
                llm_text = '{"action_type": "hold", "quantity": null, "sell_price": 265.0, "negotiation_message": ""}'
                error = str(exc)[:120]

            action, is_valid = parse_action(llm_text)
            total_actions += 1
            if is_valid:
                valid_actions += 1

            # Cost tracking using new Rs formula (Rs 2000 fixed + Rs 200/unit)
            if action.action_type == "order" and action.quantity:
                total_cost += 2000 + action.quantity * 200
            elif action.action_type == "emergency_restock" and action.quantity:
                # Emergency surcharge depends on loyalty tier — use 4x (bronze worst case) for tracking
                total_cost += 2000 + action.quantity * 200 * 4

            step_result = env.step(action)
            next_obs = step_result.observation

            # Read actuals from observation metadata (set by environment step)
            meta           = next_obs.metadata or {}
            actual_demand    = float(meta.get("actual_demand",    next_obs.demand_forecast))
            actual_fulfilled = float(meta.get("units_fulfilled",  0.0))
            units_spoiled    = float(meta.get("units_spoiled",    0.0))
            gross_profit     = float(meta.get("reward_components", {}).get("gross_profit", 0.0))

            demand_history.append(actual_demand)
            fulfilled_history.append(actual_fulfilled)
            spoilage_history.append(units_spoiled)
            revenue_history.append(gross_profit)
            rewards.append(step_result.reward)
            steps_taken = step

            action_str = json.dumps(
                {
                    "action_type": action.action_type,
                    "quantity": action.quantity,
                    "sell_price": action.sell_price,
                    "negotiation_message": action.negotiation_message[:60] + "..."
                    if len(action.negotiation_message) > 60
                    else action.negotiation_message,
                },
                separators=(",", ":"),
            )
            log_step(step=step, action=action_str, reward=step_result.reward, done=step_result.done, error=error)

            obs  = next_obs
            done = step_result.done

        history = PhaseHistory(
            phase=difficulty,
            demand_history=demand_history,
            fulfilled_history=fulfilled_history,
            spoilage_history=spoilage_history,
            revenue_history=revenue_history,
            total_cost=total_cost,
            valid_actions=valid_actions,
            total_actions=total_actions,
        )
        score = GRADERS[difficulty](history)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    base_url = os.environ.get(
        "ENV_URL",
        "https://ayush-dave-asc-agent-under-demand-uncertainity-rl-env.hf.space",
    )
    async_env = AscAgentUnderDemandUncertainityRlEnv(base_url=base_url)
    scores: dict[str, float] = {}

    with async_env.sync() as env:
        for difficulty in ("easy", "medium", "hard"):
            scores[difficulty] = run_episode(env, difficulty)

    overall = sum(scores.values()) / len(scores)
    print(f"\nOverall score: {overall:.4f}", flush=True)


if __name__ == "__main__":
    main()
