# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline inference script for the Adaptive Supply Chain RL Environment.

Uses a Hugging Face model via the OpenAI-compatible Inference API.
Defaults to meta-llama/Meta-Llama-3.1-8B-Instruct (override with HF_MODEL env var).

Runs 3 fixed-difficulty episodes (easy → medium → hard) and prints grader scores.

Reproducibility:
  - numpy seed: 42
  - episode seed: 0
  - LLM temperature: 0.0

Prerequisites:
  1. Server running at http://localhost:8000:
       uvicorn server.app:app --host 0.0.0.0 --port 8000

  2. Hugging Face token (free at https://huggingface.co/settings/tokens):
       export HF_TOKEN=hf_your_token_here

  3. (Optional) Override the model:
       export HF_MODEL=Qwen/Qwen2.5-7B-Instruct

Usage:
  HF_TOKEN=hf_... python inference.py
"""

import json
import os
import re
import sys

import numpy as np
import openai

from asc_agent_under_demand_uncertainity_rl_env import (
    AscAgentUnderDemandUncertainityRlEnv,
    SupplyChainAction,
)
from graders import PhaseHistory, grade_easy_phase, grade_hard_phase, grade_medium_phase

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)
EPISODE_SEED = 0

# ── Hugging Face client (OpenAI-compatible Inference API) ────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    sys.exit(
        "Error: HF_TOKEN is not set.\n"
        "Get a free token at https://huggingface.co/settings/tokens and run:\n"
        "    export HF_TOKEN=hf_your_token_here"
    )

# Default: Meta-Llama-3.1-8B-Instruct (free serverless tier on HF)
# Override via:  export HF_MODEL=Qwen/Qwen2.5-7B-Instruct
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

llm = openai.OpenAI(
    api_key=HF_TOKEN,
    base_url="https://api-inference.huggingface.co/v1/",
)

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


# ── Action parsing ────────────────────────────────────────────────────────────

def parse_action(content: str) -> tuple[SupplyChainAction, bool]:
    """
    Extract a SupplyChainAction from the LLM's text response.

    Returns (action, is_valid). Falls back to hold on any parse failure.

    LLM is prompted to output {"action": "order", "quantity": 100};
    we remap the "action" key to "action_type" for our Pydantic model.
    """
    try:
        match = re.search(r"\{[^{}]+\}", content, re.DOTALL)
        if not match:
            return SupplyChainAction(action_type="hold"), False

        data = json.loads(match.group())
        action_type = data.get("action", data.get("action_type", "hold"))
        quantity = data.get("quantity")

        if action_type in ("order", "emergency_restock") and isinstance(quantity, (int, float)) and quantity > 0:
            return SupplyChainAction(action_type=action_type, quantity=int(quantity)), True
        elif action_type == "hold":
            return SupplyChainAction(action_type="hold"), True
        else:
            return SupplyChainAction(action_type="hold"), False

    except (json.JSONDecodeError, ValueError, KeyError):
        return SupplyChainAction(action_type="hold"), False


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env: AscAgentUnderDemandUncertainityRlEnv, difficulty: str) -> PhaseHistory:
    """
    Run a 30-step episode at the given difficulty and return a PhaseHistory.

    Uses actual_demand / actual_fulfilled from observation metadata for
    accurate (non-forecast-proxy) grader computation.
    """
    task_id = TASK_IDS[difficulty]
    result = env.reset(task=task_id, seed=EPISODE_SEED)
    obs = result.observation

    demand_history: list[float] = []
    fulfilled_history: list[float] = []
    total_cost: float = 0.0
    valid_actions: int = 0
    total_actions: int = 0
    episode_reward: float = 0.0

    print(f"\n{'─' * 70}")
    print(
        f"  Phase : {difficulty.upper():6s}  |  Model: {HF_MODEL}\n"
        f"  Stock : {obs.current_stock}   |  Budget: ${obs.budget_remaining:.0f}"
    )
    print(f"{'─' * 70}")
    print(f"  {'Day':>3}  {'Stock':>5}  {'Action':<22}  {'Qty':>5}  {'Reward':>8}  {'SL':>6}  {'Score':>6}")
    print(f"  {'─'*3}  {'─'*5}  {'─'*22}  {'─'*5}  {'─'*8}  {'─'*6}  {'─'*6}")

    done = result.done

    while not done:
        # Query HF model
        try:
            response = llm.chat.completions.create(
                model=HF_MODEL,
                messages=[{"role": "user", "content": obs.prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            llm_text = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM error day {obs.day}]: {exc} — defaulting to hold")
            llm_text = '{"action": "hold"}'

        action, is_valid = parse_action(llm_text)
        total_actions += 1
        if is_valid:
            valid_actions += 1

        # Accumulate ordering cost for grader
        if action.action_type == "order" and action.quantity:
            total_cost += 20 + action.quantity * 2
        elif action.action_type == "emergency_restock" and action.quantity:
            total_cost += 20 + action.quantity * 6

        step_result = env.step(action)
        next_obs = step_result.observation

        # Accurate demand/fulfilled from metadata (not forecast proxies)
        meta = next_obs.metadata or {}
        actual_demand    = meta.get("actual_demand",    obs.demand_forecast)
        actual_fulfilled = meta.get("actual_fulfilled", 0.0)
        phase_score      = meta.get("phase_score",      0.0)

        demand_history.append(actual_demand)
        fulfilled_history.append(actual_fulfilled)
        episode_reward += step_result.reward

        qty_str = str(action.quantity) if action.quantity is not None else "—"
        print(
            f"  {obs.day:3d}  {obs.current_stock:5d}  {action.action_type:<22s}  "
            f"{qty_str:>5}  {step_result.reward:+8.1f}  "
            f"{next_obs.last_7_day_service_level:6.0%}  {phase_score:6.3f}"
        )

        obs  = next_obs
        done = step_result.done

    print(f"{'─' * 70}")
    print(f"  Episode total reward: {episode_reward:+.1f}  |  Valid actions: {valid_actions}/{total_actions}")

    return PhaseHistory(
        phase=difficulty,
        demand_history=demand_history,
        fulfilled_history=fulfilled_history,
        total_cost=total_cost,
        valid_actions=valid_actions,
        total_actions=total_actions,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 70)
    print(f"  ADAPTIVE SUPPLY CHAIN RL — BASELINE INFERENCE")
    print(f"  Model : {HF_MODEL}")
    print("═" * 70)

    env = AscAgentUnderDemandUncertainityRlEnv(base_url="http://localhost:8000")
    scores: dict[str, float] = {}

    with env:
        for difficulty in ("easy", "medium", "hard"):
            history = run_episode(env, difficulty)
            score   = GRADERS[difficulty](history)
            scores[difficulty] = score
            print(f"  → Grader score ({difficulty}): {score:.4f}\n")

    print("═" * 70)
    print("  FINAL GRADER SCORES")
    print("═" * 70)
    for phase, score in scores.items():
        bar = "█" * int(score * 30)
        print(f"  {phase:<8s}: {score:.4f}  {bar}")
    overall = sum(scores.values()) / len(scores)
    print(f"{'─' * 70}")
    print(f"  Overall  : {overall:.4f}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
