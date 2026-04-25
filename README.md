---
title: Adaptive Supply Chain Agent Under Demand Uncertainty RL Environment
emoji: 🏭
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - supply-chain
  - curriculum-learning
  - multi-agent
  - world-modeling
---

# Adaptive Supply Chain RL Environment

An **OpenEnv-compliant RL environment** for the Meta / PyTorch / Hugging Face OpenEnv Hackathon.

The agent plays a **warehouse manager** making four simultaneous daily decisions over a **30-day episode**: how much stock to order, what price to charge customers, how to manage perishable inventory before it expires, and what to write in a supplier negotiation message — all under uncertain demand, variable lead times, and a mid-episode supplier crisis.

**Themes covered:** Multi-Agent (reactive supplier with hidden state) · World Modeling (partially observable supplier tier) · Long-Horizon Planning (30-day episode with crisis preparation) · Adaptive Difficulty (curriculum + supplier raises expectations as performance improves)

---

## The Problem

> *"You manage a warehouse. Every day you decide how much stock to order, what price to sell at, and how to maintain your supplier relationship. Stock expires after 15 days. Your supplier has hidden loyalty tiers that determine your costs and crisis allocation. On day 21 a supply disruption cuts factory output to 30% — only customers with strong supplier relationships get priority."*

The agent must balance four simultaneous pressures:

- **Perishable inventory** — stock too much and it expires (penalty per unit); stock too little and demand goes unfulfilled (stockout penalty)
- **Sell-side pricing** — setting price above market reduces demand; below market erodes margin via price elasticity
- **Supplier relationship** — the supplier has a hidden loyalty tier (Bronze/Silver/Gold) that determines lead times, emergency surcharge rates, and crisis allocation; negotiation messages are the primary lever to improve it
- **Crisis survival** — days 21–25: supplier capacity drops to 30%; Gold-tier customers get 100% of their order, Bronze-tier get 50%

---

## Curriculum Phases

Three phases advance by **day number**, giving the agent predictable windows to plan ahead:

```
Day 1–7   → EASY   (stable demand, cooperative supplier, fixed lead time)
Day 8–15  → MEDIUM (seasonal demand, neutral supplier, variable lead time)
Day 16–30 → HARD   (volatile demand + spikes, adversarial supplier, long lead times)
Day 21–25 → CRISIS (supply disruption — loyalty tier determines allocation fraction)
```

| Phase  | Demand Pattern               | Lead Time          | Forecast Noise | Service Target |
|--------|------------------------------|--------------------|----------------|----------------|
| Easy   | Stable ~80 units/day         | Fixed 3 days       | ±10%           | 95%            |
| Medium | Seasonal wave (peak day 15)  | 2–5 days random    | ±25%           | 85%            |
| Hard   | Baseline + random spikes     | 2–10 days + delays | ±40%           | 75%            |

Each phase is also exposed as an **independent task** for direct evaluation:
- `easy_phase_inventory`
- `medium_phase_inventory`
- `hard_phase_inventory`

---

## Supplier Hidden State

The supplier maintains hidden internal state the agent **never observes directly**. The agent must infer its relationship status from observable signals and use negotiation messages to improve it — this is theory-of-mind reasoning in a business context.

**Hidden variables (never shown to agent):**

| Variable | Description |
|---|---|
| `loyalty_tier` | `"bronze"` / `"silver"` / `"gold"` — drives all cost and allocation effects |
| `order_regularity` | Rolling consistency score; improves with standard orders, drops with emergencies/holds |
| `trust_score` | Driven by negotiation quality; improves with high rubric scores each day |

**Observable signals (agent infers tier from these):**

| Signal | What it reveals |
|---|---|
| `emergency_surcharge_rate` | 2.5× = Gold, 3.0× = Silver, 4.0× = Bronze |
| `supplier_last_message` | Warm/cooperative tone = Gold; cold/delayed = Bronze |
| `lead_time_accuracy` | Consistent delays indicate Bronze tier |
| `recent_neg_scores` | Last 3 negotiation scores (0.0–1.0) |
| `proactive_discount_offered` | Only offered to Gold-tier customers |

**Tier effects:**

| Effect | Bronze | Silver | Gold |
|--------|--------|--------|------|
| Lead time modifier | +1–2 extra days | As promised | −1 day faster |
| Emergency surcharge | 4× unit cost | 3× unit cost | 2.5× unit cost |
| Crisis allocation (days 21–25) | 50% of order | 80% of order | 100% of order |

---

## Action Space

**`SupplyChainAction`** — four fields required every day:

| Field | Type | Description |
|---|---|---|
| `action_type` | `"order"` \| `"emergency_restock"` \| `"hold"` | Buy-side decision |
| `quantity` | `int` \| `null` | Units to order (required for order/emergency) |
| `sell_price` | `float` | Price per unit charged to customers — affects demand via price elasticity |
| `negotiation_message` | `str` | Natural-language message to the supplier — scored by rubric, drives loyalty tier |

```json
{
  "action_type": "order",
  "quantity": 120,
  "sell_price": 268.0,
  "negotiation_message": "As a consistent partner with zero payment defaults, we request 120 units and offer advance payment to secure this order."
}
```

Missing or malformed fields → `−10` reward penalty.

---

## Observation Space

**`SupplyChainObservation`** — delivered as a formatted natural-language prompt plus structured fields:

| Field | Type | Description |
|---|---|---|
| `day` | `int` | Current day (1–30) |
| `current_stock` | `int` | Total units across all live batches |
| `batch_count` | `int` | Number of distinct stock batches |
| `days_until_nearest_expiry` | `int` | Days until earliest batch expires (999 if no stock) |
| `expiring_soon_qty` | `int` | Units expiring within 3 days |
| `expiry_warning` | `str` | Human-readable urgency alert |
| `units_spoiled_today` | `int` | Units that expired this step |
| `demand_forecast` | `float` | Forecasted demand for today |
| `forecast_noise` | `"low"` \| `"medium"` \| `"high"` | Forecast uncertainty level |
| `market_price` | `float` | Today's reference market price |
| `last_sell_price` | `float` | What the agent charged yesterday |
| `pending_orders` | `List[PendingOrder]` | Orders placed but not yet arrived |
| `last_7_day_service_level` | `float` | Fraction of demand fulfilled over last 7 days |
| `budget_remaining` | `float` | Remaining budget |
| `supplier_status` | `"normal"` \| `"delayed"` | Current supplier reliability |
| `trust_score` | `float` | 0.0–1.0 visible trust signal |
| `supplier_last_message` | `str` | Supplier's 1-sentence response (tone signals loyalty tier) |
| `lead_time_accuracy` | `str` | `"on time"` / `"1 day late"` / `"N days late"` |
| `emergency_surcharge_rate` | `float` | 2.5 / 3.0 / 4.0 — directly signals loyalty tier |
| `proactive_discount_offered` | `bool` | True only for Gold-tier customers |
| `recent_neg_scores` | `List[float]` | Last 3 negotiation scores (0.0–1.0) |
| `crisis_active` | `bool` | True on days 21–25 |
| `current_phase` | `"easy"` \| `"medium"` \| `"hard"` | Active difficulty phase |
| `prompt` | `str` | Formatted natural-language prompt for LLM agents |

---

## Reward Function

Reward is **dense** — the agent receives a signal every step:

```
reward  = (sell_price − unit_cost) × units_fulfilled   # sell-side gross profit
reward -= 50.0                                          # if stockout occurred
reward -= 20.0 × units_spoiled                          # spoilage penalty per expired unit
reward -= 0.5 × max(0, stock − 300)                    # overstock penalty
reward -= (fixed_cost + quantity × unit_cost)           # if action = order
reward -= (fixed_cost + quantity × unit_cost × surcharge_rate)  # if action = emergency_restock
                                                        # surcharge: 4× Bronze, 3× Silver, 2.5× Gold
reward += 5.0                                           # if stock in [50, 300] (efficiency bonus)
reward += negotiation_bonus                             # up to 10/step, capped at 30/episode
reward -= 10.0                                          # if action was malformed
reward -= 100.0                                         # if budget goes negative (signal only)
```

**Optimal stock range: 50–300 units.** Below 50 risks stockouts; above 300 triggers overstock penalty.

---

## Negotiation Rubric

Every `negotiation_message` is scored by an **LLM-as-Judge** (same model as inference) on 3 criteria. Phase difficulty controls how many checks must pass for full reward:

| Check | What it requires | Easy | Medium | Hard |
|---|---|---|---|---|
| `relationship` | References past order history, reliability, or long-term commitment | ✓ needed | ✓ needed | ✓ needed |
| `concrete_offer` | Makes a specific commitment (amount, advance payment, quantity guarantee) | — | ✓ needed | ✓ needed |
| `tone` | Professional, respectful, ≥ 20 characters, no threats | — | — | ✓ needed |

`total_score = checks_passed / 3` — always in {0.0, 0.33, 0.67, 1.0}

A keyword-based fallback scorer activates automatically when the LLM judge is unavailable. Scores are MD5-cached — same message is never scored twice.

---

## Grading (0.0–1.0)

Each phase is scored independently on five metrics:

```
score = 0.30 × service_score
      + 0.25 × profit_score
      + 0.20 × cost_score
      + 0.15 × spoilage_score
      + 0.10 × validity_score

service_score  = min(total_fulfilled / total_demand / phase_target, 1.0)
profit_score   = min(sum(revenue_history) / max_revenue, 1.0)
cost_score     = max(0.0, 1.0 − total_cost / max_allowed_cost)
spoilage_score = max(0.0, 1.0 − spoilage_rate × 5.0)   # 20%+ spoilage → 0
validity_score = valid_actions / total_actions
```

| Phase  | Service Target | Max Revenue | Max Cost  |
|--------|---------------|-------------|-----------|
| Easy   | 95%           | 25,000      | 15,000    |
| Medium | 85%           | 35,000      | 25,000    |
| Hard   | 75%           | 30,000      | 30,000    |

Live `phase_score` is available in every observation's `metadata` field.

---

## Baseline Scores

Baseline scores measured with `meta-llama/Llama-3.3-70B-Instruct` via HF router, `numpy seed=42`, `episode seed=0`:

| Phase       | **Final Score** |
|-------------|-----------------|
| Easy        | **0.6334**      |
| Medium      | **0.5152**      |
| Hard        | **0.5290**      |
| **Overall** | **0.5592**      |

---

## Quick Start

### Python Client

```python
from asc_agent_under_demand_uncertainity_rl_env import (
    AscAgentUnderDemandUncertainityRlEnv,
    SupplyChainAction,
)

with AscAgentUnderDemandUncertainityRlEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation
    print(f"Day {obs.day} | Stock: {obs.current_stock} | Phase: {obs.current_phase}")
    print(f"Trust: {obs.trust_score:.2f} | Surcharge: {obs.emergency_surcharge_rate}x")

    while not result.done:
        action = SupplyChainAction(
            action_type="order",
            quantity=100,
            sell_price=268.0,
            negotiation_message="As a consistent partner with zero defaults, we request 100 units and offer advance payment to secure this order.",
        )
        result = env.step(action)
        obs = result.observation
        print(
            f"Day {obs.day:2d} | Stock: {obs.current_stock:4d} "
            f"| Reward: {result.reward:+.1f} "
            f"| SL: {obs.last_7_day_service_level:.0%} "
            f"| Phase score: {obs.metadata['phase_score']:.3f}"
        )
```

### Start a Specific Phase

```python
result = env.reset(task="hard_phase_inventory", seed=42)
```

---

## Setup

### Prerequisites

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker (for containerised deployment)

### Install & Run Locally

```bash
# Install dependencies
uv sync

# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Build & Run with Docker

```bash
# Build
docker build -t asc-supply-chain:latest -f Dockerfile .

# Run
docker run -p 8000:8000 asc-supply-chain:latest
```

### Run Baseline Inference

```bash
# Required: Hugging Face API key (free at https://huggingface.co/settings/tokens)
export HF_TOKEN=hf_your_token_here

# Optional overrides (defaults shown):
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export ENV_URL=http://localhost:8000  # override to use local server instead of HF Space

# Run:
python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment (accepts `task`, `seed` params) |
| `/step`  | POST | Execute an action |
| `/state` | GET  | Current episode state |
| `/schema`| GET  | Action / observation JSON schemas |
| `/ws`    | WS   | Persistent WebSocket session (low latency) |
| `/health`| GET  | Container health check |
| `/docs`  | GET  | Interactive Swagger UI |
| `/web`   | GET  | Web interface for manual exploration |

---

## Project Structure

```
asc_agent_under_demand_uncertainity_rl_env/
├── Dockerfile              # Container image (at project root — hackathon requirement)
├── openenv.yaml            # OpenEnv manifest with 3 task definitions
├── pyproject.toml          # Project metadata and dependencies
├── models.py               # SupplyChainAction, SupplyChainObservation, StockBatch, PendingOrder
├── client.py               # Python client (WebSocket-based)
├── graders.py              # Phase graders returning 0.0–1.0 (5-metric formula)
├── negotiation_rubric.py   # LLM-as-Judge rubric with keyword fallback and MD5 cache
├── inference.py            # Baseline agent (meta-llama/Llama-3.3-70B-Instruct via HF router)
├── __init__.py             # Package exports
└── server/
    ├── app.py              # FastAPI application
    └── asc_agent_under_demand_uncertainity_rl_env_environment.py  # Core simulation
```

---

## Deploying to Hugging Face Spaces

```bash
# Push to your namespace
openenv push

# Push to a specific repo
openenv push --repo-id my-org/asc-supply-chain

# Push as private
openenv push --private
```

The deployed space includes a web UI at `/web` and full API docs at `/docs`.
