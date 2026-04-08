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
---

# Adaptive Supply Chain RL Environment

An **OpenEnv-compliant RL environment** for the Meta / PyTorch / Hugging Face OpenEnv Hackathon.

The agent plays a **warehouse manager** making daily inventory ordering decisions over a **30-day episode** under uncertain demand, variable supplier lead times, and a multi-component reward function. Difficulty escalates automatically through three curriculum phases.

---

## The Problem

> *"You manage a warehouse. Every day you decide how much stock to order. Demand is uncertain, suppliers are unreliable, and holding too much stock is as costly as running out."*

The agent must balance:
- **Ordering cost** — fixed + per-unit cost to place orders
- **Stockout risk** — large penalty if demand exceeds available stock
- **Overstock waste** — penalty for carrying more than 200 units
- **Service level** — fraction of demand successfully fulfilled

---

## Curriculum Phases

A **single environment** automatically advances through 3 difficulty phases based on recent performance:

```
Day 1 ──► [EASY] ──(7-day service level > 90%)──► [MEDIUM] ──(90% again)──► [HARD] ──► Day 30
```

| Phase  | Demand Pattern               | Lead Time       | Forecast Noise | Service Target |
|--------|------------------------------|-----------------|----------------|----------------|
| Easy   | Stable ~80 units/day         | Fixed 3 days    | ±10%           | 95%            |
| Medium | Seasonal wave (peak day 15)  | 2–5 days random | ±25%           | 85%            |
| Hard   | Baseline + random spikes     | 2–10 days + delays | ±40%        | 75%            |

Each phase is also exposed as an **independent task** for direct evaluation:
- `easy_phase_inventory`
- `medium_phase_inventory`
- `hard_phase_inventory`

---

## Action Space

**`SupplyChainAction`**

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"order"` \| `"emergency_restock"` \| `"hold"` | What to do today |
| `quantity` | `int` \| `None` | Units to order (required for order/emergency; `None` for hold) |

```json
{"action_type": "order", "quantity": 150}
{"action_type": "emergency_restock", "quantity": 50}
{"action_type": "hold"}
```

---

## Observation Space

**`SupplyChainObservation`**

| Field | Type | Description |
|-------|------|-------------|
| `day` | `int` | Current day (1–30) |
| `current_stock` | `int` | Units currently in warehouse |
| `demand_forecast` | `float` | Forecasted demand for today |
| `forecast_noise` | `"low"` \| `"medium"` \| `"high"` | Forecast uncertainty level |
| `pending_orders` | `List[PendingOrder]` | Orders placed but not yet arrived |
| `last_7_day_service_level` | `float` | Fraction of demand fulfilled over last 7 days |
| `holding_cost_per_unit` | `float` | $0.50 per unit per day |
| `stockout_penalty` | `float` | $50 per stockout event |
| `budget_remaining` | `float` | Remaining budget in dollars |
| `supplier_status` | `"normal"` \| `"delayed"` | Current supplier reliability |
| `current_phase` | `"easy"` \| `"medium"` \| `"hard"` | Active difficulty phase |
| `prompt` | `str` | Formatted natural-language prompt for LLM agents |
| `done` | `bool` | True after day 30 |
| `reward` | `float` | Step reward (dense, every step) |
| `metadata` | `dict` | `phase_score`, `actual_demand`, `actual_fulfilled`, etc. |

---

## Reward Function

Reward is **dense** — the agent receives a signal every step:

```
reward  = fulfilled_units × 3.0          # sell at $3/unit → $1 margin over $2 unit cost
reward -= 50.0                            # if stockout occurred
reward -= 0.5 × max(0, stock − 200)      # overstock penalty
reward -= (20 + qty × 2)                 # if action = order
reward -= (20 + qty × 6)                 # if action = emergency_restock
reward += 5.0                            # if stock in [50, 200]  (efficiency bonus)
reward -= 10.0                           # if action was malformed
```

**Optimal stock range: 50–200 units.** Below 50 risks stockouts; above 200 wastes holding cost.

---

## Grading (0.0–1.0)

Each phase is scored independently:

```
score = 0.5 × service_score + 0.3 × cost_score + 0.2 × validity_score

service_score  = min(avg_service_level / phase_target, 1.0)
cost_score     = max(0.0, 1.0 − total_cost / max_allowed_cost)
validity_score = valid_actions / total_actions
```

Live `phase_score` is available in every observation's `metadata` field.

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

    while not result.done:
        action = SupplyChainAction(action_type="order", quantity=100)
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

### Run Baseline Inference (Gemini 2.0 Flash)

```bash
# Requires a free API key from https://aistudio.google.com
export GEMINI_API_KEY=your_key

# Start server first, then:
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
├── Dockerfile          # Container image (at project root — hackathon requirement)
├── openenv.yaml        # OpenEnv manifest with 3 task definitions
├── pyproject.toml      # Project metadata and dependencies
├── models.py           # SupplyChainAction, SupplyChainObservation, PendingOrder
├── client.py           # Python client (WebSocket-based)
├── graders.py          # Phase graders returning 0.0–1.0
├── inference.py        # Gemini 2.0 Flash baseline agent
├── __init__.py         # Package exports
└── server/
    ├── app.py          # FastAPI application
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
