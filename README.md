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
  - grpo
  - llm-training
---

# 🏭 Adaptive Supply Chain Agent Under Demand Uncertainty

> **OpenEnv Hackathon 2026** | Scaler × Meta × HuggingFace × PyTorch  
> **Team: AI Pilots**  
> **Theme:** World Modeling (3.1) + Long-Horizon Planning (2) + Self-Improvement (4)

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🤗 HF Space (Environment) | https://huggingface.co/spaces/Ayush-Dave/OpenENV_hackathon_finale_round |
| 💻 GitHub Repository | https://github.com/AyushDave32/AI-Pilots-OpenEnv-RL-Hackathon |
| 📓 Training Notebook (Colab) | See `training_colab_final.ipynb` in repository |
| 🎥 Demo Blog | See `Blog.md` in repository |
| 🤖 Trained Model | https://huggingface.co/Ayush-Dave/adaptive-supply-chain-qwen25-0.5b |

---

## 🎯 The Problem

Supply chain management is one of the most economically costly problems in the world. A warehouse manager must make daily decisions under uncertainty:

- **How much to order** when demand is unpredictable
- **When to emergency restock** vs wait for cheaper standard delivery
- **How to negotiate** with suppliers to unlock better terms
- **How to survive** sudden supply disruptions

Current LLMs are poor at this — they lack the ability to balance multi-objective tradeoffs, model hidden supplier state, and plan across delayed consequences.

**We built an RL environment that trains LLMs to do exactly this.**

---

## 🌍 Environment Design

### What the Agent Sees (Observation)

Each day the agent receives a natural language prompt containing:

```
ADAPTIVE SUPPLY CHAIN — Day 15 of 30
═══════════════════════════════════════

INVENTORY STATUS
  Current stock         : 127 units across 2 batches
  Nearest expiry        : 3 days (45 units) ⚠️ Warning
  
MARKET CONDITIONS
  Market price today    : 285.0/unit
  Demand forecast       : ~100 units (medium uncertainty)
  Price elasticity      : 1.5x shift per 10% price change

SUPPLIER RELATIONSHIP
  Trust score           : 0.82 / 1.00
  Emergency surcharge   : 3.0x unit cost  (signals loyalty tier)
  Supplier message      : "Your order confirmed, delivery in 2 days"

FINANCIALS
  Budget remaining      : $7,240
  Last 7-day service SL : 91%
```

### What the Agent Does (Action)

The agent outputs a single JSON action:

```json
{
  "action_type": "order",
  "quantity": 150,
  "sell_price": 285.0,
  "negotiation_message": "We have maintained consistent orders for 2 weeks. Requesting priority allocation during the upcoming disruption period."
}
```

### Reward Function (4 Components)

```
reward = fulfilled_units × 3.0          # revenue at $3/unit margin
       - 50.0                            # if stockout occurred  
       - 0.5 × max(0, stock - 300)       # overstock holding cost
       - (20 + qty × 2)                  # standard order cost
       - (20 + qty × 6 × surcharge)      # emergency order cost
       + 5.0                             # efficiency bonus (stock in optimal range)
```

This creates **genuine multi-objective tension** — the agent cannot simply maximize one metric. It must balance all four simultaneously.

---

## 🧠 What Makes This Environment Novel

### 1. Supplier Hidden State (Theory-of-Mind)
The supplier has a hidden **loyalty tier** (bronze/silver/gold) that the agent cannot observe directly. It must infer the tier from signals:
- Emergency surcharge rate shown in observation
- Whether proactive discounts are offered
- Lead time accuracy (on time vs delayed)

Higher loyalty → lower emergency costs, priority during crisis, proactive discounts.

### 2. Day 21-25 Supply Disruption Crisis
On days 21-25, supplier capacity drops to 30%. **Only loyal customers get full allocation.** Bronze-tier customers receive only 50% of their order. This creates a genuine long-horizon dependency — day-1 negotiation behavior affects day-21 survival.

### 3. Price Elasticity
The agent sets daily sell prices. Higher prices reduce demand (elasticity = 1.5x per 10% deviation from market). The agent must learn that pricing too high kills revenue even if margins look good.

### 4. FEFO Inventory (Perishable Goods)
Stock expires after 15 days (First Expired First Out). The agent must track expiry warnings and avoid ordering too much that spoils.

---

## 📚 Curriculum Learning (3 Difficulty Phases)

| Phase | Demand Pattern | Lead Time | Forecast Noise | Service Target |
|-------|---------------|-----------|----------------|----------------|
| Easy | Stable ~80/day | Fixed 3 days | ±10% | 95% |
| Medium | Seasonal wave | 2-5 days random | ±25% | 85% |
| Hard | Spikes + baseline | 2-10 days + delays | ±40% | 75% |

The environment automatically progresses through phases. Each phase is also available as an independent task:
- `easy_phase_inventory`
- `medium_phase_inventory`  
- `hard_phase_inventory`

---

## 🚀 Training Results

### GRPO Training — Reward Over Steps

![Training Reward Curves](reward_curves.png)
*Left: GRPO step rewards consistently above +260 vs untrained baseline of -1500. Right: Episode total reward before (red) vs after (green) GRPO training.*

### Episode Reward: Before vs After Training

| Phase | Baseline Reward | Trained Reward | Improvement |
|-------|----------------|----------------|-------------|
| Easy | -1,500 | **+533** | **+135.5%** |
| Medium | -1,500 | **+569** | **+137.9%** |
| Hard | -1,500 | **-133** | **+91.1%** |
| **Overall** | **-1,500** | **+323** | **+121.5%** |

### Grade Score (Service Level + Cost Efficiency)

| Phase | Baseline | Trained | Delta |
|-------|----------|---------|-------|
| Easy | 0.7500 | 0.7402 | -0.010 |
| Medium | 0.7500 | 0.6956 | -0.054 |
| Hard | 0.7500 | 0.6591 | -0.091 |

> **Note on grade vs reward:** Grade measures 30-day cumulative service level. GRPO was trained on single-step rewards. The agent learned to maximize immediate profit (reward +121.5%) but grade — which requires long-horizon multi-step optimization — shows slight regression. This is a known limitation of single-step GRPO vs multi-step rollout training, and a direction for future work.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-0.5B-Instruct (4-bit QLoRA) |
| Algorithm | GRPO — HuggingFace TRL |
| Training steps | 400 |
| Epochs | 2 |
| Dataset size | 200 prompts (30% easy, 30% medium, 40% hard) |
| Hardware | NVIDIA T4 GPU |
| Training time | ~60 minutes |
| LoRA rank | 16 |
| Learning rate | 3e-6 |

---

## 🎭 Qualitative Demo — Day 21 Crisis Response

When the supply disruption hits on Day 21 with zero stock, the **trained agent correctly responds**:

```
TRAINED AGENT (Day 21, stock = 0, crisis active):
Action: EMERGENCY_RESTOCK
Message: "We need to replenish our stock immediately due to the supply 
          disruption. As a consistent partner, we request priority 
          allocation for our order."
```

The agent correctly identifies:
1. Emergency restock is needed (not hold)
2. Professional negotiation message referencing partnership history
3. Crisis context understanding

---

## 🏗️ Technical Architecture

### OpenEnv Compliance

```
reset() → SupplyChainObservation (natural language prompt)
step()  → SupplyChainObservation (next state + reward + done)
state() → State (episode metadata)
```

### Project Structure

```
├── Dockerfile                    # Container image
├── openenv.yaml                  # OpenEnv manifest (3 tasks)
├── pyproject.toml                # Package metadata
├── models.py                     # Action/Observation/State dataclasses
├── client.py                     # Python client (WebSocket)
├── graders.py                    # Phase graders (0.0-1.0)
├── negotiation_rubric.py         # LLM-scored negotiation quality
├── inference.py                  # Baseline agent
├── reward_curves.png             # Training results plot
└── server/
    ├── app.py                    # FastAPI application
    └── *_environment.py          # Core simulation (1,200 lines)
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute action |
| `/state` | GET | Episode metadata |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |
| `/web` | GET | Interactive web UI |

---

## ⚡ Quick Start

### Python Client

```python
from asc_agent_under_demand_uncertainity_rl_env import (
    AscAgentUnderDemandUncertainityRlEnv,
    SupplyChainAction,
)

with AscAgentUnderDemandUncertainityRlEnv(
    base_url="https://Ayush-Dave-openenv-hackathon-finale-round.hf.space"
).sync() as env:
    result = env.reset(task="easy_phase_inventory", seed=42)
    print(result.observation.prompt)
    
    result = env.step(SupplyChainAction(
        action_type="order",
        quantity=100,
        sell_price=265.0,
        negotiation_message="Requesting priority delivery as a loyal partner."
    ))
    print(f"Reward: {result.reward}")
```

### Run Locally

```bash
git clone https://github.com/AyushDave32/AI-Pilots-OpenEnv-RL-Hackathon.git
cd AI-Pilots-OpenEnv-RL-Hackathon
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker run -p 8000:8000 registry.hf.space/ayush-dave-openenv-hackathon-finale-round:latest
```

---

## 🎓 Why This Environment Matters

The supply chain problem costs global industry **hundreds of billions of dollars** annually in poor decisions. Every mechanic in this environment maps to real economic pain:

- **Stockout penalty (-50)** = customer loss, production downtime
- **Overstock penalty** = drug expiry, food spoilage, capital waste  
- **Emergency restock premium** = airfreight expediting costs
- **Supplier loyalty** = 2-3 years of relationship building compressed into 30 days

An LLM trained on this environment develops **generalizable resource management skills** — not just supply chain knowledge, but the deeper capability of multi-objective optimization under uncertainty with hidden state.

---

## 📋 Submission Checklist

- ✅ OpenEnv (latest release) — built on `openenv-core>=0.2.2`
- ✅ Training script — Colab notebook with Unsloth + HF TRL
- ✅ Evidence of training — reward curves + before/after plots
- ✅ Mini writeup — this README + video
- ✅ HF Space — environment hosted and running
- ✅ `openenv.yaml` manifest with 3 task definitions
- ✅ Valid Gym-style API (reset, step, state)
