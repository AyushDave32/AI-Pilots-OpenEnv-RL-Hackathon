---
title: Adaptive Supply Chain RL Environment
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - supply-chain
  - multi-agent
  - world-modeling
  - theory-of-mind
  - inventory-management
  - negotiation
---

# Adaptive Supply Chain RL Environment

**A general-purpose OpenEnv RL environment where LLM agents manage perishable inventory, negotiate with a reactive supplier, and survive supply disruptions. Domain-agnostic — configure for any industry.**

> *"You manage perishable goods inventory over 30 days. Every unit expires 15 days after arrival. Your supplier has hidden loyalty tiers that determine your costs and crisis allocation. On day 21, supply capacity drops to 30%. Your relationship history decides how much you receive."*

---

## What This Environment Tests

Managing perishable goods under supplier uncertainty forces an LLM to reason in ways that pure numerical RL cannot:

- **Order the right quantity** given stock that expires 15 days after arrival
- **Set the right sell price** daily — price elasticity means pricing too high shrinks demand, too low shrinks margin
- **Model a reactive supplier's hidden state** — infer loyalty tier from observable signals (surcharge rates, message tone, lead time accuracy)
- **Write professional negotiation messages** that build long-term supplier trust and determine crisis allocation

Traditional RL policies can optimise the numerical decisions. They cannot write the negotiation messages that determine supplier priority during a crisis. This is why the environment is built for LLMs.

**Themes:** Theme #1 (Multi-Agent — theory-of-mind supplier modeling) + Theme #3.1 (World Modeling — professional supply chain task)

---

## Configurable For Any Industry

The environment logic, reward function, and graders are completely domain-agnostic. To adapt to a specific industry, only the prompt text and constants need updating — no code logic changes required.

| Industry | Perishable goods example | Crisis analog |
|----------|--------------------------|---------------|
| Pharmaceutical distribution | Medicines, diagnostics | Factory shutdown |
| Cold-chain food logistics | Dairy, fresh produce | Harvest failure |
| Electronics / semiconductors | Components with shelf life | Chip fab disruption |
| Blood bank / medical | Blood products, vaccines | Donation shortage |
| Fast fashion / apparel | Seasonal inventory | Factory capacity cuts |

---

## Core Mechanics

### 1. Perishable Inventory — Batch Tracking + FEFO

Every order creates a stock batch expiring 15 days after arrival. Demand is fulfilled FEFO (First Expired First Out). Unsold expired stock incurs a spoilage penalty of Rs 20/unit.

### 2. Sell-Side Pricing with Demand Elasticity

The agent sets a daily sell price. Actual demand is computed as:
```
actual_demand = base_demand × (market_price / sell_price)^1.5
```
Price below market increases demand but compresses margin. Price above market shrinks demand.

### 3. Supplier Hidden State — Theory-of-Mind

The supplier maintains three hidden variables the agent never sees directly:

```
loyalty_tier    : "bronze" | "silver" | "gold"   ← never shown
supplier_mood   : float 0.0–1.0                   ← never shown
order_regularity: float 0.0–1.0                   ← never shown
```

The agent observes only the **consequences** of these hidden variables:

| Signal | What it reveals |
|--------|----------------|
| `emergency_surcharge_rate` | 2.5×=Gold, 3.0×=Silver, 4.0×=Bronze |
| `supplier_last_message` tone | Warm=Gold, Neutral=Silver, Cold=Bronze |
| `lead_time_accuracy` | Consistent delays = Bronze |
| `proactive_discount_offered` | Only Gold/Silver get proactive discounts |
| `recent_neg_scores` | Your last 3 negotiation scores (0.0–1.0) |

### 4. Day-21 Supply Disruption

Days 21–25: factory capacity drops to 30%. Crisis allocation by loyalty tier:

| Tier | Order Fulfilled | Emergency Surcharge | Lead Time |
|------|----------------|---------------------|-----------|
| Gold | 100% | 2.5× unit cost | −1 day (faster) |
| Silver | 80% | 3.0× unit cost | As promised |
| Bronze | 50% | 4.0× unit cost | +1–2 days extra |

### 5. LLM-Native Negotiation Action

Every day the agent writes a natural language message to the supplier. Scored by a 3-check rubric:
1. **Relationship referenced** — mentions past order history, track record, partnership
2. **Concrete offer made** — specific payment amount, advance payment, quantity guarantee
3. **Tone appropriate** — professional, respectful, ≥ 20 characters

Score drives loyalty tier changes and earns negotiation bonus (capped at Rs 30/episode). A traditional numerical RL policy cannot do this.

---

## Action Space

```json
{
  "action_type": "order" | "emergency_restock" | "hold",
  "quantity": 120,
  "sell_price": 268.0,
  "negotiation_message": "As a consistent zero-default partner, we request priority allocation of 120 units with immediate advance payment of $24,000 to secure our position during this disruption."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"order"` \| `"emergency_restock"` \| `"hold"` | Buy-side decision |
| `quantity` | `int` \| `null` | Units to buy (null for hold) |
| `sell_price` | `float` | Price per unit sold to customers today |
| `negotiation_message` | `str` | Natural language message to your supplier |

Missing any field → −10 penalty, day does NOT advance.

---

## Reward Function

```
daily_reward = (sell_price - Rs 200) × units_fulfilled   # gross profit
             - Rs 50.0                                     # if stockout
             - Rs 20.0 × units_spoiled                    # spoilage penalty
             - 0.5 × max(0, stock - 300)                  # overstock penalty
             - order_cost                                  # fixed + unit × tier surcharge
             + Rs 5.0                                      # if stock in [50, 300]
             + neg_bonus                                   # negotiation score × 10 (capped Rs 30/ep)
             - Rs 10.0                                     # malformed action
             - Rs 100.0                                    # budget negative (signal only)
```

---

## Grading (0.0–1.0)

Each phase is scored independently:

```
score = 0.30 × service_score
      + 0.25 × profit_score
      + 0.20 × cost_score
      + 0.15 × spoilage_score
      + 0.10 × validity_score
```

| Component | Formula |
|-----------|---------|
| `service_score` | `min(fulfilled / demand / phase_target, 1.0)` |
| `profit_score` | `min(total_revenue / phase_max_revenue, 1.0)` |
| `cost_score` | `max(0, 1 - total_cost / phase_max_cost)` |
| `spoilage_score` | `max(0, 1 - spoilage_rate × 5)` — 20%+ spoilage → 0 |
| `validity_score` | `valid_actions / total_actions` |

Phase targets: Easy SL ≥ 95% | Medium SL ≥ 85% | Hard SL ≥ 75%

---

## Results

| Phase | Untrained Grade | Trained Grade | Improvement |
|-------|----------------|---------------|-------------|
| Easy   | X.XXX | X.XXX | +X.XXX |
| Medium | X.XXX | X.XXX | +X.XXX |
| Hard   | X.XXX | X.XXX | +X.XXX |
| **Overall** | **X.XXX** | **X.XXX** | **+X.XXX** |

*(Run Cell 8 of `training_colab.ipynb` to fill these with real numbers)*

![Training Reward](reward_curves.png)
*GRPO training reward over steps — x: training step, y: step reward (Rs)*

![Grade Comparison](grade_comparison.png)
*Grade score before vs after GRPO training across all three phases*

---

## Setup

### Install & Run Locally

```bash
uv sync
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Run Baseline Inference

```bash
export HF_TOKEN=hf_your_token_here
export ENV_URL=http://localhost:8000   # or your HF Space URL
python inference.py
```

### Docker

```bash
docker build -t supply-negotiate:latest -f Dockerfile .
docker run -p 8000:8000 supply-negotiate:latest
```

### Deploy to Hugging Face Spaces

```bash
openenv push
openenv push --repo-id my-org/supply-negotiate
```

---

## Customising for a Specific Domain

To configure the environment for a specific industry:

1. Update the `prompt` field in `_build_observation()` to use your domain's terminology (product name, unit name, customer type)
2. Adjust `SHELF_LIFE_DAYS`, `MARKET_PRICE`, and `BASE_DEMAND` constants to match your product's economics
3. Update `SUPPLIER_MESSAGES` templates to match your industry's communication style

No reward logic, grader logic, or rubric code changes are needed.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start episode — accepts `task`, `seed` params |
| `/step`  | POST | Submit 4-field action, receive observation + reward |
| `/state` | GET  | Current episode state |
| `/schema`| GET  | Action / observation JSON schemas |
| `/ws`    | WS   | Persistent WebSocket session (up to 4 concurrent) |
| `/health`| GET  | Container health check |
| `/docs`  | GET  | Interactive Swagger UI |
| `/web`   | GET  | Browser UI for manual exploration |

---

## Links

- [HuggingFace Space](#) — live environment
- [Training Notebook](training_colab.ipynb) — GRPO training with Qwen2.5-7B
- [Demo Video](#)
- [HF Blog Post](#)
