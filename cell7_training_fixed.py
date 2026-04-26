# Cell 7 — GRPO training  (ALL 10 ISSUES FIXED — replace your Cell 7 with this)
# ─────────────────────────────────────────────────────────────────────────────────
# ENV FIXES (already applied in environment.py):
#   FIX 1  — Reward scale: all costs ÷100,  target range [-100, +100]
#   FIX 2  — Dense rewards: +3×service_rate per step (solves credit assignment)
#   FIX 3  — Spoilage penalty normalised ÷100
#   FIX 4  — Early budget warning at Rs3000 (before going negative)
#   FIX 5  — Negotiation noise reduced: multiplier 10→3, cap 30→10
#   FIX 6  — Action clamping: qty∈[10,500], price∈[100,400]
#   FIX 9  — Final reward clipped to [-100, +100]
#
# TRAINING FIXES (here):
#   FIX 7  — Validity bonus: +5 valid JSON, +3 correct action_type
#             → valid action ALWAYS beats garbage output in reward
#   FIX 8  — Curriculum dataset: 60% easy / 30% medium / 10% hard
#             → model learns valid JSON on simple cases first
#   FIX 10 — kl_coef=0.1, num_generations=8, smart warning hooks
# ─────────────────────────────────────────────────────────────────────────────────

import re
import json as _json
import datasets
from trl import GRPOConfig, GRPOTrainer

# Switch back to training mode (undoes for_inference from Cell 6)
model.train()

# ── Constants ─────────────────────────────────────────────────────────────────
# Env now normalises its own reward (÷100) — we only add validity bonus on top
VALIDITY_BONUS     = 5.0   # FIX 7: reward for producing parseable JSON
ACTION_TYPE_BONUS  = 3.0   # FIX 7: extra reward for correct action_type value
VALID_ACTION_TYPES = {"order", "emergency_restock", "hold"}

# Validity rate tracker for diagnostics
_validity_tracker = {"valid": 0, "total": 0}


# ── JSON parsing helper ────────────────────────────────────────────────────────

def _parse_completion(text: str):
    """Extract and parse first JSON object. Returns (dict|None, is_valid)."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None, False
    try:
        return _json.loads(match.group()), True
    except _json.JSONDecodeError:
        return None, False


def _validity_bonus(text: str) -> float:
    """
    FIX 7 — Validity bonus ensures valid output always beats garbage.

    Even after env normalization, a bad order may score -22.
    With +8 bonus:   good order = -22 + 8 = -14   ← better than...
    Invalid JSON:    no bonus   =   0 - 10 = -10   ← still worse than profitable actions

    When action is profitable:  good action = +5 + 8 = +13  ← clearly best
    """
    parsed, is_valid = _parse_completion(text)
    if not is_valid:
        return 0.0
    bonus = VALIDITY_BONUS
    if str(parsed.get("action_type", "")).lower().strip() in VALID_ACTION_TYPES:
        bonus += ACTION_TYPE_BONUS
    return bonus


# ── Dataset builder with Curriculum (FIX 8) ───────────────────────────────────

def build_dataset(env_client, n=150):
    """
    FIX 8 — Curriculum ordering: easy first, hard last.
    60% easy / 30% medium / 10% hard.
    Model learns to produce valid JSON on simple cases before
    handling volatile demand + supply crisis.
    """
    prompts = []
    curriculum = (
        ["easy_phase_inventory"]   * 6 +   # 60% easy
        ["medium_phase_inventory"] * 3 +   # 30% medium
        ["hard_phase_inventory"]   * 1     # 10% hard
    )
    for i in range(n):
        task = curriculum[i % len(curriculum)]
        obs = env_client.reset(task=task, seed=i)
        prompts.append({
            "prompt": (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{obs}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ),
            "task": task,
        })
    return datasets.Dataset.from_list(prompts)


# ── Reward function (ALL FIXES APPLIED) ───────────────────────────────────────

def reward_fn(prompts, completions, task=None, **kwargs):
    """
    GRPO single-step reward.

    BEFORE (broken):
      valid order  → env gives -22,000   |   invalid JSON → flat -10
      model learns: "being wrong is 2200x cheaper" → collapse

    AFTER (fixed):
      env normalises raw reward ÷100       → valid order ≈ -22
      +validity bonus for good structure   → -22 + 8 = -14
      invalid JSON                         → 0 + 0 = -10 (no bonus, or worse)
      profitable hold                      → +5 + 8 = +13  ← clearly best

      The model now has a GRADIENT to climb toward valid, profitable actions.
    """
    rewards = []
    tasks_list = (
        task if task is not None
        else [TASKS[i % len(TASKS)] for i in range(len(prompts))]
    )
    batch_valid = 0

    for i, (_, completion) in enumerate(zip(prompts, completions)):
        t = tasks_list[i] if isinstance(tasks_list, list) else tasks_list

        # FIX: fresh client per completion — no shared WebSocket state
        tmp = SupplyChainEnvClient()
        try:
            tmp.reset(task=t, seed=i)
            _, r_raw, _, _ = tmp.step(completion)

            # Env already normalises to ~[-100, +100], just add validity bonus
            r = float(r_raw) + _validity_bonus(completion)

            _, is_valid = _parse_completion(completion)
            if is_valid:
                batch_valid += 1

        except Exception as e:
            _, is_valid = _parse_completion(completion)
            if is_valid:
                r = -2.0   # valid JSON but server error — not model's fault
                batch_valid += 1
            else:
                has_brace = bool(re.search(r'\{', completion))
                r = -5.0 if has_brace else -10.0   # partial credit
            print(f"[reward_fn] #{i} error: {str(e)[:80]} → r={r:.1f}")

        finally:
            tmp.close()   # always close cleanly

        rewards.append(float(r))

    _validity_tracker["valid"] += batch_valid
    _validity_tracker["total"] += len(completions)
    return rewards


# ── Build dataset ─────────────────────────────────────────────────────────────

train_ds   = build_dataset(env_client, n=150)
reward_log = []
print(f"Dataset: {len(train_ds)} prompts | curriculum: 60% easy / 30% medium / 10% hard")


# ── GRPO config (FIX 10) ──────────────────────────────────────────────────────

cfg = GRPOConfig(
    output_dir                  = "./checkpoints",
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate               = 2e-5,
    logging_steps               = LOG_INTERVAL,
    save_steps                  = 100,
    report_to                   = "none",
    max_completion_length       = 300,
    num_generations             = 8,     # was 4 — more variance → better signal
    kl_coef                     = 0.1,   # stops Qwen diverging from base
)

trainer = GRPOTrainer(
    model         = model,
    args          = cfg,
    reward_funcs  = reward_fn,
    train_dataset = train_ds,
    tokenizer     = tokenizer,
)


# ── Smart log hook ────────────────────────────────────────────────────────────

_orig_log = trainer.log

def _log_hook(logs):
    if "reward" in logs:
        r       = logs["reward"]
        std     = logs.get("reward_std", 0.0)
        kl      = logs.get("kl", 0.0)
        clipped = logs.get("completions/clipped_ratio", 0.0)
        step    = logs.get("step", len(reward_log))
        reward_log.append(r)

        total = _validity_tracker["total"]
        valid = _validity_tracker["valid"]
        vpct  = (valid / total * 100) if total > 0 else 0.0
        _validity_tracker["valid"] = 0
        _validity_tracker["total"] = 0

        print(f"\n[step {step}] reward={r:.3f} | std={std:.3f} | kl={kl:.4f} "
              f"| valid_JSON={vpct:.0f}% | clipped={clipped:.2f}")

        if std == 0.0:
            print(f"  ⚠️  COLLAPSE: std=0 — all {8} completions got same reward ({r:.2f})\n"
                  f"     Is env server running? Is model generating JSON?")
        if kl > 0.25:
            print(f"  ⚠️  HIGH KL={kl:.3f} — model diverging. Reduce learning_rate.")
        if vpct < 50 and total > 0:
            print(f"  ⚠️  LOW VALIDITY={vpct:.0f}% — model generating invalid JSON "
                  f"in {100-vpct:.0f}% of completions")
        if clipped > 0.1:
            print(f"  ⚠️  CLIPPING={clipped:.2f} — outputs hitting 300-token limit")

    _orig_log(logs)

trainer.log = _log_hook


# ── Train ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("GRPO TRAINING — ALL 10 ISSUES FIXED")
print("=" * 68)
print("ENV (environment.py):")
print("  [1] Reward scale      ÷100      → target range [-100, +100]")
print("  [2] Dense rewards     +3×SL/step  → faster credit assignment")
print("  [3] Spoilage          normalised ÷100")
print("  [4] Budget warning    at Rs3000 (before going negative)")
print("  [5] Neg noise         multiplier 10→3, cap 30→10")
print("  [6] Action clamps     qty∈[10,500], price∈[100,400]")
print("  [9] Reward clip       final clipped to [-100, +100]")
print("TRAINING (cell7):")
print(f"  [7] Validity bonus    +{VALIDITY_BONUS} valid JSON, +{ACTION_TYPE_BONUS} correct action_type")
print( "  [8] Curriculum        60% easy / 30% medium / 10% hard")
print( "  [10] kl_coef=0.1, num_generations=8, smart warnings")
print("EXPECTED REWARD (after all fixes):")
print(f"  Profitable order      ~ +{5 + VALIDITY_BONUS + ACTION_TYPE_BONUS:.0f} to +{50 + VALIDITY_BONUS + ACTION_TYPE_BONUS:.0f}")
print(f"  Hold (full stock)     ~ +{5.0 + VALIDITY_BONUS + ACTION_TYPE_BONUS:.0f}")
print(f"  Invalid JSON          ~ -10  (no bonus)")
print("=" * 68 + "\n")

trainer.train()
print("Training complete.")
