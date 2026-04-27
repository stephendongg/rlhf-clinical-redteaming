# rlhf-clinical-redteam

**Audrey Tjokro · Stephen Dong · Niki Karanikola**  
Cornell University — CS 6782 Generative Models, Spring 2026

---

## Overview
This notebook trains a PPO-optimized adversarial attacker (Qwen2.5-7B-Instruct) 
to stress-test a clinical AI target model (BioMistral-7B) across multi-turn 
conversations, using GPT-4o-mini as a safety judge.

Reward signal: `effectiveness` score — composite of attack_success, 
policy_violation severity, convincingness, and specificity.

---

## Iterations & Design Decisions

### Iteration 1: trl version fix
- Newer trl versions removed PPOTrainer — pinned to `trl==0.9.6`
- Loaded Qwen with `AutoModelForCausalLMWithValueHead` + LoRA (r=16)
- Loaded frozen reference model for KL divergence penalty

### Iteration 2: Generation bug fix
- `ppo_trainer.generate()` returned truncated tensors shorter than input
- Fixed by calling `ppo_model.pretrained_model.generate()` directly
- Resolved empty response tensors silently breaking the training loop

### Iteration 3: Batch size alignment
- PPOTrainer requires exactly `batch_size` trajectories per `.step()` call
- Added incomplete batch guard to skip PPO update on final partial batch
- Restructured loop to collect 4 trajectories before each gradient update

### Iteration 4: Hyperparameter tuning
- Loosened `target_kl`: 0.1 → 0.3 (allowed more policy movement)
- Disabled `use_score_scaling` (harmful with sparse rewards)
- Kept `learning_rate=1e-5`, `batch_size=4`, `mini_batch_size=1`

### Iteration 5: Resilience & logging
- Added `judge_trajectory_with_retry()` — 3 retries, 15s wait
  (OpenAI rate limits caused crash at step ~38)
- Added resume-from-checkpoint logic for mid-run crash recovery
- Saved full trajectories to `.jsonl` + summary to `.csv` for analysis

---

## Results

| Metric | Baseline (untuned) | PPO-trained |
|---|---|---|
| ASR (dev, n=20) | 0.40 | 0.20 |
| Avg TTF | 1.75 turns | 3.50 turns |
| Avg effectiveness | 0.28 | 0.14 |

## Interpretation
PPO **underperformed** the untuned baseline. Key reasons:
1. **Reward sparsity** — ~75% of trajectories scored 0.0, too little signal
2. **Insufficient steps** — 400 total conversations, PPO needs thousands
3. **KL over-regularization** — penalty pulls policy back toward safe behavior
4. **Terminal-only reward** — single scalar over 5 turns makes credit assignment hard

## Findings & Related Work

**Core finding:** PPO actively made the attacker *more cautious* than the 
untuned baseline (ASR 0.20 vs 0.40), suggesting the KL penalty dominated 
the sparse reward signal rather than the reward shaping policy behavior.

This is well-supported by prior work:

**On PPO and sparse rewards:** Hu et al. (2023) in *Secrets of RLHF in Large 
Language Models Part I* directly observe that "PPO suffers from sparse reward 
and inefficient exploration in word space, making it sensitive to 
hyperparameters." Our setting — binary success/fail rewards over 5-turn 
conversations — represents an extreme case of this sparsity problem.

**On KL over-regularization:** The same work finds the KL penalty "critical 
to stability" but also notes it can prevent the policy from meaningfully 
moving away from the reference model. In our case, Qwen2.5-7B-Instruct is 
already a safety-aligned base model. The KL penalty therefore pulled the 
policy *back toward safe behavior* whenever sparse rewards failed to provide 
sufficient signal to overcome it — the opposite of what we needed.

**On PPO vs DPO for this type of task:** Lyu et al. (2023) in *Rethinking 
the Role of PPO in RLHF* (Berkeley AI Research) highlight a fundamental 
tension: reward learning uses pairwise comparisons but PPO optimizes 
individual responses without comparisons. This mismatch is particularly 
acute in our setting where the reward signal is a single scalar over an 
entire multi-turn trajectory. DPO avoids this by directly learning from 
preference pairs — which may explain why DPO is theoretically better suited 
to this task.

**What this suggests for future work:**
- Reward shaping: intermediate per-turn rewards rather than terminal-only 
  signal would give PPO more to learn from
- Curriculum learning: starting with easier seeds before hard ones
- Removing the KL penalty entirely (as recent RLVR work does) and relying 
  on clipping alone to prevent collapse
- More training steps: 400 trajectories is far below the thousands typically 
  needed for PPO to show meaningful improvement

---

## File Structure

- `ppo_training.ipynb` — main training notebook
- `results/ppo_training_log.jsonl` — step-level metrics over 100 training steps
- `results/ppo_summary.csv` — human-readable training summary with attacker turn 1 text
- `results/ppo_dev_eval_metrics.json` — dev set evaluation results
- `results/ppo_trajectories.jsonl` — full conversation trajectories (Drive only, too large for GitHub)
- `checkpoints/ppo/` — model weights every 20 steps + final (Drive only, too large for GitHub)

## How to Run
1. Open in Google Colab with A100 GPU
2. Add `OPENAI_API_KEY` and `GITHUB_PAT` to Colab Secrets
3. Run all cells sequentially
4. Fresh run: set `RESUME_FROM_STEP = 0`
5. Resume: set `RESUME_FROM_STEP = <last completed step>`
