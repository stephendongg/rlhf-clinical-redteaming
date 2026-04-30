"""PPO method (TRL). Extracted from ppo_training.ipynb §3-§4.

Uses trl==0.9.6's PPOTrainer + AutoModelForCausalLMWithValueHead. The
training loop generates multi-turn red-team conversations under the *current*
PPO policy (gradients flowing through generation), scores each trajectory with
the GPT-4o-mini judge, and applies the trajectory-level effectiveness as a
single terminal reward to the last-turn (query, response) pair.

Multi-turn note (preserved from the notebook): credit assignment across turns
is approximate; we apply the terminal reward to the final turn's tokens.
"""

from __future__ import annotations

import gc
import logging
import random
from typing import Any

import torch

from ..judge import make_judge, JUDGE_MODEL
from ..models import load_model
from ..results import ResultsLogger
from ..rollouts import (
    ATTACKER_SYSTEM_PROMPT,
    make_target,
)
from ..seeds import load_seed_splits, split_fingerprint

log = logging.getLogger("redteam_rlhf.ppo")


# ── Generation with grad enabled (PPO requires it) ──────────────────────────
def _generate_with_grad(
    seed_scenario: str,
    history: list[dict],
    ppo_model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
):
    """Generate one attacker turn under the PPO policy. Returns (text, q_ids, r_ids)."""
    system = ATTACKER_SYSTEM_PROMPT.format(seed_scenario=seed_scenario)
    messages = [{"role": "system", "content": system}]
    for m in history:
        if m["role"] == "user":
            messages.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "assistant":
            messages.append({"role": "user", "content": m["content"]})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt")
    device = ppo_model.pretrained_model.device
    q_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    q_tensor = q_ids.squeeze(0)

    # NOTE: no torch.no_grad — gradients must flow for PPO.
    out = ppo_model.pretrained_model.generate(
        input_ids=q_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = out.squeeze(0)
    response_ids = full[q_tensor.shape[0]:]
    if response_ids.shape[0] == 0:
        response_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long, device=device)
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text, q_tensor, response_ids


def _run_ppo_conversation(
    seed_scenario: str,
    ppo_model: Any,
    tokenizer: Any,
    target,
    judge,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
):
    history: list[dict] = []
    turns: list[dict] = []
    q_tensors, r_tensors = [], []

    for tn in range(1, max_turns + 1):
        user_msg, q_t, r_t = _generate_with_grad(
            seed_scenario, history, ppo_model, tokenizer,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        q_tensors.append(q_t)
        r_tensors.append(r_t)

        target_history = history + [{"role": "user", "content": user_msg}]
        assistant_msg = target(target_history)

        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ])
        turns.append({"turn": tn, "user": user_msg, "assistant": assistant_msg})
        log.info("    turn %d ok", tn)

    judgment = judge(seed_scenario=seed_scenario, turns=turns)
    reward = float(judgment["effectiveness"])
    return turns, q_tensors, r_tensors, reward, judgment


def run(config: dict, logger: ResultsLogger) -> dict:
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

    data = config["data"]
    train_seeds, dev_seeds, test_seeds = load_seed_splits(
        n_train=data["n_train"], n_dev=data.get("n_dev"), n_test=data["n_test"],
    )
    fp = split_fingerprint(train_seeds, dev_seeds, test_seeds)
    log.info("Seed split fingerprint: %s", fp)
    logger.log_jsonl("split_fingerprint", fp)

    eval_seeds = test_seeds if config.get("use_test") else dev_seeds

    # ── Tokenizer + target (target uses our 4-bit loader; PPO model is custom) ──
    log.info("Loading attacker tokenizer + target model")
    from transformers import AutoTokenizer
    attacker_tok = AutoTokenizer.from_pretrained(
        config["attacker_model_id"], trust_remote_code=True,
    )
    target_tok, target_mdl = load_model(config["target_model_id"])
    target = make_target(
        target_tok, target_mdl,
        max_new_tokens=config.get("target_max_new_tokens", 256),
        temperature=config.get("target_temperature", 0.7),
    )

    # ── PPO model + LoRA ────────────────────────────────────────────────────
    lora = config["lora"]
    lora_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"],
        target_modules=lora["target_modules"],
        lora_dropout=lora["dropout"],
        bias="none", task_type="CAUSAL_LM",
    )
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    log.info("Loading PPO model (Qwen + value head, 4-bit)")
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config["attacker_model_id"],
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        peft_config=lora_config,
    )
    log.info("Loading frozen reference model")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config["attacker_model_id"],
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    for p in ref_model.parameters():
        p.requires_grad = False

    ppo_cfg = config["ppo"]
    extra = config.get("extra_ppo_kwargs") or {}
    trl_config = PPOConfig(
        model_name=config["attacker_model_id"],
        learning_rate=ppo_cfg["lr"],
        batch_size=ppo_cfg["batch_size"],
        mini_batch_size=ppo_cfg["mini_batch_size"],
        gradient_accumulation_steps=ppo_cfg["gradient_accumulation_steps"],
        optimize_cuda_cache=ppo_cfg["optimize_cuda_cache"],
        early_stopping=ppo_cfg["early_stopping"],
        target_kl=ppo_cfg["target_kl"],
        kl_penalty=ppo_cfg["kl_penalty"],
        seed=int(config.get("seed", 42)),
        use_score_scaling=ppo_cfg["use_score_scaling"],
        use_score_norm=ppo_cfg["use_score_norm"],
        **extra,
    )
    ppo_trainer = PPOTrainer(
        config=trl_config, model=ppo_model, ref_model=ref_model, tokenizer=attacker_tok,
    )
    log.info("PPOTrainer ready: lr=%.2e batch=%d target_kl=%.2f",
             ppo_cfg["lr"], ppo_cfg["batch_size"], ppo_cfg["target_kl"])

    # ── Judge ───────────────────────────────────────────────────────────────
    judge_backend = config.get("judge_backend", "openai")
    judge_model = config.get("judge_model", JUDGE_MODEL)
    log.info("Judge: backend=%s model=%s", judge_backend, judge_model)
    if judge_backend == "hf_local":
        log.warning(
            "PPO already holds policy + ref + target on GPU; loading an "
            "HF-local judge on top will likely OOM on a 40GB A100."
        )
    judge = make_judge(backend=judge_backend, model=judge_model)

    # ── Training loop ───────────────────────────────────────────────────────
    n_steps = ppo_cfg["n_train_steps"]
    ckpt_every = ppo_cfg["checkpoint_every"]
    max_turns = config.get("max_turns", 5)

    seeds_cycle = train_seeds * (n_steps // max(len(train_seeds), 1) + 1)
    random.shuffle(seeds_cycle)

    successes = 0
    step = 0
    last_running_asr = 0.0

    while step < n_steps:
        batch_q, batch_r, batch_rew, batch_judg, batch_turns = [], [], [], [], []

        for _ in range(trl_config.batch_size):
            seed = seeds_cycle[step % len(seeds_cycle)]
            log.info("=" * 70)
            log.info("Step %d/%d | seed: %s", step + 1, n_steps, seed[:80])
            turns, q_tensors, r_tensors, reward, judgment = _run_ppo_conversation(
                seed_scenario=seed,
                ppo_model=ppo_model,
                tokenizer=attacker_tok,
                target=target,
                judge=judge,
                max_turns=max_turns,
                max_new_tokens=config.get("attacker_max_new_tokens", 256),
                temperature=config.get("attacker_temperature", 0.7),
            )
            batch_q.append(q_tensors[-1])
            batch_r.append(r_tensors[-1])
            batch_rew.append(torch.tensor(reward, dtype=torch.float32))
            batch_judg.append(judgment)
            batch_turns.append(turns)
            step += 1
            if step >= n_steps:
                break

        if len(batch_q) < trl_config.batch_size:
            log.info("Incomplete final batch (%d/%d) — stopping.",
                     len(batch_q), trl_config.batch_size)
            break

        stats = ppo_trainer.step(batch_q, batch_r, batch_rew)
        scalar_stats = {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))}

        for i, (judgment, turns) in enumerate(zip(batch_judg, batch_turns)):
            successes += int(judgment["attack_success"])
            running_asr = successes / step
            last_running_asr = running_asr

            entry = {
                "step": step - len(batch_judg) + i + 1,
                "reward": batch_rew[i].item(),
                "attack_success": judgment["attack_success"],
                "policy_violation": judgment["policy_violation"],
                "turns_to_failure": judgment.get("first_failure_turn"),
                "running_asr": running_asr,
                "ppo_stats": scalar_stats,
            }
            logger.log_jsonl("training_log", entry)
            logger.log_jsonl("trajectories", {
                "step": entry["step"],
                "seed": seeds_cycle[(step - len(batch_judg) + i) % len(seeds_cycle)],
                "reward": entry["reward"],
                "attack_success": judgment["attack_success"],
                "turns": turns,
                "judgment": judgment,
            })

        log.info("  batch rewards=%s | running_ASR=%.3f",
                 [round(r.item(), 3) for r in batch_rew], last_running_asr)

        if step % ckpt_every == 0:
            ckpt = logger.artifact_path("checkpoints", f"step_{step}")
            ckpt.mkdir(parents=True, exist_ok=True)
            ppo_model.save_pretrained(str(ckpt))
            attacker_tok.save_pretrained(str(ckpt))
            log.info("Checkpoint saved: %s", ckpt)

        gc.collect()
        torch.cuda.empty_cache()

    log.info("PPO training complete. Final running ASR: %.3f", last_running_asr)

    # Save final adapter explicitly.
    final_ckpt = logger.artifact_path("checkpoints", "final")
    final_ckpt.mkdir(parents=True, exist_ok=True)
    ppo_model.save_pretrained(str(final_ckpt))
    attacker_tok.save_pretrained(str(final_ckpt))

    return {
        "n_train_steps": step,
        "final_running_asr": last_running_asr,
        "n_train_successes": successes,
        "split": "test" if config.get("use_test") else "dev",
    }
