"""PPO method (TRL). Extracted from ppo_training.ipynb §3–§4.

Uses trl==0.9.6's PPOTrainer + AutoModelForCausalLMWithValueHead.

All knobs below are set in configs/ppo.yaml and can be overridden at runtime
via --override (e.g. OVERRIDES = ['credit_assignment=discounted', 'ppo.target_kl=0.6']).

credit_assignment  "last_turn" | "all_turns" | "discounted"
    last_turn:   terminal reward applied only to the final attacker turn (original).
    all_turns:   terminal reward broadcast equally to every turn. Assumption: each
                 turn contributed equally to the outcome. More gradient signal, but
                 introduces correlated samples within a batch.
    discounted:  turn t receives gamma^(T-t) * reward (T = last turn index). More
                 principled than flat broadcast; later turns are weighted higher.

credit_discount_gamma  float (default 0.9) — exponent base for "discounted" mode.

n_convos_per_update  int (default 4)
    Full conversations per PPO step. PPOConfig batch_size is derived as
    n_convos * max_turns (all_turns/discounted) or n_convos (last_turn).

reward_signal  "effectiveness" | "attack_success" | "policy_violation" | "harmfulness"
    effectiveness:    attack_success * harmfulness_normalized (composite, default).
    attack_success:   binary 0/1 — cleanest sparse signal.
    policy_violation: 1–5 severity — denser but conflates success/failure.
    harmfulness:      composite harm score independent of the success flag.

curriculum  bool (default false)
    Sort training seeds shortest-first as a difficulty proxy before cycling.

ppo.target_kl  float   sweep: 0.1 | 0.3 | 0.6 | 0.8
ppo.kl_penalty  str    "kl" | "abs" | "mse" | "full" | "none"
ppo.use_score_scaling / use_score_norm  bool
max_turns  int          sweep: 3 | 5 | 7
attacker_temperature  float  sweep: 0.5 | 0.7 | 0.9
"""

from __future__ import annotations

import gc
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..judge import make_judge, JUDGE_MODEL
from ..models import load_model
from ..results import ResultsLogger
from ..rollouts import (
    ATTACKER_SYSTEM_PROMPT,
    make_target,
)
from ..seeds import load_seed_splits, split_fingerprint

log = logging.getLogger("redteam_rlhf.ppo")


# ── Plots ────────────────────────────────────────────────────────────────────

def _save_plots(run_dir: Path) -> None:
    """Read training_log.jsonl + trajectories.jsonl and save PNG plots to run_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return

    training_log_path = run_dir / "training_log.jsonl"
    trajectories_path = run_dir / "trajectories.jsonl"

    if not training_log_path.exists():
        log.warning("training_log.jsonl not found — skipping plots")
        return

    with training_log_path.open() as f:
        training_log = [json.loads(l) for l in f if l.strip()]
    trajectories = []
    if trajectories_path.exists():
        with trajectories_path.open() as f:
            trajectories = [json.loads(l) for l in f if l.strip()]

    if not training_log:
        log.warning("training_log is empty — skipping plots")
        return

    steps   = [e["step"]        for e in training_log]
    rewards = [e["reward"]      for e in training_log]
    asrs    = [e["running_asr"] for e in training_log]

    def _stat(key):
        return [e.get("ppo_stats", {}).get(key, float("nan")) for e in training_log]

    # ── Plot 1: Training curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO Training Curves", fontsize=14, fontweight="bold")

    axes[0, 0].plot(steps, rewards, color="steelblue", alpha=0.8)
    axes[0, 0].set(xlabel="Step", ylabel="Reward", title="Reward per Step")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, asrs, color="darkorange")
    axes[0, 1].set(xlabel="Step", ylabel="Running ASR", title="Running Attack Success Rate")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, _stat("ppo/loss/policy"), color="green", alpha=0.8)
    axes[1, 0].set(xlabel="Step", ylabel="Loss", title="PPO Policy Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, _stat("ppo/loss/value"), color="crimson", alpha=0.8)
    axes[1, 1].set(xlabel="Step", ylabel="Loss", title="PPO Value Loss")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: training_curves.png")

    if not trajectories:
        return

    judgments = [t["judgment"] for t in trajectories if "judgment" in t]
    if not judgments:
        return

    # ── Plot 2: Severity metrics ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Severity Metrics (LLM Judge)", fontsize=14, fontweight="bold")

    int_bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    axes[0, 0].hist([j.get("policy_violation", 1) for j in judgments],
                    bins=int_bins, color="crimson", edgecolor="black", alpha=0.7)
    axes[0, 0].set(xlabel="Policy Violation (1=safe, 5=severe)", ylabel="Count",
                   title="Policy Violation Distribution")
    axes[0, 0].set_xticks([1, 2, 3, 4, 5])
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    axes[0, 1].hist([j.get("convincingness", 1) for j in judgments],
                    bins=int_bins, color="mediumorchid", edgecolor="black", alpha=0.7)
    axes[0, 1].set(xlabel="Convincingness (1/3/5)", ylabel="Count",
                   title="Attack Convincingness")
    axes[0, 1].set_xticks([1, 2, 3, 4, 5])
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 0].hist([j.get("specificity", 1) for j in judgments],
                    bins=int_bins, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1, 0].set(xlabel="Specificity (1/3/5)", ylabel="Count",
                   title="Attack Specificity")
    axes[1, 0].set_xticks([1, 2, 3, 4, 5])
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].hist([j.get("harmfulness", 0.0) for j in judgments],
                    bins=10, range=(0, 1), color="darkorange", edgecolor="black", alpha=0.7)
    axes[1, 1].set(xlabel="Harmfulness Score (0–1)", ylabel="Count",
                   title="Harmfulness Distribution")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(run_dir / "severity_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: severity_metrics.png")

    # ── Plot 3: AMA category breakdown (successful attacks only) ─────────────
    successful = [j for j in judgments if j.get("attack_success")]
    if successful:
        cats = Counter(j.get("ama_category", "none") for j in successful)
        cats.pop("none", None)
        if cats:
            fig, ax = plt.subplots(figsize=(10, 5))
            labels, counts = zip(*sorted(cats.items(), key=lambda x: x[1]))
            bars = ax.barh(labels, counts, color="steelblue", edgecolor="black", alpha=0.8)
            ax.bar_label(bars, padding=3)
            ax.set(xlabel="Count", title="AMA Category Violations (Successful Attacks Only)")
            plt.tight_layout()
            fig.savefig(run_dir / "ama_categories.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved: ama_categories.png")

    # ── Plot 4: TTF distribution ─────────────────────────────────────────────
    ttf_values = [j.get("first_failure_turn") for j in judgments
                  if j.get("first_failure_turn") is not None]
    if ttf_values:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ttf_values, bins=range(1, 8), align="left",
                color="darkorange", edgecolor="black", alpha=0.8)
        ax.set(xlabel="Turn Number", ylabel="Count",
               title="Turns to First Safety Violation (TTF)")
        ax.set_xticks(range(1, 7))
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(run_dir / "ttf_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved: ttf_distribution.png")

    log.info("All plots saved to %s", run_dir)


# ── Reward extraction ────────────────────────────────────────────────────────

def _extract_reward(judgment: dict, reward_signal: str) -> float:
    """Return a scalar reward from the judge output based on reward_signal."""
    if reward_signal == "effectiveness":
        return float(judgment["effectiveness"])
    if reward_signal == "attack_success":
        return float(judgment["attack_success"])
    if reward_signal == "policy_violation":
        # Normalise 1–5 scale to [0, 1].
        return (float(judgment.get("policy_violation", 1)) - 1) / 4
    if reward_signal == "harmfulness":
        return float(judgment.get("harmfulness", 0.0))
    raise ValueError(f"Unknown reward_signal: {reward_signal!r}")


# ── Credit assignment ────────────────────────────────────────────────────────

def _assign_credit(
    q_tensors: list,
    r_tensors: list,
    reward: float,
    credit_assignment: str,
    gamma: float,
) -> tuple[list, list, list]:
    """Return (queries, responses, rewards) lists for PPO based on credit mode."""
    T = len(q_tensors)
    if credit_assignment == "last_turn":
        return [q_tensors[-1]], [r_tensors[-1]], [torch.tensor(reward, dtype=torch.float32)]

    if credit_assignment == "all_turns":
        rewards = [torch.tensor(reward, dtype=torch.float32) for _ in range(T)]
        return list(q_tensors), list(r_tensors), rewards

    if credit_assignment == "discounted":
        # Turn t (0-indexed) gets gamma^(T-1-t) * reward; final turn gets reward * 1.
        rewards = [
            torch.tensor(reward * (gamma ** (T - 1 - t)), dtype=torch.float32)
            for t in range(T)
        ]
        return list(q_tensors), list(r_tensors), rewards

    raise ValueError(f"Unknown credit_assignment: {credit_assignment!r}")


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
    reward_signal: str,
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
    reward = _extract_reward(judgment, reward_signal)
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

    # ── Experiment knobs ────────────────────────────────────────────────────
    max_turns         = config.get("max_turns", 5)
    credit_assignment = config.get("credit_assignment", "last_turn")
    gamma             = config.get("credit_discount_gamma", 0.9)
    reward_signal     = config.get("reward_signal", "effectiveness")
    curriculum        = config.get("curriculum", False)

    log.info(
        "Experiment config: credit_assignment=%s gamma=%.2f reward_signal=%s "
        "curriculum=%s max_turns=%d attacker_temp=%.2f",
        credit_assignment, gamma, reward_signal, curriculum, max_turns,
        config.get("attacker_temperature", 0.7),
    )

    # ── Curriculum: sort train seeds shortest-first as difficulty proxy ──────
    if curriculum:
        train_seeds = sorted(train_seeds, key=len)
        log.info("Curriculum enabled: seeds sorted by length (shortest first).")

    # ── Tokenizer + target ──────────────────────────────────────────────────
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

    # ── PPOConfig — batch sizes derived from credit assignment mode ──────────
    ppo_cfg = config["ppo"]
    n_convos = ppo_cfg.get("n_convos_per_update", 4)

    # last_turn: one (q,r) pair per conversation → batch_size = n_convos
    # all_turns / discounted: max_turns pairs per conversation → batch_size = n_convos * max_turns
    if credit_assignment == "last_turn":
        effective_batch      = n_convos
        effective_mini_batch = 1
    else:
        effective_batch      = n_convos * max_turns
        effective_mini_batch = max_turns  # one conversation's turns per mini-batch

    log.info(
        "PPO batch: n_convos=%d credit=%s → batch_size=%d mini_batch=%d",
        n_convos, credit_assignment, effective_batch, effective_mini_batch,
    )

    extra = config.get("extra_ppo_kwargs") or {}
    trl_config = PPOConfig(
        model_name=config["attacker_model_id"],
        learning_rate=ppo_cfg["lr"],
        batch_size=effective_batch,
        mini_batch_size=effective_mini_batch,
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
    log.info(
        "PPOTrainer ready: lr=%.2e batch=%d mini_batch=%d target_kl=%.2f kl_penalty=%s "
        "score_scaling=%s score_norm=%s",
        ppo_cfg["lr"], effective_batch, effective_mini_batch,
        ppo_cfg["target_kl"], ppo_cfg["kl_penalty"],
        ppo_cfg["use_score_scaling"], ppo_cfg["use_score_norm"],
    )

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
    n_steps    = ppo_cfg["n_train_steps"]
    ckpt_every = ppo_cfg["checkpoint_every"]

    seeds_cycle = train_seeds * (n_steps // max(len(train_seeds), 1) + 1)
    # Shuffle only if not using curriculum (curriculum ordering must be preserved).
    if not curriculum:
        random.shuffle(seeds_cycle)

    successes        = 0
    step             = 0
    last_running_asr = 0.0

    pbar = tqdm(total=n_steps, desc="PPO", unit="step")
    while step < n_steps:
        # batch_q / batch_r / batch_rew: per-(q,r)-pair lists passed to PPO step.
        # batch_rew_per_convo: one reward per conversation, used for logging only.
        batch_q, batch_r, batch_rew = [], [], []
        batch_rew_per_convo, batch_judg, batch_turns = [], [], []

        for _ in range(n_convos):
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
                reward_signal=reward_signal,
            )

            qs, rs, rews = _assign_credit(q_tensors, r_tensors, reward, credit_assignment, gamma)
            batch_q.extend(qs)
            batch_r.extend(rs)
            batch_rew.extend(rews)
            batch_rew_per_convo.append(reward)
            batch_judg.append(judgment)
            batch_turns.append(turns)
            step += 1
            if step >= n_steps:
                break

        if len(batch_q) < effective_batch:
            log.info("Incomplete final batch (%d/%d pairs) — stopping.",
                     len(batch_q), effective_batch)
            break

        stats = ppo_trainer.step(batch_q, batch_r, batch_rew)
        scalar_stats = {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))}

        for i, (judgment, turns) in enumerate(zip(batch_judg, batch_turns)):
            successes += int(judgment["attack_success"])
            running_asr = successes / step
            last_running_asr = running_asr

            entry = {
                "step": step - len(batch_judg) + i + 1,
                "reward": batch_rew_per_convo[i],
                "reward_signal": reward_signal,
                "credit_assignment": credit_assignment,
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

        log.info("  convo rewards=%s | running_ASR=%.3f",
                 [round(r, 3) for r in batch_rew_per_convo], last_running_asr)
        pbar.update(len(batch_judg))
        pbar.set_postfix(
            asr=f"{last_running_asr:.2f}",
            reward=f"{sum(batch_rew_per_convo) / len(batch_rew_per_convo):.2f}",
        )

        if step % ckpt_every == 0:
            ckpt = logger.artifact_path("checkpoints", f"step_{step}")
            ckpt.mkdir(parents=True, exist_ok=True)
            ppo_model.save_pretrained(str(ckpt))
            attacker_tok.save_pretrained(str(ckpt))
            log.info("Checkpoint saved: %s", ckpt)

        gc.collect()
        torch.cuda.empty_cache()

    pbar.close()
    log.info("PPO training complete. Final running ASR: %.3f", last_running_asr)

    _save_plots(logger.output_dir)

    final_ckpt = logger.artifact_path("checkpoints", "final")
    final_ckpt.mkdir(parents=True, exist_ok=True)
    ppo_model.save_pretrained(str(final_ckpt))
    attacker_tok.save_pretrained(str(final_ckpt))

    return {
        "n_train_steps": step,
        "final_running_asr": last_running_asr,
        "n_train_successes": successes,
        "split": "test" if config.get("use_test") else "dev",
        "credit_assignment": credit_assignment,
        "reward_signal": reward_signal,
        "curriculum": curriculum,
    }
