"""DPO method wrapper. Thin adapter over the existing src.dpo.iterative_dpo_train.

The training loop itself lives in src/dpo.py and is imported as-is so we don't
fork two copies of the math. This module just:
  - loads models + LoRA + tokenizer per the config
  - resolves the seed split
  - calls iterative_dpo_train with checkpoints written under the run dir
  - keeps every adapter (per project decision; checkpoint_every iter)
"""

from __future__ import annotations

import logging

from ..dpo import iterative_dpo_train
from ..eval import evaluate_attacker
from ..judge import make_judge, JUDGE_MODEL
from ..lora_setup import attach_lora
from ..models import load_model
from ..results import ResultsLogger
from ..rollouts import make_target
from ..seeds import load_seed_splits, split_fingerprint

log = logging.getLogger("redteam_rlhf.dpo")


def run(config: dict, logger: ResultsLogger) -> dict:
    data = config["data"]
    train_seeds, dev_seeds, test_seeds = load_seed_splits(
        n_train=data["n_train"],
        n_dev=data.get("n_dev"),
        n_test=data["n_test"],
    )
    fp = split_fingerprint(train_seeds, dev_seeds, test_seeds)
    log.info("Seed split fingerprint: %s", fp)
    logger.log_jsonl("split_fingerprint", fp)

    eval_seeds = test_seeds if config.get("use_test") else dev_seeds

    # ── Models ──────────────────────────────────────────────────────────────
    log.info("Loading attacker base: %s", config["attacker_model_id"])
    attacker_tok, attacker_base = load_model(config["attacker_model_id"])

    log.info("Loading target: %s", config["target_model_id"])
    target_tok, target_mdl = load_model(config["target_model_id"])

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora = config["lora"]
    log.info("Attaching LoRA: r=%d alpha=%d targets=%s",
             lora["r"], lora["alpha"], lora["target_modules"])
    policy = attach_lora(
        attacker_base,
        r=lora["r"], alpha=lora["alpha"],
        target_modules=lora["target_modules"],
        dropout=lora["dropout"],
    )

    target = make_target(
        target_tok, target_mdl,
        max_new_tokens=config.get("target_max_new_tokens", 256),
        temperature=config.get("target_temperature", 0.7),
    )
    judge_backend = config.get("judge_backend", "openai")
    judge_model = config.get("judge_model", JUDGE_MODEL)
    log.info("Judge: backend=%s model=%s", judge_backend, judge_model)
    judge = make_judge(backend=judge_backend, model=judge_model)

    # ── Train ───────────────────────────────────────────────────────────────
    dpo_cfg = config["dpo"]
    ckpt_dir = str(logger.artifact_path("checkpoints"))
    log.info("=" * 70)
    log.info("Starting DPO: n_outer=%d n_per_seed=%d n_epochs=%d beta=%.3f lr=%.2e",
             dpo_cfg["n_outer"], dpo_cfg["n_per_seed"], dpo_cfg["n_epochs"],
             dpo_cfg["beta"], dpo_cfg["lr"])
    log.info("=" * 70)

    history = iterative_dpo_train(
        policy=policy,
        tokenizer=attacker_tok,
        target=target,
        judge=judge,
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        n_outer=dpo_cfg["n_outer"],
        n_per_seed=dpo_cfg["n_per_seed"],
        n_epochs=dpo_cfg["n_epochs"],
        beta=dpo_cfg["beta"],
        lr=dpo_cfg["lr"],
        max_turns=config.get("max_turns", 5),
        grad_accum=dpo_cfg["grad_accum"],
        checkpoint_dir=ckpt_dir,
    )

    for entry in history:
        logger.log_jsonl("training_log", entry)

    # ── Final eval on the held-out split ────────────────────────────────────
    log.info("Running final evaluation on %s split (n=%d)",
             "test" if config.get("use_test") else "dev", len(eval_seeds))
    final = evaluate_attacker(
        policy=policy, tokenizer=attacker_tok,
        target=target, judge=judge,
        seeds=eval_seeds,
        max_turns=config.get("max_turns", 5),
        verbose=True,
    )
    final_metrics = {k: v for k, v in final.items() if k != "trajectories"}
    final_metrics["split"] = "test" if config.get("use_test") else "dev"
    log.info("Final DPO metrics: %s", final_metrics)
    return final_metrics
