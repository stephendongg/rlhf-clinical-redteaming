"""Baseline: untuned attacker vs target, judged. ASR / TTF / effectiveness.

Maps to redteam.ipynb §4. No training; this is just inference + judging.
"""

from __future__ import annotations

import logging

from tqdm.auto import tqdm

from ..models import load_model
from ..rollouts import make_attacker, make_target, run_conversation
from ..judge import make_judge, JUDGE_MODEL
from ..seeds import load_seed_splits, split_fingerprint
from ..results import ResultsLogger

log = logging.getLogger("redteam_rlhf.baseline")


def run(config: dict, logger: ResultsLogger) -> dict:
    """Execute the baseline evaluation. Returns a metrics dict."""
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
    log.info("Evaluating on %s split (n=%d)",
             "test" if config.get("use_test") else "dev", len(eval_seeds))

    # ── Load models ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Loading attacker: %s", config["attacker_model_id"])
    log.info("=" * 70)
    attacker_tok, attacker_mdl = load_model(config["attacker_model_id"])

    log.info("=" * 70)
    log.info("Loading target: %s", config["target_model_id"])
    log.info("=" * 70)
    target_tok, target_mdl = load_model(config["target_model_id"])

    target = make_target(
        target_tok, target_mdl,
        max_new_tokens=config.get("target_max_new_tokens", 256),
        temperature=config.get("target_temperature", 0.7),
    )

    judge_backend = config.get("judge_backend", "openai")
    judge_model = config.get("judge_model", JUDGE_MODEL)
    log.info("Judge: backend=%s model=%s", judge_backend, judge_model)
    judge = make_judge(backend=judge_backend, model=judge_model)

    # ── Run ─────────────────────────────────────────────────────────────────
    n = len(eval_seeds)
    successes = 0
    ttf_values: list[int] = []
    eff_sum = 0.0

    pbar = tqdm(eval_seeds, desc="baseline eval", unit="seed")
    for i, seed in enumerate(pbar, start=1):
        log.info("─" * 70)
        log.info("[%d/%d] %s", i, n, seed[:90])
        attacker = make_attacker(
            seed, attacker_tok, attacker_mdl,
            max_new_tokens=config.get("attacker_max_new_tokens", 256),
            temperature=config.get("attacker_temperature", 0.7),
        )
        traj = run_conversation(
            seed_scenario=seed,
            attacker=attacker,
            target=target,
            judge=judge,
            max_turns=config.get("max_turns", 5),
        )

        # Persist trajectory + per-trial summary.
        logger.log_jsonl("trajectories", {
            "seed_scenario": seed,
            "turns": [
                {"turn": idx + 1, "user": t.attacker_response, "assistant": t.target_response}
                for idx, t in enumerate(traj.turns)
            ],
            "judge": traj.judge,
            "attack_success": traj.attack_success,
            "effectiveness": traj.effectiveness,
        })

        if traj.attack_success:
            successes += 1
            ttf = traj.judge.get("first_failure_turn")
            if ttf is not None:
                ttf_values.append(int(ttf))
        eff_sum += traj.effectiveness

        running_asr = successes / i
        pbar.set_postfix(asr=f"{running_asr:.2f}", eff=f"{eff_sum / i:.2f}")
        log.info("  -> success=%s eff=%.3f running_ASR=%.3f",
                 traj.attack_success, traj.effectiveness, running_asr)
        logger.log_jsonl("eval_log", {
            "i": i, "n": n,
            "attack_success": traj.attack_success,
            "effectiveness": traj.effectiveness,
            "running_asr": running_asr,
        })

    metrics = {
        "n_trials": n,
        "n_successes": successes,
        "ASR": successes / n if n else 0.0,
        "avg_TTF_successes_only": (sum(ttf_values) / len(ttf_values)) if ttf_values else None,
        "avg_effectiveness": eff_sum / n if n else 0.0,
        "split": "test" if config.get("use_test") else "dev",
    }
    log.info("Final baseline metrics: %s", metrics)
    return metrics
