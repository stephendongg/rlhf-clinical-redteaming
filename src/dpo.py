"""Trajectory-level multi-turn DPO loss and outer iterative training loop."""

import os
import random
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .eval import evaluate_attacker
from .logprobs import attacker_logprobs
from .preference import PreferencePair, build_pairs
from .rollouts import collect_rollouts, make_attacker


def dpo_loss(
    policy: Any,
    tokenizer: Any,
    pair: PreferencePair,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Trajectory-level DPO loss for one (chosen, rejected) pair.

        L = -log σ( β · ((logπ_θ(yw) - logπ_ref(yw)) - (logπ_θ(yl) - logπ_ref(yl))) )

    Reference log-probs are read from the cached fields on `pair`.
    """
    chosen_logp = attacker_logprobs(policy, tokenizer, pair.chosen)
    rejected_logp = attacker_logprobs(policy, tokenizer, pair.rejected)

    chosen_logr = chosen_logp - pair.chosen_ref_logp
    rejected_logr = rejected_logp - pair.rejected_ref_logp

    logits = beta * (chosen_logr - rejected_logr)
    loss = -F.logsigmoid(logits)

    metrics = {
        "loss": loss.item(),
        "chosen_logr": chosen_logr.item(),
        "rejected_logr": rejected_logr.item(),
        "margin": (chosen_logr - rejected_logr).item(),
        "accuracy": float((chosen_logr > rejected_logr).item()),
    }
    return loss, metrics


def iterative_dpo_train(
    policy: Any,
    tokenizer: Any,
    target: Callable,
    judge: Callable,
    train_seeds: list[str],
    eval_seeds: list[str],
    n_outer: int = 4,
    n_per_seed: int = 4,
    n_epochs: int = 2,
    beta: float = 0.1,
    lr: float = 5e-6,
    max_turns: int = 5,
    grad_accum: int = 4,
    checkpoint_dir: str | None = None,
) -> list[dict]:
    """Outer iterative DPO loop. Returns per-iteration history.

    Each outer iteration:
      1. Roll out `n_per_seed` trajectories per seed under the current policy.
      2. Build (chosen, rejected) pairs and cache reference log-probs once
         (LoRA disabled, no grad).
      3. Run `n_epochs` passes over the pairs, computing dpo_loss per pair,
         backprop, AdamW step with `grad_accum`-step accumulation.
      4. Evaluate on `eval_seeds`, log metrics, save the LoRA adapter to
         `checkpoint_dir/iter_{outer:03d}` if provided.
    """
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    def ref_logp_fn(traj):
        with torch.no_grad(), policy.disable_adapter():
            return attacker_logprobs(policy, tokenizer, traj)

    history: list[dict] = []

    for outer in range(n_outer):
        print(f"\n{'=' * 80}\nOuter iteration {outer + 1}/{n_outer}\n{'=' * 80}")

        # --- 1. Collect rollouts under the current policy ---
        policy.eval()
        attacker_factory = lambda seed: make_attacker(seed, tokenizer, policy)

        with torch.no_grad():
            trajectories = collect_rollouts(
                seeds=train_seeds,
                attacker_factory=attacker_factory,
                target=target,
                judge=judge,
                n_per_seed=n_per_seed,
                max_turns=max_turns,
                verbose=False,
            )

        n_succ = sum(1 for t in trajectories if t.attack_success)
        print(f"  collected {len(trajectories)} trajectories, {n_succ} successful")

        # --- 2. Build preference pairs (caches ref log-probs) ---
        with torch.no_grad():
            pairs = build_pairs(trajectories, ref_logp_fn)
        print(f"  built {len(pairs)} preference pairs")

        if not pairs:
            print("  ⚠️ no pairs this iteration — skipping training")
            history.append({"outer": outer, "n_trajectories": len(trajectories), "n_pairs": 0})
            continue

        # --- 3. Train over the cached pairs ---
        policy.train()
        step_metrics: dict[str, list[float]] = {
            "loss": [], "chosen_logr": [], "rejected_logr": [],
            "margin": [], "accuracy": [],
        }

        for epoch in range(n_epochs):
            random.shuffle(pairs)
            optimizer.zero_grad()
            for i, pair in enumerate(pairs):
                loss, m = dpo_loss(policy, tokenizer, pair, beta=beta)
                (loss / grad_accum).backward()

                if (i + 1) % grad_accum == 0 or (i + 1) == len(pairs):
                    optimizer.step()
                    optimizer.zero_grad()

                for k, v in m.items():
                    step_metrics[k].append(v)

            recent = step_metrics["loss"][-len(pairs):]
            recent_acc = step_metrics["accuracy"][-len(pairs):]
            print(
                f"  epoch {epoch + 1}/{n_epochs}: "
                f"avg_loss={sum(recent) / len(recent):.4f} "
                f"avg_acc={sum(recent_acc) / len(recent_acc):.2f}"
            )

        # --- 4. Held-out eval ---
        policy.eval()
        eval_metrics = evaluate_attacker(
            policy=policy, tokenizer=tokenizer,
            target=target, judge=judge,
            seeds=eval_seeds, max_turns=max_turns, verbose=False,
        )
        print(
            f"  eval ASR={eval_metrics['ASR']:.3f} "
            f"avg_eff={eval_metrics['avg_effectiveness']:.3f}"
        )

        # --- 5. Checkpoint ---
        if checkpoint_dir is not None:
            ckpt_path = os.path.join(checkpoint_dir, f"iter_{outer:03d}")
            os.makedirs(ckpt_path, exist_ok=True)
            policy.save_pretrained(ckpt_path)
            print(f"  saved adapter to {ckpt_path}")

        history.append({
            "outer": outer,
            "n_trajectories": len(trajectories),
            "n_pairs": len(pairs),
            "train_avg_loss": sum(step_metrics["loss"]) / len(step_metrics["loss"]),
            "train_avg_accuracy": sum(step_metrics["accuracy"]) / len(step_metrics["accuracy"]),
            "train_mean_chosen_logr": sum(step_metrics["chosen_logr"]) / len(step_metrics["chosen_logr"]),
            "train_mean_rejected_logr": sum(step_metrics["rejected_logr"]) / len(step_metrics["rejected_logr"]),
            **{f"eval_{k}": v for k, v in eval_metrics.items() if k != "trajectories"},
        })

    return history
