"""Trajectory-level multi-turn DPO loss and outer iterative training loop."""

from typing import Any, Callable

import torch

from .preference import PreferencePair


def dpo_loss(
    policy: Any,
    tokenizer: Any,
    pair: PreferencePair,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Trajectory-level DPO loss for one (chosen, rejected) pair.

        L = -log σ( β · ((logπ_θ(yw) - logπ_ref(yw)) - (logπ_θ(yl) - logπ_ref(yl))) )

    where logπ(·) is the sum of attacker-token log-probs over all turns
    of the trajectory (see `attacker_logprobs`). Reference log-probs are
    read from `pair.chosen_ref_logp` and `pair.rejected_ref_logp`.

    Returns (loss, metrics_dict). Metrics include chosen_logr, rejected_logr,
    margin, and pairwise accuracy.
    """
    raise NotImplementedError


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
    """Outer iterative DPO loop.

    Each outer iteration:
      1. Roll out `n_per_seed` trajectories per seed under the current policy.
      2. Build (chosen, rejected) pairs and cache reference log-probs (with
         the LoRA adapter disabled — π_ref shares base weights with π_θ).
      3. Run `n_epochs` passes over the pairs, computing dpo_loss per pair,
         backprop, AdamW step. Gradient accumulation over `grad_accum` pairs.
      4. Evaluate on `eval_seeds`, log metrics, save adapter to
         `checkpoint_dir` if provided.

    Returns the per-iteration history (list of dicts).
    """
    raise NotImplementedError
