"""Compute summed log-probabilities of attacker tokens across a multi-turn trajectory."""

from typing import Any

import torch

from .rollouts import Trajectory


def attacker_logprobs(
    model: Any,
    tokenizer: Any,
    trajectory: Trajectory,
) -> torch.Tensor:
    """Return Σ_t log π(attacker_response_t | conversation_state_t).

    Sum is taken over every attacker-generated token, across every turn.
    Result is a scalar tensor; gradient flows iff the model is in train mode.

    Implementation contract:
      - Reconstruct the prompt from `turn.attacker_prompt_messages` via the
        tokenizer's chat template, with `add_generation_prompt=True`.
      - Tokenize prompt and response separately to know the boundary.
      - Mask all non-attacker positions out of the sum.
    """
    raise NotImplementedError


def verify_logprob_roundtrip(
    model: Any,
    tokenizer: Any,
    trajectory: Trajectory,
    tol_per_token: float = 1e-2,
) -> dict:
    """Sanity check called during development.

    Recomputes attacker_logprobs on `trajectory` against `model` and
    returns diagnostics: per-turn log-prob, per-token log-prob, total.
    Use this to confirm tokenization round-trips correctly before
    trusting any DPO loss values.
    """
    raise NotImplementedError
