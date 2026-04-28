"""Compute summed log-probabilities of attacker tokens across a multi-turn trajectory."""

from typing import Any

import torch
import torch.nn.functional as F

from .rollouts import Trajectory


def attacker_logprobs(
    model: Any,
    tokenizer: Any,
    trajectory: Trajectory,
) -> torch.Tensor:
    """Return Σ_t log π(attacker_response_t | conversation_state_t).

    Sum is taken over every attacker-generated token, across every turn of the
    trajectory. Result is a scalar tensor; gradient flows iff the model is in
    train mode and the surrounding caller did not wrap in torch.no_grad().
    """
    device = next(model.parameters()).device
    total_logp = torch.zeros((), device=device)

    for turn in trajectory.turns:
        if not turn.attacker_response_ids:
            continue

        # Reconstruct the same prompt the attacker conditioned on at generation
        # time. The default add_special_tokens=True must match rollouts.py to
        # avoid a boundary off-by-one when slicing logits to response positions.
        prompt_text = tokenizer.apply_chat_template(
            turn.attacker_prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]

        response_ids = torch.tensor(turn.attacker_response_ids, dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, response_ids]).to(device)

        logits = model(full_ids.unsqueeze(0)).logits[0]  # (T, V)

        prompt_len = prompt_ids.shape[0]
        # logits[i] predicts full_ids[i+1]. Response tokens occupy positions
        # [prompt_len .. T-1]; the predicting logits sit at [prompt_len-1 .. T-2].
        resp_logits = logits[prompt_len - 1 : -1].float()
        resp_targets = full_ids[prompt_len:]

        log_probs = F.log_softmax(resp_logits, dim=-1)
        token_logp = log_probs.gather(1, resp_targets.unsqueeze(1)).squeeze(1)
        total_logp = total_logp + token_logp.sum()

    return total_logp


def verify_logprob_roundtrip(
    model: Any,
    tokenizer: Any,
    trajectory: Trajectory,
) -> dict:
    """Diagnostic: compute attacker_logprobs and report per-turn magnitudes.

    A fluent attacker turn typically lands in [-3, -1] avg log-prob per token
    (per-token probability ~0.05-0.4). Values outside [-5, 0] usually indicate
    a tokenization mismatch with rollout time and should be debugged before
    trusting any DPO loss values.
    """
    with torch.no_grad():
        total = attacker_logprobs(model, tokenizer, trajectory).item()

    n_tokens = sum(len(t.attacker_response_ids) for t in trajectory.turns)
    avg_per_token = total / n_tokens if n_tokens > 0 else 0.0

    return {
        "n_turns": len(trajectory.turns),
        "n_response_tokens": n_tokens,
        "total_logp": total,
        "avg_logp_per_token": avg_per_token,
        "in_sane_range": -5.0 <= avg_per_token <= 0.0,
    }
