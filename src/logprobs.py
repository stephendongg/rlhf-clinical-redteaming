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

    One padded forward pass over all turns of the trajectory at once. Sum is
    taken over every attacker-generated token across every turn. Result is a
    scalar tensor; gradient flows iff the model is in train mode and the
    surrounding caller did not wrap in torch.no_grad().

    A100 perf notes:
      - Prompt token IDs are read from `turn.attacker_prompt_ids` (cached at
        rollout time) — no apply_chat_template + tokenizer round-trip per
        gradient step.
      - All turns of the trajectory are padded into one (B, T) batch and
        consumed by a single model forward — vs N sequential bs=1 forwards.
      - F.cross_entropy is fused: it never materializes a full-vocab
        log-softmax matrix, and accumulates internally in fp32 from bf16
        logits without an explicit `.float()` upcast.
      - Attention mask is built from sequence lengths, *not* from
        `(batch != pad_id)`, so a real eos token inside a response is never
        accidentally masked out.
    """
    device = next(model.parameters()).device

    seqs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for turn in trajectory.turns:
        if not turn.attacker_response_ids:
            continue
        full = turn.attacker_prompt_ids + turn.attacker_response_ids
        # Mask = 1 on attacker response tokens, 0 elsewhere. We shift this
        # by one when applying it to predicted-token positions below.
        m = [0] * len(turn.attacker_prompt_ids) + [1] * len(turn.attacker_response_ids)
        seqs.append(torch.tensor(full, dtype=torch.long))
        masks.append(torch.tensor(m, dtype=torch.long))

    if not seqs:
        return torch.zeros((), device=device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    batch = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=pad_id,
    ).to(device)
    resp_mask = torch.nn.utils.rnn.pad_sequence(
        masks, batch_first=True, padding_value=0,
    ).to(device)

    seq_lens = torch.tensor([s.size(0) for s in seqs], device=device)
    arange = torch.arange(batch.size(1), device=device).unsqueeze(0)
    attn_mask = (arange < seq_lens.unsqueeze(1)).long()

    logits = model(batch, attention_mask=attn_mask).logits  # (B, T, V)

    # logits[:, i] predicts batch[:, i+1]. Slice to predict positions
    # [1 .. T-1] from logits at positions [0 .. T-2].
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = batch[:, 1:].contiguous()
    shift_resp_mask = resp_mask[:, 1:].contiguous()

    # Per-token NLL with no full log-softmax materialization.
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        reduction="none",
    ).reshape_as(shift_targets)

    return -(nll * shift_resp_mask.to(nll.dtype)).sum()


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
