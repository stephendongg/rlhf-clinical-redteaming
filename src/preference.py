"""Preference pair construction from a batch of trajectories."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import torch

from .rollouts import Trajectory


@dataclass
class PreferencePair:
    seed: str
    chosen: Trajectory
    rejected: Trajectory
    chosen_ref_logp: float | None = None
    rejected_ref_logp: float | None = None


def build_pairs(
    trajectories: list[Trajectory],
    ref_logprob_fn: Callable[[Trajectory], torch.Tensor],
    strategy: str = "best_vs_worst",
) -> list[PreferencePair]:
    """Group trajectories by seed, form pairs, cache reference log-probs.

    Strategies:
      - "best_vs_worst": within each seed, pair the highest-effectiveness
        trajectory with the lowest. Skip seeds where all trajectories tie.

    Reference log-probs are cached at construction so the inner training
    loop does not re-forward π_ref on every gradient step. The caller
    supplies `ref_logprob_fn` as a closure that already wraps the model in
    a `disable_adapter()` context (see Cell 5.4 for the standard pattern).
    """
    if strategy != "best_vs_worst":
        raise ValueError(f"unknown strategy: {strategy!r}")

    by_seed: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_seed[t.seed_scenario].append(t)

    pairs: list[PreferencePair] = []
    for seed, trajs in by_seed.items():
        if len(trajs) < 2:
            continue
        srt = sorted(trajs, key=lambda t: t.effectiveness)
        chosen, rejected = srt[-1], srt[0]
        if chosen.effectiveness == rejected.effectiveness:
            continue
        pair = PreferencePair(seed=seed, chosen=chosen, rejected=rejected)
        pair.chosen_ref_logp = float(ref_logprob_fn(chosen).item())
        pair.rejected_ref_logp = float(ref_logprob_fn(rejected).item())
        pairs.append(pair)

    return pairs
