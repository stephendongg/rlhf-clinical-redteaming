"""Preference pair construction from a batch of trajectories."""

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
    loop does not re-forward π_ref on every gradient step.
    """
    raise NotImplementedError
