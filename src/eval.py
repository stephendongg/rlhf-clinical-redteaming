"""Held-out evaluation of an attacker policy: ASR, TTF, mean effectiveness."""

from typing import Any, Callable


def evaluate_attacker(
    policy: Any,
    tokenizer: Any,
    target: Callable,
    judge: Callable,
    seeds: list[str],
    max_turns: int = 5,
) -> dict:
    """Run one rollout per seed under `policy`, judge each trajectory,
    return a dict with keys: ASR, avg_TTF_successes_only, avg_effectiveness,
    n_trials, n_successes, trajectories.
    """
    raise NotImplementedError
