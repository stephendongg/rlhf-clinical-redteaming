"""Held-out evaluation of an attacker policy: ASR, TTF, mean effectiveness."""

from typing import Any, Callable

from .rollouts import Trajectory, make_attacker, run_conversation


def evaluate_attacker(
    policy: Any,
    tokenizer: Any,
    target: Callable,
    judge: Callable,
    seeds: list[str],
    max_turns: int = 5,
    verbose: bool = True,
) -> dict:
    """Run one rollout per seed under `policy`, judge each trajectory,
    return a dict with ASR, avg_TTF_successes_only, avg_effectiveness, etc.
    """
    results: list[Trajectory] = []
    for i, seed in enumerate(seeds):
        attacker = make_attacker(seed, tokenizer, policy)
        traj = run_conversation(seed, attacker, target, judge, max_turns=max_turns)
        results.append(traj)
        if verbose:
            print(
                f"[eval {i + 1}/{len(seeds)}] "
                f"success={traj.attack_success} eff={traj.effectiveness:.3f}"
            )

    n = len(results)
    successes = [r for r in results if r.attack_success]
    ttf_values = [
        r.judge.get("first_failure_turn")
        for r in successes
        if r.judge.get("first_failure_turn") is not None
    ]

    return {
        "n_trials": n,
        "n_successes": len(successes),
        "ASR": len(successes) / n if n > 0 else 0.0,
        "avg_TTF_successes_only": (sum(ttf_values) / len(ttf_values)) if ttf_values else None,
        "avg_effectiveness": sum(r.effectiveness for r in results) / n if n > 0 else 0.0,
        "trajectories": results,
    }
