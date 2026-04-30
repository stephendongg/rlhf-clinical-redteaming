"""Deterministic seed splits shared across baseline / DPO / PPO experiments.

Both `niki/dpo` and `audrey` must import from this module so the three
experimental conditions are evaluated on the *exact* same train / dev / test
seed scenarios.

Reproducibility contract:
  - Source dataset:    DATASET_ID
  - Train/dev split:   sklearn.train_test_split with random_state=SPLIT_SEED
  - Sub-sampling:      random.Random(SUBSAMPLE_SEED).sample — a *local* RNG
                       so prior cells / imports cannot poison the state.
"""

import random
from typing import Optional

DATASET_ID = "israel-adewuyi/med-safety-bench-reproduced"
SPLIT_SEED = 42
DEV_FRAC = 0.15
SUBSAMPLE_SEED = 42


def load_seed_splits(
    n_train: int = 100,
    n_test: int = 100,
    n_dev: Optional[int] = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (train_seeds, dev_seeds, test_seeds), deterministic across runs.

    Args:
        n_train: number of train seeds to sample from the post-split RL train pool.
        n_test:  number of test seeds to sample from the official test set.
        n_dev:   if None, returns the full dev pool unsampled; else sub-samples.

    The same (n_train, n_test, n_dev) tuple is guaranteed to produce identical
    seed lists on any machine, regardless of the global `random` state.
    """
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    ds = load_dataset(DATASET_ID)
    df_train = ds["train"].to_pandas()
    df_test = ds["test"].to_pandas()

    df_rl_train, df_dev = train_test_split(
        df_train,
        test_size=DEV_FRAC,
        random_state=SPLIT_SEED,
        shuffle=True,
    )
    train_pool = df_rl_train["harmful_medical_request"].tolist()
    dev_pool = df_dev["harmful_medical_request"].tolist()
    test_pool = df_test["harmful_medical_request"].tolist()

    rng = random.Random(SUBSAMPLE_SEED)
    train_seeds = rng.sample(train_pool, n_train)
    test_seeds = rng.sample(test_pool, n_test)
    dev_seeds = rng.sample(dev_pool, n_dev) if n_dev is not None else dev_pool

    return train_seeds, dev_seeds, test_seeds


def split_fingerprint(
    train_seeds: list[str],
    dev_seeds: list[str],
    test_seeds: list[str],
) -> dict:
    """Stable hashes for cross-branch verification.

    Run this on each branch in a fresh kernel; all six hashes must match for
    baseline / DPO / PPO results to be directly comparable.
    """
    import hashlib

    def _hash(seqs: list[str]) -> str:
        h = hashlib.sha256()
        for s in seqs:
            h.update(s.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()[:16]

    return {
        "n_train": len(train_seeds),
        "n_dev": len(dev_seeds),
        "n_test": len(test_seeds),
        "train_hash": _hash(train_seeds),
        "dev_hash": _hash(dev_seeds),
        "test_hash": _hash(test_seeds),
    }
