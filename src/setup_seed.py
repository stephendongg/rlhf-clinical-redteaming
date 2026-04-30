"""Seed every RNG and toggle CUDA determinism flags."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, torch (CPU + CUDA) and set CUDA determinism flags.

    NB: full determinism on CUDA disables some fast kernels and will slow
    training. We set `deterministic=True` and `benchmark=False`, which is the
    standard "reproducible" stance for RLHF research code.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
