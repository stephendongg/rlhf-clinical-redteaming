"""Capture run environment: git state, package versions, GPU, CUDA, OS."""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_sha() -> str:
    return _run(["git", "rev-parse", "HEAD"]) or "unknown"


def _git_dirty() -> bool:
    """True iff there are uncommitted modifications to *tracked* files.

    Untracked files (e.g. *.egg-info from `pip install -e .`, __pycache__,
    .venv) are intentionally ignored — they don't change what code ran.
    """
    return bool(_run(["git", "status", "--porcelain", "--untracked-files=no"]))


def _git_diff() -> str:
    """Return uncommitted diff (truncated). Empty string if tree is clean."""
    return _run(["git", "diff", "HEAD"])[:200_000]


def _pip_freeze() -> list[str]:
    out = _run([sys.executable, "-m", "pip", "freeze"])
    return [line for line in out.splitlines() if line.strip()]


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpu_name": None, "cuda_version": None,
                             "n_gpus": 0, "gpu_mem_gb": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["n_gpus"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            info["gpu_mem_gb"] = round(props.total_memory / (1024 ** 3), 1)
    except ImportError:
        pass
    return info


def capture(allow_dirty: bool = False) -> dict:
    """Snapshot every reproducibility-critical fact about the environment.

    Raises if the git tree is dirty and `allow_dirty` is False.
    """
    sha = _git_sha()
    dirty = _git_dirty()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "Git tree is dirty. Commit changes or pass --allow-dirty to proceed."
        )

    env: dict[str, Any] = {
        "git_sha": sha,
        "git_dirty": dirty,
        "git_diff": _git_diff() if dirty else "",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pip_freeze": _pip_freeze(),
    }
    env.update(_gpu_info())
    return env
