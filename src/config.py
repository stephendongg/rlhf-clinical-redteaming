"""YAML config loader + override merge + content-addressable hash.

Contract:
  - YAML on disk is the source of truth for defaults.
  - CLI flags are folded in via `apply_overrides` as dotted-path keys.
  - The resolved dict is hashed (SHA256) so two runs with identical configs
    produce the same `config_hash` regardless of key ordering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict:
    """Load a YAML config file as a plain dict."""
    with Path(path).open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(data)}")
    return data


def parse_kv_overrides(raw_overrides: list[str]) -> dict[str, Any]:
    """Parse a list of `--override key.path=value` strings.

    Values are parsed as JSON when possible (so `lr=1e-5`, `flag=true`,
    `list=[1,2]` all work), falling back to raw string.
    """
    out: dict[str, Any] = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(f"--override must be KEY=VAL, got {raw!r}")
        key, _, val = raw.partition("=")
        try:
            out[key] = json.loads(val)
        except json.JSONDecodeError:
            out[key] = val
    return out


def apply_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    """Apply dotted-path overrides into a (deep-copied) config."""
    out = json.loads(json.dumps(config))  # cheap deep copy of JSON-serializable dict
    for dotted, value in overrides.items():
        _set_dotted(out, dotted, value)
    return out


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def hash_config(config: dict) -> str:
    """Stable SHA256 hex digest of the config (key-sorted JSON)."""
    canonical = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
