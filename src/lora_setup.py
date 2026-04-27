"""LoRA attachment for the 4-bit quantized attacker policy."""

from typing import Any


def attach_lora(
    base_model: Any,
    r: int = 16,
    alpha: int = 32,
    target_modules: list[str] | None = None,
    dropout: float = 0.05,
) -> Any:
    """Wrap a 4-bit base model with trainable LoRA adapters via peft.

    The reference policy is the same returned object with the adapter
    disabled — use `with policy.disable_adapter(): ...` at log-prob time
    to access π_ref without loading a second copy of the base weights.
    """
    raise NotImplementedError
