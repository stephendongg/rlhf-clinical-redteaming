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
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Required when training on a bitsandbytes 4-bit base: casts layer norms
    # and lm_head to fp32 and enables gradient checkpointing. Without this,
    # gradients silently fail to flow through the quantized layers.
    base_model = prepare_model_for_kbit_training(base_model)

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    policy = get_peft_model(base_model, config)
    policy.print_trainable_parameters()
    return policy
