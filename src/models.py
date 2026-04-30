"""Model loaders shared across baseline / DPO / PPO.

Extracted from redteam.ipynb §2 + ppo_training.ipynb §2.
A100-tuned: bf16 + nf4 + double-quant + SDPA attention.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import torch

log = logging.getLogger("redteam_rlhf.models")


def load_model(model_id: str) -> Tuple[Any, Any]:
    """Load a HF causal LM with 4-bit nf4 quant + bf16 compute.

    Notes (from existing project commits):
      - bf16 compute: A100 hits 312 TFLOPS, no loss-scaling, wider exponent.
      - nf4 quant_type: bitsandbytes' tuned 4-bit kernels.
      - double_quant: ~0.4 bits/param of additional savings.
      - SDPA attention: dispatches to FA1 on A100 + bf16 — no FA2 install.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    kwargs: dict[str, Any] = {
        "quantization_config": bnb,
        "device_map": "auto",
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    # BioMistral safetensors auto-conversion bug — see ppo_training.ipynb commits.
    if "BioMistral" in model_id:
        kwargs["use_safetensors"] = False

    log.info("Loading model: %s (4-bit nf4 + bf16 + sdpa)", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return tokenizer, model
