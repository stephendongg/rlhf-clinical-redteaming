"""Local HuggingFace judge backend.

Stateful: loads the judge model once and reuses it across calls. Same return
shape as the OpenAI backend (`src.judge.judge_trajectory`) so methods can swap
backends via config without changing call sites.

Default model (recommended): meta-llama/Llama-3.1-8B-Instruct
Alternative (ungated):        Qwen/Qwen2.5-14B-Instruct

GPU memory: a 7-8B judge at 4-bit nf4 is ~5-6 GB. Fits alongside attacker +
target on a 40 GB A100 for baseline / DPO. PPO already holds 3 models
(policy + ref + target) so adding the HF judge will likely OOM on 40 GB —
use the OpenAI backend for PPO, or upgrade to an 80 GB A100.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .judge import TRAJECTORY_JUDGE_PROMPT, score_trajectory_judgment
from .models import load_model

log = logging.getLogger("redteam_rlhf.judge_hf")


def _extract_json(text: str) -> dict:
    """Extract the first balanced {...} JSON object from `text`.

    Robust to: leading prose, trailing prose, markdown ``` fences, and
    extra whitespace. Raises ValueError if no balanced object is found.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    start = text.find("{")
    if start == -1:
        raise ValueError("no '{' found in judge output")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in judge output")


def _coerce(raw: dict) -> dict:
    """Coerce HF-judge output into the schema score_trajectory_judgment expects.

    Local models are messy: they return strings ("1"), floats (3.0), nulls
    (when they think a field is "not applicable"), or out-of-range values.
    We coerce defensively. Missing or null rubric scores default to 1
    (= "fully safe" per the AMA-9 rubric), which is the right semantic when
    the judge couldn't decide. Range-clamped to [1, 5].
    """
    required = ["attack_success", "policy_violation", "convincingness", "specificity"]
    for k in required:
        if k not in raw:
            raise KeyError(f"judge output missing required field: {k}")

    def _to_int(v: Any, default: int) -> int:
        if v is None:
            return default
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    out = dict(raw)
    out["attack_success"] = int(bool(_to_int(raw["attack_success"], 0)))
    out["policy_violation"] = max(1, min(5, _to_int(raw["policy_violation"], 1)))
    out["convincingness"]   = max(1, min(5, _to_int(raw["convincingness"], 1)))
    out["specificity"]      = max(1, min(5, _to_int(raw["specificity"], 1)))
    out.setdefault("first_failure_turn", None)
    out.setdefault("ama_category", "none")
    out.setdefault("reason", "")
    return out


def _zero_reward_judgment(reason: str) -> dict:
    """Same fallback shape as src.judge.judge_trajectory_with_retry."""
    return {
        "attack_success": False,
        "first_failure_turn": None,
        "policy_violation": 1,
        "ama_category": "none",
        "convincingness": 1,
        "specificity": 1,
        "reason": f"hf judge failure: {reason}",
        "harmfulness": 0.0,
        "effectiveness": 0.0,
        "safe": True,
        "borderline": False,
    }


class HFJudge:
    """Stateful HF-local judge.

    Construct once per run; call `(seed_scenario, turns)` per trajectory.
    Greedy decoding (`do_sample=False`) for deterministic judgments.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens: int = 400,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        log.info("Loading HF judge: %s", model_id)
        self.tokenizer, self.model = load_model(model_id)

    def __call__(self, seed_scenario: str, turns: list) -> dict:
        prompt = TRAJECTORY_JUDGE_PROMPT.format(
            seed_scenario=seed_scenario,
            trajectory=json.dumps(turns, indent=2),
        )
        # Cap at one retry: on first parse failure we re-prompt with an
        # explicit "JSON only" prefix; on second failure we return the
        # zero-reward judgment so a single judge hiccup doesn't abort training.
        attempts = [
            prompt,
            "Respond with valid JSON only. No prose, no code fences.\n\n" + prompt,
        ]
        last_err: Exception | None = None
        for attempt_text in attempts:
            try:
                raw_output = self._generate(attempt_text)
                parsed = _extract_json(raw_output)
                coerced = _coerce(parsed)
                return score_trajectory_judgment(coerced)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("HF judge parse failed (%s); retrying with strict prompt", e)

        log.error("HF judge gave up after retries: %s", last_err)
        return _zero_reward_judgment(repr(last_err))

    def _generate(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)
