"""Prompt and inference helpers for a frozen A/B observation gate."""

from __future__ import annotations

from typing import Any

from lightnav.inference.engine import _build_prompt_dict
from lightnav.prompts import build_video_block
from lightnav.slowfast import slowfast_video_segments


GATE_PROMPT = (
    "You are a mobile robot. You are given visual observations over time, "
    "ordered from earliest to most recent: {videos}. "
    "Your assigned task is: <navigation_task>{task}</navigation_task>. "
    "You may be at the beginning, middle, or end of the task. "
    "Before predicting your next navigation action, determine whether the current "
    "visual observations provide sufficient information to decide how to proceed. "
    "You may perform an additional observation scan by stopping and rotating to "
    "observe the surrounding environment. However, this scan has a physical cost: "
    "it requires additional actions and execution time. Request it only when the "
    "current observations are insufficient or ambiguous for deciding the next "
    "navigation action. Choose exactly one option: "
    "A. The current observations are sufficient. Proceed without an additional scan. "
    "B. The current observations are insufficient or ambiguous. Perform an additional scan. "
    "Answer with only A or B."
)


def build_gate_sample(policy: Any, instruction: str) -> dict[str, Any]:
    """Build a gate query from the policy's current history without mutating it."""
    agent = policy.agent
    video_tensor = agent._get_video_tensor()
    bundle = policy.engine.bundle
    if bundle.slowfast_tiers:
        total = int(video_tensor.shape[0])
        segments = slowfast_video_segments(
            video_tensor,
            total - 1,
            total,
            bundle.slowfast_tiers,
        )
        absolute_indices = True
    else:
        total = int(video_tensor.shape[0])
        segments = [
            {
                "video": video_tensor,
                "frame_indices": list(agent._history_frame_ids),
                "total_frames": total,
                "pool_spatial": 1,
                "pool_mode": bundle.pool_mode,
            }
        ]
        absolute_indices = False

    return {
        "video_segments": segments,
        "conversations": [
            {
                "from": "human",
                "value": GATE_PROMPT.format(
                    videos=build_video_block(len(segments)),
                    task=instruction,
                ),
            },
            {"from": "gpt", "value": "placeholder"},
        ],
        "video_fps": bundle.video_fps,
        "slowfast_abs_frame_indices": absolute_indices,
        "_allow_vit_cache": False,
        "_skip_normalize": True,
    }


def free_gate_answer(policy: Any, instruction: str, max_new_tokens: int = 16) -> str:
    """Return the checkpoint's unconstrained answer to the gate prompt."""
    text, _latency_ms = policy.engine.generate(
        build_gate_sample(policy, instruction),
        max_new_tokens=max_new_tokens,
    )
    return text.strip()


def constrained_gate_answer(policy: Any, instruction: str) -> str:
    """Greedily choose exactly one tokenizer token: A or B."""
    from vllm import SamplingParams

    engine = policy.engine
    if engine.backend != "vllm_local":
        raise ValueError("constrained A/B probing currently requires backend=vllm_local")
    tokenizer = engine.bundle.tokenizer
    candidates: dict[int, str] = {}
    for label in ("A", "B"):
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"gate label {label!r} is not a single token: {token_ids}")
        candidates[int(token_ids[0])] = label
    if len(candidates) != 2:
        raise ValueError("A and B resolve to the same tokenizer id")

    result = engine.vit_forward(build_gate_sample(policy, instruction))
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        allowed_token_ids=list(candidates),
    )
    outputs = engine.vllm_engine.generate(
        [_build_prompt_dict(result.prompt_ids, result.video_embeds, result.video_grid_thw)],
        params,
        use_tqdm=False,
    )
    token_ids = list(outputs[0].outputs[0].token_ids)
    if len(token_ids) != 1 or int(token_ids[0]) not in candidates:
        raise RuntimeError(f"unexpected constrained gate output ids: {token_ids}")
    return candidates[int(token_ids[0])]
