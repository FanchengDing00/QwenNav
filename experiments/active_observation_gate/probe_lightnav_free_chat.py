"""Probe whether the frozen LightNav checkpoint can answer unconstrained chat prompts.

This is deliberately isolated from both production evaluation and the active-gate
runner.  It uses the checkpoint's normal video processor but does not use any of
LightNav's navigation/tracking prompt templates or trajectory-output contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from lightnav.inference import InferenceConfig, NavigationPolicy, build_engine
from lightnav.prompts import build_video_block
from lightnav.slowfast import slowfast_video_segments


def _load_first_frames(path: str, target_fps: float, max_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or target_fps)
    stride = max(1, int(round(source_fps / target_fps)))
    frames: list[np.ndarray] = []
    source_index = 0
    while len(frames) < max_frames:
        ok, bgr = capture.read()
        if not ok:
            break
        if source_index % stride == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        source_index += 1
    capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from: {path}")
    return frames


def _free_chat_sample(policy: NavigationPolicy, user_text: str) -> dict:
    video = policy._get_video_tensor()
    bundle = policy.engine.bundle
    total = int(video.shape[0])
    if bundle.slowfast_tiers:
        segments = slowfast_video_segments(
            video, total - 1, total, bundle.slowfast_tiers
        )
        absolute_indices = True
    else:
        segments = [
            {
                "video": video,
                "frame_indices": list(policy._history_frame_ids),
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
                "value": f"{build_video_block(len(segments))} {user_text}",
            },
            {"from": "gpt", "value": "placeholder"},
        ],
        "video_fps": bundle.video_fps,
        "slowfast_abs_frame_indices": absolute_indices,
        "_allow_vit_cache": False,
        "_skip_normalize": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", default="lightnav_free_chat_probe.json")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    args = parser.parse_args()

    # The experiment is specifically testing the checkpoint's normal language
    # distribution. Refuse an inherited action-vocabulary constraint rather than
    # silently producing a meaningless negative result.
    if os.environ.get("VLN_EVAL_TRAJ_TOP1", "0").lower() in {"1", "true", "yes"}:
        raise SystemExit("unset VLN_EVAL_TRAJ_TOP1 for an unconstrained chat probe")
    os.environ["VLN_EVAL_TEMPERATURE"] = "0.0"
    os.environ["VLN_EVAL_TOP_P"] = "1.0"
    os.environ["VLN_EVAL_TOP_K"] = "0"

    engine, bundle = build_engine(
        InferenceConfig(
            model_path=args.model_path,
            backend="vllm_local",
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=1,
        ),
        task_type="vlnce",
        max_new_tokens=args.max_new_tokens,
    )
    frames = _load_first_frames(args.video, float(bundle.video_fps), args.max_frames)
    policy = NavigationPolicy(
        engine,
        num_history_frames=int(bundle.num_history_frames),
        predict_horizon=int(bundle.predict_horizon),
    )
    policy.reset()
    for frame in frames:
        policy.observe(frame)

    prompts = {
        "scene_description": (
            "Describe what is visible in this visual history and how the scene changes. "
            "Answer freely in one to three sentences."
        ),
        "gate_reasoning": (
            f"The navigation instruction is: {args.instruction!r} Based on the visual "
            "history, is there enough information to decide the next navigation move, "
            "or would an additional left-right observation scan likely help? Explain "
            "your reasoning freely in one to three sentences."
        ),
    }
    answers: dict[str, dict] = {}
    for name, prompt in prompts.items():
        sample = _free_chat_sample(policy, prompt)
        answer, latency_ms = engine.generate(sample, max_new_tokens=args.max_new_tokens)
        answers[name] = {
            "user_text": prompt,
            "raw_answer": answer,
            "latency_ms": round(float(latency_ms), 3),
        }
        print(f"\n[{name}]\n{answer}", flush=True)

    result = {
        "experiment": "lightnav_without_navigation_prompt_free_chat",
        "model_path": args.model_path,
        "video": args.video,
        "instruction": args.instruction,
        "frames": len(frames),
        "backend": "vllm_local",
        "temperature": 0.0,
        "trajectory_token_allowlist": False,
        "system_message_present": False,
        "navigation_prompt_template_used": False,
        "max_new_tokens": args.max_new_tokens,
        "answers": answers,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
