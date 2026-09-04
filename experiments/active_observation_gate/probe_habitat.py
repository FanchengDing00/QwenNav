"""Probe frozen LightNav A/B answers along one ordinary Habitat episode."""

from __future__ import annotations

import argparse
import json

from lightnav.habitat.policy import extract_instruction
from lightnav.habitat.runner import (
    HabitatEvalConfig,
    _default_engine_factory,
    _default_env_factory,
    build_velocity_policy,
)

from .gate import constrained_gate_answer, free_gate_answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server", default="tcp://localhost:5755")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--query-steps", type=int, nargs="+", default=[1, 20, 40])
    parser.add_argument("--output", default="gate_probe.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    query_steps = set(args.query_steps)
    if not query_steps or min(query_steps) < 1:
        raise SystemExit("--query-steps must contain positive navigation step numbers")

    cfg = HabitatEvalConfig(
        model_path=args.model_path,
        server=args.server,
        backend="vllm_local",
        output_dir=".",
        episodes=1,
        max_steps=max(query_steps),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    engine, bundle = _default_engine_factory(cfg)
    env = _default_env_factory(cfg)
    records: list[dict] = []
    try:
        obs, info = env.reset()
        instruction = extract_instruction(obs)
        policy = build_velocity_policy(cfg, engine, bundle, info)
        policy.reset(obs)
        for step in range(1, max(query_steps) + 1):
            action = policy.act(obs, info)
            if step in query_steps:
                free = free_gate_answer(policy, instruction)
                constrained = constrained_gate_answer(policy, instruction)
                record = {
                    "navigation_step": step,
                    "frames_observed": int(policy.agent._buffer_len),
                    "instruction": instruction,
                    "free_answer": free,
                    "constrained_answer": constrained,
                    "meaning": "proceed" if constrained == "A" else "observe",
                }
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
