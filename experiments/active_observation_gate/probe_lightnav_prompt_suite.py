"""Compare prompt styles for using frozen LightNav as a language-level gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from lightnav.inference import InferenceConfig, NavigationPolicy, VitResult, build_engine

from .probe_lightnav_free_chat import _free_chat_sample, _load_first_frames


LABELS = {"ADEQUATE", "MORE_VIEW", "UNCERTAIN"}
ACTION_TOKEN_RE = re.compile(r"<(?:apos|opos|act_l\d+|traj)_")


def _prompts(instruction: str) -> list[dict]:
    return [
        {
            "id": "p0_direct_navigation_free",
            "max_new_tokens": 64,
            "text": (
                f"The navigation instruction is: {instruction!r} Based on the visual "
                "history, is there enough information to decide the next navigation move, "
                "or would an additional left-right observation scan likely help? Explain "
                "your reasoning freely in one to three sentences."
            ),
        },
        {
            "id": "p1_visual_evidence_free",
            "max_new_tokens": 64,
            "text": (
                f"Reference description: {instruction!r} Review the visual evidence only. "
                "Is the currently visible evidence sufficient to make a confident choice, "
                "or are important surrounding areas still unseen? Answer in ordinary "
                "English in one or two sentences. Do not produce coordinates, control "
                "commands, special tokens, a route, or a future trajectory."
            ),
        },
        {
            "id": "p2_coverage_auditor_free",
            "max_new_tokens": 64,
            "text": (
                "This is a visual coverage audit, not a robot-control request. "
                f"Reference description: {instruction!r} Assess only whether the images "
                "provide adequate visual coverage, whether seeing more surrounding areas "
                "would be useful, or whether the evidence is too uncertain to judge. "
                "Explain briefly in ordinary English. Do not output a path, movement, "
                "coordinates, control commands, or special tokens."
            ),
        },
        {
            "id": "p3_coverage_labels_described",
            "max_new_tokens": 8,
            "text": (
                "This is visual coverage classification, not robot control. "
                f"Reference description: {instruction!r} Choose ADEQUATE if the images "
                "contain enough relevant visual evidence, MORE_VIEW if seeing currently "
                "unseen surroundings would likely resolve missing evidence, or UNCERTAIN "
                "if this cannot be judged. Reply with exactly ADEQUATE, MORE_VIEW, or "
                "UNCERTAIN and no other text."
            ),
        },
        {
            "id": "p4_coverage_labels_minimal",
            "max_new_tokens": 8,
            "text": (
                f"Reference description: {instruction!r} Classify the visual coverage. "
                "Reply with one label only: ADEQUATE, MORE_VIEW, or UNCERTAIN."
            ),
        },
        {
            "id": "p5_system_auditor_free",
            "max_new_tokens": 64,
            "system": (
                "You are a visual-evidence auditor, not a robot controller. Respond in "
                "ordinary English. Never predict a route, movement, coordinates, future "
                "trajectory, or robot-control tokens."
            ),
            "text": (
                f"Reference description: {instruction!r} Do the images provide enough "
                "relevant visual coverage, would seeing more surrounding areas be useful, "
                "or is the evidence too uncertain to judge? Explain briefly."
            ),
        },
        {
            "id": "p6_system_auditor_labels",
            "max_new_tokens": 8,
            "system": (
                "You are a visual-evidence classifier, not a robot controller. Never "
                "predict a route, movement, coordinates, future trajectory, or special "
                "robot-control tokens. Follow the requested answer format exactly."
            ),
            "text": (
                f"Reference description: {instruction!r} Classify the visual coverage. "
                "ADEQUATE means enough relevant evidence is visible. MORE_VIEW means unseen "
                "surroundings would likely resolve missing evidence. UNCERTAIN means this "
                "cannot be judged. Reply with exactly one label: ADEQUATE, MORE_VIEW, or "
                "UNCERTAIN."
            ),
        },
        {
            "id": "p7_caption_then_text_gate",
            "max_new_tokens": 8,
            "two_stage": True,
            "system": (
                "You are a visual-evidence classifier. You receive a reference "
                "description and a textual summary of observed images. Classify only "
                "whether the summary contains enough relevant evidence. Never produce "
                "robot-control tokens. Follow the requested output format exactly."
            ),
            "text": "",
        },
    ]


def _classify_output(text: str) -> dict:
    stripped = text.strip()
    normalized = stripped.upper()
    return {
        "exact_label": normalized if normalized in LABELS else None,
        "contains_action_token": bool(ACTION_TOKEN_RE.search(stripped)),
        "empty": not bool(stripped),
    }


def _enable_probe_system_role() -> None:
    """Patch only this short-lived process so samples may contain a system turn.

    Production LightNav samples never contain a system turn and the repository's
    shared data processor is not edited by this experiment.
    """
    import lightnav.data_processor as data_processor

    def build_messages(item: dict[str, Any], videos: list[Any]) -> list[dict[str, Any]]:
        video_pool = [{"type": "video", "video": video} for video in videos]
        messages: list[dict[str, Any]] = []
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        for turn in item["conversations"]:
            source = str(turn["from"])
            if source not in role_map:
                raise ValueError(f"unsupported probe conversation role: {source}")
            role = role_map[source]
            text = str(turn["value"])
            content = []
            for part in re.split(r"(<video>)", text):
                if part == "<video>":
                    if role != "user" or not video_pool:
                        raise ValueError("video placeholders are permitted only in user turns")
                    content.append(video_pool.pop(0))
                elif part.strip():
                    content.append({"type": "text", "text": part.strip()})
            messages.append({"role": role, "content": content})
        if video_pool:
            raise ValueError(f"{len(video_pool)} unused probe videos")
        return messages

    data_processor._build_messages = build_messages


def _text_only_generate(engine, system: str, user: str, max_new_tokens: int) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt_ids = engine.bundle.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    started = time.monotonic()
    answer = engine.llm_generate_batch(
        [VitResult(prompt_ids=list(prompt_ids), video_embeds=None, video_grid_thw=None)],
        max_new_tokens,
    )[0]
    return answer, (time.monotonic() - started) * 1000.0


def _load_cases(results_jsonl: str, episode_ids: list[str], video_root: str) -> list[dict]:
    wanted = set(episode_ids)
    cases: dict[str, dict] = {}
    with open(results_jsonl, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            episode_id = str(record.get("habitat_episode_id", ""))
            if episode_id not in wanted:
                continue
            video_rel = record.get("video")
            if not video_rel:
                raise ValueError(f"episode {episode_id} has no video field")
            cases[episode_id] = {
                "episode_id": episode_id,
                "instruction": str(record.get("instruction", "")),
                "video": str(Path(video_root) / Path(video_rel).name),
                "baseline_success": bool(record.get("success", False)),
            }
    missing = wanted - set(cases)
    if missing:
        raise ValueError(f"episodes absent from results: {sorted(missing)}")
    return [cases[episode_id] for episode_id in episode_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--episode-ids", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--prompt-ids", nargs="*", default=None)
    args = parser.parse_args()

    if os.environ.get("VLN_EVAL_TRAJ_TOP1", "0").lower() in {"1", "true", "yes"}:
        raise SystemExit("unset VLN_EVAL_TRAJ_TOP1 for an unconstrained prompt suite")
    os.environ["VLN_EVAL_TEMPERATURE"] = "0.0"
    os.environ["VLN_EVAL_TOP_P"] = "1.0"
    os.environ["VLN_EVAL_TOP_K"] = "0"
    _enable_probe_system_role()

    engine, bundle = build_engine(
        InferenceConfig(
            model_path=args.model_path,
            backend="vllm_local",
            max_new_tokens=64,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=1,
        ),
        task_type="vlnce",
        max_new_tokens=64,
    )
    cases = _load_cases(args.results_jsonl, args.episode_ids, args.video_root)
    records: list[dict] = []
    for case in cases:
        engine.reset_episode_state()
        frames = _load_first_frames(case["video"], float(bundle.video_fps), args.max_frames)
        policy = NavigationPolicy(
            engine,
            num_history_frames=int(bundle.num_history_frames),
            predict_horizon=int(bundle.predict_horizon),
        )
        policy.reset()
        for frame in frames:
            policy.observe(frame)
        print(f"\n===== episode {case['episode_id']} =====", flush=True)
        answers = []
        prompts = _prompts(case["instruction"])
        if args.prompt_ids:
            wanted_prompts = set(args.prompt_ids)
            prompts = [prompt for prompt in prompts if prompt["id"] in wanted_prompts]
            missing_prompts = wanted_prompts - {prompt["id"] for prompt in prompts}
            if missing_prompts:
                raise ValueError(f"unknown prompt ids: {sorted(missing_prompts)}")
        for prompt in prompts:
            intermediate_caption = None
            if prompt.get("two_stage"):
                caption_sample = _free_chat_sample(
                    policy,
                    "Describe the visible places, objects, openings, and changes across "
                    "these images in ordinary English. Do not suggest any movement or route.",
                )
                caption_sample["_allow_vit_cache"] = True
                intermediate_caption, caption_latency_ms = engine.generate(
                    caption_sample, max_new_tokens=96
                )
                user_text = (
                    f"Reference description: {case['instruction']!r}\n"
                    f"Observed-image summary: {intermediate_caption!r}\n"
                    "Reply with ADEQUATE if this summary contains enough relevant visual "
                    "evidence, MORE_VIEW if unseen surroundings would likely add important "
                    "missing evidence, or UNCERTAIN if this cannot be judged. Reply with "
                    "exactly one label and no other text."
                )
                answer, latency_ms = _text_only_generate(
                    engine, prompt["system"], user_text, int(prompt["max_new_tokens"])
                )
                latency_ms += caption_latency_ms
            else:
                user_text = prompt["text"]
                sample = _free_chat_sample(policy, user_text)
                if prompt.get("system"):
                    sample["conversations"].insert(
                        0, {"from": "system", "value": prompt["system"]}
                    )
                sample["_allow_vit_cache"] = True
                answer, latency_ms = engine.generate(
                    sample, max_new_tokens=int(prompt["max_new_tokens"])
                )
            parsed = _classify_output(answer)
            row = {
                "prompt_id": prompt["id"],
                "prompt": user_text,
                "system_prompt": prompt.get("system"),
                "max_new_tokens": prompt["max_new_tokens"],
                "intermediate_caption": intermediate_caption,
                "raw_answer": answer,
                **parsed,
                "latency_ms": round(float(latency_ms), 3),
            }
            answers.append(row)
            print(
                f"[{prompt['id']}] label={parsed['exact_label']} "
                f"action={parsed['contains_action_token']}\n{answer}",
                flush=True,
            )
        records.append({**case, "frames": len(frames), "answers": answers})

    prompt_ids = [prompt["id"] for prompt in _prompts("")]
    if args.prompt_ids:
        prompt_ids = [prompt_id for prompt_id in prompt_ids if prompt_id in args.prompt_ids]
    aggregate = {}
    for prompt_id in prompt_ids:
        rows = [
            answer
            for record in records
            for answer in record["answers"]
            if answer["prompt_id"] == prompt_id
        ]
        aggregate[prompt_id] = {
            "cases": len(rows),
            "natural_language_count": sum(
                not row["contains_action_token"] and not row["empty"] for row in rows
            ),
            "action_token_count": sum(row["contains_action_token"] for row in rows),
            "exact_label_count": sum(row["exact_label"] is not None for row in rows),
        }
    result = {
        "experiment": "lightnav_language_gate_prompt_suite",
        "model_path": args.model_path,
        "temperature": 0.0,
        "trajectory_token_allowlist": False,
        "system_message_present": any(
            answer.get("system_prompt")
            for record in records
            for answer in record["answers"]
        ),
        "navigation_prompt_template_used": False,
        "aggregate": aggregate,
        "cases": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
