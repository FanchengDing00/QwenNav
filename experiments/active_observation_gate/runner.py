"""Independent Habitat loop with a remote frozen-Qwen observation gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lightnav.habitat.policy import extract_instruction
from lightnav.habitat.results import make_json_safe, print_vlnce_summary
from lightnav.habitat.runner import (
    HabitatEvalConfig,
    _EvalVisualizer,
    _default_engine_factory,
    _default_env_factory,
    _format_action,
    _safe_scalar,
    build_velocity_policy,
)
from lightnav.velocity import first_waypoint_to_velocity_cmd

from .gate_rpc import GateClient
from .sampling import select_gate_frames


def _draw_gate_overlay(
    frame: np.ndarray,
    *,
    decision: str,
    query_index: int,
    observation_added: bool,
) -> np.ndarray:
    """Draw the latest gate decision without changing the official HUD renderer."""
    import cv2

    out = np.ascontiguousarray(frame).copy()
    height, width = out.shape[:2]
    scale = max(0.55, height / 490.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(scale * 1.35)))
    decision_text = (
        "GATE: NOT_QUERIED"
        if query_index <= 0
        else f"GATE #{query_index}: {decision}"
    )
    observation_text = f"EXTRA OBS: {'YES' if observation_added else 'NO'}"
    lines = (decision_text, observation_text)
    text_sizes = [
        cv2.getTextSize(line, font, scale, thickness)[0] for line in lines
    ]
    pad = max(5, int(round(8 * scale)))
    line_gap = max(4, int(round(7 * scale)))
    panel_width = max(size[0] for size in text_sizes) + 2 * pad
    panel_height = sum(size[1] for size in text_sizes) + line_gap + 2 * pad
    x0 = max(0, width - panel_width - pad)
    y0 = max(0, height - panel_height - pad)
    x1 = min(width, x0 + panel_width)
    y1 = min(height, y0 + panel_height)

    panel = out[y0:y1, x0:x1].copy()
    panel[:] = (8, 18, 24)
    out[y0:y1, x0:x1] = cv2.addWeighted(
        out[y0:y1, x0:x1], 0.18, panel, 0.82, 0.0
    )
    border_colour = {
        "NEED": (255, 165, 70),
        "NO_NEED": (65, 225, 165),
        "UNKNOWN": (255, 215, 90),
        "INVALID": (80, 80, 255),
    }.get(decision, (125, 155, 170))
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), border_colour, thickness)

    baseline_y = y0 + pad + text_sizes[0][1]
    cv2.putText(
        out,
        decision_text,
        (x0 + pad, baseline_y),
        font,
        scale,
        border_colour,
        thickness,
        cv2.LINE_AA,
    )
    observation_colour = (255, 125, 90) if observation_added else (185, 205, 210)
    cv2.putText(
        out,
        observation_text,
        (x0 + pad, baseline_y + line_gap + text_sizes[1][1]),
        font,
        scale,
        observation_colour,
        thickness,
        cv2.LINE_AA,
    )
    return out


class _ActiveGateVisualizer(_EvalVisualizer):
    """Experiment-only visualizer that exposes the latest gate state in videos."""

    def __init__(self, cfg: HabitatEvalConfig, output_dir: str) -> None:
        super().__init__(cfg, output_dir)
        self._gate_decision = "NOT_QUERIED"
        self._gate_query_index = 0
        self._gate_observation_added = False

    def begin_episode(self, habitat_episode_id: str) -> None:
        self._gate_decision = "NOT_QUERIED"
        self._gate_query_index = 0
        self._gate_observation_added = False
        super().begin_episode(habitat_episode_id)

    def set_gate_state(
        self, *, decision: str, query_index: int, observation_added: bool
    ) -> None:
        self._gate_decision = decision
        self._gate_query_index = query_index
        self._gate_observation_added = observation_added

    def _render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        frame = super()._render(*args, **kwargs)
        return _draw_gate_overlay(
            frame,
            decision=self._gate_decision,
            query_index=self._gate_query_index,
            observation_added=self._gate_observation_added,
        )


@dataclass(frozen=True)
class ActiveGateConfig:
    gate_server: str = "tcp://localhost:6755"
    horizon_multiplier: int = 2
    temporal_stride: int = 4
    dense_history_limit: int = 16
    max_gate_frames: int = 20
    video_fps: float = 4.0
    gate_frame_height: int = 224
    gate_frame_width: int = 384
    unknown_policy: str = "skip"  # skip | scan
    order_seed: int = 0
    side_yaw_degrees: int = 90
    rotation_step_degrees: int = 30
    jpeg_quality: int = 85
    timeout_ms: int = 180000

    def __post_init__(self) -> None:
        if self.horizon_multiplier < 1 or self.temporal_stride < 1:
            raise ValueError("horizon_multiplier and temporal_stride must be >= 1")
        if self.dense_history_limit < 2 or self.max_gate_frames < 2:
            raise ValueError("gate frame limits must be >= 2")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if self.unknown_policy not in {"skip", "scan"}:
            raise ValueError("unknown_policy must be 'skip' or 'scan'")
        if self.side_yaw_degrees % self.rotation_step_degrees:
            raise ValueError("side_yaw_degrees must be divisible by rotation_step_degrees")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")


def _scan_direction(config: ActiveGateConfig, episode_key: str, event_index: int) -> str:
    payload = f"{config.order_seed}|{episode_key}|{event_index}|gate".encode("utf-8")
    return "left_first" if hashlib.sha256(payload).digest()[0] & 1 else "right_first"


def _rotation_plan(direction: str, config: ActiveGateConfig) -> list[int]:
    side_steps = config.side_yaw_degrees // config.rotation_step_degrees
    first = 1 if direction == "left_first" else -1
    return [first] * side_steps + [-first] * (2 * side_steps) + [first] * side_steps


def _rotation_action(policy: Any, sign: int, config: ActiveGateConfig) -> dict[str, Any]:
    waypoint = np.asarray(
        [0.0, 0.0, math.radians(sign * config.rotation_step_degrees)],
        dtype=np.float32,
    )
    return first_waypoint_to_velocity_cmd(
        waypoint, policy.dt, policy.lin_vel_range, policy.ang_vel_range
    )


def _rotation_policy_info(sign: int, config: ActiveGateConfig) -> dict[str, Any]:
    yaw = math.radians(sign * config.rotation_step_degrees)
    return {
        "predicted_traj": np.asarray([[0.0, 0.0, yaw]], dtype=np.float32),
        "raw_text": f"gate_scan_rotate_{'left' if sign > 0 else 'right'}",
    }


def _gate_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = {key: 0 for key in ("NEED", "NO_NEED", "UNKNOWN", "INVALID")}
    for event in events:
        decisions[event["decision"]] += 1
    total = len(events)
    return {
        "query_count": len(events),
        "decision_counts": decisions,
        "decision_ratios": {
            key: round(count / total, 6) if total else 0.0
            for key, count in decisions.items()
        },
        "invalid_format_count": sum(not event["valid_format"] for event in events),
        "scan_count": sum(bool(event["scan_executed"]) for event in events),
        "rotation_action_count": sum(int(event["rotation_actions"]) for event in events),
        "events": events,
    }


def gate_due(
    completed_navigation_actions: int,
    resolved_interval: int,
    last_queried_action: int,
) -> bool:
    """Trigger before action 1, then after every N completed navigation actions."""
    return bool(
        completed_navigation_actions >= 0
        and completed_navigation_actions % resolved_interval == 0
        and completed_navigation_actions != last_queried_action
    )


def run_active_gate_eval(
    cfg: HabitatEvalConfig, gate_cfg: ActiveGateConfig
) -> list[dict[str, Any]]:
    """Run LightNav unchanged; query the remote gate every N navigation actions."""
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    results_jsonl = os.path.join(output_dir, "results.jsonl")
    gate_episode_jsonl = os.path.join(output_dir, "gate_episode_summary.jsonl")
    viz = (
        _ActiveGateVisualizer(cfg, output_dir)
        if (cfg.save_video or cfg.record_dir)
        else None
    )
    engine, bundle = _default_engine_factory(cfg)
    if not bundle.slowfast_tiers:
        raise ValueError("active gate experiment requires LightNav SlowFast tiers")
    if abs(float(bundle.video_fps) - gate_cfg.video_fps) > 1e-9:
        raise ValueError(
            f"gate video_fps={gate_cfg.video_fps} must match the LightNav checkpoint "
            f"video_fps={bundle.video_fps} so absolute timestamps stay identical"
        )
    lightnav_horizon = int(bundle.predict_horizon)
    if lightnav_horizon < 1:
        raise ValueError(f"invalid LightNav predict_horizon: {lightnav_horizon}")
    gate_interval = lightnav_horizon * gate_cfg.horizon_multiplier
    manifest = {
        "experiment": "qwen_active_observation_gate",
        "lightnav_model": cfg.model_path,
        "lightnav_backend": cfg.backend,
        "lightnav_horizon": lightnav_horizon,
        "gate_horizon_multiplier": gate_cfg.horizon_multiplier,
        "resolved_gate_interval": gate_interval,
        "gate": asdict(gate_cfg),
        "trigger_point": (
            "at episode start before the first LightNav inference, then after each "
            "interval of completed actions before the next inference"
        ),
        "trigger_unit": "LightNav navigation actions (rotation actions excluded)",
        "unknown_mapping": gate_cfg.unknown_policy,
        "observation_camera": "official front RGB only",
        "gate_lightnav_schedule": "gate_then_lightnav_sequential",
    }
    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    env = _default_env_factory(cfg)
    gate = GateClient(gate_cfg.gate_server, timeout_ms=gate_cfg.timeout_ms)
    print("\n" + "=" * 72)
    print("Qwen Active Observation Gate Evaluation")
    print("=" * 72)
    print(f"  LightNav:          {cfg.model_path}")
    print(f"  Gate server:       {gate_cfg.gate_server}")
    print(f"  LightNav horizon:  {lightnav_horizon}")
    print(f"  Gate multiplier:   {gate_cfg.horizon_multiplier}x")
    print(f"  Gate schedule:     start, then every {gate_interval} navigation actions")
    print(f"  Temporal stride:   {gate_cfg.temporal_stride} (absolute timestamps retained)")
    print(f"  Dense history:     <= {gate_cfg.dense_history_limit} frames")
    print(f"  Gate frame cap:    {gate_cfg.max_gate_frames} frames")
    print(f"  Gate frame size:   {gate_cfg.gate_frame_height}x{gate_cfg.gate_frame_width}")
    print(f"  UNKNOWN behavior:  {gate_cfg.unknown_policy}")
    print(f"  Output:            {output_dir}")
    print("=" * 72 + "\n", flush=True)

    results: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    skipped_lang = 0
    lang_filter = set(cfg.languages) if cfg.languages else None
    policy: Any = None
    n_run = 0
    start_time = time.time()

    try:
        while True:
            obs, info = env.reset()
            habitat_ep_id = info.get("episode_id", "")
            habitat_scene_id = info.get("scene_id", "")
            episode_key = (habitat_scene_id, habitat_ep_id)
            if episode_key in seen:
                break
            seen.add(episode_key)
            if lang_filter and info.get("language", "") not in lang_filter:
                skipped_lang += 1
                continue

            instruction = extract_instruction(obs)
            if policy is None:
                policy = build_velocity_policy(cfg, engine, bundle, info)
                if int(policy.H) != lightnav_horizon:
                    raise ValueError(
                        f"model bundle horizon {lightnav_horizon} != action decoder horizon {policy.H}"
                    )
            policy.reset(obs)
            if viz is not None:
                viz.begin_episode(str(habitat_ep_id))
            print(
                f"Episode {n_run + 1} [{habitat_scene_id} / {habitat_ep_id}]...",
                flush=True,
            )

            final_success: Any = False
            forced_stop_used = False
            min_distance_to_goal = float("inf")
            action_step = 0
            navigation_actions = 0
            last_gate_navigation_action = -1
            gate_events: list[dict[str, Any]] = []
            video_rel: str | None = None
            episode_done = False

            try:
                while action_step < cfg.max_steps and obs.get("rgb") is not None:
                    post_scan_event: dict[str, Any] | None = None
                    should_query_gate = gate_due(
                        navigation_actions, gate_interval, last_gate_navigation_action
                    ) and action_step < cfg.max_steps - 1
                    if should_query_gate:
                        # Both branches see the same current post-action observation.
                        # Gate gets a copied/JPEG history; policy.act inserts it once
                        # into LightNav history while computing the candidate action.
                        history_frames = list(policy.agent._history) + [obs["rgb"]]
                        history_ids = list(policy.agent._history_frame_ids) + [
                            int(policy.agent._next_frame_id)
                        ]
                        selection = select_gate_frames(
                            history_frames,
                            history_ids,
                            bundle.slowfast_tiers,
                            stride=gate_cfg.temporal_stride,
                            dense_history_limit=gate_cfg.dense_history_limit,
                            max_selected_frames=gate_cfg.max_gate_frames,
                            lightnav_frame_size=tuple(bundle.video_size),
                            gate_frame_size=(
                                gate_cfg.gate_frame_height,
                                gate_cfg.gate_frame_width,
                            ),
                        )
                        # Each GPU hosts both frozen models. Run Gate first so Qwen
                        # and LightNav activation peaks never overlap on a 24 GB card.
                        answer = gate.decide(
                            frames=selection.frames,
                            frame_ids=selection.frame_ids,
                            fps=gate_cfg.video_fps,
                            instruction=instruction,
                            jpeg_quality=gate_cfg.jpeg_quality,
                        )
                        last_gate_navigation_action = navigation_actions
                        should_scan = answer["decision"] == "NEED" or (
                            answer["decision"] == "UNKNOWN"
                            and gate_cfg.unknown_policy == "scan"
                        )
                        event = {
                            "navigation_actions_before_query": navigation_actions,
                            "lightnav_horizon": lightnav_horizon,
                            "horizon_multiplier": gate_cfg.horizon_multiplier,
                            "resolved_gate_interval": gate_interval,
                            "env_action_step": action_step,
                            **answer,
                            "selected_frame_ids": selection.frame_ids,
                            "selected_unique_frames": selection.unique_frame_count,
                            "slowfast_candidate_frames": selection.slowfast_candidate_count,
                            "temporally_downsampled": selection.downsampled,
                            "temporal_padding": selection.padded,
                            "lightnav_visual_tokens_estimate": selection.lightnav_visual_tokens_estimate,
                            "gate_visual_tokens_estimate": selection.gate_visual_tokens_estimate,
                            "scan_requested": should_scan,
                            "scan_executed": False,
                            "direction": None,
                            "rotation_actions": 0,
                            "execution_schedule": "gate_then_lightnav_sequential",
                            "lightnav_computed_after_scan": False,
                        }
                        plan: list[int] = []
                        if should_scan:
                            direction = _scan_direction(
                                gate_cfg,
                                f"{habitat_scene_id}:{habitat_ep_id}",
                                len(gate_events),
                            )
                            plan = _rotation_plan(direction, gate_cfg)
                            event["direction"] = direction

                        # Reserve a genuine post-scan LightNav inference/action plus
                        # the official final forced-STOP slot.
                        if plan and action_step + len(plan) < cfg.max_steps - 1:
                            event["scan_executed"] = True
                            event["rotation_actions"] = len(plan)
                            if viz is not None:
                                viz.set_gate_state(
                                    decision=answer["decision"],
                                    query_index=len(gate_events) + 1,
                                    observation_added=True,
                                )
                            # Gate does not mutate LightNav history. Insert the live
                            # front frame once before adding rotated observations.
                            policy.observe(obs, info)
                            for plan_index, sign in enumerate(plan):
                                rotation = _rotation_action(policy, sign, gate_cfg)
                                if viz is not None:
                                    viz.step(
                                        obs["rgb"],
                                        step=action_step,
                                        instruction=instruction,
                                        policy_info=_rotation_policy_info(sign, gate_cfg),
                                        latency_ms=0.0,
                                        exploration=True,
                                    )
                                obs, _reward, terminated, truncated, info = env.step(rotation)
                                action_step += 1
                                min_distance_to_goal = min(
                                    min_distance_to_goal,
                                    _safe_scalar(info.get("distance_to_goal"), float("inf")),
                                )
                                if terminated or truncated:
                                    final_success = info.get("success", False)
                                    episode_done = True
                                    break
                                if plan_index < len(plan) - 1:
                                    policy.observe(obs, info)
                            post_scan_event = event
                        elif plan:
                            event["budget_suppressed"] = True
                            if viz is not None:
                                viz.set_gate_state(
                                    decision=answer["decision"],
                                    query_index=len(gate_events) + 1,
                                    observation_added=False,
                                )
                        else:
                            if viz is not None:
                                viz.set_gate_state(
                                    decision=answer["decision"],
                                    query_index=len(gate_events) + 1,
                                    observation_added=False,
                                )
                        gate_events.append(event)
                        print(
                            f"  gate@nav={navigation_actions}: {answer['decision']} "
                            f"valid={answer['valid_format']} scan={event['scan_executed']} "
                            f"frames={selection.unique_frame_count} "
                            f"tokens~{selection.gate_visual_tokens_estimate} "
                            f"latency={answer['latency_ms']:.0f}ms",
                            flush=True,
                        )
                        if episode_done:
                            break

                    forced_stop = bool(
                        cfg.force_stop_at_max_steps
                        and action_step == cfg.max_steps - 1
                        and hasattr(policy, "stop_action")
                    )
                    if forced_stop:
                        action = policy.stop_action()
                        forced_stop_used = True
                        policy_info = {}
                        latency_ms = 0.0
                    else:
                        started = time.monotonic()
                        action = policy.act(obs, info)
                        latency_ms = (time.monotonic() - started) * 1000.0
                        policy_info = policy.get_info() if (
                            cfg.verbose or viz is not None or post_scan_event is not None
                        ) else {}
                        if post_scan_event is not None:
                            post_scan_event["lightnav_computed_after_scan"] = True
                            post_scan_event["post_scan_lightnav_latency_ms"] = round(
                                latency_ms, 3
                            )
                            post_scan_event["post_scan_lightnav_raw_text"] = policy_info.get(
                                "raw_text", ""
                            )
                    navigation_actions += 1
                    if not policy_info and (cfg.verbose or viz is not None):
                        policy_info = policy.get_info()
                    if cfg.verbose:
                        print(
                            f"    Action {action_step}: {_format_action(action)} "
                            f"raw={policy_info.get('raw_text', '')!r}",
                            flush=True,
                        )
                    if viz is not None:
                        viz.step(
                            obs["rgb"],
                            step=action_step,
                            instruction=instruction,
                            policy_info=policy_info,
                            latency_ms=latency_ms,
                            exploration=False,
                        )
                    obs, _reward, terminated, truncated, info = env.step(action)
                    action_step += 1
                    min_distance_to_goal = min(
                        min_distance_to_goal,
                        _safe_scalar(info.get("distance_to_goal"), float("inf")),
                    )
                    if terminated or truncated:
                        final_success = info.get("success", False)
                        break
                if viz is not None and obs.get("rgb") is not None:
                    viz.final_frame(obs["rgb"], step=action_step, instruction=instruction)
            finally:
                if viz is not None:
                    video_rel = viz.end_episode(success=bool(final_success))

            oracle_metric = info.get("oracle_success")
            oracle_success = (
                min_distance_to_goal < 3.0
                if oracle_metric is None
                else _safe_scalar(oracle_metric) > 0.0
            )
            result = {
                "episode_id": f"episode_{n_run:03d}",
                "habitat_episode_id": habitat_ep_id,
                "raw_episode_id": str(info.get("raw_episode_id", "")),
                "scene_id": habitat_scene_id,
                "rollout_idx": 0,
                "success": final_success,
                "oracle_success": oracle_success,
                "spl": info.get("spl", 0.0),
                "ndtw": info.get("ndtw", 0.0),
                "soft_spl": float(info.get("soft_spl", 0.0)),
                "steps": action_step,
                "navigation_actions": navigation_actions,
                "final_distance": info.get("distance_to_goal", float("inf")),
                "min_distance": min_distance_to_goal,
                "instruction": instruction,
                "termination_reason": info.get("termination_reason", "unknown"),
                "forced_stop": forced_stop_used,
                "termination_details": info.get("termination_details", {}),
                "active_observation_gate": _gate_stats(gate_events),
            }
            if video_rel is not None:
                result["video"] = video_rel
            results.append(result)
            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(make_json_safe(result), ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            episode_gate_summary = {
                "episode_id": result["episode_id"],
                "habitat_episode_id": habitat_ep_id,
                "raw_episode_id": result["raw_episode_id"],
                "scene_id": habitat_scene_id,
                "instruction": instruction,
                "success": final_success,
                **result["active_observation_gate"],
            }
            with open(gate_episode_jsonl, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        make_json_safe(episode_gate_summary), ensure_ascii=False
                    )
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())
            gate_stats = result["active_observation_gate"]
            print(
                f"  [{'OK' if result['success'] else 'FAIL'}] "
                f"SPL={float(result['spl']):.3f}, actions={action_step}, "
                f"queries={len(gate_events)}, scans={gate_stats['scan_count']}, "
                f"answers={gate_stats['decision_counts']}",
                flush=True,
            )
            n_run += 1
            if cfg.episodes > 0 and n_run >= cfg.episodes:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        gate.close()
        if viz is not None:
            viz.close()
        env.close()

    elapsed = time.time() - start_time
    if skipped_lang:
        print(f"Skipped {skipped_lang} episodes due to language filter.")
    print_vlnce_summary(
        results,
        elapsed,
        output_dir,
        extra_info={
            "experiment": "qwen_active_observation_gate",
            "active_gate_config": asdict(gate_cfg),
            "lightnav_horizon": lightnav_horizon,
            "resolved_gate_interval": gate_interval,
        },
    )
    return results
