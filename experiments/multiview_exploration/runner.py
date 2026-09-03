"""Independent Habitat evaluation loop for real-action visual exploration."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
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

from .exploration import (
    ExplorationConfig,
    ExplorationController,
    ScanRequest,
    rotation_action,
    rotation_plan,
)


def _rotation_policy_info(turn_sign: int, config: ExplorationConfig) -> dict[str, Any]:
    """Give the normal visualizer a one-step yaw waypoint for rotation frames."""
    yaw = np.deg2rad(turn_sign * config.rotation_step_degrees)
    return {
        "predicted_traj": np.asarray([[0.0, 0.0, yaw]], dtype=np.float32),
        "raw_text": f"explore_rotate_{'left' if turn_sign > 0 else 'right'}",
    }


def run_multiview_eval(
    cfg: HabitatEvalConfig,
    exploration_cfg: ExplorationConfig,
) -> list[dict[str, Any]]:
    """Run an isolated eval loop while keeping the LightNav policy untouched."""
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    results_jsonl = os.path.join(output_dir, "results.jsonl")
    manifest = {
        "experiment": "real_rotation_exploration",
        "observation_camera": "official_front_rgb_only",
        "exploration": asdict(exploration_cfg),
        "model_path": cfg.model_path,
        "backend": cfg.backend,
        "max_steps": cfg.max_steps,
        "languages": cfg.languages,
    }
    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    viz = _EvalVisualizer(cfg, output_dir) if (cfg.save_video or cfg.record_dir) else None
    engine, bundle = _default_engine_factory(cfg)
    env = _default_env_factory(cfg)

    print("\n" + "=" * 68)
    print("Real Rotation Exploration Evaluation (left/right 90 deg; return front)")
    print("=" * 68)
    print(f"  Model:               {cfg.model_path}")
    print(f"  Server:              {cfg.server}")
    print(f"  Episodes:            {cfg.episodes if cfg.episodes > 0 else 'full split'}")
    print(f"  Action interval:     {exploration_cfg.action_interval or 'disabled'}")
    print(f"  Reference trigger:   {exploration_cfg.reference_enabled}")
    print(f"  Reference threshold: {exploration_cfg.reference_threshold_m:.2f} m")
    print(f"  Initial 360 scan:    {exploration_cfg.initial_360_enabled}")
    print(f"  Rotation action:     {exploration_cfg.rotation_step_degrees} deg")
    print(f"  Order seed:          {exploration_cfg.order_seed}")
    print(f"  Output:              {output_dir}")
    print("=" * 68 + "\n", flush=True)

    lang_filter = set(cfg.languages) if cfg.languages else None
    results: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    skipped_lang = 0
    n_run = 0
    policy: Any = None
    start_time = time.time()

    try:
        while True:
            obs, info = env.reset()
            habitat_ep_id = info.get("episode_id", "")
            habitat_scene_id = info.get("scene_id", "")
            ep_key = (habitat_scene_id, habitat_ep_id)
            if ep_key in seen:
                print(f"Full split completed: {len(seen)} unique episodes seen.", flush=True)
                break
            seen.add(ep_key)

            if lang_filter and info.get("language", "") not in lang_filter:
                skipped_lang += 1
                continue

            episode_id = f"episode_{n_run:03d}"
            print(
                f"Episode {n_run + 1} ({episode_id}) "
                f"[{habitat_scene_id} / {habitat_ep_id}]...",
                flush=True,
            )
            instruction = extract_instruction(obs)
            if policy is None:
                policy = build_velocity_policy(cfg, engine, bundle, info)
            policy.reset(obs)
            controller = ExplorationController(exploration_cfg)
            controller.reset(info)
            if viz is not None:
                viz.begin_episode(str(habitat_ep_id))

            final_success: Any = False
            forced_stop_used = False
            min_distance_to_goal = float("inf")
            action_step = 0
            navigation_actions = 0
            exploration_suppressed_for_budget = False
            video_rel: str | None = None
            episode_done = False

            try:
                while action_step < cfg.max_steps:
                    if obs.get("rgb") is None:
                        break

                    request: ScanRequest | None = None
                    if not exploration_suppressed_for_budget:
                        request = controller.request(action_step, info)
                    if request is not None:
                        plan = rotation_plan(request, exploration_cfg)
                        # Reserve one final action for the policy (or forced STOP).
                        if action_step + len(plan) < cfg.max_steps:
                            scan_start = action_step
                            # The current forward view belongs to the scan. Intermediate
                            # rotation results are observed below; the final forward frame
                            # remains for the unchanged policy.act call.
                            policy.observe(obs, info)
                            inserted_frames = 1
                            for plan_index, turn_sign in enumerate(plan):
                                action = rotation_action(policy, turn_sign, exploration_cfg)
                                if cfg.verbose:
                                    print(
                                        f"    Action {action_step}: explore "
                                        f"{request.direction} {_format_action(action)}",
                                        flush=True,
                                    )
                                if viz is not None:
                                    viz.step(
                                        obs["rgb"],
                                        step=action_step,
                                        instruction=instruction,
                                        policy_info=_rotation_policy_info(
                                            turn_sign, exploration_cfg
                                        ),
                                        latency_ms=0.0,
                                        episode_id=episode_id,
                                        habitat_episode_id=str(habitat_ep_id),
                                        scene_id=str(habitat_scene_id),
                                        exploration=True,
                                    )
                                obs, _reward, terminated, truncated, info = env.step(action)
                                action_step += 1
                                dtg = _safe_scalar(
                                    info.get("distance_to_goal"), default=float("inf")
                                )
                                min_distance_to_goal = min(min_distance_to_goal, dtg)
                                if terminated or truncated:
                                    final_success = info.get("success", False)
                                    episode_done = True
                                    break
                                if plan_index < len(plan) - 1:
                                    policy.observe(obs, info)
                                    inserted_frames += 1

                            controller.complete_scan(
                                request,
                                start_action_step=scan_start,
                                end_action_step=action_step,
                                inserted_frame_count=inserted_frames,
                            )
                            if episode_done:
                                break
                        else:
                            # Do not start a scan that cannot return to front before the
                            # episode limit. Continue normal navigation for the remainder.
                            exploration_suppressed_for_budget = True

                    forced_stop = bool(
                        cfg.force_stop_at_max_steps
                        and action_step == cfg.max_steps - 1
                        and hasattr(policy, "stop_action")
                    )
                    t_act = time.monotonic()
                    if forced_stop:
                        action = policy.stop_action()
                        forced_stop_used = True
                    else:
                        action = policy.act(obs, info)
                    act_ms = (time.monotonic() - t_act) * 1000.0
                    navigation_actions += 1

                    policy_info: dict[str, Any] = {}
                    if cfg.verbose or viz is not None:
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
                            latency_ms=act_ms,
                            episode_id=episode_id,
                            habitat_episode_id=str(habitat_ep_id),
                            scene_id=str(habitat_scene_id),
                            exploration=False,
                        )

                    obs, _reward, terminated, truncated, info = env.step(action)
                    action_step += 1
                    dtg = _safe_scalar(info.get("distance_to_goal"), default=float("inf"))
                    min_distance_to_goal = min(min_distance_to_goal, dtg)
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
                "episode_id": episode_id,
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
                "exploration": controller.get_episode_stats(),
            }
            if video_rel is not None:
                result["video"] = video_rel
            results.append(result)
            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(make_json_safe(result), ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            status = "OK" if result["success"] else "FAIL"
            print(
                f"  [{status}] success={bool(result['success'])}, "
                f"SPL={float(result['spl']):.3f}, NDTW={float(result['ndtw']):.3f}, "
                f"actions={result['steps']}, navigation={navigation_actions}, "
                f"rotations={result['exploration']['rotation_action_count']}",
                flush=True,
            )
            n_run += 1
            if cfg.episodes > 0 and n_run >= cfg.episodes:
                print(f"Reached --episodes {cfg.episodes}.", flush=True)
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if viz is not None:
            viz.close()
        env.close()

    elapsed = time.time() - start_time
    if skipped_lang:
        print(f"Skipped {skipped_lang} episodes due to language filter.")
    extra_info = {
        "model": cfg.model_path,
        "backend": cfg.backend,
        "experiment": "real_rotation_exploration",
        "exploration_config": asdict(exploration_cfg),
    }
    print_vlnce_summary(results, elapsed, output_dir, extra_info=extra_info)
    return results
