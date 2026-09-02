"""Trigger logic and unmodified-policy adapter for three-view exploration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExplorationConfig:
    """The two triggers are independent and combined with logical OR."""

    step_interval: int = 5  # 0 disables; 1 scans before every model action
    reference_enabled: bool = True
    reference_threshold_m: float = 0.75
    order_seed: int = 0

    def __post_init__(self) -> None:
        if self.step_interval < 0:
            raise ValueError("step_interval must be >= 0")
        if self.reference_threshold_m <= 0:
            raise ValueError("reference_threshold_m must be > 0")
        if self.step_interval == 0 and not self.reference_enabled:
            raise ValueError("at least one exploration trigger must be enabled")


def _points(value: Any) -> np.ndarray:
    """Normalize Habitat's reference_path to finite XYZ rows."""
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return np.empty((0, 3), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    arr = arr[:, :3]
    return arr[np.all(np.isfinite(arr), axis=1)]


def _position(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


class ExplorationPolicyAdapter:
    """Insert auxiliary views, then call the original LightNav policy unchanged.

    The original policy's ``act(front_obs)`` always runs last, so it appends the
    official front RGB as the newest/current frame before generating an action.
    """

    def __init__(self, policy: Any, config: ExplorationConfig) -> None:
        self.policy = policy
        self.config = config
        self._episode_key = ""
        self._reference_points = np.empty((0, 3), dtype=np.float32)
        self._reference_reached = np.empty((0,), dtype=bool)
        self._event_index = 0
        self._events: list[dict[str, Any]] = []

    def reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        self.policy.reset(obs)
        self._episode_key = f"{info.get('scene_id', '')}:{info.get('episode_id', '')}"
        self._reference_points = _points(info.get("reference_path"))
        self._reference_reached = np.zeros(len(self._reference_points), dtype=bool)
        self._event_index = 0
        self._events = []

    def _new_reference_indices(self, info: dict[str, Any]) -> list[int]:
        if not self.config.reference_enabled or not len(self._reference_points):
            return []
        pos = _position(info.get("agent_position"))
        if pos is None:
            return []
        remaining = np.flatnonzero(~self._reference_reached)
        if not len(remaining):
            return []
        distances = np.linalg.norm(self._reference_points[remaining] - pos[None, :], axis=1)
        reached = remaining[distances <= self.config.reference_threshold_m]
        if len(reached):
            self._reference_reached[reached] = True
        return [int(i) for i in reached]

    def _left_first(self) -> bool:
        payload = (
            f"{self.config.order_seed}|{self._episode_key}|{self._event_index}"
        ).encode("utf-8")
        return bool(hashlib.sha256(payload).digest()[0] & 1)

    def act(self, obs: dict[str, Any], info: dict[str, Any], step: int) -> Any:
        periodic = self.config.step_interval > 0 and step % self.config.step_interval == 0
        reached = self._new_reference_indices(info)
        reference = bool(reached)

        if periodic or reference:
            missing = [key for key in ("rgb_left", "rgb_right") if key not in obs]
            if missing:
                raise KeyError(f"multiview observation missing: {missing}")
            left_first = self._left_first()
            order = ("rgb_left", "rgb_right") if left_first else ("rgb_right", "rgb_left")
            for key in order:
                self.policy.observe({"rgb": obs[key]}, info)
            self._events.append(
                {
                    "step": int(step),
                    "reasons": [
                        reason
                        for enabled, reason in ((periodic, "periodic"), (reference, "reference"))
                        if enabled
                    ],
                    "auxiliary_order": [key.removeprefix("rgb_") for key in order],
                    "reference_indices": reached,
                }
            )
            self._event_index += 1

        # This appends the front RGB last and invokes the untouched model/policy path.
        return self.policy.act(obs, info)

    def stop_action(self) -> Any:
        return self.policy.stop_action()

    def get_info(self) -> dict[str, Any]:
        info = self.policy.get_info() if hasattr(self.policy, "get_info") else {}
        return {
            **info,
            "exploration_events": len(self._events),
            "last_exploration_event": self._events[-1] if self._events else None,
        }

    def get_episode_stats(self) -> dict[str, Any]:
        reached = np.flatnonzero(self._reference_reached).tolist()
        return {
            "exploration_event_count": len(self._events),
            "auxiliary_frame_count": 2 * len(self._events),
            "reference_point_count": int(len(self._reference_points)),
            "reached_reference_indices": reached,
            "events": list(self._events),
        }
