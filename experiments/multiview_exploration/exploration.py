"""Trigger and rotation-plan logic for the real-action exploration experiment."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lightnav.velocity import first_waypoint_to_velocity_cmd


@dataclass(frozen=True)
class ExplorationConfig:
    """Periodic/reference triggers use OR; the optional landing scan is independent."""

    action_interval: int = 5
    reference_enabled: bool = True
    reference_threshold_m: float = 0.5
    initial_360_enabled: bool = True
    order_seed: int = 0
    side_yaw_degrees: int = 90
    rotation_step_degrees: int = 30

    def __post_init__(self) -> None:
        if self.action_interval < 0:
            raise ValueError("action_interval must be >= 0")
        if self.reference_threshold_m <= 0:
            raise ValueError("reference_threshold_m must be > 0")
        if self.side_yaw_degrees <= 0 or self.rotation_step_degrees <= 0:
            raise ValueError("rotation angles must be positive")
        if self.side_yaw_degrees % self.rotation_step_degrees:
            raise ValueError("side_yaw_degrees must be divisible by rotation_step_degrees")
        if 360 % self.rotation_step_degrees:
            raise ValueError("360 must be divisible by rotation_step_degrees")
        if (
            self.action_interval == 0
            and not self.reference_enabled
            and not self.initial_360_enabled
        ):
            raise ValueError("at least one exploration trigger must be enabled")


@dataclass(frozen=True)
class ScanRequest:
    reasons: tuple[str, ...]
    direction: str
    reference_indices: tuple[int, ...]
    initial_360: bool = False


def _points(value: Any) -> np.ndarray:
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


def rotation_plan(request: ScanRequest, config: ExplorationConfig) -> list[int]:
    """Return +1 (left) / -1 (right) for every real Habitat rotation action."""
    if request.initial_360:
        sign = -1 if request.direction == "clockwise" else 1
        return [sign] * (360 // config.rotation_step_degrees)

    side_steps = config.side_yaw_degrees // config.rotation_step_degrees
    first = 1 if request.direction == "left_first" else -1
    # front -> first side -> opposite side -> front
    return [first] * side_steps + [-first] * (2 * side_steps) + [first] * side_steps


def rotation_action(policy: Any, turn_sign: int, config: ExplorationConfig) -> dict[str, Any]:
    """Build a pure normalized velocity action using the policy/server velocity contract."""
    waypoint = np.array(
        [0.0, 0.0, math.radians(turn_sign * config.rotation_step_degrees)],
        dtype=np.float32,
    )
    return first_waypoint_to_velocity_cmd(
        waypoint, policy.dt, policy.lin_vel_range, policy.ang_vel_range
    )


class ExplorationController:
    """Schedule scans by actual ``env.step`` count without modifying LightNav policy."""

    def __init__(self, config: ExplorationConfig) -> None:
        self.config = config
        self._episode_key = ""
        self._reference_points = np.empty((0, 3), dtype=np.float32)
        self._reference_reached = np.empty((0,), dtype=bool)
        self._event_index = 0
        self._initial_pending = False
        self._next_periodic_action = 0
        self._events: list[dict[str, Any]] = []

    def reset(self, info: dict[str, Any]) -> None:
        self._episode_key = f"{info.get('scene_id', '')}:{info.get('episode_id', '')}"
        self._reference_points = _points(info.get("reference_path"))
        self._reference_reached = np.zeros(len(self._reference_points), dtype=bool)
        self._event_index = 0
        self._initial_pending = self.config.initial_360_enabled
        self._next_periodic_action = self.config.action_interval
        self._events = []
        # The path normally starts at the spawn position. Treat those points as
        # already reached without firing a reference scan, keeping "initial" and
        # "reference" as separable ablation factors.
        self._new_reference_indices(info)

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

    def _bit(self, label: str) -> bool:
        payload = (
            f"{self.config.order_seed}|{self._episode_key}|{self._event_index}|{label}"
        ).encode("utf-8")
        return bool(hashlib.sha256(payload).digest()[0] & 1)

    def request(self, action_step: int, info: dict[str, Any]) -> ScanRequest | None:
        reached = self._new_reference_indices(info)
        if self._initial_pending:
            self._initial_pending = False
            return ScanRequest(
                reasons=("initial_360",),
                direction="clockwise" if self._bit("initial") else "counterclockwise",
                reference_indices=tuple(reached),
                initial_360=True,
            )

        periodic = (
            self.config.action_interval > 0
            and action_step >= self._next_periodic_action
        )
        reference = bool(reached)
        if not periodic and not reference:
            return None
        return ScanRequest(
            reasons=tuple(
                reason
                for enabled, reason in ((periodic, "periodic"), (reference, "reference"))
                if enabled
            ),
            direction="left_first" if self._bit("regular") else "right_first",
            reference_indices=tuple(reached),
        )

    def complete_scan(
        self,
        request: ScanRequest,
        *,
        start_action_step: int,
        end_action_step: int,
        inserted_frame_count: int,
    ) -> None:
        plan = rotation_plan(request, self.config)
        self._events.append(
            {
                "action_step": int(start_action_step),
                "end_action_step": int(end_action_step),
                "reasons": list(request.reasons),
                "direction": request.direction,
                "rotation_actions": len(plan),
                "rotation_step_degrees": self.config.rotation_step_degrees,
                "inserted_frame_count": int(inserted_frame_count),
                "reference_indices": list(request.reference_indices),
            }
        )
        self._event_index += 1
        if self.config.action_interval > 0:
            # A scan never recursively triggers itself. Count every rotation action,
            # then wait N further real actions before the next periodic eligibility.
            self._next_periodic_action = end_action_step + self.config.action_interval

    def get_episode_stats(self) -> dict[str, Any]:
        reached = np.flatnonzero(self._reference_reached).tolist()
        return {
            "exploration_event_count": len(self._events),
            "initial_360_used": any("initial_360" in event["reasons"] for event in self._events),
            "rotation_action_count": sum(int(event["rotation_actions"]) for event in self._events),
            "auxiliary_frame_count": sum(
                int(event["inserted_frame_count"]) for event in self._events
            ),
            "reference_point_count": int(len(self._reference_points)),
            "reached_reference_indices": reached,
            "events": list(self._events),
        }
