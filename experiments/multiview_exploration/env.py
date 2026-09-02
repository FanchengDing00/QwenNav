"""Experiment-only VLN-CE environment exposing left/front/right RGB views."""

from __future__ import annotations

from typing import Any

import numpy as np

from lightnav_habitat.vlnce import VLNCEEnv

# Importing registers the two custom sensor types before habitat.Env is built.
from . import sensors as _sensors  # noqa: F401


def _rgb_uint8(value: Any) -> np.ndarray:
    rgb = np.asarray(value)
    if rgb.dtype in (np.float32, np.float64):
        rgb = (rgb * 255).astype(np.uint8)
    return rgb


class MultiviewVLNCEEnv(VLNCEEnv):
    """Keep the official front observation and append two experiment-only views."""

    def _build_observation(self, habitat_obs: dict) -> dict[str, Any]:
        obs = super()._build_observation(habitat_obs)
        missing = [key for key in ("rgb_left", "rgb_right") if key not in habitat_obs]
        if missing:
            raise KeyError(f"multiview Habitat sensors missing observations: {missing}")
        obs["rgb_left"] = _rgb_uint8(habitat_obs["rgb_left"])
        obs["rgb_right"] = _rgb_uint8(habitat_obs["rgb_right"])
        return obs

    def _compute_info(self, habitat_obs: dict) -> dict[str, Any]:
        info = super()._compute_info(habitat_obs)
        if self._habitat_env is not None and hasattr(self._habitat_env, "sim"):
            state = self._habitat_env.sim.get_agent_state()
            info["agent_position"] = np.asarray(state.position, dtype=np.float32).tolist()
        return info
