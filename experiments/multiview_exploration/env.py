"""Experiment-only VLN-CE environment exposing position for reference triggers."""

from __future__ import annotations

from typing import Any

import numpy as np

from lightnav_habitat.vlnce import VLNCEEnv


class ExplorationVLNCEEnv(VLNCEEnv):
    """Use the official sensors unchanged and add agent XYZ to ``info``."""

    def _compute_info(self, habitat_obs: dict) -> dict[str, Any]:
        info = super()._compute_info(habitat_obs)
        if self._habitat_env is not None and hasattr(self._habitat_env, "sim"):
            state = self._habitat_env.sim.get_agent_state()
            info["agent_position"] = np.asarray(state.position, dtype=np.float32).tolist()
        return info
