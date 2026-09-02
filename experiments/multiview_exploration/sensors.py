"""Experiment-only Habitat RGB sensors at yaw offsets of +/-60 degrees."""

from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimRGBSensor


@registry.register_sensor(name="ExplorationLeftRGBSensor")
class ExplorationLeftRGBSensor(HabitatSimRGBSensor):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "rgb_left"


@registry.register_sensor(name="ExplorationRightRGBSensor")
class ExplorationRightRGBSensor(HabitatSimRGBSensor):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "rgb_right"
