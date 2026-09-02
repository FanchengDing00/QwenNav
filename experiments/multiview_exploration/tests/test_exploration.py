from __future__ import annotations

import unittest

import numpy as np

from experiments.multiview_exploration.exploration import (
    ExplorationConfig,
    ExplorationPolicyAdapter,
)


class FakePolicy:
    def __init__(self):
        self.frames = []

    def reset(self, obs):
        self.frames = []

    def observe(self, obs, info=None):
        self.frames.append(int(obs["rgb"][0, 0, 0]))

    def act(self, obs, info=None):
        self.frames.append(int(obs["rgb"][0, 0, 0]))
        return "action"

    def stop_action(self):
        return "stop"

    def get_info(self):
        return {}


def _obs():
    return {
        "rgb_left": np.full((1, 1, 3), 1, dtype=np.uint8),
        "rgb": np.full((1, 1, 3), 2, dtype=np.uint8),
        "rgb_right": np.full((1, 1, 3), 3, dtype=np.uint8),
    }


def _info(position=(0.0, 0.0, 0.0)):
    return {
        "scene_id": "scene",
        "episode_id": "episode",
        "agent_position": position,
        "reference_path": [[0, 0, 0], [2, 0, 0]],
    }


class ExplorationPolicyTests(unittest.TestCase):
    def test_front_is_always_last_and_period_one_scans_every_step(self):
        base = FakePolicy()
        policy = ExplorationPolicyAdapter(
            base, ExplorationConfig(step_interval=1, reference_enabled=False, order_seed=7)
        )
        policy.reset(_obs(), _info())
        policy.act(_obs(), _info(), 0)
        self.assertEqual(len(base.frames), 3)
        self.assertEqual(base.frames[-1], 2)
        policy.act(_obs(), _info(), 1)
        self.assertEqual(len(base.frames), 6)
        self.assertEqual(base.frames[-1], 2)

    def test_periodic_and_reference_triggers_are_or_and_do_not_duplicate_scan(self):
        base = FakePolicy()
        policy = ExplorationPolicyAdapter(
            base,
            ExplorationConfig(
                step_interval=5, reference_enabled=True, reference_threshold_m=0.75
            ),
        )
        policy.reset(_obs(), _info())
        policy.act(_obs(), _info(), 0)
        stats = policy.get_episode_stats()
        self.assertEqual(stats["exploration_event_count"], 1)
        self.assertEqual(stats["events"][0]["reasons"], ["periodic", "reference"])

        policy.act(_obs(), _info((1.0, 0.0, 0.0)), 1)
        self.assertEqual(policy.get_episode_stats()["exploration_event_count"], 1)
        policy.act(_obs(), _info((1.4, 0.0, 0.0)), 2)
        self.assertEqual(policy.get_episode_stats()["exploration_event_count"], 2)

    def test_random_order_is_reproducible(self):
        orders = []
        for _ in range(2):
            base = FakePolicy()
            policy = ExplorationPolicyAdapter(
                base, ExplorationConfig(step_interval=1, reference_enabled=False, order_seed=123)
            )
            policy.reset(_obs(), _info())
            for step in range(4):
                policy.act(_obs(), _info(), step)
            orders.append(
                [event["auxiliary_order"] for event in policy.get_episode_stats()["events"]]
            )
        self.assertEqual(orders[0], orders[1])


if __name__ == "__main__":
    unittest.main()
