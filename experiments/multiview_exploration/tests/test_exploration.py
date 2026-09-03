from __future__ import annotations

import unittest

from experiments.multiview_exploration.exploration import (
    ExplorationConfig,
    ExplorationController,
    ScanRequest,
    rotation_action,
    rotation_plan,
)


def _info(position=(0.0, 0.0, 0.0)):
    return {
        "scene_id": "scene-a",
        "episode_id": "episode-7",
        "agent_position": list(position),
        "reference_path": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    }


class FakePolicy:
    dt = 0.1
    lin_vel_range = (0.0, 2.5)
    ang_vel_range = (-300.0, 300.0)


class ExplorationTests(unittest.TestCase):
    def test_regular_scan_reaches_both_90_degree_sides_and_returns_front(self):
        cfg = ExplorationConfig(reference_enabled=False, initial_360_enabled=False)
        request = ScanRequest(("periodic",), "left_first", ())
        plan = rotation_plan(request, cfg)
        headings = []
        heading = 0
        for turn in plan:
            heading += turn * cfg.rotation_step_degrees
            headings.append(heading)
        self.assertEqual(len(plan), 12)
        self.assertEqual(max(headings), 90)
        self.assertEqual(min(headings), -90)
        self.assertEqual(headings[-1], 0)

    def test_initial_scan_is_one_full_turn(self):
        cfg = ExplorationConfig(action_interval=0, reference_enabled=False)
        clockwise = rotation_plan(
            ScanRequest(("initial_360",), "clockwise", (), initial_360=True), cfg
        )
        counterclockwise = rotation_plan(
            ScanRequest(("initial_360",), "counterclockwise", (), initial_360=True), cfg
        )
        self.assertEqual(clockwise, [-1] * 12)
        self.assertEqual(counterclockwise, [1] * 12)

    def test_rotation_action_is_pure_rotation_not_midrange_forward(self):
        cfg = ExplorationConfig(reference_enabled=False, initial_360_enabled=False)
        left = rotation_action(FakePolicy(), 1, cfg)["action_args"]
        right = rotation_action(FakePolicy(), -1, cfg)["action_args"]
        self.assertAlmostEqual(left["linear_velocity"], -1.0)
        self.assertAlmostEqual(right["linear_velocity"], -1.0)
        self.assertAlmostEqual(left["angular_velocity"], 1.0, places=6)
        self.assertAlmostEqual(right["angular_velocity"], -1.0, places=6)

    def test_interval_counts_real_actions_and_does_not_retrigger_inside_scan(self):
        cfg = ExplorationConfig(
            action_interval=5, reference_enabled=False, initial_360_enabled=False
        )
        controller = ExplorationController(cfg)
        controller.reset(_info())
        for action_step in range(5):
            self.assertIsNone(controller.request(action_step, _info()))
        request = controller.request(5, _info())
        self.assertIsNotNone(request)
        controller.complete_scan(
            request, start_action_step=5, end_action_step=17, inserted_frame_count=12
        )
        for action_step in range(17, 22):
            self.assertIsNone(controller.request(action_step, _info()))
        self.assertIsNotNone(controller.request(22, _info()))

    def test_interval_one_scans_after_each_following_navigation_action(self):
        cfg = ExplorationConfig(
            action_interval=1, reference_enabled=False, initial_360_enabled=False
        )
        controller = ExplorationController(cfg)
        controller.reset(_info())
        self.assertIsNone(controller.request(0, _info()))
        request = controller.request(1, _info())
        controller.complete_scan(
            request, start_action_step=1, end_action_step=13, inserted_frame_count=12
        )
        self.assertIsNone(controller.request(13, _info()))
        self.assertIsNotNone(controller.request(14, _info()))

    def test_reference_trigger_uses_threshold_and_combines_with_periodic(self):
        cfg = ExplorationConfig(
            action_interval=5,
            reference_enabled=True,
            reference_threshold_m=0.5,
            initial_360_enabled=False,
        )
        controller = ExplorationController(cfg)
        controller.reset(_info(position=(1.0, 0.0, 0.0)))
        request = controller.request(5, _info(position=(1.6, 0.0, 0.0)))
        self.assertEqual(request.reasons, ("periodic", "reference"))
        self.assertEqual(request.reference_indices, (1,))

    def test_reference_point_at_spawn_is_marked_without_landing_scan(self):
        controller = ExplorationController(
            ExplorationConfig(
                action_interval=0,
                reference_enabled=True,
                initial_360_enabled=False,
            )
        )
        controller.reset(_info())
        self.assertIsNone(controller.request(0, _info()))
        self.assertEqual(controller.get_episode_stats()["reached_reference_indices"], [0])

    def test_order_is_reproducible_per_episode_and_seed(self):
        directions = []
        for _ in range(2):
            controller = ExplorationController(
                ExplorationConfig(
                    action_interval=1,
                    reference_enabled=False,
                    initial_360_enabled=False,
                    order_seed=123,
                )
            )
            controller.reset(_info())
            directions.append(controller.request(1, _info()).direction)
        self.assertEqual(directions[0], directions[1])


if __name__ == "__main__":
    unittest.main()
