from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.active_observation_gate.gate_server import parse_decision
from experiments.active_observation_gate.runner import (
    _draw_gate_overlay,
    _gate_stats,
    gate_due,
)
from experiments.active_observation_gate.sampling import select_gate_frames


def _tiers():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "checkpoints/LightNav-0/eval_config.json").read_text())
    return config["tasks"]["vlnce"]["slowfast_tiers"]


def test_gate_decision_parser_strictly_validates_unconstrained_output():
    assert parse_decision("NEED") == ("NEED", True)
    assert parse_decision(" no_need\n") == ("NO_NEED", True)
    assert parse_decision("UNKNOWN") == ("UNKNOWN", True)
    assert parse_decision("NEED.") == ("INVALID", False)
    assert parse_decision("I choose NEED") == ("INVALID", False)


def test_sampling_preserves_absolute_time_and_budget():
    count = 500
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * count
    selection = select_gate_frames(
        frames,
        list(range(count)),
        _tiers(),
        stride=4,
        dense_history_limit=16,
        max_selected_frames=20,
        gate_frame_size=(224, 384),
    )
    assert selection.frame_ids[0] == 0
    assert selection.frame_ids[-1] == 499
    assert selection.unique_frame_count <= 20
    assert selection.gate_visual_tokens_estimate == 840
    assert selection.lightnav_visual_tokens_estimate == 952


def test_short_history_is_not_temporally_downsampled():
    count = 16
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * count
    selection = select_gate_frames(frames, list(range(count)), _tiers())
    assert not selection.downsampled
    assert selection.frame_ids == list(range(count))


def test_gate_triggers_at_start_then_after_twenty_before_twenty_first_inference():
    assert gate_due(0, 20, -1)
    assert not gate_due(0, 20, 0)
    assert not gate_due(19, 20, -1)
    assert gate_due(20, 20, -1)
    assert not gate_due(20, 20, 20)
    assert not gate_due(21, 20, 20)
    assert gate_due(40, 20, 20)


def test_gate_stats_include_all_decision_counts_and_ratios():
    events = [
        {"decision": "NEED", "valid_format": True, "scan_executed": True,
         "rotation_actions": 12},
        {"decision": "NO_NEED", "valid_format": True, "scan_executed": False,
         "rotation_actions": 0},
        {"decision": "NO_NEED", "valid_format": True, "scan_executed": False,
         "rotation_actions": 0},
    ]
    stats = _gate_stats(events)
    assert stats["decision_counts"] == {
        "NEED": 1, "NO_NEED": 2, "UNKNOWN": 0, "INVALID": 0
    }
    assert stats["decision_ratios"] == {
        "NEED": 0.333333,
        "NO_NEED": 0.666667,
        "UNKNOWN": 0.0,
        "INVALID": 0.0,
    }


def test_gate_overlay_marks_decision_and_observation_without_mutating_input():
    frame = np.zeros((270, 480, 3), dtype=np.uint8)
    rendered = _draw_gate_overlay(
        frame,
        decision="NEED",
        query_index=2,
        observation_added=True,
    )
    assert rendered.shape == frame.shape
    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(frame) == 0
