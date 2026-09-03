"""Merge real-rotation experiment shards and retain exploration statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lightnav.habitat.merge import find_shard_dirs, load_results, merge_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    shard_dirs = find_shard_dirs(args.shards)
    records = [record for shard in shard_dirs for record in load_results(shard)]
    configs = []
    for shard in shard_dirs:
        path = shard / "experiment_config.json"
        if path.is_file():
            configs.append(json.loads(path.read_text(encoding="utf-8")))
    if configs and any(config != configs[0] for config in configs[1:]):
        raise ValueError("shards were produced with different experiment configurations")

    events = [event for record in records for event in record.get("exploration", {}).get("events", [])]
    reason_counts = Counter(reason for event in events for reason in event.get("reasons", []))
    direction_counts = Counter(event.get("direction", "unknown") for event in events)
    total_aux = sum(
        int(record.get("exploration", {}).get("auxiliary_frame_count", 0))
        for record in records
    )
    total_reached = sum(
        len(record.get("exploration", {}).get("reached_reference_indices", []))
        for record in records
    )
    total_rotations = sum(
        int(record.get("exploration", {}).get("rotation_action_count", 0))
        for record in records
    )
    n = len(records)
    extra = {
        "experiment": "real_rotation_exploration",
        "observation_camera": "official_front_rgb_only",
        "exploration_config": configs[0].get("exploration", {}) if configs else {},
        "exploration_stats": {
            "total_events": len(events),
            "avg_events_per_episode": round(len(events) / n, 3) if n else 0.0,
            "total_auxiliary_frames": total_aux,
            "total_rotation_actions": total_rotations,
            "total_reached_reference_points": total_reached,
            "event_reason_counts": dict(reason_counts),
            "direction_counts": dict(direction_counts),
        },
    }
    merge_results(shard_dirs, Path(args.output), extra_info=extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
