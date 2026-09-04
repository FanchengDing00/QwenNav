"""Merge active-gate shards and aggregate gate decisions and scan costs."""

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
    shards = find_shard_dirs(args.shards)
    records = [record for shard in shards for record in load_results(shard)]
    gates = [record.get("active_observation_gate", {}) for record in records]
    events = [event for gate in gates for event in gate.get("events", [])]
    decisions = Counter(event.get("decision", "UNKNOWN") for event in events)
    decision_counts = {
        decision: int(decisions.get(decision, 0))
        for decision in ("NEED", "NO_NEED", "UNKNOWN", "INVALID")
    }
    total_queries = len(events)
    decision_ratios = {
        decision: round(count / total_queries, 6) if total_queries else 0.0
        for decision, count in decision_counts.items()
    }
    n = len(records)
    extra = {
        "experiment": "qwen_active_observation_gate",
        "gate_stats": {
            "total_queries": total_queries,
            "avg_queries_per_episode": round(total_queries / n, 3) if n else 0.0,
            "decision_counts": decision_counts,
            "decision_ratios": decision_ratios,
            "invalid_format_count": sum(not event.get("valid_format", False) for event in events),
            "scan_count": sum(bool(event.get("scan_executed", False)) for event in events),
            "rotation_action_count": sum(int(event.get("rotation_actions", 0)) for event in events),
            "mean_gate_latency_ms": round(
                sum(float(event.get("latency_ms", 0.0)) for event in events) / len(events), 3
            ) if events else 0.0,
        },
    }
    output = Path(args.output)
    merge_results(shards, output, extra_info=extra)
    episode_summaries = [
        {
            "episode_id": record.get("episode_id"),
            "habitat_episode_id": record.get("habitat_episode_id"),
            "raw_episode_id": record.get("raw_episode_id"),
            "scene_id": record.get("scene_id"),
            "instruction": record.get("instruction"),
            "success": record.get("success"),
            **record.get("active_observation_gate", {}),
        }
        for record in records
    ]
    episode_summaries.sort(
        key=lambda item: (str(item.get("scene_id", "")), str(item.get("habitat_episode_id", "")))
    )
    with (output / "gate_episode_summary.jsonl").open("w", encoding="utf-8") as handle:
        for summary in episode_summaries:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
