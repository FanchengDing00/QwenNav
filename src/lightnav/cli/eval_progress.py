"""Terminal progress bars for parallel Habitat evaluation.

The evaluator appends one JSON record to each shard's ``results.jsonl`` after an
episode completes. This monitor counts completed records and their ``success`` values;
it never modifies evaluation state or supplies metrics to the final summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm


def shard_targets(total_episodes: int, num_shards: int, per_shard_limit: int) -> list[int]:
    """Mirror the Habitat server's floor-sized chunks with remainder in the last shard."""
    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    chunk_size = total_episodes // num_shards
    targets = [chunk_size] * max(0, num_shards - 1)
    targets.append(total_episodes - chunk_size * (num_shards - 1))
    if per_shard_limit > 0:
        targets = [min(target, per_shard_limit) for target in targets]
    return targets


def shard_stats(path: Path, target: int) -> tuple[int, int]:
    """Return ``(completed, successful)`` valid records, capped at the shard target."""
    if not path.is_file():
        return 0, 0

    completed = 0
    successful = 0
    with path.open("rb") as f:
        for line in f:
            if completed >= target:
                break
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A concurrently appended or damaged line does not affect evaluation;
                # retry it from the file on the next polling pass.
                continue
            if not isinstance(record, dict):
                continue
            completed += 1
            successful += int(bool(record.get("success", False)))
    return completed, successful


def monitor(
    output_root: Path,
    targets: list[int],
    stop_file: Path,
    poll_seconds: float,
) -> None:
    enabled = sys.stderr.isatty()
    shard_bars = [
        tqdm(
            total=target,
            desc=f"shard_{index}",
            position=index,
            dynamic_ncols=True,
            leave=True,
            disable=not enabled,
        )
        for index, target in enumerate(targets)
    ]
    total_bar = tqdm(
        total=sum(targets),
        desc="total",
        position=len(targets),
        dynamic_ncols=True,
        leave=True,
        disable=not enabled,
    )

    try:
        while True:
            stats = [
                shard_stats(output_root / f"shard_{index}" / "results.jsonl", target)
                for index, target in enumerate(targets)
            ]
            counts = [completed for completed, _successful in stats]
            for bar, count in zip(shard_bars, counts):
                if count > bar.n:
                    bar.update(count - bar.n)
                else:
                    bar.refresh()

            total_count = sum(counts)
            total_success = sum(successful for _completed, successful in stats)
            sr = 100.0 * total_success / total_count if total_count else 0.0
            total_bar.set_postfix_str(f"SR={sr:.1f}% ({total_success}/{total_count})", refresh=False)
            if total_count > total_bar.n:
                total_bar.update(total_count - total_bar.n)
            else:
                total_bar.refresh()

            if stop_file.exists() or total_count >= total_bar.total:
                break
            time.sleep(poll_seconds)
    finally:
        total_bar.close()
        for bar in shard_bars:
            bar.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--total-episodes", type=int, required=True)
    parser.add_argument("--per-shard-limit", type=int, default=-1)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = shard_targets(args.total_episodes, args.num_shards, args.per_shard_limit)
    monitor(args.output_root, targets, args.stop_file, args.poll_seconds)


if __name__ == "__main__":
    main()
