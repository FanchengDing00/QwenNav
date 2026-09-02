"""Serve the experiment-only three-view R2R Habitat environment over ZeroMQ."""

from __future__ import annotations

import argparse
import logging
import os

from lightnav_habitat.remote_server import RemoteEnvServer

from .env import MultiviewVLNCEEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--split-id", type=int, default=None)
    parser.add_argument("--split-num", type=int, default=None)
    parser.add_argument("--ready-file", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (args.split_id is None) != (args.split_num is None):
        raise SystemExit("--split-id and --split-num must be provided together")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    kwargs = {
        "config_path": args.config,
        "gpu_id": int(os.environ.get("HABITAT_SIM_GPU_ID", "0")),
        "max_steps": args.max_steps,
        "split": args.split,
    }
    if args.split_id is not None:
        kwargs.update(split_id=args.split_id, split_num=args.split_num)

    env = MultiviewVLNCEEnv(**kwargs)
    RemoteEnvServer(env, address=f"tcp://*:{args.port}").start(ready_file=args.ready_file)


if __name__ == "__main__":
    main()
