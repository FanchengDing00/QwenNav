"""CLI for the independent real-rotation exploration evaluation loop."""

from __future__ import annotations

import argparse
import sys

from lightnav.cli.eval_habitat import build_parser as build_official_parser
from lightnav.cli.eval_habitat import config_from_args

from .exploration import ExplorationConfig
from .runner import run_multiview_eval


def build_parser():
    parser = build_official_parser()
    parser.prog = "python -m experiments.multiview_exploration.eval"
    group = parser.add_argument_group("multiview exploration")
    group.add_argument(
        "--exploration-action-interval",
        type=int,
        default=5,
        help="Scan after N real env actions; rotations count, 0 disables this trigger.",
    )
    group.add_argument(
        "--exploration-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan when the agent comes within the threshold of an unreached reference point.",
    )
    group.add_argument("--reference-threshold-m", type=float, default=0.5)
    group.add_argument(
        "--initial-360",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At episode start, execute a reproducible clockwise/counterclockwise 360-degree scan.",
    )
    group.add_argument("--exploration-order-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    exploration_cfg = ExplorationConfig(
        action_interval=args.exploration_action_interval,
        reference_enabled=args.exploration_reference,
        reference_threshold_m=args.reference_threshold_m,
        initial_360_enabled=args.initial_360,
        order_seed=args.exploration_order_seed,
    )
    run_multiview_eval(cfg, exploration_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
