"""CLI for the isolated Qwen-gated Habitat evaluation."""

from __future__ import annotations

import argparse

from lightnav.cli.eval_habitat import build_parser as build_official_parser
from lightnav.cli.eval_habitat import config_from_args

from .runner import ActiveGateConfig, run_active_gate_eval


def build_parser() -> argparse.ArgumentParser:
    parser = build_official_parser()
    parser.prog = "python -m experiments.active_observation_gate.eval"
    group = parser.add_argument_group("active observation gate")
    group.add_argument("--gate-server", default="tcp://localhost:6755")
    group.add_argument("--gate-horizon-multiplier", type=int, default=2)
    group.add_argument("--gate-temporal-stride", type=int, default=4)
    group.add_argument("--gate-dense-history-limit", type=int, default=16)
    group.add_argument("--gate-max-frames", type=int, default=20)
    group.add_argument("--gate-video-fps", type=float, default=4.0)
    group.add_argument("--gate-frame-height", type=int, default=224)
    group.add_argument("--gate-frame-width", type=int, default=384)
    group.add_argument("--gate-unknown-policy", choices=("skip", "scan"), default="skip")
    group.add_argument("--gate-order-seed", type=int, default=0)
    group.add_argument("--gate-jpeg-quality", type=int, default=85)
    group.add_argument("--gate-timeout-ms", type=int, default=180000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    gate_cfg = ActiveGateConfig(
        gate_server=args.gate_server,
        horizon_multiplier=args.gate_horizon_multiplier,
        temporal_stride=args.gate_temporal_stride,
        dense_history_limit=args.gate_dense_history_limit,
        max_gate_frames=args.gate_max_frames,
        video_fps=args.gate_video_fps,
        gate_frame_height=args.gate_frame_height,
        gate_frame_width=args.gate_frame_width,
        unknown_policy=args.gate_unknown_policy,
        order_seed=args.gate_order_seed,
        jpeg_quality=args.gate_jpeg_quality,
        timeout_ms=args.gate_timeout_ms,
    )
    run_active_gate_eval(cfg, gate_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
