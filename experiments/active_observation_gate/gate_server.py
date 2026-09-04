"""Serve frozen Qwen3-VL-4B three-way observation-gate decisions over ZeroMQ."""

from __future__ import annotations

import argparse
import json
import signal
import socket
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata

from .gate_rpc import (
    VALID_DECISIONS,
    decode_jpegs,
    parse_tcp_address,
    receive_message,
    send_message,
)


GATE_PROMPT_ID = "active_observation_three_way_prompt_v4"
GATE_PROMPT = (
    "You are controlling a mobile robot. The timestamped visual observations are "
    "ordered from the beginning of the episode to the current moment. Your "
    "navigation task is: <navigation_task>{task}</navigation_task>. Before the "
    "robot makes its next navigation decision, assess whether the visual history "
    "contains enough evidence to choose how to proceed. The robot may stop and "
    "perform a left-right observation scan, but doing so consumes additional "
    "actions and execution time. Choose NEED only when a scan is likely to supply "
    "missing or ambiguous information that matters for the next move. Choose "
    "NO_NEED when the existing history is sufficient. Choose UNKNOWN only when "
    "you cannot judge whether a scan would help. Reply with exactly one of NEED, "
    "NO_NEED, or UNKNOWN, and no other text."
)


def parse_decision(text: str) -> tuple[str, bool]:
    """Strictly validate normal, unconstrained generation after it completes."""
    normalized = text.strip().upper()
    if normalized in VALID_DECISIONS:
        return normalized, True
    # INVALID is deliberately distinct from a valid UNKNOWN judgment. It never
    # triggers a scan, even when the configurable UNKNOWN policy is "scan".
    return "INVALID", False


class QwenGate:
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        frame_size: tuple[int, int],
        max_frames: int,
        max_new_tokens: int,
    ) -> None:
        self.device = torch.device(device)
        self.frame_size = tuple(int(value) for value in frame_size)
        self.max_frames = int(max_frames)
        self.total_pixel_budget = self.max_frames * self.frame_size[0] * self.frame_size[1]
        self.max_new_tokens = int(max_new_tokens)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def _prepare(self, frames: list[Any], frame_ids: list[int], fps: float, task: str):
        if len(frames) > self.max_frames:
            raise ValueError(f"received {len(frames)} frames, budget permits {self.max_frames}")
        height, width = self.frame_size
        resized = [
            Image.fromarray(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
            for frame in frames
        ]
        metadata = VideoMetadata(
            total_num_frames=max(frame_ids) + 1,
            fps=float(fps),
            width=width,
            height=height,
            duration=(max(frame_ids) + 1) / float(fps),
            frames_indices=list(frame_ids),
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": resized},
                    {"type": "text", "text": GATE_PROMPT.format(task=task)},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[rendered],
            videos=[resized],
            videos_kwargs={
                "do_sample_frames": False,
                "do_resize": True,
                "size": {
                    "shortest_edge": 4096,
                    "longest_edge": self.total_pixel_budget,
                },
                "video_metadata": [metadata],
            },
            return_tensors="pt",
        )
        return inputs.to(self.device)

    @torch.inference_mode()
    def decide(
        self, frames: list[Any], frame_ids: list[int], fps: float, task: str
    ) -> dict[str, Any]:
        started = time.monotonic()
        inputs = self._prepare(frames, frame_ids, fps, task)
        output = self.model.generate(
            **inputs,
            # Greedy decoding is the Transformers equivalent of temperature 0.
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )
        new_ids = output[:, inputs["input_ids"].shape[1] :]
        raw_text = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        decision, valid_format = parse_decision(raw_text)
        return {
            "prompt_id": GATE_PROMPT_ID,
            "decision": decision,
            "valid_format": valid_format,
            "raw_text": raw_text,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
            "generated_token_count": int(new_ids.shape[1]),
            "temperature": 0.0,
            "do_sample": False,
            "attention_implementation": "flash_attention_2",
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--address", default="tcp://*:6755")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-height", type=int, default=224)
    parser.add_argument("--frame-width", type=int, default=384)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--ready-file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frame_height % 32 or args.frame_width % 32:
        raise SystemExit("gate frame dimensions must be divisible by 32")
    gate = QwenGate(
        args.model_path,
        device=args.device,
        frame_size=(args.frame_height, args.frame_width),
        max_frames=args.max_frames,
        max_new_tokens=args.max_new_tokens,
    )
    host, port = parse_tcp_address(args.address)
    if host == "*":
        host = "0.0.0.0"
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(8)
    listener.settimeout(1.0)
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if args.ready_file:
        Path(args.ready_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.ready_file).write_text("ready\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "ready",
                "model": args.model_path,
                "address": args.address,
                "device": args.device,
                "frame_size": [args.frame_height, args.frame_width],
                "max_frames": args.max_frames,
                "total_pixel_budget": args.max_frames * args.frame_height * args.frame_width,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        while running:
            try:
                connection, _peer = listener.accept()
            except socket.timeout:
                continue
            with connection:
                try:
                    request = receive_message(connection)
                    if request.get("command") != "decide":
                        raise ValueError(f"unknown command: {request.get('command')!r}")
                    frames = decode_jpegs(request["jpeg_frames"])
                    frame_ids = [int(value) for value in request["frame_ids"]]
                    if len(frames) != len(frame_ids) or not frames:
                        raise ValueError("frames/frame_ids must be non-empty and equally sized")
                    result = gate.decide(
                        frames, frame_ids, float(request["fps"]), str(request["instruction"])
                    )
                    response = {"status": "success", "result": result}
                except Exception as exc:
                    response = {
                        "status": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                send_message(connection, response)
    finally:
        listener.close()


if __name__ == "__main__":
    main()
