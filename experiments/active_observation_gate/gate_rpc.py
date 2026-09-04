"""Small ZeroMQ protocol for the frozen Qwen observation-gate service."""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any, Sequence

import cv2
import numpy as np

VALID_DECISIONS = {"NEED", "NO_NEED", "UNKNOWN"}
_PICKLE_PROTOCOL = 4
_HEADER = struct.Struct("!Q")


def parse_tcp_address(address: str) -> tuple[str, int]:
    if not address.startswith("tcp://"):
        raise ValueError(f"gate address must start with tcp://, got {address!r}")
    host, separator, port = address[6:].rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid gate address: {address!r}")
    return host, int(port)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("gate connection closed before the message completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=_PICKLE_PROTOCOL)
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def receive_message(sock: socket.socket) -> Any:
    size = _HEADER.unpack(_recv_exact(sock, _HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, size))


def encode_jpegs(frames: Sequence[np.ndarray], quality: int = 85) -> list[bytes]:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    encoded: list[bytes] = []
    for frame in frames:
        # Habitat observations are RGB; OpenCV's encoder expects BGR.
        ok, data = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), params)
        if not ok:
            raise RuntimeError("could not JPEG-encode a gate frame")
        encoded.append(data.tobytes())
    return encoded


def decode_jpegs(payloads: Sequence[bytes]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for payload in payloads:
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode a gate JPEG frame")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


class GateClient:
    def __init__(self, address: str, timeout_ms: int = 180000) -> None:
        self.address = address
        self.host, self.port = parse_tcp_address(address)
        self.timeout_s = float(timeout_ms) / 1000.0

    def decide(
        self,
        *,
        frames: Sequence[np.ndarray],
        frame_ids: Sequence[int],
        fps: float,
        instruction: str,
        jpeg_quality: int = 85,
    ) -> dict[str, Any]:
        request = {
            "command": "decide",
            "jpeg_frames": encode_jpegs(frames, jpeg_quality),
            "frame_ids": [int(value) for value in frame_ids],
            "fps": float(fps),
            "instruction": str(instruction),
        }
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        ) as connection:
            connection.settimeout(self.timeout_s)
            send_message(connection, request)
            response = receive_message(connection)
        if response.get("status") != "success":
            raise RuntimeError(f"gate server error: {response.get('message', 'unknown')}")
        return response["result"]

    def close(self) -> None:
        return None
