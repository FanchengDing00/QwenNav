"""SlowFast-aligned temporal sampling for the independent gate model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from lightnav.slowfast import slowfast_segments, validate_slowfast_tiers


@dataclass(frozen=True)
class GateFrameSelection:
    """Frames selected for Qwen while retaining original episode timestamps."""

    frames: list[np.ndarray]
    frame_ids: list[int]
    unique_frame_count: int
    slowfast_candidate_count: int
    downsampled: bool
    padded: bool
    lightnav_visual_tokens_estimate: int
    gate_visual_tokens_estimate: int


def _tokens_per_temporal_pair(
    frame_size: tuple[int, int], pool_spatial: int = 1
) -> int:
    # Qwen3-VL uses patch_size=16 and spatial_merge_size=2: one LLM visual
    # token covers a 32x32 area for each temporal pair.
    merged_h = int(frame_size[0]) // 32
    merged_w = int(frame_size[1]) // 32
    return math.ceil(merged_h / pool_spatial) * math.ceil(
        merged_w / pool_spatial
    )


def select_gate_frames(
    frames: Sequence[np.ndarray],
    frame_ids: Sequence[int],
    slowfast_tiers: list[dict[str, Any]],
    *,
    stride: int = 4,
    dense_history_limit: int = 16,
    max_selected_frames: int = 20,
    lightnav_frame_size: tuple[int, int] = (256, 448),
    gate_frame_size: tuple[int, int] = (224, 384),
) -> GateFrameSelection:
    """Apply LightNav's temporal tiers, then thin each tier without retiming it.

    ``frame_ids`` are absolute observation indices. Sampling never renumbers them;
    the gate server renders timestamps as ``frame_id / fps``. Only the first
    episode frame and newest/current frame are mandatory. A duplicate
    of the newest frame pads odd frame counts for Qwen's temporal patch size 2.
    """
    if stride < 1 or dense_history_limit < 2 or max_selected_frames < 2:
        raise ValueError("stride >= 1 and frame limits >= 2 are required")
    if not frames or len(frames) != len(frame_ids):
        raise ValueError("frames and frame_ids must be non-empty and equally sized")
    ids = [int(value) for value in frame_ids]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("frame_ids must be unique and increasing")

    current_abs = ids[-1]
    segments = slowfast_segments(
        current_abs, len(ids), validate_slowfast_tiers(slowfast_tiers)
    )
    available = set(ids)
    candidates: set[int] = set()
    per_segment_ids: list[list[int]] = []
    lightnav_tokens = 0
    for segment in segments:
        segment_ids = [int(value) for value in segment["frame_ids"] if value in available]
        if not segment_ids:
            continue
        per_segment_ids.append(segment_ids)
        candidates.update(segment_ids)
        lightnav_tokens += math.ceil(len(segment_ids) / 2) * _tokens_per_temporal_pair(
            lightnav_frame_size, int(segment["pool_spatial"])
        )

    downsampled = len(ids) > dense_history_limit
    selected: set[int]
    if downsampled:
        selected = {
            frame_id
            for segment_ids in per_segment_ids
            for frame_id in segment_ids[::stride]
        }
    else:
        selected = set(candidates)

    # Preserve exactly the episode anchor and live frame regardless of thinning.
    selected.update((ids[0], ids[-1]))
    selected_ids = sorted(selected)
    if len(selected_ids) > max_selected_frames:
        mandatory = {ids[0], ids[-1]}
        optional = [value for value in selected_ids if value not in mandatory]
        slots = max_selected_frames - len(mandatory)
        if slots > 0 and optional:
            positions = np.linspace(0, len(optional) - 1, min(slots, len(optional)))
            optional = [optional[int(round(position))] for position in positions]
        else:
            optional = []
        selected_ids = sorted(mandatory | set(optional))
        downsampled = True
    frame_by_id = {frame_id: frame for frame_id, frame in zip(ids, frames)}
    selected_frames = [frame_by_id[frame_id] for frame_id in selected_ids]
    unique_count = len(selected_ids)
    padded = bool(unique_count % 2)
    if padded:
        selected_ids.append(selected_ids[-1])
        selected_frames.append(selected_frames[-1])

    gate_tokens = (len(selected_ids) // 2) * _tokens_per_temporal_pair(gate_frame_size)
    return GateFrameSelection(
        frames=selected_frames,
        frame_ids=selected_ids,
        unique_frame_count=unique_count,
        slowfast_candidate_count=len(candidates),
        downsampled=downsampled,
        padded=padded,
        lightnav_visual_tokens_estimate=lightnav_tokens,
        gate_visual_tokens_estimate=gate_tokens,
    )
