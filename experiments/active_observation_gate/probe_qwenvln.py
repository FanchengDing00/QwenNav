"""Compare explanation and direct A/B gate prompting on Qwen(VLN)-4B."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration


CORE_PROMPT = (
    "You are a mobile robot. You are given visual observations over time, "
    "ordered from earliest to most recent. "
    "Your assigned task is: <navigation_task>{task}</navigation_task>. "
    "You may be at the beginning, middle, or end of the task. "
    "Before predicting your next navigation action, determine whether the current "
    "visual observations provide sufficient information to decide how to proceed. "
    "You may perform an additional observation scan by stopping and rotating to "
    "observe the surrounding environment. However, this scan has a physical cost: "
    "it requires additional actions and execution time. Request it only when the "
    "current observations are insufficient or ambiguous for deciding the next "
    "navigation action. Choose exactly one option: "
    "A. The current observations are sufficient. Proceed without an additional scan. "
    "B. The current observations are insufficient or ambiguous. Perform an additional scan. "
)

DIRECT_SUFFIX = "Answer with only A or B."
EXPLAIN_SUFFIX = (
    "Briefly explain what in the visual history makes the next navigation decision "
    "clear or ambiguous. Then end with exactly one separate line in the form "
    "Final answer: A or Final answer: B."
)


def sample_video(path: str, num_frames: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        raise ValueError(f"video contains no frames: {path}")
    indices = np.linspace(0, count - 1, min(num_frames, count), dtype=np.int64)
    frames: list[Image.Image] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"failed to decode frame {int(index)} from {path}")
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def prepare_inputs(processor, frames, prompt: str, device: torch.device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": 4.0},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


@torch.inference_mode()
def generate_answer(model, tokenizer, inputs, max_new_tokens: int) -> dict:
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    new_ids = generated.sequences[:, prompt_len:]
    text = tokenizer.decode(new_ids[0].tolist(), skip_special_tokens=True).strip()
    token_ids = new_ids[0].tolist()
    result = {
        "text": text,
        "generated_token_count": int(new_ids.shape[1]),
        "generated_token_ids": token_ids,
        "generated_tokens": [tokenizer.id_to_token(token_id) for token_id in token_ids],
    }
    if generated.scores:
        label_ids = {
            label: tokenizer.encode(label, add_special_tokens=False).ids
            for label in ("A", "B")
        }
        if all(len(ids) == 1 for ids in label_ids.values()):
            logits = generated.scores[0][0].float()
            pair = torch.stack([logits[label_ids["A"][0]], logits[label_ids["B"][0]]])
            probs = torch.softmax(pair, dim=0).cpu().tolist()
            result["first_token_ab_normalized"] = {"A": probs[0], "B": probs[1]}
    return result


def load_as_language_model(path: str, reference_config_path: str, device: str):
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    compatibility_fixes: list[str] = []
    if getattr(config.text_config, "rope_scaling", None) is None:
        reference_config = AutoConfig.from_pretrained(
            reference_config_path, local_files_only=True
        )
        config.text_config.rope_scaling = copy.deepcopy(
            reference_config.text_config.rope_scaling
        )
        compatibility_fixes.append(
            "filled missing text_config.rope_scaling from Qwen3-VL-4B-Instruct"
        )
    model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        config=config,
        dtype=torch.bfloat16,
        local_files_only=True,
        output_loading_info=True,
    )
    model = model.to(device)
    model.eval()
    return model, {
        "missing_keys": loading.get("missing_keys", []),
        "unexpected_key_count": len(loading.get("unexpected_keys", [])),
        "mismatched_keys": loading.get("mismatched_keys", []),
        "compatibility_fixes": compatibility_fixes,
    }


def run_model(
    label: str, path: str, processor_path: str, frames, task: str, device: str
) -> dict:
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    # Use 32 continuous frames rather than SlowFast sampling and permit a moderately
    # larger per-frame budget than QwenVLN's navigation-time defaults.
    processor.video_processor.max_frames = max(32, len(frames))
    processor.video_processor.max_pixels = 256 * 448
    # LightNav's tokenizer_config stores extra_special_tokens in an older list
    # format that current AutoTokenizer rejects. The underlying tokenizer.json is
    # valid and is sufficient for decoding and A/B token scoring.
    tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    model, loading = load_as_language_model(path, processor_path, device)
    model_device = next(model.parameters()).device
    direct_inputs = prepare_inputs(processor, frames, CORE_PROMPT.format(task=task) + DIRECT_SUFFIX, model_device)
    explain_inputs = prepare_inputs(processor, frames, CORE_PROMPT.format(task=task) + EXPLAIN_SUFFIX, model_device)
    result = {
        "label": label,
        "path": str(Path(path).resolve()),
        "loading": loading,
        "direct": generate_answer(model, tokenizer, direct_inputs, 16),
        "explain_then_choose": generate_answer(model, tokenizer, explain_inputs, 192),
    }
    del direct_inputs, explain_inputs, model, processor, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--qwenvln-checkpoint", required=True)
    parser.add_argument("--qwen-base", required=True)
    parser.add_argument("--lightnav-checkpoint")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--only",
        choices=("all", "lightnav", "qwenvln", "base"),
        default="all",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_frames < 2:
        raise SystemExit("--num-frames must be >= 2")

    frames = sample_video(args.video, args.num_frames)
    output = {
        "video": str(Path(args.video).resolve()),
        "num_continuous_uniform_frames": len(frames),
        "slowfast": False,
        "video_max_pixels_per_frame": 256 * 448,
        "task": args.task,
        "models": [],
    }
    models = [
        ("QwenVLN-4B VLM backbone", args.qwenvln_checkpoint),
        ("Qwen3-VL-4B-Instruct base", args.qwen_base),
    ]
    if args.lightnav_checkpoint:
        models.insert(0, ("LightNav-0 loaded as stock Qwen3-VL", args.lightnav_checkpoint))
    if args.only != "all":
        wanted = {
            "lightnav": "LightNav-0 loaded as stock Qwen3-VL",
            "qwenvln": "QwenVLN-4B VLM backbone",
            "base": "Qwen3-VL-4B-Instruct base",
        }[args.only]
        models = [model for model in models if model[0] == wanted]
        if not models:
            raise SystemExit(f"requested model is unavailable: {args.only}")
    for label, path in models:
        result = run_model(label, path, args.qwen_base, frames, args.task, args.device)
        output["models"].append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
