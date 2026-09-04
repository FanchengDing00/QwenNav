# Active observation gate experiment

This experiment leaves the production evaluation, LightNav checkpoint, and
`TrajVocabVLNCEPolicy` unchanged. At episode step 0 and then every 20
**LightNav navigation actions**, a
separate frozen Qwen3-VL-4B-Instruct process receives a temporally sampled history.
Its prompt requests exactly one of:

- `NEED`: a left/right scan is likely to resolve insufficient or ambiguous evidence;
- `NO_NEED`: the history is sufficient for the next navigation move;
- `UNKNOWN`: it cannot determine whether a scan would help.

This is a prompt-level instruction only. Generation uses the model's normal full
vocabulary: there is no candidate-token mask, constrained decoding, logit remapping,
or nearest-label correction. After generation, the program accepts only an exact
label (ignoring surrounding whitespace and letter case). Any other text is recorded
as the separate `INVALID` class and does not trigger a scan. Every raw output is
retained for auditing.

The user-facing period is a horizon multiplier rather than a hard-coded step count:

```text
resolved gate interval = loaded LightNav policy.H * GATE_HORIZON_MULTIPLIER
                       = 10 * 2
                       = 20 completed navigation actions
```

The runner reads `H=10` from the loaded model bundle and verifies that the RVQ action
decoder reports the same `policy.H`. The Gate first runs at step 0 immediately before
LightNav's first inference. It then runs after action 20/before inference 21, after
action 40/before inference 41, and so on. Inserted rotation actions neither advance
this counter nor recursively trigger the Gate.

Only `NEED` causes a scan by default. `UNKNOWN` is logged and skipped; this can be
changed to `scan` for a conservative uncertainty policy. A scan uses the same real
Habitat velocity-control action as the earlier multiview experiment:

```text
front -> left/right 90 deg -> opposite side 90 deg -> front
```

The 90-degree legs consist of 30-degree actions. All resulting observations enter
LightNav's normal history, while the final observation is again the front view.

## Video and token budget

Qwen's stock processor defaults to a total-video budget of 25,165,824 pixels. With
many input frames it reduces every frame's spatial resolution, which makes late
history progressively blurry. This experiment separates the controls:

1. frames are explicitly kept at `224x384`;
2. the processor's total-video pixel ceiling is set to
   `20 * 224 * 384 = 1,720,320`;
3. histories of at most 16 observations retain every frame;
4. above 16 observations, LightNav's SlowFast temporal tiers first determine the
   candidate timestamps, then every fourth candidate in each tier is retained;
5. at most 20 frames are sent; only the first episode frame and newest/current
   frame are mandatory;
6. selected frames retain their **absolute frame ids**. Qwen renders timestamps as
   `absolute_frame_id / 4 FPS`; downsampling never compresses or renumbers time.

Qwen3-VL uses a temporal patch of two frames and one LLM visual token per 32x32
spatial area. The maximum gate budget is therefore:

```text
(20 / 2) * (224 / 32) * (384 / 32) = 840 visual tokens
```

This is close to LightNav's approximately 952-token mature SlowFast history. Qwen's
budget is about 11.8% smaller and does not alter LightNav's own frames, visual
tokens, timestamps, or SlowFast policy.

Threshold checks using the shipped LightNav tiers produced:

| History observations | Gate frames | LightNav tokens (est.) | Gate tokens (est.) |
|---:|---:|---:|---:|
| step 0 | 1 (+1 temporal pad) | 112 | 84 |
| 20 | 7 (+1 temporal pad) | 392 | 336 |
| 40 | 12 | 596 | 504 |
| 64 | 14 | 708 | 588 |
| 100 | 18 | 880 | 756 |
| 200+ | 20 | 952 | 840 |

The step-0 query duplicates its sole frame only to satisfy Qwen's temporal patch size
of two. The default dense threshold is 16; the query after 20 actions is already
temporally thinned, while the episode anchor, latest frame, and their absolute
timestamps remain mandatory.

## Process layout and memory

Every visible GPU runs one independent shard containing Habitat, one Qwen Gate, and
one LightNav process. Four visible GPUs therefore evaluate four scene-sorted shards
without a shared Gate queue:

```text
GPU 0: Habitat shard 0 <-> runner 0 <-> Gate 0 + LightNav 0
GPU 1: Habitat shard 1 <-> runner 1 <-> Gate 1 + LightNav 1
...
```

The model processes use separate environments: LightNav uses `qwennav_model`, while
the Qwen Gate uses `qwen3vl_py310` with FlashAttention 2. LightNav still receives
the official `gpu_memory_utilization=0.65`; its loader also explicitly sets
a 2 GiB KV cache, so vLLM reports that this explicit cache size overrides percentage-
based memory profiling. No LightNav memory or inference setting is reduced.

An earlier 24-frame SDPA Qwen measurement reached 8,968 MiB peak reserved. The new
Gate cap is 20 frames (840 rather than 1,008 visual tokens) and uses FlashAttention 2;
its new joint peak still needs to be measured on an otherwise idle GPU. Runtime GPU
measurements put the LightNav/Habitat increment around 12.7 GiB and the earlier
24-frame Qwen Gate increment around 9.2 GiB. Their conservative sum fits narrowly on
a clean 24,564 MiB RTX 4090,
but allocator fragmentation and simultaneous activations leave too little safety
margin. The launcher therefore requires at least 23,000 MiB free on every selected
GPU and uses a sequential trigger schedule. Resident weights remain colocated, but
temporary activation peaks do not overlap:

```text
current observation -> Qwen Gate -> decision -> LightNav inference
```

For `NO_NEED`/`UNKNOWN`, LightNav runs once on its unchanged official history. For
`NEED`, the current front frame is inserted once, the real scan augments LightNav's
history, and LightNav runs once on the augmented history. No speculative LightNav
action is computed or discarded.

Apart from real scan observations requested by Qwen, LightNav uses the production R2R
Habitat YAML, `vllm_local`, 500-step budget, greedy decoding, checkpoint SlowFast
tiers, video size, history, prompt, RVQ decoder, and the official 0.65 vLLM setting.

## Gate result records

Each episode stores its complete Gate history under `active_observation_gate` in the
normal `results.jsonl`, and also gets one inspectable row in
`gate_episode_summary.jsonl`. The episode summary contains:

- counts and ratios for `NEED`, `NO_NEED`, `UNKNOWN`, and `INVALID`;
- every trigger's completed navigation-action count and environment-action count;
- the raw unconstrained Qwen answer and whether the parser found a recognized judgment;
- selected absolute frame ids and visual-token estimates;
- whether a scan was requested/executed and its reproducible direction;
- the sequential execution schedule and the post-scan LightNav output when applicable.

After shard merging, root `summary.json` contains overall counts and ratios, while
root `gate_episode_summary.jsonl` contains every episode and every Gate response in
one file.

## Run

From the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0,1  # two colocated Gate+LightNav evaluation shards
./experiments/active_observation_gate/eval_r2r.sh checkpoints/LightNav-0
```

Four-card evaluation launches four independent shards:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
./experiments/active_observation_gate/eval_r2r.sh checkpoints/LightNav-0 gate_h2_4gpu
```

The optional second argument changes only the result suffix:

```bash
./experiments/active_observation_gate/eval_r2r.sh checkpoints/LightNav-0 gate20_trial2
```

Outputs are isolated under
`experiments/active_observation_gate/output/LightNav-0_gate/r2r_<suffix>/` and are
ignored by Git. The launcher refuses to overwrite an existing result directory.

The two models run from separate Python environments. By default, LightNav uses
`~/.conda/envs/qwennav_model/bin/python`, while the frozen Qwen Gate uses
`~/.conda/envs/qwen3vl_py310/bin/python`. The Gate environment provides its own
PyTorch/CUDA stack and FlashAttention 2; it does not import or modify LightNav's
environment. These paths can be overridden independently with `LIGHTNAV_PYTHON`
and `GATE_PYTHON`.

For a server installation that keeps all three prefix environments under the
repository's `conda_envs/` directory, follow
[INSTALL_CONDA_ENVS.md](INSTALL_CONDA_ENVS.md). The guide covers the LightNav,
Habitat, and frozen Qwen3-VL Gate environments and shows the launcher overrides
needed for repository-local environments.
