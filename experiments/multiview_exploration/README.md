# Real-rotation exploration evaluation

This disposable experiment evaluates an unchanged LightNav checkpoint by executing
real Habitat rotation actions. It uses only the official front RGB/depth cameras;
no extra simulator cameras and no changes to the LightNav model or policy class are
required.

Production code under `src/lightnav`, `habitat_server/lightnav_habitat`, and
`scripts/eval` is not modified.

## Exploration action

A normal scan reproducibly chooses left-first or right-first, rotates to both sides,
and returns to the original forward heading:

```text
left-first:  front -> left 90 -> right 90 -> front
right-first: front -> right 90 -> left 90 -> front
```

The server allows at most 30 degrees of rotation per `velocity_control` action, so
each 90-degree leg is three 30-degree actions. A complete scan is therefore
`3 + 6 + 3 = 12` real `env.step` actions. Every resulting front-camera observation
is added to LightNav's existing history; after returning forward, the untouched
`TrajVocabVLNCEPolicy.act` generates the navigation action.

The optional landing scan executes 12 rotations in one reproducibly selected direction
to make a real 360-degree turn and finish at the original forward heading.

## Trigger and step semantics

The variables near the top of `eval_r2r.sh` are:

- `ACTION_INTERVAL=N`: first scan after N real Habitat actions, then wait N real actions
  after each completed scan. Rotation and navigation actions both count. `1` scans
  after the next navigation action;
  `0` disables this trigger.
- `REFERENCE_ENABLED=true`: scan when the current position first comes within
  `REFERENCE_THRESHOLD_M` of a point in `reference_path`.
- `REFERENCE_THRESHOLD_M=0.5`: reference-point arrival radius.
- `INITIAL_360_ENABLED=true`: perform the real 360-degree scan on landing.
- `ORDER_SEED=0`: reproducibly controls direction order per episode and event.

Periodic and reference triggers are OR-combined. Trigger checks happen only outside a
complete scan, so the scan's own 12 actions cannot recursively trigger another scan.
After any completed scan, the next periodic eligibility is `scan_end_action + N`.
All rotations consume the same 500-action episode budget as navigation actions.
Reference points already within the threshold at the spawn position are marked reached
without triggering a scan, so landing exploration remains an independent factor.

## Run and optional result suffix

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
./experiments/multiview_exploration/eval_r2r.sh checkpoints/LightNav-0

# Manually append a suffix to the r2r directory.
./experiments/multiview_exploration/eval_r2r.sh checkpoints/LightNav-0 rotate90

# The suffix must come immediately after CHECKPOINT_DIR; options follow it.
./experiments/multiview_exploration/eval_r2r.sh checkpoints/LightNav-0 rotate90 \
    --reference-threshold-m 0.7 --no-initial-360
```

The corresponding outputs are isolated under:

```text
experiments/multiview_exploration/output/LightNav-0_mv/r2r/
experiments/multiview_exploration/output/LightNav-0_mv/r2r_rotate90/
```

The `_mv` checkpoint-directory suffix and optional `r2r_<suffix>` only distinguish
experimental outputs. They do not rename or alter the loaded checkpoint.

There is no additional condition subdirectory. Each `r2r_<suffix>` directory contains
a root `config.json` describing its checkpoint, triggers, threshold, seed, action limit,
and video setting, alongside `shard_*`, `logs`, `videos`, and the merged summary.

Each result stores total real actions in `steps`, model-produced navigation actions in
`navigation_actions`, and scan reasons, directions, rotation-action count, inserted
frame count, and reached reference indices under `exploration`.

## Four-group R2R ablation

Run all four groups sequentially with the same visible GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
./experiments/multiview_exploration/eval_r2r_ablations.sh checkpoints/LightNav-0
```

| Group | Periodic | Initial 360 | Reference | Result directory |
|---|---:|---:|---:|---|
| Every 20 actions | 20 | off | off | `r2r_turn20` |
| Initial only | off | on | off | `r2r_turn_init` |
| Reference only | off | off | on | `r2r_turn_ref` |
| All triggers | 20 | on | on | `r2r_turn_all` |

The four runs are intentionally sequential. If one output path already exists,
`eval_r2r.sh` refuses to overwrite it and the ablation launcher stops at that group.
