# Multiview exploration evaluation

This disposable experiment evaluates an unchanged LightNav checkpoint with two
additional RGB observations rendered at the same agent pose at yaw offsets of
`-60` and `+60` degrees. The auxiliary frames are inserted in a deterministic
pseudo-random left/right order; the official front RGB is always appended last by
the untouched `TrajVocabVLNCEPolicy.act` method.

Production code under `src/lightnav`, `habitat_server/lightnav_habitat`, and
`scripts/eval` is not modified.

## Trigger semantics

`eval_r2r.sh` exposes four variables near the top:

- `STEP_INTERVAL=N`: scan at steps `0, N, 2N, ...`; `1` means every step and `0` disables it.
- `REFERENCE_ENABLED=true`: scan when the current position first comes within
  `REFERENCE_THRESHOLD_M` of each point in the episode's `reference_path`.
- `REFERENCE_THRESHOLD_M=0.75`: reference-point arrival radius.
- `ORDER_SEED=0`: controls the reproducible left-first/right-first choice per event.

When both triggers are enabled they are combined with logical OR. If both fire at
the same step, exactly one pair of auxiliary frames is inserted.

The three supported conditions require no Python changes:

```text
periodic only:   STEP_INTERVAL=N, REFERENCE_ENABLED=false
reference only:  STEP_INTERVAL=0, REFERENCE_ENABLED=true
combined OR:     STEP_INTERVAL=N, REFERENCE_ENABLED=true
```

## Run

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
./experiments/multiview_exploration/eval_r2r.sh checkpoints/LightNav-0
```

Outputs are isolated under:

```text
experiments/multiview_exploration/output/<checkpoint>/r2r/<condition>/
```

Each result row contains the trigger reasons, left/right order, reached reference
indices, event count, and auxiliary-frame count under its `exploration` field.
