#!/usr/bin/env bash
# Isolated R2R evaluation: frozen Qwen3-VL gate decides whether LightNav scans.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 LIGHTNAV_CHECKPOINT [RESULT_SUFFIX]" >&2
    exit 2
fi
MODEL_PATH="$1"
RESULT_SUFFIX="${2:-gate20}"

# Experiment settings. Edit these values rather than the production eval scripts.
GATE_HORIZON_MULTIPLIER=2        # actual policy.H * multiplier; 10 * 2 = 20 here
GATE_TEMPORAL_STRIDE=4          # applied only after DENSE_HISTORY_LIMIT is exceeded
DENSE_HISTORY_LIMIT=16          # keep every frame at or below this history length
GATE_MAX_FRAMES=20              # ~83% of the old Qwen visual-token budget
GATE_FRAME_HEIGHT=224           # Qwen-only input size; divisible by 32
GATE_FRAME_WIDTH=384            # Qwen-only input size; divisible by 32
GATE_MAX_NEW_TOKENS=4           # labels use 1-2 tokens; vocabulary remains unconstrained
GATE_TEMPERATURE=0.0            # implemented as deterministic greedy decoding
GATE_PROMPT_ID=active_observation_three_way_prompt_v4
GATE_UNKNOWN_POLICY=skip        # skip | scan
GATE_ORDER_SEED=0
GATE_JPEG_QUALITY=85
LIGHTNAV_GPU_MEM_UTIL=0.65      # identical to the official Habitat evaluation
MAX_STEPS=500
EPISODES=-1
SAVE_VIDEO=true
BASE_HABITAT_PORT=5855
BASE_GATE_PORT=6855
READY_TIMEOUT_S=900

HF_HOME_DIR=${HF_HOME:-/mnt/disk0/dfc_documents/HF_home}
GATE_MODEL_PATH=${GATE_MODEL_PATH:-$HF_HOME_DIR/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}
LIGHTNAV_PYTHON=${LIGHTNAV_PYTHON:-$HOME/.conda/envs/qwennav_model/bin/python}
GATE_PYTHON=${GATE_PYTHON:-$HOME/.conda/envs/qwen3vl_py310/bin/python}
HABITAT_ENV=${HABITAT_ENV:-qwennav_habitat}
CONDA_SH=${CONDA_SH:-/opt/anaconda3/etc/profile.d/conda.sh}
HABITAT_CONFIG="$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml"

[[ -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/eval_config.json" ]] || {
    echo "[active_gate] ERROR: invalid LightNav checkpoint: $MODEL_PATH" >&2; exit 1; }
[[ -f "$GATE_MODEL_PATH/config.json" && -f "$GATE_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "[active_gate] ERROR: Qwen gate model is not local: $GATE_MODEL_PATH" >&2; exit 1; }
[[ -x "$LIGHTNAV_PYTHON" ]] || { echo "[active_gate] ERROR: missing LightNav Python: $LIGHTNAV_PYTHON" >&2; exit 1; }
[[ -x "$GATE_PYTHON" ]] || { echo "[active_gate] ERROR: missing Gate Python: $GATE_PYTHON" >&2; exit 1; }
"$GATE_PYTHON" -c 'import flash_attn; from transformers import Qwen3VLForConditionalGeneration' || {
    echo "[active_gate] ERROR: Gate environment lacks Qwen3-VL or FlashAttention 2: $GATE_PYTHON" >&2
    exit 1
}
[[ "$RESULT_SUFFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "[active_gate] ERROR: invalid result suffix: $RESULT_SUFFIX" >&2; exit 2; }

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    DETECTED=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
    (( DETECTED > 0 )) || { echo "[active_gate] ERROR: no GPUs detected" >&2; exit 1; }
    GPUS=(); for ((i=0; i<DETECTED; i++)); do GPUS+=("$i"); done
fi
for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]//[[:space:]]/}
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "[active_gate] ERROR: invalid GPU: $gpu" >&2; exit 2; }
    GPUS[$i]=$((10#$gpu))
done
unset CUDA_VISIBLE_DEVICES
EXECUTION_LAYOUT=colocated_per_shard_sequential
LIGHTNAV_GPUS=("${GPUS[@]}")
GATE_GPUS=("${GPUS[@]}")
N=${#LIGHTNAV_GPUS[@]}
for gpu in "${GPUS[@]}"; do
    FREE_MIB=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    [[ "$FREE_MIB" =~ ^[0-9]+$ ]] || {
        echo "[active_gate] ERROR: cannot read free memory for GPU $gpu" >&2; exit 1; }
    (( FREE_MIB >= 23000 )) || {
        echo "[active_gate] ERROR: colocated mode requires at least 23000 MiB free per GPU; GPU $gpu has ${FREE_MIB} MiB." >&2
        echo "[active_gate] Stop other GPU jobs before launching this experiment." >&2
        exit 1
    }
done
GATE_MAX_VISUAL_TOKENS=$(( ((GATE_MAX_FRAMES + 1) / 2) * (GATE_FRAME_HEIGHT / 32) * (GATE_FRAME_WIDTH / 32) ))

CHECKPOINT_NAME="$(basename "${MODEL_PATH%/}")_gate"
OUTPUT_ROOT="$REPO_ROOT/experiments/active_observation_gate/output/$CHECKPOINT_NAME/r2r_$RESULT_SUFFIX"
LOG_DIR="$OUTPUT_ROOT/logs"
READY_DIR="$OUTPUT_ROOT/.ready"
[[ ! -e "$OUTPUT_ROOT" ]] || {
    echo "[active_gate] ERROR: refusing to overwrite $OUTPUT_ROOT" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$READY_DIR"

set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$HABITAT_ENV"
HABITAT_PYTHON="$CONDA_PREFIX/bin/python"
HABITAT_LD="$CONDA_PREFIX/lib"
conda deactivate
set -u

HABITAT_PIDS=()
GATE_PIDS=()
CLIENT_PIDS=()
PROGRESS_PID=""
cleanup() {
    touch "$READY_DIR/progress.stop" 2>/dev/null || true
    [[ -n "$PROGRESS_PID" ]] && kill "$PROGRESS_PID" 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]:-}" "${GATE_PIDS[@]:-}" "${HABITAT_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

"$LIGHTNAV_PYTHON" -c '
import json, sys
from pathlib import Path
keys = ["lightnav_model", "gate_model", "result_suffix", "gate_horizon_multiplier",
        "dense_history_limit", "temporal_stride", "max_gate_frames",
        "gate_frame_height", "gate_frame_width", "unknown_policy",
        "lightnav_gpu_memory_utilization", "episodes", "max_steps", "execution_layout",
        "gate_lightnav_schedule", "gate_prompt_id", "gate_temperature",
        "gate_do_sample", "gate_attention_implementation", "lightnav_python",
        "gate_python", "gate_max_new_tokens", "invalid_output_policy"]
values = [sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]),
          int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10]),
          sys.argv[11], float(sys.argv[12]), int(sys.argv[13]), int(sys.argv[14]),
          sys.argv[15], "gate_then_lightnav_sequential", sys.argv[16],
          float(sys.argv[17]), False, "flash_attention_2", sys.argv[18], sys.argv[19],
          int(sys.argv[20]), "no_scan"]
Path(sys.argv[1]).write_text(json.dumps(dict(zip(keys, values)), indent=2) + "\n")
' "$OUTPUT_ROOT/config.json" "$MODEL_PATH" "$GATE_MODEL_PATH" "$RESULT_SUFFIX" \
  "$GATE_HORIZON_MULTIPLIER" "$DENSE_HISTORY_LIMIT" "$GATE_TEMPORAL_STRIDE" \
  "$GATE_MAX_FRAMES" "$GATE_FRAME_HEIGHT" "$GATE_FRAME_WIDTH" \
  "$GATE_UNKNOWN_POLICY" "$LIGHTNAV_GPU_MEM_UTIL" "$EPISODES" "$MAX_STEPS" \
  "$EXECUTION_LAYOUT" "$GATE_PROMPT_ID" "$GATE_TEMPERATURE" \
  "$LIGHTNAV_PYTHON" "$GATE_PYTHON" "$GATE_MAX_NEW_TOKENS"

echo "[active_gate] LightNav:      $MODEL_PATH"
echo "[active_gate] Qwen gate:     $GATE_MODEL_PATH"
echo "[active_gate] LightNav env:  $LIGHTNAV_PYTHON"
echo "[active_gate] Gate env:      $GATE_PYTHON (FlashAttention 2)"
echo "[active_gate] colocated GPUs: ${LIGHTNAV_GPUS[*]} ($N independent shards)"
echo "[active_gate] layout:        $EXECUTION_LAYOUT"
echo "[active_gate] schedule:      Gate first, then LightNav (no activation overlap)"
echo "[active_gate] prompt:        $GATE_PROMPT_ID"
echo "[active_gate] decoding:      greedy (temperature=$GATE_TEMPERATURE, do_sample=false)"
echo "[active_gate] gate period:   LightNav horizon x $GATE_HORIZON_MULTIPLIER"
echo "[active_gate] history:       dense<=${DENSE_HISTORY_LIMIT}, stride=${GATE_TEMPORAL_STRIDE}, cap=${GATE_MAX_FRAMES}"
echo "[active_gate] pixel budget:  $((GATE_MAX_FRAMES * GATE_FRAME_HEIGHT * GATE_FRAME_WIDTH)) total pixels (~${GATE_MAX_VISUAL_TOKENS} max visual tokens)"
echo "[active_gate] WARNING: each shard colocates Gate, LightNav, and Habitat on one GPU."
echo "[active_gate] output:        $OUTPUT_ROOT"

# Habitat servers use the production R2R configuration unchanged.
for ((i=0; i<N; i++)); do
    gpu=${LIGHTNAV_GPUS[$i]}; port=$((BASE_HABITAT_PORT+i))
    HABITAT_SIM_GPU_ID=$gpu PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT/habitat_server${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="$HABITAT_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$HABITAT_PYTHON" -m lightnav_habitat.serve --task vlnce \
      --config "$HABITAT_CONFIG" --port "$port" --max-steps "$MAX_STEPS" \
      --split val_unseen --split-id "$i" --split-num "$N" \
      --ready-file "$READY_DIR/habitat_$i.ready" >"$LOG_DIR/habitat_$i.log" 2>&1 &
    HABITAT_PIDS+=("$!")
done

# Every evaluation shard gets its own frozen Qwen Gate on the same physical GPU.
# Gate finishes before LightNav starts at a trigger, avoiding activation overlap.
for ((i=0; i<N; i++)); do
    gpu=${GATE_GPUS[$i]}; port=$((BASE_GATE_PORT+i))
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src" \
      "$GATE_PYTHON" -m experiments.active_observation_gate.gate_server \
      --model-path "$GATE_MODEL_PATH" --address "tcp://*:$port" --device cuda:0 \
      --frame-height "$GATE_FRAME_HEIGHT" --frame-width "$GATE_FRAME_WIDTH" \
      --max-frames "$GATE_MAX_FRAMES" --max-new-tokens "$GATE_MAX_NEW_TOKENS" \
      --ready-file "$READY_DIR/gate_$i.ready" >"$LOG_DIR/gate_$i.log" 2>&1 &
    GATE_PIDS+=("$!")
done

deadline=$((SECONDS+READY_TIMEOUT_S))
while :; do
    ready_h=$(find "$READY_DIR" -maxdepth 1 -type f -name 'habitat_*.ready' | wc -l)
    ready_g=$(find "$READY_DIR" -maxdepth 1 -type f -name 'gate_*.ready' | wc -l)
    echo "[active_gate] ready: habitat $ready_h/$N, gate $ready_g/$N"
    (( ready_h >= N && ready_g >= N )) && break
    for pid in "${HABITAT_PIDS[@]}" "${GATE_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || {
            echo "[active_gate] ERROR: a server exited; inspect $LOG_DIR" >&2
            tail -n 60 "$LOG_DIR"/*.log >&2 || true
            exit 1
        }
    done
    (( SECONDS < deadline )) || { echo "[active_gate] ERROR: server timeout" >&2; exit 1; }
    sleep 2
done

PYTHONPATH="$REPO_ROOT/src" "$LIGHTNAV_PYTHON" -m lightnav.cli.eval_progress \
  --output-root "$OUTPUT_ROOT" --num-shards "$N" --total-episodes 1839 \
  --per-shard-limit "$EPISODES" --stop-file "$READY_DIR/progress.stop" &
PROGRESS_PID=$!

video_args=()
[[ "$SAVE_VIDEO" == true ]] && video_args=(--save_video --video_dir "$OUTPUT_ROOT/videos" --video_episode_count 1839)
for ((i=0; i<N; i++)); do
    gpu=${LIGHTNAV_GPUS[$i]}; habitat_port=$((BASE_HABITAT_PORT+i)); gate_port=$((BASE_GATE_PORT+i))
    shard_dir="$OUTPUT_ROOT/shard_$i"; mkdir -p "$shard_dir"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$LIGHTNAV_PYTHON" -m experiments.active_observation_gate.eval \
      --model_path "$MODEL_PATH" --server "tcp://localhost:$habitat_port" \
      --backend vllm_local --episodes "$EPISODES" --max_steps "$MAX_STEPS" \
      --output_dir "$shard_dir" --gpu_memory_utilization "$LIGHTNAV_GPU_MEM_UTIL" \
      "${video_args[@]}" --gate-server "tcp://localhost:$gate_port" \
      --gate-horizon-multiplier "$GATE_HORIZON_MULTIPLIER" --gate-temporal-stride "$GATE_TEMPORAL_STRIDE" \
      --gate-dense-history-limit "$DENSE_HISTORY_LIMIT" --gate-max-frames "$GATE_MAX_FRAMES" \
      --gate-frame-height "$GATE_FRAME_HEIGHT" --gate-frame-width "$GATE_FRAME_WIDTH" \
      --gate-unknown-policy "$GATE_UNKNOWN_POLICY" --gate-order-seed "$GATE_ORDER_SEED" \
      --gate-jpeg-quality "$GATE_JPEG_QUALITY" >"$LOG_DIR/client_$i.log" 2>&1 &
    CLIENT_PIDS+=("$!")
done

failed=0
for pid in "${CLIENT_PIDS[@]}"; do wait "$pid" || failed=$((failed+1)); done
touch "$READY_DIR/progress.stop"
wait "$PROGRESS_PID" 2>/dev/null || true
PROGRESS_PID=""
(( failed == 0 )) || { echo "[active_gate] ERROR: $failed client shard(s) failed" >&2; exit 1; }

PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src" "$LIGHTNAV_PYTHON" \
  -m experiments.active_observation_gate.merge "$OUTPUT_ROOT"/shard_* --output "$OUTPUT_ROOT" \
  | tee "$OUTPUT_ROOT/merge.log"
echo "[active_gate] done: $OUTPUT_ROOT/summary.json"
