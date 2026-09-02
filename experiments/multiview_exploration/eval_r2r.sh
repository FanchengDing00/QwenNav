#!/usr/bin/env bash
# Independent parallel R2R evaluation for the disposable three-view experiment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if (( $# != 1 )); then
    echo "Usage: $0 CHECKPOINT_DIR" >&2
    exit 2
fi

# Experiment variables. The periodic and reference triggers are combined with OR.
MODEL_PATH="$1"
STEP_INTERVAL=5              # 1 = three views at every model step; 0 = disabled
REFERENCE_ENABLED=true       # true / false
REFERENCE_THRESHOLD_M=0.75
ORDER_SEED=0
MAX_STEPS=500
EPISODES=-1
GPU_MEM_UTIL=0.65
BASE_PORT=5655
READY_TIMEOUT_S=900
SAVE_VIDEO=true

HABITAT_CONFIG="$REPO_ROOT/experiments/multiview_exploration/configs/vlnce_r2r_multiview.yaml"
HABITAT_ENV=qwennav_habitat
CONDA_SH=/opt/anaconda3/etc/profile.d/conda.sh
CLIENT_PYTHON="$HOME/.conda/envs/qwennav_model/bin/python"
CHECKPOINT_NAME=$(basename "${MODEL_PATH%/}")

if (( STEP_INTERVAL < 0 )); then
    echo "[multiview_eval] ERROR: STEP_INTERVAL must be >= 0" >&2
    exit 2
fi
case "$REFERENCE_ENABLED" in true|false) ;; *)
    echo "[multiview_eval] ERROR: REFERENCE_ENABLED must be true or false" >&2
    exit 2
esac
if (( STEP_INTERVAL == 0 )) && [[ "$REFERENCE_ENABLED" == false ]]; then
    echo "[multiview_eval] ERROR: both exploration triggers are disabled" >&2
    exit 2
fi

if (( STEP_INTERVAL > 0 )) && [[ "$REFERENCE_ENABLED" == true ]]; then
    CONDITION="interval_${STEP_INTERVAL}_or_reference_thr_${REFERENCE_THRESHOLD_M}_seed_${ORDER_SEED}"
elif (( STEP_INTERVAL > 0 )); then
    CONDITION="interval_${STEP_INTERVAL}_seed_${ORDER_SEED}"
else
    CONDITION="reference_thr_${REFERENCE_THRESHOLD_M}_seed_${ORDER_SEED}"
fi
CONDITION=${CONDITION//./p}
OUTPUT_ROOT="$REPO_ROOT/experiments/multiview_exploration/output/$CHECKPOINT_NAME/r2r/$CONDITION"
LOG_DIR="$OUTPUT_ROOT/logs"
READY_DIR="$OUTPUT_ROOT/.ready"

required_checkpoint_files=(
    "$MODEL_PATH/config.json"
    "$MODEL_PATH/eval_config.json"
    "$MODEL_PATH/model-00001-of-00001.safetensors"
    "$MODEL_PATH/action_tokenizer/manifest.json"
)
for path in "${required_checkpoint_files[@]}"; do
    [[ -f "$path" ]] || { echo "[multiview_eval] ERROR: missing $path" >&2; exit 1; }
done
[[ -x "$CLIENT_PYTHON" ]] || {
    echo "[multiview_eval] ERROR: inference Python not found: $CLIENT_PYTHON" >&2
    exit 1
}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    DETECTED=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
    (( DETECTED > 0 )) || { echo "[multiview_eval] ERROR: no GPUs detected" >&2; exit 1; }
    GPUS=(); for ((i=0; i<DETECTED; i++)); do GPUS+=("$i"); done
fi
for i in "${!GPUS[@]}"; do
    g=${GPUS[$i]//[[:space:]]/}
    [[ "$g" =~ ^[0-9]+$ ]] || { echo "[multiview_eval] ERROR: invalid GPU id: $g" >&2; exit 2; }
    GPUS[$i]=$((10#$g))
done
unset CUDA_VISIBLE_DEVICES
N=${#GPUS[@]}

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "[multiview_eval] ERROR: refusing to overwrite: $OUTPUT_ROOT" >&2
    exit 1
fi
mkdir -p "$LOG_DIR" "$READY_DIR"

set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$HABITAT_ENV"
HABITAT_PYTHON="$CONDA_PREFIX/bin/python"
HABITAT_LD="$CONDA_PREFIX/lib"
conda deactivate
set -u

SERVER_PIDS=()
CLIENT_PIDS=()
PROGRESS_PID=""
cleanup() {
    touch "$READY_DIR/progress.stop" 2>/dev/null || true
    [[ -n "$PROGRESS_PID" ]] && kill "$PROGRESS_PID" 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "[multiview_eval] checkpoint: $MODEL_PATH"
echo "[multiview_eval] GPUs:       ${GPUS[*]}"
echo "[multiview_eval] condition:  $CONDITION"
echo "[multiview_eval] output:     $OUTPUT_ROOT"

for ((i=0; i<N; i++)); do
    g=${GPUS[$i]}
    port=$((BASE_PORT+i))
    log="$LOG_DIR/server_gpu${g}_shard${i}.log"
    HABITAT_SIM_GPU_ID=$g PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT:$REPO_ROOT/habitat_server${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="$HABITAT_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$HABITAT_PYTHON" -m experiments.multiview_exploration.serve \
        --config "$HABITAT_CONFIG" --port "$port" --max-steps "$MAX_STEPS" \
        --split val_unseen --split-id "$i" --split-num "$N" \
        --ready-file "$READY_DIR/shard${i}.ready" >"$log" 2>&1 &
    SERVER_PIDS+=("$!")
done

deadline=$((SECONDS+READY_TIMEOUT_S))
while :; do
    ready=$(find "$READY_DIR" -maxdepth 1 -type f -name 'shard*.ready' | wc -l)
    echo "[multiview_eval] habitat servers ready: $ready/$N"
    (( ready >= N )) && break
    for pid in "${SERVER_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || {
            echo "[multiview_eval] ERROR: a server exited; see $LOG_DIR" >&2
            tail -n 40 "$LOG_DIR"/server_*.log >&2
            exit 1
        }
    done
    (( SECONDS < deadline )) || { echo "[multiview_eval] ERROR: server timeout" >&2; exit 1; }
    sleep 2
done

targets=()
if (( EPISODES > 0 )); then
    for ((i=0; i<N; i++)); do targets+=("$EPISODES"); done
else
    chunk=$((1839/N))
    for ((i=0; i<N; i++)); do
        if (( i < N-1 )); then targets+=("$chunk"); else targets+=("$((1839-chunk*(N-1)))"); fi
    done
fi
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$CLIENT_PYTHON" -m lightnav.cli.eval_progress \
    --output-root "$OUTPUT_ROOT" --targets "${targets[@]}" \
    --stop-file "$READY_DIR/progress.stop" &
PROGRESS_PID=$!

reference_flag=--exploration-reference
[[ "$REFERENCE_ENABLED" == true ]] || reference_flag=--no-exploration-reference
video_args=()
if [[ "$SAVE_VIDEO" == true ]]; then
    video_args=(--save_video --video_dir "$OUTPUT_ROOT/videos" --video_episode_count 1839)
fi
for ((i=0; i<N; i++)); do
    g=${GPUS[$i]}
    port=$((BASE_PORT+i))
    shard_dir="$OUTPUT_ROOT/shard_$i"
    mkdir -p "$shard_dir"
    log="$LOG_DIR/client_gpu${g}_shard${i}.log"
    CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$CLIENT_PYTHON" -m experiments.multiview_exploration.eval \
        --model_path "$MODEL_PATH" --server "tcp://localhost:$port" \
        --backend vllm_local --episodes "$EPISODES" --max_steps "$MAX_STEPS" \
        --output_dir "$shard_dir" --gpu_memory_utilization "$GPU_MEM_UTIL" \
        "${video_args[@]}" \
        --exploration-step-interval "$STEP_INTERVAL" "$reference_flag" \
        --reference-threshold-m "$REFERENCE_THRESHOLD_M" \
        --exploration-order-seed "$ORDER_SEED" >"$log" 2>&1 &
    CLIENT_PIDS+=("$!")
done

failed=0
for pid in "${CLIENT_PIDS[@]}"; do
    wait "$pid" || failed=$((failed+1))
done
touch "$READY_DIR/progress.stop"
wait "$PROGRESS_PID" 2>/dev/null || true
PROGRESS_PID=""
(( failed == 0 )) || { echo "[multiview_eval] ERROR: $failed client shard(s) failed" >&2; exit 1; }

PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$CLIENT_PYTHON" -m experiments.multiview_exploration.merge \
    "$OUTPUT_ROOT"/shard_* --output "$OUTPUT_ROOT" \
    | tee "$OUTPUT_ROOT/merge.log"
echo "[multiview_eval] done: $OUTPUT_ROOT/summary.json"
