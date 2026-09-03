#!/usr/bin/env bash
# Independent parallel R2R evaluation for the disposable real-rotation experiment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if (( $# < 1 )); then
    echo "Usage: $0 CHECKPOINT_DIR [R2R_SUFFIX] [--action-interval N] [--reference|--no-reference] [--reference-threshold-m METRES] [--initial-360|--no-initial-360]" >&2
    exit 2
fi

# Experiment variables. The periodic and reference triggers are combined with OR.
MODEL_PATH="$1"
shift
R2R_SUFFIX=""
if (( $# > 0 )) && [[ "$1" != --* ]]; then
    R2R_SUFFIX="$1"
    shift
fi
ACTION_INTERVAL=5            # measured in every env.step; 0 = disabled
REFERENCE_ENABLED=true       # true / false
REFERENCE_THRESHOLD_M=0.5
INITIAL_360_ENABLED=true
ORDER_SEED=0
MAX_STEPS=500
EPISODES=-1
GPU_MEM_UTIL=0.65
BASE_PORT=5655
READY_TIMEOUT_S=900
SAVE_VIDEO=true

while (( $# > 0 )); do
    case "$1" in
        --action-interval)
            (( $# >= 2 )) || { echo "[multiview_eval] ERROR: $1 needs a value" >&2; exit 2; }
            ACTION_INTERVAL="$2"
            shift 2
            ;;
        --reference)
            REFERENCE_ENABLED=true
            shift
            ;;
        --no-reference)
            REFERENCE_ENABLED=false
            shift
            ;;
        --reference-threshold-m)
            (( $# >= 2 )) || { echo "[multiview_eval] ERROR: $1 needs a value" >&2; exit 2; }
            REFERENCE_THRESHOLD_M="$2"
            shift 2
            ;;
        --initial-360)
            INITIAL_360_ENABLED=true
            shift
            ;;
        --no-initial-360)
            INITIAL_360_ENABLED=false
            shift
            ;;
        *)
            echo "[multiview_eval] ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

HABITAT_CONFIG="$REPO_ROOT/experiments/multiview_exploration/configs/vlnce_r2r_multiview.yaml"
HABITAT_ENV=qwennav_habitat
CONDA_SH=/opt/anaconda3/etc/profile.d/conda.sh
CLIENT_PYTHON="$HOME/.conda/envs/qwennav_model/bin/python"
CHECKPOINT_NAME="$(basename "${MODEL_PATH%/}")_mv"

if ! [[ "$ACTION_INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "[multiview_eval] ERROR: ACTION_INTERVAL must be a non-negative integer" >&2
    exit 2
fi
ACTION_INTERVAL=$((10#$ACTION_INTERVAL))
case "$REFERENCE_ENABLED" in true|false) ;; *)
    echo "[multiview_eval] ERROR: REFERENCE_ENABLED must be true or false" >&2
    exit 2
esac
case "$INITIAL_360_ENABLED" in true|false) ;; *)
    echo "[multiview_eval] ERROR: INITIAL_360_ENABLED must be true or false" >&2
    exit 2
esac
if ! [[ "$REFERENCE_THRESHOLD_M" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || [[ "$REFERENCE_THRESHOLD_M" =~ ^0*([.]0*)?$ ]]; then
    echo "[multiview_eval] ERROR: reference threshold must be a positive number" >&2
    exit 2
fi
if (( ACTION_INTERVAL == 0 )) && [[ "$REFERENCE_ENABLED" == false ]] && [[ "$INITIAL_360_ENABLED" == false ]]; then
    echo "[multiview_eval] ERROR: all exploration features are disabled" >&2
    exit 2
fi

if [[ -n "$R2R_SUFFIX" ]] && ! [[ "$R2R_SUFFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[multiview_eval] ERROR: R2R_SUFFIX may contain only letters, digits, '.', '_' and '-'" >&2
    exit 2
fi
R2R_DIR="r2r"
[[ -n "$R2R_SUFFIX" ]] && R2R_DIR="r2r_${R2R_SUFFIX}"

OUTPUT_ROOT="$REPO_ROOT/experiments/multiview_exploration/output/$CHECKPOINT_NAME/$R2R_DIR"
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

"$CLIENT_PYTHON" -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = {
    "experiment": "real_rotation_exploration",
    "benchmark": "r2r",
    "checkpoint": sys.argv[2],
    "result_directory": sys.argv[3],
    "action_interval": int(sys.argv[4]),
    "reference_enabled": sys.argv[5] == "true",
    "reference_threshold_m": float(sys.argv[6]),
    "initial_360_enabled": sys.argv[7] == "true",
    "order_seed": int(sys.argv[8]),
    "max_steps": int(sys.argv[9]),
    "episodes_per_shard": int(sys.argv[10]),
    "save_video": sys.argv[11] == "true",
}
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
' "$OUTPUT_ROOT/config.json" "$MODEL_PATH" "$R2R_DIR" "$ACTION_INTERVAL" \
    "$REFERENCE_ENABLED" "$REFERENCE_THRESHOLD_M" "$INITIAL_360_ENABLED" \
    "$ORDER_SEED" "$MAX_STEPS" "$EPISODES" "$SAVE_VIDEO"

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
echo "[multiview_eval] dataset dir: $R2R_DIR"
echo "[multiview_eval] interval:   $ACTION_INTERVAL actions"
echo "[multiview_eval] reference:  $REFERENCE_ENABLED"
echo "[multiview_eval] initial360: $INITIAL_360_ENABLED"
echo "[multiview_eval] ref radius: $REFERENCE_THRESHOLD_M m"
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

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$CLIENT_PYTHON" -m lightnav.cli.eval_progress \
    --output-root "$OUTPUT_ROOT" \
    --num-shards "$N" \
    --total-episodes 1839 \
    --per-shard-limit "$EPISODES" \
    --stop-file "$READY_DIR/progress.stop" &
PROGRESS_PID=$!

reference_flag=--exploration-reference
[[ "$REFERENCE_ENABLED" == true ]] || reference_flag=--no-exploration-reference
initial_flag=--initial-360
[[ "$INITIAL_360_ENABLED" == true ]] || initial_flag=--no-initial-360
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
        --exploration-action-interval "$ACTION_INTERVAL" "$reference_flag" \
        --reference-threshold-m "$REFERENCE_THRESHOLD_M" \
        "$initial_flag" \
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
