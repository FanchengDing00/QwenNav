#!/usr/bin/env bash
# Run the official LightNav-0 RxR val_unseen guide evaluation (English only) with
# this repository's local environments and checkpoint layout.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if (( $# != 1 )); then
    echo "Usage: $0 CHECKPOINT_DIR" >&2
    exit 2
fi

# The checkpoint is the only command-line input. Adjust the remaining evaluation
# settings here; they stay local to this script and are passed to eval_habitat.sh below.
MODEL_PATH="$1"
CONDA_SH=/opt/anaconda3/etc/profile.d/conda.sh
HABITAT_CONDA_ENV=qwennav_habitat
INFER_VENV="$HOME/.conda/envs/qwennav_model"
TASK=vlnce
HABITAT_CONFIG="$REPO_ROOT/habitat_server/configs/vlnce_rxr.yaml"
SPLIT=val_unseen
LANGUAGES="en-US en-IN"
EPISODES=-1
MAX_STEPS=500
BACKEND=vllm_local
GPU_MEM_UTIL=0.65
BASE_PORT=5555
READY_TIMEOUT_S=900
CLIENT_ARGS="--save_video"

# Fail early with actionable errors instead of waiting for worker logs.
required_checkpoint_files=(
    "$MODEL_PATH/config.json"
    "$MODEL_PATH/eval_config.json"
    "$MODEL_PATH/model-00001-of-00001.safetensors"
    "$MODEL_PATH/action_tokenizer/manifest.json"
)
for path in "${required_checkpoint_files[@]}"; do
    [[ -f "$path" ]] || {
        echo "[eval_rxr] ERROR: checkpoint file not found: $path" >&2
        exit 1
    }
done

[[ -x "$INFER_VENV/bin/python" ]] || {
    echo "[eval_rxr] ERROR: model environment not found: $INFER_VENV" >&2
    exit 1
}

required_data_files=(
    "$REPO_ROOT/data/datasets/RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz"
    "$REPO_ROOT/data/datasets/RxR_VLNCE_v0/val_unseen/val_unseen_guide_gt.json.gz"
)
for path in "${required_data_files[@]}"; do
    [[ -f "$path" ]] || {
        echo "[eval_rxr] ERROR: RxR file not found: $path" >&2
        exit 1
    }
done

[[ -d "$REPO_ROOT/data/scene_datasets/mp3d" ]] || {
    echo "[eval_rxr] ERROR: MP3D scenes not found under data/scene_datasets/mp3d" >&2
    exit 1
}

echo "[eval_rxr] model:          $MODEL_PATH"
echo "[eval_rxr] habitat env:    $HABITAT_CONDA_ENV"
echo "[eval_rxr] inference env:  $INFER_VENV"
echo "[eval_rxr] split:          $SPLIT guide (languages=$LANGUAGES, episodes=3669)"
echo "[eval_rxr] max steps:      $MAX_STEPS"
echo "[eval_rxr] GPUs:           all detected GPUs"
echo "[eval_rxr] output:         determined by eval_habitat.sh"
echo "[eval_rxr] videos:         $REPO_ROOT/output/$(basename "${MODEL_PATH%/}")/habitat_vlnce/rxr/videos/"

# Pass this script's settings only to the evaluator process. Remove inherited optional
# overrides so GPU discovery and the fixed RxR configuration remain deterministic.
exec env \
    -u GPU_IDS -u NUM_GPUS -u OUTPUT_ROOT -u LOG_DIR -u CUDA_VISIBLE_DEVICES \
    -u ACTION_TOKENIZER_BUNDLE -u SUCCESS_DISTANCE -u DATA_PATH -u SCENES_DIR \
    -u SERVER_ARGS -u CLIENT_PYTHON -u HABITAT_PYTHON \
    MODEL_PATH="$MODEL_PATH" \
    CONDA_SH="$CONDA_SH" \
    HABITAT_CONDA_ENV="$HABITAT_CONDA_ENV" \
    INFER_VENV="$INFER_VENV" \
    TASK="$TASK" \
    HABITAT_CONFIG="$HABITAT_CONFIG" \
    SPLIT="$SPLIT" \
    LANGUAGES="$LANGUAGES" \
    EPISODES="$EPISODES" \
    MAX_STEPS="$MAX_STEPS" \
    BACKEND="$BACKEND" \
    GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    BASE_PORT="$BASE_PORT" \
    READY_TIMEOUT_S="$READY_TIMEOUT_S" \
    CLIENT_ARGS="$CLIENT_ARGS" \
    bash "$REPO_ROOT/scripts/eval/eval_habitat.sh"
