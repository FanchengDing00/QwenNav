#!/usr/bin/env bash
# Run the full R2R evaluation first, then run the full RxR evaluation with the
# same checkpoint. RxR starts only after R2R exits successfully.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if (( $# != 1 )); then
    echo "Usage: $0 CHECKPOINT_DIR" >&2
    exit 2
fi

MODEL_PATH="$1"

echo "[eval_r2r_rxr] checkpoint: $MODEL_PATH"
echo "[eval_r2r_rxr] stage 1/2: R2R"
"$REPO_ROOT/scripts/eval/eval_r2r.sh" "$MODEL_PATH"

echo "[eval_r2r_rxr] stage 2/2: RxR"
"$REPO_ROOT/scripts/eval/eval_rxr.sh" "$MODEL_PATH"

echo "[eval_r2r_rxr] all evaluations completed"
