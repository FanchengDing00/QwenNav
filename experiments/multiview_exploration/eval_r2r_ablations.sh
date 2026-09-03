#!/usr/bin/env bash
# Run the four R2R real-rotation exploration ablations sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if (( $# != 1 )); then
    echo "Usage: $0 CHECKPOINT_DIR" >&2
    exit 2
fi

MODEL_PATH="$1"
EVAL_SCRIPT="$SCRIPT_DIR/eval_r2r.sh"

run_group() {
    local label="$1"
    local suffix="$2"
    shift 2

    echo
    echo "===================================================================="
    echo "[r2r_ablations] group:  $label"
    echo "[r2r_ablations] output: r2r_${suffix}"
    echo "===================================================================="
    "$EVAL_SCRIPT" "$MODEL_PATH" "$suffix" "$@"
}

# 1. Periodic rotation only: first eligible after 20 real actions, then 20 more
#    actions after each completed scan. Navigation and rotation actions both count.
run_group "periodic every 20 actions" "turn20" \
    --action-interval 20 --no-reference --no-initial-360

# 2. Landing rotation only: one real 360-degree scan at episode reset.
run_group "initial 360 only" "turn_init" \
    --action-interval 0 --no-reference --initial-360

# 3. Reference-point rotation only: threshold is eval_r2r.sh's default 0.5 m.
run_group "reference point only" "turn_ref" \
    --action-interval 0 --reference --no-initial-360

# 4. Full exploration: periodic 20 + landing 360 + reference point (OR triggers).
run_group "all rotation triggers" "turn_all" \
    --action-interval 20 --reference --initial-360

echo
echo "[r2r_ablations] all four R2R groups completed."
