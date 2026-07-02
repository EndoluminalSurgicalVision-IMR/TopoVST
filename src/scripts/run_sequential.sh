#!/usr/bin/env bash
# Run train or test configs sequentially within each GPU; GPUs run in parallel.
#
# Activate the right Python environment first (e.g. `conda activate tubular`),
# then edit the Settings block below and run:
#   bash src/scripts/run_sequential.sh
#
# Per-run stdout -> $DATA_ROOT/logs/runner_stdout/<mode>/<run_id>.log
# Per-GPU stdout -> $DATA_ROOT/logs/runner_stdout/<mode>/gpu<N>.out

set -uo pipefail
clear

# --- Settings --------------------------------------------------------------

MODE="test"  # "train" or "test"

# One line per GPU: "<csv_path> <gpu_id>".
# Each GPU works through its CSV sequentially, top to bottom.
ASSIGNMENTS=(
    "src/scripts/configs/test_ablate_cow_alpha_pos_weight_gpu0.csv 0"
    "src/scripts/configs/test_ablate_cow_alpha_pos_weight_gpu1.csv 1"
)

# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="/home/lyyu/data/202605_topovst_rev"
SCRIPT="src/scripts/${MODE}/run_${MODE}.py"
LOG_DIR="$DATA_ROOT/logs/runner_stdout/$MODE"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

run_on_gpu() {
    local csv="$1" gpu="$2"
    mapfile -t ids < <(
        tail -n +2 "$csv" | awk -F',' 'NF>0 && $1 !~ /^[[:space:]]*$/ {print $1}')
    echo "[gpu=$gpu] csv=$csv runs=${#ids[@]}: ${ids[*]}"
    for id in "${ids[@]}"; do
        local log="$LOG_DIR/${id}.log"
        local start; start=$(date +%s)
        echo "[gpu=$gpu] >>> $id (log: $log)"
        if python "$SCRIPT" --config-csv "$csv" --run-id "$id" \
                --device "cuda:$gpu" >"$log" 2>&1; then
            echo "[gpu=$gpu] OK   $id ($(( $(date +%s) - start ))s)"
        else
            echo "[gpu=$gpu] FAIL $id ($(( $(date +%s) - start ))s) -- see $log" >&2
        fi
    done
}

PIDS=()
for assignment in "${ASSIGNMENTS[@]}"; do
    read -r csv gpu <<<"$assignment"
    gpu_log="$LOG_DIR/gpu${gpu}.out"
    echo "[launch] gpu=$gpu csv=$csv -> $gpu_log"
    ( run_on_gpu "$csv" "$gpu" ) >"$gpu_log" 2>&1 &
    PIDS+=($!)
done

trap 'for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait; exit 130' INT TERM

for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "[launch] all GPUs done."
