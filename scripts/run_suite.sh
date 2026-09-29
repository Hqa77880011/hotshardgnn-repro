#!/usr/bin/env bash
set -euo pipefail
DATA=${1:?Usage: bash scripts/run_suite.sh DATA OUTPUT GPUS}
OUT=${2:?Output directory required}
GPUS=${3:-4}
CONFIG=${CONFIG:-configs/temporal.json}
WINDOW_BATCHES=${WINDOW_BATCHES:-2000}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run() {
  local policy=$1 seed=$2 budget=$3 tag=$4
  torchrun --standalone --nproc_per_node="$GPUS" -m hotshardgnn.train \
    --config "$CONFIG" --data "$DATA" --policy "$policy" --seed "$seed" \
    --budget-bytes "$budget" --window-batches "$WINDOW_BATCHES" --output "$OUT/$tag"
}
for seed in 7 17 27 37 47; do
  run static "$seed" 2147483648 "static-$seed"
  run hotshard "$seed" 2147483648 "hotshard-$seed"
done
for policy in periodic no_forecast cut_only load_only no_penalty no_cooldown random; do
  run "$policy" 7 2147483648 "$policy-7"
done
for budget in 0 65536 1048576 16777216 268435456 1073741824 4294967296; do
  run hotshard 7 "$budget" "budget-$budget"
done
