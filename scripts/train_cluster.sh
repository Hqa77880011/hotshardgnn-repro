#!/usr/bin/env bash
set -euo pipefail
: "${NNODES:?Set number of servers}"
: "${NODE_RANK:?Set this server rank, starting at zero}"
: "${MASTER_ADDR:?Set rank-zero server's reachable address}"
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS_PER_NODE" --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" -m hotshardgnn.train "$@"
