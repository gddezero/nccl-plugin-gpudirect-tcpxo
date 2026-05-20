#!/usr/bin/env bash
# D1 wrapper for test_barrier.py
# Coordinated by caller env: RANK WORLD_SIZE MASTER_ADDR MASTER_PORT N_LOCAL
set -euo pipefail

N_LOCAL=${N_LOCAL:-1}
NUM_QPS=${NUM_QPS:-1}
HYBRID=${HYBRID:-0}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET,INIT}
export NCCL_NET_PLUGIN=/work/epv2_gin_plugin/build_deploy/libnccl-net-tcpxo-gin.so
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth1}
export EP_DISABLE_BARRIER_PROFILING=1

LOG=/tmp/d1_test_barrier_rank${RANK}.log
: > "$LOG"

cd /work/DeepEP_v2
echo "[run_test_barrier_d1] rank=$RANK/$WORLD_SIZE n_local=$N_LOCAL qps=$NUM_QPS hybrid=$HYBRID log=$LOG"

python tests/elastic/test_barrier.py \
  --num-processes "$N_LOCAL" \
  --num-allocated-qps "$NUM_QPS" \
  --allow-hybrid-mode "$HYBRID" 2>&1 | tee -a "$LOG"
