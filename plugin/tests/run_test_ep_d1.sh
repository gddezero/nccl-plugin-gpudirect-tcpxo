#!/usr/bin/env bash
set -euo pipefail
N_LOCAL=${N_LOCAL:-1}
HYBRID=${HYBRID:-1}
NUM_QPS=${NUM_QPS:-0}
NUM_TOKENS=${NUM_TOKENS:-256}
HIDDEN=${HIDDEN:-512}
NUM_TOPK=${NUM_TOPK:-2}
NUM_EXPERTS=${NUM_EXPERTS:-2}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET,INIT}
export NCCL_NET_PLUGIN=/work/epv2_gin_plugin/build_deploy/libnccl-net-tcpxo-gin.so
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth1}
export EP_DISABLE_BARRIER_PROFILING=1

LOG=/tmp/d1_test_ep_rank${RANK}.log
: > "$LOG"
cd /work/DeepEP_v2
echo "[run_test_ep_d1] rank=$RANK/$WORLD_SIZE n_local=$N_LOCAL hybrid=$HYBRID qps=$NUM_QPS tokens=$NUM_TOKENS hidden=$HIDDEN topk=$NUM_TOPK experts=$NUM_EXPERTS"
python tests/elastic/test_ep.py \
  --num-processes "$N_LOCAL" \
  --allow-hybrid-mode "$HYBRID" \
  --num-allocated-qps "$NUM_QPS" \
  --num-tokens "$NUM_TOKENS" \
  --hidden "$HIDDEN" \
  --num-topk "$NUM_TOPK" \
  --num-experts "$NUM_EXPERTS" \
  --skip-perf-test 2>&1 | tee -a "$LOG"
