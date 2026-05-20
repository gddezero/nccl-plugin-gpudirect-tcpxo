#!/usr/bin/env bash
set -euo pipefail
N_LOCAL=${N_LOCAL:-1}
INFLIGHT=${INFLIGHT:-4}
STRESS=${STRESS:-1}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET,INIT}
export NCCL_NET_PLUGIN=/work/epv2_gin_plugin/build_deploy/libnccl-net-tcpxo-gin.so
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth1}
export EP_DISABLE_BARRIER_PROFILING=1

LOG=/tmp/d1_test_agrs_rank${RANK}.log
: > "$LOG"
cd /work/DeepEP_v2
echo "[run_test_agrs_d1] rank=$RANK/$WORLD_SIZE n_local=$N_LOCAL inflight=$INFLIGHT stress=$STRESS"
python tests/elastic/test_agrs.py \
  --num-processes "$N_LOCAL" \
  --num-max-inflight-agrs "$INFLIGHT" \
  --num-stress-iterations "$STRESS" 2>&1 | tee -a "$LOG"
