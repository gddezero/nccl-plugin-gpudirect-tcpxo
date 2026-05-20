#!/usr/bin/env bash
# D1 wrapper for test_pp.py: runs with NCCL_DEBUG=INFO + plugin from /work/epv2_gin_plugin/build_deploy,
# and pipes stderr to a node-local log so we can grep [D1] markers post-run.
#
# Caller env (same as run_test_pp.sh):
#   RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
#   N_LOCAL TOKENS HIDDEN INFLIGHT STRESS SENDS
set -euo pipefail

N_LOCAL=${N_LOCAL:-1}
TOKENS=${TOKENS:-64}
HIDDEN=${HIDDEN:-128}
# Profiling iterates concurrent ∈ (1,2,3), so buffer must support ≥3 inflight.
INFLIGHT=${INFLIGHT:-4}
STRESS=${STRESS:-1}
SENDS=${SENDS:-4}

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET,INIT}
export NCCL_NET_PLUGIN=/work/epv2_gin_plugin/build_deploy/libnccl-net-tcpxo-gin.so
export NCCL_IB_DISABLE=1
# Use GPUDirect NIC eth1 (TCP-level, not yet libfastrak) instead of eth0
# control plane. eth0 single-stream caps at ~1-5 Gbps; eth1 is 200 Gbps fabric.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth1}
export EP_DISABLE_BARRIER_PROFILING=1
# Increase CUDA HW queue slots — default is 8, our plugin uses 17 ctxs × stream + user kernels.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}

LOG=/tmp/d1_test_pp_rank${RANK}.log
: > "$LOG"

cd /work/DeepEP_v2
echo "[run_test_pp_d1] rank=$RANK/$WORLD_SIZE n_local=$N_LOCAL tokens=$TOKENS hidden=$HIDDEN inflight=$INFLIGHT stress=$STRESS sends=$SENDS log=$LOG"

python tests/elastic/test_pp.py \
  --num-processes "$N_LOCAL" \
  --num-tokens "$TOKENS" \
  --hidden "$HIDDEN" \
  --num-max-inflight-tensors "$INFLIGHT" \
  --num-stress-iterations "$STRESS" \
  --num-sends "$SENDS" 2>&1 | tee -a "$LOG"
