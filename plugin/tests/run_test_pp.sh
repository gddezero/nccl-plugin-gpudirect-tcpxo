#!/usr/bin/env bash
# Launch EPv2 test_pp.py on this node. Coordinated by env from caller:
#   RANK         : node rank (0=master, 1=worker)
#   WORLD_SIZE   : number of NODES (2)
#   MASTER_ADDR  : master node IP
#   MASTER_PORT  : tcp port for torch dist init
#   N_LOCAL      : local ranks per node (default 1 for first run)
#   TOKENS HIDDEN INFLIGHT STRESS SENDS : EPv2 test_pp.py knobs
set -euo pipefail

N_LOCAL=${N_LOCAL:-1}
TOKENS=${TOKENS:-64}
HIDDEN=${HIDDEN:-128}
INFLIGHT=${INFLIGHT:-2}
STRESS=${STRESS:-1}
SENDS=${SENDS:-4}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_NET_PLUGIN=/work/epv2_gin_plugin/build_deploy/libnccl-net-tcpxo-gin.so
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export EP_DISABLE_BARRIER_PROFILING=1

cd /work/DeepEP_v2
echo "[run_test_pp] rank=$RANK/$WORLD_SIZE n_local=$N_LOCAL tokens=$TOKENS hidden=$HIDDEN inflight=$INFLIGHT stress=$STRESS sends=$SENDS"
python tests/elastic/test_pp.py \
  --num-processes "$N_LOCAL" \
  --num-tokens "$TOKENS" \
  --hidden "$HIDDEN" \
  --num-max-inflight-tensors "$INFLIGHT" \
  --num-stress-iterations "$STRESS" \
  --num-sends "$SENDS"
