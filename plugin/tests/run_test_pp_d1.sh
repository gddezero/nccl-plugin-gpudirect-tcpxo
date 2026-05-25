#!/usr/bin/env bash
# D1 wrapper for test_pp.py — adapted for fresh-install env (NGC PyTorch 25.04,
# Python 3.12, NCCL 2.30.4 built into /work/nccl_dir, plugin in
# /work/epv2_gin_plugin/plugin/build_deploy/).
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

# PyTorch's bundled NCCL (libnccl.so.2 in /usr/lib/x86_64-linux-gnu/) has
# been replaced with our 2.30.4 build (which has the GIN PROXY ABI). Make
# sure DeepEP's check_nccl_so() finds the matching NCCL root via
# EP_NCCL_ROOT_DIR.
export EP_NCCL_ROOT_DIR=/work/nccl_dir
export LD_LIBRARY_PATH=/work/nccl_dir/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET,INIT}
export NCCL_NET_PLUGIN=/work/epv2_gin_plugin/plugin/build_deploy/libnccl-net-tcpxo-gin.so
export NCCL_IB_DISABLE=1
# Use GPUDirect NIC eth1 (TCP-level), not eth0 control plane.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth1}
export EP_DISABLE_BARRIER_PROFILING=1
# CUDA HW queue slots — default 8, plugin uses ~17 ctxs × stream + user kernels.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
# DeepEP build went to build/lib.linux-x86_64-cpython-312/deep_ep/
export PYTHONPATH=/work/DeepEP/build/lib.linux-x86_64-cpython-312:${PYTHONPATH:-}

LOG=/tmp/d1_test_pp_rank${RANK}.log
: > "$LOG"

cd /work/DeepEP
echo "[run_test_pp_d1] rank=$RANK/$WORLD_SIZE n_local=$N_LOCAL tokens=$TOKENS hidden=$HIDDEN inflight=$INFLIGHT stress=$STRESS sends=$SENDS log=$LOG"

python3 tests/elastic/test_pp.py \
  --num-processes "$N_LOCAL" \
  --num-tokens "$TOKENS" \
  --hidden "$HIDDEN" \
  --num-max-inflight-tensors "$INFLIGHT" \
  --num-stress-iterations "$STRESS" \
  --num-sends "$SENDS" 2>&1 | tee -a "$LOG"
