#!/usr/bin/env bash
set -euo pipefail
N_LOCAL=${N_LOCAL:-1}
NUM_QPS=${NUM_QPS:-1}
HYBRID=${HYBRID:-0}

export EP_NCCL_ROOT_DIR=/work/nccl_dir
export LD_LIBRARY_PATH=/work/nccl_dir/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}
export EP_RDMA_GBS=${EP_RDMA_GBS:-25}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-NET,INIT}
export NCCL_NET_PLUGIN=/work/epv2_gin_plugin/plugin/build_deploy/libnccl-net-tcpxo-gin.so
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth1}
export EP_DISABLE_BARRIER_PROFILING=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export PYTHONPATH=/work/DeepEP/build/lib.linux-x86_64-cpython-312:${PYTHONPATH:-}

LOG=/tmp/d1_test_barrier_rank${RANK}.log
: > "$LOG"

cd /work/DeepEP
echo "[run_test_barrier_d1] rank=$RANK/$WORLD_SIZE n_local=$N_LOCAL qps=$NUM_QPS hybrid=$HYBRID log=$LOG"

python3 tests/elastic/test_barrier.py \
  --num-processes "$N_LOCAL" \
  --num-allocated-qps "$NUM_QPS" \
  --allow-hybrid-mode "$HYBRID" 2>&1 | tee -a "$LOG"
