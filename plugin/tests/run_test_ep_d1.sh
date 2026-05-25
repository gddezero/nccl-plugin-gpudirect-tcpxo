#!/usr/bin/env bash
# test_ep wrapper — supports SKIP_PERF=1 to keep the old --skip-perf-test
# behaviour (correctness only), or unset to run the real dispatch/combine
# perf measurement.
set -euo pipefail
N_LOCAL=${N_LOCAL:-1}
HYBRID=${HYBRID:-1}
NUM_QPS=${NUM_QPS:-0}
NUM_TOKENS=${NUM_TOKENS:-256}
HIDDEN=${HIDDEN:-512}
NUM_TOPK=${NUM_TOPK:-2}
NUM_EXPERTS=${NUM_EXPERTS:-2}
SKIP_PERF=${SKIP_PERF:-0}

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

LOG=/tmp/d1_test_ep_rank${RANK}.log
: > "$LOG"
cd /work/DeepEP

PERF_FLAG=""
if [ "$SKIP_PERF" = "1" ]; then PERF_FLAG="--skip-perf-test"; fi

echo "[run_test_ep_d1] rank=$RANK/$WORLD_SIZE n_local=$N_LOCAL hybrid=$HYBRID qps=$NUM_QPS tokens=$NUM_TOKENS hidden=$HIDDEN topk=$NUM_TOPK experts=$NUM_EXPERTS skip_perf=$SKIP_PERF"
python3 tests/elastic/test_ep.py \
  --num-processes "$N_LOCAL" \
  --allow-hybrid-mode "$HYBRID" \
  --num-allocated-qps "$NUM_QPS" \
  --num-tokens "$NUM_TOKENS" \
  --hidden "$HIDDEN" \
  --num-topk "$NUM_TOPK" \
  --num-experts "$NUM_EXPERTS" \
  $PERF_FLAG 2>&1 | tee -a "$LOG"
