/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Single-rank NCCL initialization test that exercises NCCL >= 2.30.4's
 * GIN plugin probe path. Run with:
 *   NCCL_GIN_PLUGIN=/path/to/libnccl-gin.so \
 *   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL \
 *   ./nccl_init_test
 *
 * Expected: NCCL_DEBUG=INFO logs "NET/FasTrak ... FasTrak GIN provider
 * (PROXY mode) init" and reports our 8 NICs.
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstring>

#define CHECK_CUDA(x)                                                         \
  do {                                                                        \
    cudaError_t e = (x);                                                      \
    if (e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA err %s:%d %s\n", __FILE__, __LINE__,              \
              cudaGetErrorString(e));                                         \
      return 1;                                                               \
    }                                                                         \
  } while (0)

#define CHECK_NCCL(x)                                                         \
  do {                                                                        \
    ncclResult_t r = (x);                                                     \
    if (r != ncclSuccess) {                                                   \
      fprintf(stderr, "NCCL err %s:%d %s\n", __FILE__, __LINE__,              \
              ncclGetErrorString(r));                                         \
      return 1;                                                               \
    }                                                                         \
  } while (0)

int main(int argc, char** argv) {
  fprintf(stderr, "==> CUDA setup\n");
  CHECK_CUDA(cudaSetDevice(0));

  fprintf(stderr, "==> ncclGetUniqueId\n");
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  fprintf(stderr, "==> ncclCommInitRank (nranks=1, rank=0)\n");
  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, 1, id, 0));
  fprintf(stderr, "    comm initialized\n");

  // Tiny allreduce on 16 elements to drive at least one ncclCommSplit / probe.
  float* d = nullptr;
  CHECK_CUDA(cudaMalloc(&d, 16 * sizeof(float)));
  CHECK_CUDA(cudaMemset(d, 0, 16 * sizeof(float)));
  fprintf(stderr, "==> ncclAllReduce 16 floats\n");
  CHECK_NCCL(ncclAllReduce(d, d, 16, ncclFloat, ncclSum, comm, 0));
  CHECK_CUDA(cudaStreamSynchronize(0));
  fprintf(stderr, "    allreduce done\n");

  fprintf(stderr, "==> ncclCommDestroy\n");
  CHECK_NCCL(ncclCommDestroy(comm));
  CHECK_CUDA(cudaFree(d));
  fprintf(stderr, "==> done\n");
  return 0;
}
