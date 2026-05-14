/*
 * Copyright 2026 Google LLC
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE.md file or at
 * https://developers.google.com/open-source/licenses/bsd
 *
 * Two-rank cross-host NCCL test that finally exercises the GIN PROXY
 * data path. Single-mode binary:
 *
 *   <rank> <nranks> <file>
 *     rank 0: ncclGetUniqueId, write to <file>, then ncclCommInitRank
 *             (the listener inside the UniqueId stays alive only while the
 *             generating process is alive, so we cannot split gen and run
 *             across processes)
 *     rank N>0: spin until <file> exists, read UniqueId, ncclCommInitRank
 */

#include <cuda_runtime.h>
#include <fcntl.h>
#include <nccl.h>
#include <unistd.h>

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK(x, msg)                                                          \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA err %s:%d %s\n", __FILE__, __LINE__,               \
              cudaGetErrorString(e));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_NCCL(x)                                                          \
  do {                                                                         \
    ncclResult_t r = (x);                                                      \
    if (r != ncclSuccess) {                                                    \
      fprintf(stderr, "NCCL err %s:%d %s\n", __FILE__, __LINE__,               \
              ncclGetErrorString(r));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int WriteId(const char* path, const ncclUniqueId& id) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  CHECK(fd >= 0, "open(write)");
  ssize_t n = write(fd, &id, sizeof(id));
  CHECK(n == (ssize_t)sizeof(id), "write");
  close(fd);
  fprintf(stderr, "wrote ncclUniqueId to %s (%zu bytes)\n", path, sizeof(id));
  return 0;
}

static int ReadId(const char* path, ncclUniqueId* id) {
  int fd = open(path, O_RDONLY);
  CHECK(fd >= 0, "open(read)");
  ssize_t n = read(fd, id, sizeof(*id));
  CHECK(n == (ssize_t)sizeof(*id), "read");
  close(fd);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <rank> <nranks> <file>\n", argv[0]);
    return 2;
  }
  int rank = atoi(argv[1]);
  int nranks = atoi(argv[2]);
  const char* path = argv[3];

  fprintf(stderr, "[rank %d] CUDA setup\n", rank);
  CHECK_CUDA(cudaSetDevice(0));

  ncclUniqueId id;
  if (rank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&id));
    if (WriteId(path, id) != 0) return 1;
    fprintf(stderr, "[rank 0] generated uniqueId, wrote to %s\n", path);
  } else {
    fprintf(stderr, "[rank %d] waiting for uniqueId file %s\n", rank, path);
    auto deadline = time(nullptr) + 60;
    while (true) {
      int fd = open(path, O_RDONLY);
      if (fd >= 0) { close(fd); break; }
      if (time(nullptr) > deadline) {
        fprintf(stderr, "[rank %d] timeout waiting for %s\n", rank, path);
        return 1;
      }
      usleep(100000);
    }
    CHECK(ReadId(path, &id) == 0, "ReadId");
    fprintf(stderr, "[rank %d] read uniqueId from %s\n", rank, path);
  }

  fprintf(stderr, "[rank %d] ncclCommInitRank (nranks=%d)\n", rank, nranks);
  ncclComm_t comm = nullptr;
  CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));
  fprintf(stderr, "[rank %d] comm init OK\n", rank);

  // Tiny AllReduce on 16 floats. Rank 0 contributes 1.0 at every slot,
  // rank 1 contributes 2.0; sum reduce -> all ranks see 3.0.
  constexpr int kN = 16;
  float* d = nullptr;
  CHECK_CUDA(cudaMalloc(&d, kN * sizeof(float)));
  float h[kN];
  for (int i = 0; i < kN; ++i) h[i] = 1.0f * (rank + 1);
  CHECK_CUDA(cudaMemcpy(d, h, kN * sizeof(float), cudaMemcpyHostToDevice));

  fprintf(stderr, "[rank %d] AllReduce 16 floats\n", rank);
  CHECK_NCCL(ncclAllReduce(d, d, kN, ncclFloat, ncclSum, comm, 0));
  CHECK_CUDA(cudaStreamSynchronize(0));
  CHECK_CUDA(cudaMemcpy(h, d, kN * sizeof(float), cudaMemcpyDeviceToHost));

  fprintf(stderr, "[rank %d] result h[0]=%.2f h[15]=%.2f\n", rank, h[0],
          h[kN - 1]);
  float expected = 0;
  for (int r = 0; r < nranks; ++r) expected += (r + 1);
  bool ok = true;
  for (int i = 0; i < kN; ++i) {
    if (h[i] != expected) {
      fprintf(stderr, "[rank %d] MISMATCH at %d: got %.2f want %.2f\n", rank,
              i, h[i], expected);
      ok = false;
      break;
    }
  }

  CHECK_NCCL(ncclCommDestroy(comm));
  CHECK_CUDA(cudaFree(d));
  fprintf(stderr, "[rank %d] %s\n", rank, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
